"""Fused SiLU + multiply + UE8M0 FP8 quantize kernel for masked grouped MoE.

Vendored verbatim (trimmed to UE8M0-only path) from:
  submodules/vllm/vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py
  lines 40-268
  vllm-project/vllm @ 12a3f6454b973d7cd8806d398ba287a7e1d22c63

Why vendored:
  Phase 5 replaces minisgl's prior 3-op (silu_and_mul + Python-loop FP8 quantize +
  manual UE8M0 pack) with vLLM's fused path. The fused path is GPU-resident,
  avoids an intermediate BF16 HBM round-trip, and is cudagraph-safe.
"""
from __future__ import annotations

from importlib.util import find_spec

import torch
import triton
import triton.language as tl


def cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def scales_shape_stride_dtype(
    E: int, T: int, G: int
) -> tuple[tuple[int, ...], tuple[int, ...], torch.dtype]:
    shape = (E, T, cdiv(G, 4))
    strides = (T * cdiv(G, 4), 1, T)
    return shape, strides, torch.int32


@triton.jit
def _silu_mul_fp8_quant_psum_deep_gemm(
    input_ptr,  # 16-bit activations (M, 2*H), expert-psum layout
    topk_weights_ptr,  # optional expanded top-k weights, one per row
    y_q_ptr,  # fp8 quantized activations (M, H)
    y_s_ptr,  # packed UE8M0 scales (M, ceil(G / 4))
    psum_ptr,  # int32 end offset per expert (E)
    H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    stride_i_m,
    stride_i_h,
    stride_yq_m,
    stride_yq_h,
    stride_ys_m,
    stride_ys_g,
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    APPLY_TOPK_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE
    PACKED_G = (G + 3) // 4
    pid = tl.program_id(0)
    e = pid // PACKED_G
    pg = pid % PACKED_G

    e = e.to(tl.int64)
    pg = pg.to(tl.int64)
    prev_end = tl.where(e == 0, 0, tl.load(psum_ptr + e - 1)).to(tl.int64)
    start = ((prev_end + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    end = tl.load(psum_ptr + e).to(tl.int64)
    n_tokens = end - start

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        packed_scale = tl.full((), 0, tl.int32)
        for lane in tl.static_range(0, 4):
            g = pg * 4 + lane
            valid_g = g < G
            g_safe = tl.minimum(g, G - 1)
            base_input_offset = start * stride_i_m + g_safe * GROUP_SIZE * stride_i_h
            base_gate_offset = base_input_offset + cols * stride_i_h
            base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
            base_yq_offset = (
                start * stride_yq_m
                + g_safe * GROUP_SIZE * stride_yq_h
                + cols * stride_yq_h
            )

            gate = tl.load(
                input_ptr + base_gate_offset + t * stride_i_m,
                mask=valid_g & mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                input_ptr + base_up_offset + t * stride_i_m,
                mask=valid_g & mask,
                other=0.0,
            )
            gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
            y = gate * up
            if APPLY_TOPK_WEIGHT:
                weight = tl.load(topk_weights_ptr + start + t).to(tl.float32)
                y = y * weight
            y_s = tl.maximum(tl.max(tl.abs(y)), eps) * (1.0 / fp8_max)
            if ceil_ue8m0:
                scale_exp = tl.ceil(tl.log2(y_s)) + 127.0
                scale_exp = tl.minimum(tl.maximum(scale_exp, 1.0), 254.0).to(tl.int32)
                y_s = tl.exp2(scale_exp.to(tl.float32) - 127.0)
            else:
                scale_exp = tl.full((), 0, tl.int32)
            y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
            tl.store(
                y_q_ptr + base_yq_offset + t * stride_yq_m,
                y_q,
                mask=valid_g & mask,
            )
            packed_scale = packed_scale | (
                tl.where(valid_g, scale_exp, 0).to(tl.int32) << (lane * 8)
            )
        tl.store(y_s_ptr + (start + t) * stride_ys_m + pg * stride_ys_g, packed_scale)


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        # Use multiply-by-reciprocal to match PyTorch's tensor/scalar
        # division precision (Triton GPU fast-division for constexpr
        # divisors can introduce 1-ULP error).
        y_s = tl.maximum(tl.max(tl.abs(y)), eps) * (1.0 / fp8_max)
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def _ensure_vllm_extension_loaded() -> None:
    if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "persistent_masked_m_silu_mul_quant"):
        return
    spec = find_spec("vllm._C")
    if spec is None or spec.origin is None:
        raise RuntimeError("vLLM extension vllm._C is required for fused UE8M0 quant")
    torch.ops.load_library(spec.origin)


def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    group_size: int = 128,
    num_parallel_tokens: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with UE8M0 scales."""
    del num_parallel_tokens
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    y_q = torch.empty((E, T, H), dtype=torch.float8_e4m3fn, device=y.device)

    ys_shape, ys_strides, ys_dtype = scales_shape_stride_dtype(E, T, G)
    y_s = torch.empty_strided(
        ys_shape,
        ys_strides,
        dtype=ys_dtype,
        device=y.device,
    )

    device_capability = torch.cuda.get_device_capability(y.device)
    cuda_arch = device_capability[0] * 10 + device_capability[1]
    if cuda_arch < 80:
        raise RuntimeError("persistent_masked_m_silu_mul_quant requires CUDA arch >= 80")

    _ensure_vllm_extension_loaded()
    torch.ops._C.persistent_masked_m_silu_mul_quant(
        y, tokens_per_expert, y_q, y_s, True
    )
    return y_q, y_s


def persistent_psum_silu_mul_quant(
    y: torch.Tensor,
    psum_tokens_per_expert: torch.Tensor,
    *,
    alignment: int,
    topk_weights: torch.Tensor | None = None,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] for DeepGEMM psum layout."""
    assert y.ndim == 2, "y must be (M, 2*H)"
    from minisgl.kernel.fp8_quant import psum_silu_mul_fp8_quant_cuda

    return psum_silu_mul_fp8_quant_cuda(
        y,
        psum_tokens_per_expert,
        alignment=alignment,
        topk_weights=topk_weights,
        group_size=group_size,
    )


__all__ = ["persistent_masked_m_silu_mul_quant", "persistent_psum_silu_mul_quant"]
