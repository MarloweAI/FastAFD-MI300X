from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl


FP8_MAX = 448.0


def _ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _align(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float or x.size(-1) % 4 != 0:
        raise RuntimeError(
            "UE8M0 packing expects float scales with last dim divisible by 4, "
            f"got {tuple(x.shape)} {x.dtype}"
        )
    if not bool((x.view(torch.int) & ((1 << 23) - 1) == 0).all().item()):
        raise RuntimeError("UE8M0 packing expects exponent-only float scales")
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def per_token_cast_to_fp8_torch(
    x: torch.Tensor,
    *,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise RuntimeError(f"per_token_cast_to_fp8 expects 2D tensor, got {x.shape}")
    m, n = x.shape
    padded_n = _align(n, gran_k)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / FP8_MAX
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (
        (x_view * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_fp8, pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


@triton.jit
def _per_token_cast_to_fp8_triton_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    n_cols: tl.constexpr,
    groups: tl.constexpr,
    x_stride_m: tl.constexpr,
    y_stride_m: tl.constexpr,
    s_stride_m: tl.constexpr,
    use_ue8m0: tl.constexpr,
    gran_k: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, gran_k).to(tl.int64)
    col = group * gran_k + cols
    mask = col < n_cols
    x = tl.load(x_ptr + row * x_stride_m + col, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-4)
    scale = amax * (1.0 / 448.0)
    if use_ue8m0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    y = tl.clamp(x / scale, -448.0, 448.0).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * y_stride_m + col, y, mask=mask)
    tl.store(s_ptr + row * s_stride_m + group, scale)


def per_token_cast_to_fp8_triton(
    x: torch.Tensor,
    *,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise RuntimeError(f"per_token_cast_to_fp8_triton expects 2D tensor, got {x.shape}")
    if gran_k <= 0 or gran_k > 1024:
        raise RuntimeError(f"unsupported gran_k={gran_k}")
    if not x.is_contiguous():
        x = x.contiguous()
    m, n = x.shape
    groups = _ceil_div(n, gran_k)
    y = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, groups), dtype=torch.float32, device=x.device)
    _per_token_cast_to_fp8_triton_kernel[(m, groups)](
        x,
        y,
        s,
        n,
        groups,
        x.stride(0),
        y.stride(0),
        s.stride(0),
        bool(use_ue8m0),
        gran_k,
        num_warps=4,
    )
    return y, pack_ue8m0_to_int(s) if use_packed_ue8m0 else s


@functools.cache
def _jit_fp8_to_uint8():
    """JIT-load the FP8-to-uint8 raw-byte copy kernel."""
    from .utils import load_jit

    return load_jit(
        "fp8_to_uint8",
        cuda_files=["fp8_to_uint8.cu"],
        cuda_wrappers=[("launch", "Fp8ToUint8Kernel::run")],
    )


@functools.cache
def _jit_add_rmsnorm_fp8_quant():
    """JIT-load the fused add-RMSNorm + per-128 UE8M0 FP8 quant kernel."""
    from .utils import load_jit

    return load_jit(
        "add_rmsnorm_fp8_quant",
        cuda_files=["add_rmsnorm_fp8_quant.cu"],
        cuda_wrappers=[("launch", "AddRMSNormFp8QuantKernel::run")],
    )


def fused_add_rmsnorm_fp8_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused (residual-add) RMSNorm + per-128-group packed-UE8M0 FP8 quant.

    In-place updates ``residual`` to ``x + residual`` (bf16) and returns the
    normed activations as FP8 e4m3 + packed UE8M0 scales (DeepGEMM TMA layout,
    identical to ``per_token_cast_to_fp8_cuda``), replacing the
    ``fused_add_rmsnorm`` + ``per_token_cast`` two-kernel chain for the decode
    attention path.  Requires H % 512 == 0 and H <= 4096.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
    assert residual.shape == x.shape and residual.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    m, n = x.shape
    groups = _ceil_div(n, 128)
    packed_groups = _ceil_div(groups, 4)
    y = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty_strided(
        (m, packed_groups), (1, _align(m, 4)), dtype=torch.int32, device=x.device
    )
    _jit_add_rmsnorm_fp8_quant().launch(
        residual, x.contiguous(), weight.contiguous(),
        y.view(torch.uint8), s, float(eps),
    )
    return y, s


@functools.cache
def _jit_per_token_cast_fp8_packed():
    """JIT-load the unified packed-UE8M0 FP8 quant kernel + manual-dispatch helper."""
    from .utils import load_jit

    return load_jit(
        "per_token_fp8_packed_unified",
        cuda_files=["per_token_fp8_packed.cu"],
        cuda_wrappers=[
            ("launch", "PerTokenCastFp8PackedKernel::run"),
            ("launch_manual", "PerTokenCastFp8PackedKernelManual::run"),
        ],
    )


@functools.cache
def _jit_psum_silu_mul_fp8_packed():
    """JIT-load the unified psum-layout silu+mul+FP8-packed quant kernel."""
    from .utils import load_jit

    return load_jit(
        "psum_silu_mul_fp8_packed_unified",
        cuda_files=["psum_silu_mul_fp8_packed.cu"],
        cuda_wrappers=[
            ("launch", "PsumSiluMulFp8PackedKernel::run"),
            ("launch_manual", "PsumSiluMulFp8PackedKernelManual::run"),
        ],
        # Match legacy fp8_quant build: use fast_math so expf in silu uses the
        # same hardware-approximate path as the reference (avoids 1-ulp drift
        # that occasionally pushes fp8 quant across a code boundary).
        extra_cuda_cflags=["--use_fast_math"],
    )


def per_token_cast_to_fp8_cuda(
    x: torch.Tensor,
    *,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = True,
    manual_config: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 quantization with packed UE8M0 scales (gran_k=128 only).

    Routes through the single unified JIT CUDA kernel in
    `python/minisgl/kernel/csrc/jit/per_token_fp8_packed.cu`, which covers both
    the vLLM register-resident design (THREADS_PER_GROUP=8, 2x uint4 LDG) and
    the minisgl-v1 warp-per-group design (THREADS_PER_GROUP=32). The runtime
    auto-picks (TPG, KX, RY) for best SM utilization on the current shape and
    uses sglang/DeepEP-style `__nv_cvt_float2_to_fp8x2(__NV_SATFINITE,
    __NV_E4M3)` PTX intrinsic for the quant store (1 instr / 2 elements).

    The non-packed (fp32 per-group scale) CUDA path was removed; for non-packed
    output use `per_token_cast_to_fp8_torch` or `per_token_cast_to_fp8_triton`.

    Args:
        manual_config: (threads_per_group, groups_per_block_x, rows_per_block)
            overrides the auto-dispatcher (debug / bench only).
    """
    if x.dim() != 2:
        raise RuntimeError(f"per_token_cast_to_fp8_cuda expects 2D tensor, got {x.shape}")
    if not x.is_cuda:
        raise RuntimeError("per_token_cast_to_fp8_cuda expects a CUDA tensor")
    if not use_packed_ue8m0:
        raise RuntimeError(
            "per_token_cast_to_fp8_cuda only supports packed UE8M0 scales "
            "(use_packed_ue8m0=True). For non-packed fp32 scales use the "
            "torch or triton backends."
        )
    if not use_ue8m0:
        raise RuntimeError("packed FP8 scales require use_ue8m0=True")
    if int(gran_k) != 128:
        raise RuntimeError("unified packed-UE8M0 kernel requires gran_k=128")
    if not x.is_contiguous():
        x = x.contiguous()
    m, n = x.shape
    groups = _ceil_div(n, gran_k)
    packed_groups = _ceil_div(groups, 4)
    y = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    # Match DeepGEMM/vLLM's TMA-friendly packed-scale layout.
    s = torch.empty_strided(
        (m, packed_groups),
        (1, _align(m, 4)),
        dtype=torch.int32,
        device=x.device,
    )
    module = _jit_per_token_cast_fp8_packed()
    if manual_config is not None:
        tpg, kx, ry = manual_config
        module.launch_manual(x, y.view(torch.uint8), s, int(tpg), int(kx), int(ry))
    else:
        module.launch(x, y.view(torch.uint8), s)
    return y, s


def psum_silu_mul_fp8_quant_cuda(
    x: torch.Tensor,
    psum_tokens_per_expert: torch.Tensor,
    *,
    alignment: int,
    topk_weights: torch.Tensor | None = None,
    group_size: int = 128,
    manual_config: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MoE psum-layout silu+mul + FP8 (UE8M0 packed) quantization.

    Computes y[m, h] = quantize_fp8(silu(x[m, :h]) * x[m, h:] * (topk if any))
    with per-group=128 UE8M0 scales packed into int32 (DeepGEMM TMA layout).
    Routes through the unified JIT kernel in
    `python/minisgl/kernel/csrc/jit/psum_silu_mul_fp8_packed.cu`, sharing the
    same (TPG, KX, RY) auto-dispatch family as per_token_cast_to_fp8_cuda.
    """
    if x.dim() != 2:
        raise RuntimeError(f"psum_silu_mul_fp8_quant_cuda expects 2D tensor, got {x.shape}")
    if not x.is_cuda:
        raise RuntimeError("psum_silu_mul_fp8_quant_cuda expects a CUDA tensor")
    if not x.is_contiguous():
        x = x.contiguous()
    m, h2 = x.shape
    if h2 % 2 != 0:
        raise RuntimeError(f"last dim must be even, got {h2}")
    h = h2 // 2
    if int(group_size) != 128:
        raise RuntimeError(f"only group_size=128 supported, got {group_size}")
    if h % group_size != 0:
        raise RuntimeError(f"hidden={h} must be divisible by group_size={group_size}")
    packed_groups = _ceil_div(h // group_size, 4)
    if m == 0:
        return (
            torch.empty((0, h), dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty_strided(
                (0, packed_groups),
                (1, 0),
                dtype=torch.int32,
                device=x.device,
            ),
        )
    psum = psum_tokens_per_expert.to(device=x.device, dtype=torch.int32)
    if not psum.is_contiguous():
        psum = psum.contiguous()
    if topk_weights is None:
        weights = torch.empty((1,), dtype=torch.float32, device=x.device)
        apply_topk = False
    else:
        weights = topk_weights.to(device=x.device, dtype=torch.float32).reshape(-1)
        if weights.numel() < m:
            raise RuntimeError(
                f"topk_weights too small: weights={tuple(weights.shape)} rows={m}"
            )
        if not weights.is_contiguous():
            weights = weights.contiguous()
        apply_topk = True
    y = torch.empty((m, h), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty_strided(
        (m, packed_groups),
        (1, _align(m, 4)),
        dtype=torch.int32,
        device=x.device,
    )
    module = _jit_psum_silu_mul_fp8_packed()
    if manual_config is not None:
        tpg, kx, ry = manual_config
        module.launch_manual(
            x, y.view(torch.uint8), scales, psum, weights,
            int(alignment), bool(apply_topk), int(tpg), int(kx), int(ry),
        )
    else:
        module.launch(
            x, y.view(torch.uint8), scales, psum, weights,
            int(alignment), bool(apply_topk),
        )
    return y, scales


def fp8_to_uint8_cuda(x: torch.Tensor) -> torch.Tensor:
    """Copy FP8 e4m3 raw bytes into a uint8 tensor (for inspection / cpu xfer)."""
    if x.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"fp8_to_uint8_cuda expects float8_e4m3fn, got {x.dtype}")
    if not x.is_cuda:
        raise RuntimeError("fp8_to_uint8_cuda expects a CUDA tensor")
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    module = _jit_fp8_to_uint8()
    module.launch(x.view(torch.uint8).reshape(-1), y.reshape(-1))
    return y


def per_token_cast_to_fp8(
    x: torch.Tensor,
    *,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    backend = backend or os.environ.get(
        "MINISGL_FP8_QUANT_BACKEND",
        "cuda" if x.is_cuda else "torch",
    )
    if backend == "torch":
        return per_token_cast_to_fp8_torch(
            x,
            use_ue8m0=use_ue8m0,
            gran_k=gran_k,
            use_packed_ue8m0=use_packed_ue8m0,
        )
    if backend == "triton":
        return per_token_cast_to_fp8_triton(
            x,
            use_ue8m0=use_ue8m0,
            gran_k=gran_k,
            use_packed_ue8m0=use_packed_ue8m0,
        )
    if backend == "cuda":
        return per_token_cast_to_fp8_cuda(
            x,
            use_ue8m0=use_ue8m0,
            gran_k=gran_k,
            use_packed_ue8m0=use_packed_ue8m0,
        )
    raise RuntimeError(f"unknown FP8 quant backend: {backend!r}")


__all__ = [
    "ceil_to_ue8m0",
    "pack_ue8m0_to_int",
    "per_token_cast_to_fp8",
    "per_token_cast_to_fp8_torch",
    "per_token_cast_to_fp8_triton",
    "per_token_cast_to_fp8_cuda",
    "fp8_to_uint8_cuda",
    "psum_silu_mul_fp8_quant_cuda",
]
