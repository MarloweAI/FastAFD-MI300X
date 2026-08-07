from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def _jit_module(q_heads: int, kv_heads: int, head_dim: int) -> Module:
    args = make_cpp_args(q_heads, kv_heads, head_dim)
    return load_jit(
        "qk_norm_rope",
        *args,
        cuda_files=["qk_norm_rope.cu"],
        cuda_wrappers=[
            ("launch", f"QKNormRopeKernel<{args}>::run"),
            ("launch_store", f"QKNormRopeKernel<{args}>::run_store"),
        ],
    )


def qk_norm_rope_cuda(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """Single-kernel fused RMSNorm(Q, K) + NeoX RoPE.

    Ported from vLLM's csrc/fused_qknorm_rope_kernel.cu (TensorRT-LLM design).
    Mutates q and k in place. q/k are typically strided views from qkv.split().
    cos_sin_cache is (max_position, head_dim) fp32 with layout [cos..., sin...].
    """
    if head_dim != 128:
        raise RuntimeError(
            f"qk_norm_rope_cuda currently supports head_dim=128, got {head_dim}"
        )
    if positions.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"positions must be int32 or int64, got {positions.dtype}")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise RuntimeError(
            f"q/k must both be bf16 or fp16, got q={q.dtype}, k={k.dtype}"
        )
    if q_weight.dtype != q.dtype or k_weight.dtype != q.dtype:
        raise RuntimeError(
            f"q/k norm weights must match q dtype, got q_weight={q_weight.dtype}, "
            f"k_weight={k_weight.dtype}, q={q.dtype}"
        )
    if cos_sin_cache.dtype != torch.float32:
        raise RuntimeError(f"cos_sin_cache must be fp32, got {cos_sin_cache.dtype}")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise RuntimeError("q/k last dimension must be contiguous")
    module = _jit_module(int(q_heads), int(kv_heads), int(head_dim))
    module.launch(positions, q, k, q_weight, k_weight, cos_sin_cache, float(eps))


def qk_norm_rope_store_kv_cuda(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """qk_norm_rope_cuda + fused KV-cache store: the roped K rows and the raw
    V rows land at ``kv_cache[out_loc[token]]`` in the same kernel, replacing
    the separate ``store_kv_cache`` launch.  ``k_cache``/``v_cache`` are the
    layer's cache viewed as (capacity, kv_heads*head_dim)."""
    if head_dim != 128:
        raise RuntimeError(
            f"qk_norm_rope_store_kv_cuda currently supports head_dim=128, got {head_dim}"
        )
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("q/k/v must share a bf16 or fp16 dtype")
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise RuntimeError("kv cache dtype must match q/k/v")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise RuntimeError("q/k/v last dimension must be contiguous")
    module = _jit_module(int(q_heads), int(kv_heads), int(head_dim))
    module.launch_store(
        positions, q, k, v, q_weight, k_weight, cos_sin_cache,
        k_cache, v_cache, out_loc, float(eps),
    )
