"""Triton implementations of the FlashInfer op set, fused to cut kernel count.

Why this exists: `_torch_ops.py` is correct but expresses each op as many small ATen
calls. Measured on gfx942 (dev_log/qwen/14_performance.md), colocated decode at B=1 spent
**56.7% of GPU time in 2790 elementwise kernels per step** out of 3816 total, and was
2.0x launch-bound (CPU 37.7 ms/step vs GPU 18.8 ms/step). Attention, by contrast, was
1.5%. So the win here is not arithmetic -- it is issuing one kernel where torch issued
eight or ten.

Kernel counts per call:

| op                  | _torch_ops | here |
|---------------------|-----------:|-----:|
| rmsnorm             |        ~8  |    1 |
| fused_add_rmsnorm   |       ~11  |    1 |
| silu_and_mul        |        ~4  |    1 |
| rope (in-place q+k) |       ~14  |    1 |

Numerics follow `_torch_ops` exactly, which in turn follows HF (see that module's
docstring): the mean-square reduction is fp32 and the normalized value is rounded to
the input dtype *before* the weight multiply.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "rmsnorm",
    "fused_add_rmsnorm",
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "apply_rope_with_cos_sin_cache_inplace",
    "gptoss_swiglu",
]


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------
@triton.jit
def _row_off(row, s_outer, s_inner, INNER: tl.constexpr):
    """Byte-free element offset of logical row `row`, from the tensor's real strides.

    See `_rows()` for why this exists instead of a reshape.
    """
    if INNER == 1:
        return row * s_outer
    return (row // INNER) * s_outer + (row % INNER) * s_inner


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    xs_outer, xs_inner, os_outer, os_inner,
    n_cols, eps,
    INNER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_row_stride = _row_off(row, xs_outer, xs_inner, INNER)
    out_row_stride = _row_off(row, os_outer, os_inner, INNER)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    xhat = x * tl.rsqrt(var + eps)
    # Round to the output dtype BEFORE applying the weight, matching HF's
    # Qwen3RMSNorm (`self.weight * hidden.to(input_dtype)`).
    xhat = xhat.to(out_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + out_row_stride + cols, (xhat * w).to(out_ptr.dtype.element_ty),
             mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr,
    xs_outer, xs_inner, rs_outer, rs_inner,
    n_cols, eps,
    INNER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    xo = x_ptr + _row_off(row, xs_outer, xs_inner, INNER) + cols
    ro = res_ptr + _row_off(row, rs_outer, rs_inner, INNER) + cols
    x = tl.load(xo, mask=mask, other=0.0)
    r = tl.load(ro, mask=mask, other=0.0)
    # Sum in the INPUT dtype, matching HF's decoder layer (bf16 add), then store.
    s = (x + r).to(x_ptr.dtype.element_ty)
    tl.store(ro, s, mask=mask)
    sf = s.to(tl.float32)
    var = tl.sum(sf * sf, axis=0) / n_cols
    xhat = sf * tl.rsqrt(var + eps)
    xhat = xhat.to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(xo, (xhat * w).to(x_ptr.dtype.element_ty), mask=mask)


def _norm_block(n: int) -> tuple[int, int]:
    blk = triton.next_power_of_2(n)
    return blk, max(1, min(8, blk // 256))


def _rows(t: torch.Tensor) -> tuple[int, int, int, int]:
    """Describe `t` as `rows` rows of `t.shape[-1]` contiguous elements.

    Returns `(rows, s_outer, s_inner, inner)` such that logical row `r` starts at
    element `(r // inner) * s_outer + (r % inner) * s_inner`.

    Why not just `t.reshape(-1, t.shape[-1])`: **reshape on a non-contiguous view
    silently returns a copy**, and these ops write in place. The live case is
    `q_norm.forward_inplace(q.view(-1, num_heads, head_dim))` in
    `layers/attention.py`, where `q` is a `qkv.split(...)` view -- shape
    `(nnz, H, D)` with strides `(3*H*D, D, 1)`. Collapsing `(nnz, H) -> nnz*H`
    is not expressible in one stride, so reshape copies, the kernel normalises
    the copy, and the model silently sees un-normalised q/k. That produced fluent
    garbage end-to-end while every op still passed an isolated parity test,
    because a 2-D non-contiguous reshape to the *same* shape is a no-op and only
    the 3-D collapse copies. See dev_log/qwen/14_performance.md.
    """
    assert t.stride(-1) == 1, f"last dim must be contiguous, got strides {t.stride()}"
    lead = t.shape[:-1]
    if len(lead) == 1:
        return lead[0], t.stride(0), 0, 1
    if len(lead) == 2:
        return lead[0] * lead[1], t.stride(0), t.stride(1), lead[1]
    # >3-D never occurs on the live paths; require plain contiguity so the
    # single-stride collapse is valid rather than silently copying.
    assert t.is_contiguous(), f"{t.dim()}-D input must be contiguous"
    return t.numel() // t.shape[-1], t.shape[-1], 0, 1


def rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None):
    del enable_pdl
    if out is None:
        out = torch.empty_like(input)
    n = input.shape[-1]
    rows, xs_o, xs_i, inner = _rows(input)
    o_rows, os_o, os_i, o_inner = _rows(out)
    assert (o_rows, o_inner) == (rows, inner), "out layout must match input"
    blk, warps = _norm_block(n)
    _rmsnorm_kernel[(rows,)](
        input, weight, out, xs_o, xs_i, os_o, os_i, n, float(eps),
        INNER=inner, BLOCK=blk, num_warps=warps,
    )
    return out


def fused_add_rmsnorm(input, residual, weight, eps=1e-6, enable_pdl=None):
    del enable_pdl
    n = input.shape[-1]
    rows, xs_o, xs_i, inner = _rows(input)
    r_rows, rs_o, rs_i, r_inner = _rows(residual)
    assert (r_rows, r_inner) == (rows, inner), "residual layout must match input"
    blk, warps = _norm_block(n)
    _fused_add_rmsnorm_kernel[(rows,)](
        input, residual, weight, xs_o, xs_i, rs_o, rs_i, n, float(eps),
        INNER=inner, BLOCK=blk, num_warps=warps,
    )


# --------------------------------------------------------------------------
# Gated activations: x is (..., 2*d); gate is the FIRST half
# --------------------------------------------------------------------------
@triton.jit
def _gate_mul_kernel(
    x_ptr, out_ptr, xs_outer, xs_inner, os_outer, os_inner, d,
    INNER: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_row_stride = _row_off(row, xs_outer, xs_inner, INNER)
    out_row_stride = _row_off(row, os_outer, os_inner, INNER)
    for start in tl.range(0, d, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < d
        g = tl.load(x_ptr + x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + x_row_stride + d + cols, mask=mask, other=0.0)
        if ACT == 0:      # silu
            a = g * tl.sigmoid(g)
        elif ACT == 1:    # gelu (erf)
            a = g * 0.5 * (1.0 + tl.erf(g * 0.7071067811865476))
        else:             # gelu tanh
            inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
            a = g * 0.5 * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))
        # Round the activation to the element dtype BEFORE multiplying, because
        # `F.silu(bf16)` returns bf16 and torch then multiplies in bf16. Keeping the
        # activation in fp32 through the multiply is more accurate but differs from
        # the reference by up to one bf16 ULP (measured 3.1e-02..6.3e-02), and the
        # policy in _torch_ops is to match HF rather than to maximise accuracy.
        a = a.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + out_row_stride + cols,
                 (a * u).to(out_ptr.dtype.element_ty), mask=mask)


def _gated(x, out, act: int):
    d = x.shape[-1] // 2
    if out is None:
        out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
    # Real strides, never reshape -- see `_rows()`.
    rows, xs_o, xs_i, inner = _rows(x)
    o_rows, os_o, os_i, o_inner = _rows(out)
    assert (o_rows, o_inner) == (rows, inner), "out layout must match input"
    blk = min(1024, triton.next_power_of_2(d))
    _gate_mul_kernel[(rows,)](
        x, out, xs_o, xs_i, os_o, os_i, d,
        INNER=inner, ACT=act, BLOCK=blk, num_warps=4,
    )
    return out


# --------------------------------------------------------------------------
# gpt-oss gated activation
#
# NOT a variant of the split-half `_gate_mul_kernel` above. Four differences, all
# load-bearing (HF GptOssExperts.forward):
#   1. gate/up are INTERLEAVED -- gate = x[..., 0::2], up = x[..., 1::2] -- where Qwen3
#      uses first-half/second-half. Reusing the split-half kernel would silently
#      compute a different function, not raise.
#   2. gate is clamped ABOVE only (max=limit); up is clamped BOTH ways (+-limit).
#   3. the sigmoid is scaled: gate * sigmoid(gate * alpha), alpha = 1.702.
#   4. the multiplicand is (up + 1), not up.
# --------------------------------------------------------------------------
@triton.jit
def _gptoss_swiglu_kernel(
    x_ptr, out_ptr, xs_outer, xs_inner, os_outer, os_inner, d,
    limit, alpha,
    INNER: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_row = _row_off(row, xs_outer, xs_inner, INNER)
    o_row = _row_off(row, os_outer, os_inner, INNER)
    for start in tl.range(0, d, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < d
        # interleaved: gate at 2c, up at 2c+1
        g = tl.load(x_ptr + x_row + 2 * cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + x_row + 2 * cols + 1, mask=mask, other=0.0).to(tl.float32)
        g = tl.minimum(g, limit)                       # clamp(max=limit), no lower bound
        u = tl.minimum(tl.maximum(u, -limit), limit)   # clamp(-limit, limit)
        glu = g * tl.sigmoid(g * alpha)
        res = (u + 1.0) * glu
        tl.store(out_ptr + o_row + cols, res.to(out_ptr.dtype.element_ty), mask=mask)


def gptoss_swiglu(x, out=None, *, limit: float = 7.0, alpha: float = 1.702):
    """gpt-oss expert activation: `(clamp(up)+1) * gate*sigmoid(gate*alpha)`, interleaved."""
    d = x.shape[-1] // 2
    if out is None:
        out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
    rows, xs_o, xs_i, inner = _rows(x)
    o_rows, os_o, os_i, o_inner = _rows(out)
    assert (o_rows, o_inner) == (rows, inner), "out layout must match input"
    blk = min(1024, triton.next_power_of_2(d))
    _gptoss_swiglu_kernel[(rows,)](
        x, out, xs_o, xs_i, os_o, os_i, d, float(limit), float(alpha),
        INNER=inner, BLOCK=blk, num_warps=4,
    )
    return out


def silu_and_mul(x, out=None, **kw):
    del kw
    return _gated(x, out, 0)


def gelu_and_mul(x, out=None, **kw):
    del kw
    return _gated(x, out, 1)


def gelu_tanh_and_mul(x, out=None, **kw):
    del kw
    return _gated(x, out, 2)


# --------------------------------------------------------------------------
# RoPE -- one kernel for q and k together, in place
# --------------------------------------------------------------------------
@triton.jit
def _rope_kernel(
    q_ptr, k_ptr, pos_ptr, cache_ptr,
    q_row_stride, k_row_stride, cache_row_stride,
    q_heads, k_heads, head_size, rotary_dim,
    HALF: tl.constexpr, IS_NEOX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    total = q_heads + k_heads
    if head >= total:
        return
    pos = tl.load(pos_ptr + row).to(tl.int64)
    h = tl.arange(0, HALF)
    hmask = h < (rotary_dim // 2)
    cos = tl.load(cache_ptr + pos * cache_row_stride + h, mask=hmask, other=0.0)
    sin = tl.load(cache_ptr + pos * cache_row_stride + rotary_dim // 2 + h,
                  mask=hmask, other=0.0)

    is_q = head < q_heads
    base_ptr = q_ptr if is_q else k_ptr
    stride = q_row_stride if is_q else k_row_stride
    hh = head if is_q else head - q_heads
    off = base_ptr + row * stride + hh * head_size

    if IS_NEOX:
        i1, i2 = h, h + rotary_dim // 2
    else:
        i1, i2 = 2 * h, 2 * h + 1
    x1 = tl.load(off + i1, mask=hmask, other=0.0).to(tl.float32)
    x2 = tl.load(off + i2, mask=hmask, other=0.0).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    dt = q_ptr.dtype.element_ty
    tl.store(off + i1, o1.to(dt), mask=hmask)
    tl.store(off + i2, o2.to(dt), mask=hmask)


def apply_rope_with_cos_sin_cache_inplace(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    rotary_dim = cos_sin_cache.shape[-1]
    nnz = query.shape[0]
    q_heads = query.shape[-1] // head_size
    k_heads = key.shape[-1] // head_size
    half = triton.next_power_of_2(max(1, rotary_dim // 2))
    _rope_kernel[(nnz, q_heads + k_heads)](
        query, key, positions, cos_sin_cache,
        query.stride(0), key.stride(0), cos_sin_cache.stride(0),
        q_heads, k_heads, head_size, rotary_dim,
        HALF=half, IS_NEOX=bool(is_neox), num_warps=2,
    )
