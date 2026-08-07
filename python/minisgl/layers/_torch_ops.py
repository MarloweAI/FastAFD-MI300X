"""Pure-torch replacements for the FlashInfer op set.

FlashInfer has no ROCm build, and upstream depends on it for far more than
attention: RMSNorm, silu/gelu-and-mul, RoPE and sampling all route through it
(dev_log/qwen/02_dependency_inventory.md §B). This module supplies each of those in
plain torch so the model can run on gfx942.

Selection is by *availability*, not platform — see `minisgl.layers._ops_backend`.
On a CUDA host with FlashInfer installed nothing changes; here, these run.

Numerics policy
---------------
Where FlashInfer and HF `transformers` could differ, these follow **HF**, because
HF is the correctness reference for the end-to-end gate (dev_log/qwen/04_test_plan.md
T-10). Two places where that choice is visible:

* `rmsnorm` accumulates the mean-square in fp32 but rounds the normalized value
  back to the input dtype *before* applying the weight, because `Qwen3RMSNorm`
  does `weight * normed.to(input_dtype)`. Keeping the weight multiply in fp32
  would be slightly more accurate yet differ from HF by up to one bf16 ULP.
* `fused_add_rmsnorm` performs the residual add in the *input* dtype, stores
  that, and normalizes the stored value. HF's decoder layer does exactly this
  (`hidden_states = residual + hidden_states` in bf16, then norm). Accumulating
  the add in fp32 instead would differ by one bf16 rounding per layer, which
  compounds over 48–94 layers.

Performance
-----------
These are correctness-first. Every one is memory-bound and launches several
kernels where FlashInfer launches one; `rmsnorm` and `silu_and_mul` are the hot
ones. Triton/AITER versions are M3 work (dev_log/qwen/03_port_plan.md), not a
prerequisite for a first end-to-end number.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "rmsnorm",
    "fused_add_rmsnorm",
    "silu_and_mul",
    "gptoss_swiglu",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "apply_rope_with_cos_sin_cache_inplace",
    "softmax",
    "sampling_from_probs",
    "top_k_sampling_from_probs",
    "top_p_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,  # accepted and ignored: no PDL on CDNA3
) -> torch.Tensor:
    """`input * rsqrt(mean(input^2) + eps) * weight`, reduction in fp32.

    Rounding order matters and is chosen to match HF bit-for-bit. `Qwen3RMSNorm`
    is:

        hidden = hidden.to(float32)
        hidden = hidden * rsqrt(hidden.pow(2).mean(-1) + eps)
        return self.weight * hidden.to(input_dtype)     # <- cast BEFORE weight

    i.e. the normalized value is rounded to the input dtype *first*, and the
    weight multiply then happens in that dtype. Multiplying in fp32 and rounding
    once at the end is marginally more accurate but differs from HF by up to one
    bf16 ULP per element (measured: 3.125e-02 at these magnitudes), which is
    enough to flip a token on a near-tie and would show up as spurious T-10
    mismatches across 48-94 layers. Accuracy is not the objective here;
    agreeing with the reference is.
    """
    del enable_pdl
    orig_dtype = input.dtype
    x = input.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = weight * x.to(orig_dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    """In-place: `residual <- input + residual`, then `input <- rmsnorm(residual)`.

    Both tensors are written. The add is done in the input dtype to match HF's
    decoder-layer arithmetic (see module docstring).
    """
    del enable_pdl
    summed = input + residual
    residual.copy_(summed)
    input.copy_(rmsnorm(summed, weight, eps))


# --------------------------------------------------------------------------
# Gated activations — x is (..., 2 * d); gate is the FIRST half
# --------------------------------------------------------------------------
def _gate_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None,
    act,
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    y = act(gate) * up
    if out is not None:
        out.copy_(y)
        return out
    return y


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
    del kwargs
    return _gate_and_mul(x, out, F.silu)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
    """Exact (erf) GELU, matching FlashInfer's `gelu_and_mul`.

    FlashInfer's tanh-approximation variant is a separate symbol
    (`gelu_tanh_and_mul`), so this must NOT use approximate="tanh".
    """
    del kwargs
    return _gate_and_mul(x, out, lambda t: F.gelu(t, approximate="none"))


def gptoss_swiglu(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    limit: float = 7.0,
    alpha: float = 1.702,
    **kwargs,
) -> torch.Tensor:
    """gpt-oss expert activation, reference form. Four things differ from silu_and_mul and
    three of them fail silently, so this is deliberately spelled out rather than expressed
    as a flag on `_gate_and_mul`:

      1. gate/up are INTERLEAVED (`[..., 0::2]` / `[..., 1::2]`), not first/second half;
      2. gate is clamped ABOVE only; up is clamped BOTH ways;
      3. the sigmoid is alpha-scaled;
      4. the multiplicand is `(up + 1)`, not `up`.

    Matches HF `GptOssExperts.forward`; see dev_log/gpt_oss_120b/gptoss_ops_parity.py.
    """
    del kwargs
    gate = x[..., 0::2].clamp(max=limit)
    up = x[..., 1::2].clamp(min=-limit, max=limit)
    y = (up + 1) * (gate * torch.sigmoid(gate * alpha))
    if out is None:
        return y
    out.copy_(y)
    return out


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
    del kwargs
    return _gate_and_mul(x, out, lambda t: F.gelu(t, approximate="tanh"))


# --------------------------------------------------------------------------
# Rotary embedding
# --------------------------------------------------------------------------
def _rope_neox(x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX / rotate-half. `x_rot` is (nnz, heads, rotary_dim); cos/sin (nnz, 1, rotary_dim//2)."""
    half = x_rot.shape[-1] // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    # bf16 * fp32 promotes to fp32, so the rotation math runs in fp32.
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _rope_gptj(x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-J / interleaved: pairs are (0,1), (2,3), ..."""
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """In-place RoPE on the first `rotary_dim` elements of every head.

    `cos_sin_cache` is (max_position, rotary_dim) = cat(cos, sin) on the last
    dim, each half being rotary_dim//2 wide — the layout built in
    `layers/rotary.py`. `query`/`key` are (nnz, num_heads * head_size) and are
    written in place; when rotary_dim < head_size the tail is left untouched.
    """
    rotary_dim = cos_sin_cache.shape[-1]
    cos, sin = cos_sin_cache[positions.long()].chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rot = _rope_neox if is_neox else _rope_gptj

    for tensor in (query, key):
        if tensor is None or tensor.numel() == 0:
            continue
        view = tensor.view(tensor.shape[0], -1, head_size)
        x_rot = view[..., :rotary_dim]
        # rot() allocates, so the write cannot clobber its own inputs.
        x_rot.copy_(rot(x_rot, cos, sin))


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def softmax(
    logits: torch.Tensor,
    temperatures: torch.Tensor | float | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    del enable_pdl
    if temperatures is None:
        return torch.softmax(logits, dim=-1)
    if isinstance(temperatures, torch.Tensor):
        temperatures = temperatures.reshape(-1, *([1] * (logits.dim() - 1)))
    return torch.softmax(logits / temperatures, dim=-1)


def _as_col(v, batch: int, device, dtype) -> torch.Tensor:
    """Broadcast a per-request scalar-or-tensor parameter to (batch, 1)."""
    if not isinstance(v, torch.Tensor):
        v = torch.full((batch,), v, device=device, dtype=dtype)
    return v.to(device=device, dtype=dtype).reshape(batch, 1)


def sampling_from_probs(probs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    del args, kwargs
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _masked_sample(
    probs: torch.Tensor,
    keep_top_k: torch.Tensor | None,
    keep_top_p: torch.Tensor | None,
) -> torch.Tensor:
    """Sort descending, apply the keep-masks, renormalize, sample, map back."""
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    keep = torch.ones_like(sorted_probs, dtype=torch.bool)

    if keep_top_k is not None:
        ranks = torch.arange(probs.shape[-1], device=probs.device).expand_as(sorted_probs)
        keep &= ranks < keep_top_k

    if keep_top_p is not None:
        # Exclusive cumulative sum: a token is kept when the mass strictly
        # before it is still below p, so the token that crosses p is included
        # and at least the argmax always survives.
        cumsum_excl = sorted_probs.cumsum(dim=-1) - sorted_probs
        keep &= cumsum_excl < keep_top_p

    masked = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    total = masked.sum(dim=-1, keepdim=True)
    # Degenerate rows (all mass masked away by an extreme p) fall back to argmax.
    masked = torch.where(total > 0, masked, torch.zeros_like(masked).scatter_(-1, torch.zeros_like(sorted_idx[:, :1]), 1.0))
    picked = torch.multinomial(masked, num_samples=1)
    return sorted_idx.gather(-1, picked).squeeze(-1)


def top_k_sampling_from_probs(
    probs: torch.Tensor, top_k, *args, **kwargs
) -> torch.Tensor:
    del args, kwargs
    k = _as_col(top_k, probs.shape[0], probs.device, torch.long)
    return _masked_sample(probs, k, None)


def top_p_sampling_from_probs(
    probs: torch.Tensor, top_p, *args, **kwargs
) -> torch.Tensor:
    del args, kwargs
    p = _as_col(top_p, probs.shape[0], probs.device, probs.dtype)
    return _masked_sample(probs, None, p)


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor, top_k, top_p, *args, **kwargs
) -> torch.Tensor:
    del args, kwargs
    k = _as_col(top_k, probs.shape[0], probs.device, torch.long)
    p = _as_col(top_p, probs.shape[0], probs.device, probs.dtype)
    return _masked_sample(probs, k, p)
