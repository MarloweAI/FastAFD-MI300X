from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from ._ops_backend import get_ops

    return get_ops().silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from ._ops_backend import get_ops

    return get_ops().gelu_and_mul(x, out=out)


def glm4_silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    return silu_and_mul(x, out=out)


def gptoss_swiglu(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    limit: float = 7.0,
    alpha: float = 1.702,
):
    """gpt-oss's clamped, alpha-scaled, INTERLEAVED SwiGLU. Not a variant of
    `silu_and_mul` -- see the note in the backend implementations."""
    from ._ops_backend import get_ops

    return get_ops().gptoss_swiglu(x, out=out, limit=limit, alpha=alpha)


__all__ = ["silu_and_mul", "gelu_and_mul", "glm4_silu_and_mul", "gptoss_swiglu"]
