"""Chooses between FlashInfer and the pure-torch fallback for the shared op set.

Selection is by **availability**, not by platform:

    MINISGL_OPS_BACKEND=auto        (default) FlashInfer if importable, else Triton,
                                    falling back to torch if Triton is unusable
    MINISGL_OPS_BACKEND=flashinfer  force FlashInfer; raise if missing
    MINISGL_OPS_BACKEND=triton      force the fused Triton kernels
    MINISGL_OPS_BACKEND=torch       force the pure-torch path (the A/B reference)

Why availability and not `torch.version.hip`: a CUDA host with FlashInfer keeps
using it and nothing changes, a CUDA host *without* it still works instead of
crashing, and ROCm gets the fallback automatically because FlashInfer has no
ROCm build. Branching on HIP would have handled only the third case.

Import this module's getters rather than the ops directly, so the decision is
made once and stays consistent across call sites.
"""

from __future__ import annotations

import functools
import os

from minisgl.utils import init_logger

logger = init_logger(__name__)

_VALID = ("auto", "flashinfer", "triton", "torch")


def _requested() -> str:
    value = os.environ.get("MINISGL_OPS_BACKEND", "auto").lower()
    if value not in _VALID:
        raise RuntimeError(
            f"MINISGL_OPS_BACKEND must be one of {_VALID}, got {value!r}"
        )
    return value


@functools.cache
def use_flashinfer() -> bool:
    requested = _requested()
    if requested == "torch":
        return False

    try:
        import flashinfer  # noqa: F401
    except ImportError as exc:
        if requested == "flashinfer":
            raise RuntimeError(
                "MINISGL_OPS_BACKEND=flashinfer but flashinfer is not importable. "
                "There is no ROCm build of FlashInfer; use 'torch' (or 'auto') on AMD GPUs."
            ) from exc
        # Do NOT name a concrete fallback here: get_ops() decides between the fused
        # Triton kernels and the pure-torch reference, and this used to claim
        # "using pure-torch" while Triton was actually selected.
        logger.info(
            "flashinfer unavailable; norm/activation/rope/sampling will use the "
            "MINISGL_OPS_BACKEND fallback (see get_ops())"
        )
        return False
    return True


@functools.cache
def get_ops():
    """Return the module providing the op set."""
    if use_flashinfer():
        import flashinfer

        return flashinfer

    requested = _requested()
    if requested == "torch":
        from . import _torch_ops

        logger.info("using pure-torch ops (minisgl.layers._torch_ops)")
        return _torch_ops

    # Prefer the fused Triton kernels: the torch path issues ~8-11 kernels per op and
    # colocated decode measured 2.0x launch-bound with 2790 elementwise kernels per
    # step (dev_log/qwen/14_performance.md). Fall back to torch if Triton cannot be used,
    # since correctness must not depend on it.
    try:
        from . import _triton_ops

        logger.info("using fused Triton ops (minisgl.layers._triton_ops)")
        return _triton_ops
    except Exception as exc:  # pragma: no cover
        if requested == "triton":
            raise
        logger.warning(
            "fused Triton ops unavailable (%s); falling back to pure-torch ops", exc
        )
        from . import _torch_ops

        return _torch_ops


@functools.cache
def get_sampling_ops():
    """Sampling lives in `flashinfer.sampling`, but in the same module for the fallback."""
    if use_flashinfer():
        import flashinfer.sampling as sampling

        return sampling
    # Sampling runs once per step, not once per layer, so there is nothing to gain
    # from fusing it -- keep the torch implementation.
    from . import _torch_ops

    return _torch_ops


__all__ = ["get_ops", "get_sampling_ops", "use_flashinfer"]
