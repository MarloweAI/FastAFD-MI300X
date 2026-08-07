"""FP8 block-wise dequantization helpers for HF FP8 checkpoints.

Used in two contexts:
  1. Eager (α): weight.py at load for non-MoE FP8 tensors (attention q/k/v/o).
     Caller passes a per-tensor (out, in) FP8 weight + (out/bo, in/bi) scale.
  2. Forward (β): MoeRunner.apply() per step for MoE expert weights.
     Caller passes a batched (E, out, in) FP8 weight + (E, out/bo, in/bi) scale.

Mirrors HF Qwen3-FP8 convention: weight axis order (out, in); scale axis
order (out/bo, in/bi); scale dtype BF16. See
tasks/fp8_integration/phase_impl/phase0_preparation/results/ckpt_inspection.md.
"""
from __future__ import annotations

import torch


def dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Block-wise FP8 dequant for a single (out, in) tensor.

    Each (bo, bi) tile of the weight is multiplied by one scalar from `scale`.
    Returns BF16 with `weight_fp8`'s shape.
    """
    if weight_fp8.dim() != 2:
        raise ValueError(
            f"dequant_fp8_block expects 2D weight, got shape {tuple(weight_fp8.shape)}"
        )
    out, in_ = weight_fp8.shape
    bo, bi = block_size
    if out % bo != 0 or in_ % bi != 0:
        raise ValueError(
            f"weight shape {(out, in_)} not divisible by block_size {block_size}"
        )
    expected = (out // bo, in_ // bi)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match expected "
            f"{expected} (= weight_shape // block_size)"
        )
    w = weight_fp8.to(torch.bfloat16).view(out // bo, bo, in_ // bi, bi)
    s = scale.to(torch.bfloat16).view(out // bo, 1, in_ // bi, 1)
    return (w * s).view(out, in_)


def dequant_fp8_block_batched(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Per-expert independent block-wise dequant for (E, out, in) MoE weights.

    Returns BF16 with `weight_fp8`'s shape. Implemented as one fused multiply
    so MoeRunner.apply() pays a single op per forward instead of looping E times.
    """
    if weight_fp8.dim() != 3:
        raise ValueError(
            f"dequant_fp8_block_batched expects 3D weight (E, out, in), "
            f"got shape {tuple(weight_fp8.shape)}"
        )
    E, out, in_ = weight_fp8.shape
    bo, bi = block_size
    if out % bo != 0 or in_ % bi != 0:
        raise ValueError(
            f"weight shape {(out, in_)} not divisible by block_size {block_size}"
        )
    expected = (E, out // bo, in_ // bi)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match expected "
            f"{expected} (= (E, out//bo, in//bi))"
        )
    w = weight_fp8.to(torch.bfloat16).view(E, out // bo, bo, in_ // bi, bi)
    s = scale.to(torch.bfloat16).view(E, out // bo, 1, in_ // bi, 1)
    return (w * s).view(E, out, in_)
