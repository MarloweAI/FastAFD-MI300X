"""MI355X-native AITER runner for GPT-OSS packed MXFP4 experts."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from .base import MoeRunner, MoeRunnerBackend, MoeRunnerConfig

if TYPE_CHECKING:
    from ..layer import MoELayer


def _deinterleave_gate_up_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Convert GPT-OSS [g0,u0,g1,u1,...] rows to AITER [g...,u...]."""
    experts, rows = tensor.shape[:2]
    if rows % 2:
        raise RuntimeError(f"GPT-OSS gate/up rows must be even, got {rows}")
    tail = tensor.shape[2:]
    return (
        tensor.reshape(experts, rows // 2, 2, *tail)
        .permute(0, 2, 1, *range(3, 3 + len(tail)))
        .contiguous()
        .reshape(experts, rows, *tail)
    )


def _round_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def _pad_gptoss_for_aiter(
    w1: torch.Tensor,
    s1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    s2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, int, int]:
    """Pad GPT-OSS dimensions to AITER's gfx950 256-wide MXFP4 tiles."""
    experts, twice_intermediate, hidden_bytes = w1.shape
    intermediate = twice_intermediate // 2
    hidden = hidden_bytes * 2
    padded_hidden = _round_up(hidden, 256)
    padded_intermediate = _round_up(intermediate, 256)
    if padded_hidden == hidden and padded_intermediate == intermediate:
        return w1, s1, b1, w2, s2, 0, 0

    padded_w1 = torch.zeros(
        (experts, 2 * padded_intermediate, padded_hidden // 2),
        dtype=w1.dtype,
        device=w1.device,
    )
    padded_s1 = torch.full(
        (experts, 2 * padded_intermediate, padded_hidden // 32),
        127,
        dtype=s1.dtype,
        device=s1.device,
    )
    # w1 is already deinterleaved: gate half, then up half.
    padded_w1[:, :intermediate, :hidden_bytes] = w1[:, :intermediate]
    padded_w1[:, padded_intermediate : padded_intermediate + intermediate, :hidden_bytes] = (
        w1[:, intermediate:]
    )
    hidden_groups = hidden // 32
    padded_s1[:, :intermediate, :hidden_groups] = s1[:, :intermediate]
    padded_s1[
        :, padded_intermediate : padded_intermediate + intermediate, :hidden_groups
    ] = s1[:, intermediate:]

    padded_b1 = None
    if b1 is not None:
        padded_b1 = torch.zeros(
            (experts, 2 * padded_intermediate), dtype=b1.dtype, device=b1.device
        )
        padded_b1[:, :intermediate] = b1[:, :intermediate]
        padded_b1[:, padded_intermediate : padded_intermediate + intermediate] = b1[
            :, intermediate:
        ]

    padded_w2 = torch.zeros(
        (experts, padded_hidden, padded_intermediate // 2),
        dtype=w2.dtype,
        device=w2.device,
    )
    padded_s2 = torch.full(
        (experts, padded_hidden, padded_intermediate // 32),
        127,
        dtype=s2.dtype,
        device=s2.device,
    )
    padded_w2[:, :hidden, : intermediate // 2] = w2
    padded_s2[:, :hidden, : intermediate // 32] = s2
    return (
        padded_w1,
        padded_s1,
        padded_b1,
        padded_w2,
        padded_s2,
        padded_hidden - hidden,
        padded_intermediate - intermediate,
    )


def _prepare_aiter_mxfp4_weights(layer: "MoELayer") -> None:
    """Destructively convert one layer from checkpoint to gfx950 CK layout.

    Keeping both forms would nearly double GPT-OSS-120B expert memory. The raw
    checkpoint tensors are therefore replaced once loading is complete.
    """
    if getattr(layer, "_aiter_mxfp4_prepared", False):
        return
    if not getattr(layer, "_is_mxfp4_packed", False):
        raise RuntimeError(
            "afd_moe_runner_backend=aiter requires packed GPT-OSS MXFP4 weights; "
            "set MINISGL_MXFP4_PACKED=1"
        )
    if not torch.version.hip:
        raise RuntimeError("afd_moe_runner_backend=aiter requires a ROCm PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("afd_moe_runner_backend=aiter requires an available AMD GPU")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if not str(arch).startswith("gfx950"):
        raise RuntimeError(
            "the native AITER MXFP4 runner is intentionally gfx950-only, "
            f"detected {arch!r}"
        )

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError(
            "PyTorch lacks torch.float4_e2m1fn_x2; use the pinned MI355X ROCm image"
        )
    try:
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
    except ImportError as exc:
        raise RuntimeError(
            "AITER with gfx950 A16W4 shuffle kernels is required; install the pinned "
            "amd-aiter build"
        ) from exc

    w1 = _deinterleave_gate_up_rows(layer.gate_up_proj_blocks).view(fp4_dtype)
    output_hidden_size = int(layer.down_proj_blocks.shape[1])
    s1 = _deinterleave_gate_up_rows(layer.gate_up_proj_scales)
    b1 = getattr(layer, "gate_up_proj_bias", None)
    if b1 is not None:
        b1 = _deinterleave_gate_up_rows(b1).to(torch.float32)

    w2 = layer.down_proj_blocks.view(fp4_dtype)
    s2 = layer.down_proj_scales
    w1, s1, b1, w2, s2, hidden_pad, intermediate_pad = _pad_gptoss_for_aiter(
        w1.view(torch.uint8), s1, b1, w2.view(torch.uint8), s2
    )
    w1 = w1.view(fp4_dtype)
    w2 = w2.view(fp4_dtype)
    num_experts = int(w1.shape[0])

    w1 = shuffle_weight_a16w4(w1, 16, True)
    s1 = shuffle_scale_a16w4(s1.reshape(-1, s1.shape[-1]), num_experts, True)
    w2 = shuffle_weight_a16w4(w2, 16, False)
    s2 = shuffle_scale_a16w4(s2.reshape(-1, s2.shape[-1]), num_experts, False)
    w1.is_shuffled = True
    w2.is_shuffled = True

    layer.gate_up_proj_blocks = w1
    layer.gate_up_proj_scales = s1
    layer.down_proj_blocks = w2
    layer.down_proj_scales = s2
    if b1 is not None:
        layer.gate_up_proj_bias = b1
    layer._aiter_hidden_pad = int(hidden_pad)
    layer._aiter_intermediate_pad = int(intermediate_pad)
    layer._aiter_padded_hidden_size = int(w2.shape[1])
    layer._aiter_output_hidden_size = output_hidden_size
    layer._aiter_mxfp4_prepared = True


class AiterMxfp4Runner(MoeRunner):
    """AITER CK W4A16 fused MoE, specialized for GPT-OSS on gfx950."""

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER

    def apply(self, dispatch_output: Any, layer: "MoELayer") -> torch.Tensor:
        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return hidden_states
        if layer.activation != "gptoss_swiglu":
            raise RuntimeError(
                "the MI355X AITER runner currently supports only GPT-OSS "
                f"gptoss_swiglu, got {layer.activation!r}"
            )
        _prepare_aiter_mxfp4_weights(layer)

        padded_hidden_size = int(layer._aiter_padded_hidden_size)
        if hidden_states.shape[1] > padded_hidden_size:
            raise RuntimeError(
                "AITER activation width exceeds its padded weight width: "
                f"activation={hidden_states.shape[1]}, weight={padded_hidden_size}"
            )
        if hidden_states.shape[1] < padded_hidden_size:
            padded_hidden_states = hidden_states.new_zeros(
                (hidden_states.shape[0], padded_hidden_size)
            )
            padded_hidden_states[:, : hidden_states.shape[1]] = hidden_states
            hidden_states = padded_hidden_states

        topk_output = dispatch_output.topk_output
        if not (isinstance(topk_output, tuple) and len(topk_output) == 2):
            raise RuntimeError("AITER M:N input must provide (topk_weights, topk_ids)")
        topk_weights, topk_ids = topk_output

        try:
            from aiter import ActivationType, QuantType
            from aiter.fused_moe import fused_moe
        except ImportError as exc:
            raise RuntimeError("the pinned amd-aiter package is not importable") from exc

        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)
        # The pinned gfx950 build has tuned GPT-OSS shapes through M=4096. Larger
        # routed batches can select a pathological fallback whose first launch runs
        # for many minutes. Preserve the exact row order while tiling only M; weights,
        # routing, and accumulation within every row are unchanged.
        max_rows = int(os.environ.get("MINISGL_AITER_MAX_M", "4096"))
        if max_rows <= 0:
            raise RuntimeError(f"MINISGL_AITER_MAX_M must be positive, got {max_rows}")
        outputs: list[torch.Tensor] = []
        for row_start in range(0, hidden_states.shape[0], max_rows):
            row_end = min(row_start + max_rows, hidden_states.shape[0])
            outputs.append(
                fused_moe(
                    hidden_states=hidden_states[row_start:row_end],
                    w1=layer.gate_up_proj_blocks,
                    w2=layer.down_proj_blocks,
                    topk_weight=topk_weights[row_start:row_end],
                    topk_ids=topk_ids[row_start:row_end],
                    activation=ActivationType.Swiglu,
                    quant_type=QuantType.per_1x32,
                    doweight_stage1=bool(layer.apply_router_weight_on_input),
                    w1_scale=layer.gate_up_proj_scales,
                    w2_scale=layer.down_proj_scales,
                    dtype=hidden_states.dtype,
                    hidden_pad=layer._aiter_hidden_pad,
                    intermediate_pad=layer._aiter_intermediate_pad,
                    bias1=getattr(layer, "gate_up_proj_bias", None),
                    # Down bias is router-weighted on the AG after M:N combine.
                    bias2=None,
                )[:, : layer._aiter_output_hidden_size]
            )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


__all__ = ["AiterMxfp4Runner"]
