"""FP8-native MoE runner — Phase 2 of `tasks/fp8_integration/`.

Routes the MoeRunner.apply pipeline through the vendored vLLM Triton FP8
grouped MoE kernel (see `minisgl.moe.fused_fp8`) instead of Phase 1's
β-dequant→BF16-Triton path. Activated when the user passes
`--afd-moe-runner-backend triton_fp8`.

Differs from `TritonRunner` (Phase 1) in one place only: the kernel call.
- `TritonRunner` β-dequants `layer.gate_up_proj` / `layer.down_proj` to
  BF16 and dispatches AFD's existing BF16 fused-MoE backend.
- `TritonFp8Runner` keeps the FP8 storage as-is and dispatches the
  FP8-native kernel that consumes (FP8 weight + FP32 per-block scale)
  pairs directly, so the GEMM math runs on Blackwell FP8 tensor cores.

Pre-condition: `layer.gate_up_proj.dtype == torch.float8_e4m3fn`. Raises
`ValueError` otherwise — this backend is only valid on a quantized layer
(Phase 1 storage in place). The same MoELayer object can be served by
either runner; the choice is the user-facing knob.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import MoeRunner, MoeRunnerBackend, MoeRunnerConfig

if TYPE_CHECKING:
    from ..layer import MoELayer


class TritonFp8Runner(MoeRunner):
    """FP8-native MoE runner; wraps `minisgl.moe.fused_fp8.fused_moe_fp8_triton`.

    Mirror of `TritonRunner` for the FP8 path. Inputs/outputs are identical
    (dispatch_output, layer) → BF16 output tensor; the only swap is which
    kernel runs underneath. This is the Phase 2 deliverable.
    """

    def __init__(self, config: MoeRunnerConfig) -> None:
        super().__init__(config)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON_FP8

    def apply(self, dispatch_output: Any, layer: "MoELayer") -> torch.Tensor:
        # Hard pre-condition: FP8 storage on the MoELayer. If a user picks
        # `triton_fp8` against a BF16 ckpt, fail loudly with a fix-it message
        # rather than silently producing garbage.
        if layer.gate_up_proj.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "TRITON_FP8 backend requires FP8 storage on MoELayer; "
                "load model with quantization_config or use TRITON backend"
            )

        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return hidden_states.new_empty((0, layer.hidden_size))
        topk_output = dispatch_output.topk_output
        if isinstance(topk_output, tuple) and len(topk_output) == 2:
            topk_weights, topk_ids = topk_output
        else:
            # Same lazy topk path as TritonRunner — when the dispatcher hands
            # us raw router logits, fold the softmax here.
            from minisgl.moe.fused import fused_topk

            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=topk_output,
                topk=layer.top_k,
                renormalize=layer.renormalize,
            )

        # Phase 3.1 Fix C (2026-05-15): scales are FP32 in MoELayer storage
        # (see layer.py:99-108 + lessons.md 2026-05-15 Fix C entry). No
        # per-forward `.to(torch.float32)` here — that cast cost ~20 ms/step
        # at 96 launches/step. Use layer parameters directly.
        from minisgl.moe.fused_fp8 import fused_moe_fp8_triton

        if getattr(dispatch_output, "use_expert_map_override", False):
            expert_map = dispatch_output.expert_map_override
            global_num_experts = dispatch_output.global_num_experts_override
        else:
            expert_map = layer._expert_map_dev
            global_num_experts = layer.num_experts
        return fused_moe_fp8_triton(
            hidden_states=hidden_states,
            w1=layer.gate_up_proj,
            w1_scale=layer.gate_up_proj_scale,
            w2=layer.down_proj,
            w2_scale=layer.down_proj_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
        )


__all__ = ["TritonFp8Runner"]
