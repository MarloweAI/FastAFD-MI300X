from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from minisgl.core import get_global_ctx

from .base import MoeRunner, MoeRunnerBackend, MoeRunnerConfig

if TYPE_CHECKING:
    import torch

    from ..layer import MoELayer


class TritonRunner(MoeRunner):
    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON

    def apply(self, dispatch_output: Any, layer: "MoELayer") -> "torch.Tensor":
        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return hidden_states
        topk_output = dispatch_output.topk_output
        if isinstance(topk_output, tuple) and len(topk_output) == 2:
            topk_weights, topk_ids = topk_output
        else:
            from minisgl.moe.fused import fused_topk

            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=topk_output,
                topk=layer.top_k,
                renormalize=layer.renormalize,
            )
        if getattr(layer, "_is_mxfp4_packed", False):
            # Packed MXFP4: hand the (blocks, scales) pairs straight through.
            # `fused_experts_impl` dispatches each GEMM on the tuple, and the 4-bit values
            # are upcast inside the kernel -- they are never materialised as BF16 in HBM,
            # which is the whole point (doc 50).
            w1 = (layer.gate_up_proj_blocks, layer.gate_up_proj_scales)
            w2 = (layer.down_proj_blocks, layer.down_proj_scales)
        else:
            w1 = layer.gate_up_proj
            w2 = layer.down_proj
        if not isinstance(w1, tuple) and w1.dtype == torch.float8_e4m3fn:
            from minisgl.models.fp8_utils import dequant_fp8_block_batched

            block_size = (
                layer.quant.weight_block_size
                if getattr(layer, "quant", None) is not None
                else (128, 128)
            )
            w1 = dequant_fp8_block_batched(w1, layer.gate_up_proj_scale, block_size)
            w2 = dequant_fp8_block_batched(w2, layer.down_proj_scale, block_size)

        ctx = get_global_ctx()
        if getattr(dispatch_output, "use_expert_map_override", False):
            expert_map = dispatch_output.expert_map_override
            global_num_experts = dispatch_output.global_num_experts_override
        else:
            expert_map = layer._expert_map_dev
            global_num_experts = layer.num_experts
        return ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
            # gate_up bias belongs to the expert GEMM and rides along here. down_proj
            # bias deliberately does NOT: it is full-width and replicated, so its
            # contribution is sum_e w_e * bias_e over the ROUTER weights, which live on
            # the AG rank. Adding it per expert rank would count it once per rank. The AG
            # side folds it in after combine via topk.shared_output; see
            # models/gpt_oss_afd.py and dev_log/gpt_oss_120b/25_afd_gpt_oss.md.
            w1_bias=getattr(layer, "gate_up_proj_bias", None),
            w2_bias=None,
        )


__all__ = ["TritonRunner"]
