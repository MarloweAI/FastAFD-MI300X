from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    gelu_and_mul,
    glm4_silu_and_mul,
)
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class Glm4AfdTopK(NamedTuple):
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    router_logits: torch.Tensor | None = None
    renormalize: bool = True
    deepep_topk: bool = False
    shared_output: torch.Tensor | None = None
    routed_scaling_factor: float = 1.0
    skip_moe: bool = False
    local_output: torch.Tensor | None = None
    dispatch_fp8: bool = True


def _remote_shared_expert_enabled() -> bool:
    return os.environ.get("MINISGL_AFD_MOE_BACKEND") == "megamoe_m2n"


def _glm4_extra(config: ModelConfig, name: str, default):
    return getattr(config, "model_extra", {}).get(name, default)


def _glm4_n_shared_experts(config: ModelConfig) -> int:
    return int(_glm4_extra(config, "n_shared_experts", 0) or 0)


def _glm4_n_group(config: ModelConfig) -> int:
    return int(_glm4_extra(config, "n_group", 0) or 1)


def _glm4_topk_group(config: ModelConfig) -> int:
    return int(_glm4_extra(config, "topk_group", 0) or 1)


def _glm4_first_dense_replace(config: ModelConfig) -> int:
    return int(_glm4_extra(config, "first_k_dense_replace", 0) or 0)


def _glm4_routed_scaling_factor(config: ModelConfig) -> float:
    return float(_glm4_extra(config, "routed_scaling_factor", 1.0))


def _glm4_attention_bias(config: ModelConfig) -> bool:
    return bool(_glm4_extra(config, "attention_bias", False))


def _glm4_topk_torch(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(router_logits.float())
    choice = scores + correction_bias.float().to(device=router_logits.device)
    if num_expert_group > 1:
        num_tokens, num_experts = choice.shape
        experts_per_group = num_experts // num_expert_group
        grouped = choice.view(num_tokens, num_expert_group, experts_per_group)
        group_scores = torch.topk(grouped, k=min(2, experts_per_group), dim=-1).values.sum(dim=-1)
        selected_groups = torch.topk(group_scores, k=topk_group, dim=-1).indices
        group_mask = torch.zeros(
            (num_tokens, num_expert_group), dtype=torch.bool, device=router_logits.device
        )
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = group_mask.repeat_interleave(experts_per_group, dim=-1)
        choice = choice.masked_fill(~expert_mask, float("-inf"))
    topk_ids = torch.topk(choice, k=top_k, dim=-1).indices.to(torch.int32)
    topk_weights = scores.gather(1, topk_ids.long()).float()
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    return topk_weights.contiguous(), topk_ids.contiguous()


def glm4_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
) -> Glm4AfdTopK:
    num_tokens = int(router_logits.shape[0])
    if num_tokens == 0:
        return Glm4AfdTopK(
            topk_ids=torch.empty((0, top_k), dtype=torch.int32, device=router_logits.device),
            topk_weights=torch.empty((0, top_k), dtype=torch.float32, device=router_logits.device),
            renormalize=renormalize,
        )
    num_experts = int(router_logits.shape[1])
    supported_fused_topk = (
        ((num_experts & (num_experts - 1)) == 0 and num_experts <= 256)
        or num_experts == 160
    )
    if num_expert_group <= 1 and topk_group <= 1 and supported_fused_topk:
        from minisgl.kernel import topk_gating

        weights = torch.empty(
            (num_tokens, top_k), dtype=torch.float32, device=router_logits.device
        )
        ids = torch.empty((num_tokens, top_k), dtype=torch.int32, device=router_logits.device)
        topk_gating(
            weights,
            ids,
            router_logits.contiguous(),
            scoring="sigmoid",
            bias=correction_bias.contiguous(),
            renormalize=renormalize,
        )
        return Glm4AfdTopK(topk_ids=ids, topk_weights=weights, renormalize=renormalize)
    weights, ids = _glm4_topk_torch(
        router_logits,
        correction_bias,
        top_k=top_k,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )
    return Glm4AfdTopK(topk_ids=ids, topk_weights=weights, renormalize=renormalize)


class Glm4MoeMLP(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        intermediate_size: int | None = None,
    ):
        inter = int(config.intermediate_size if intermediate_size is None else intermediate_size)
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [inter, inter],
            has_bias=False,
            quant=config.quant,
        )
        fn_map = {"silu": glm4_silu_and_mul, "gelu": gelu_and_mul}
        if config.hidden_act not in fn_map:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = fn_map[config.hidden_act]
        self.down_proj = LinearRowParallel(
            inter,
            config.hidden_size,
            has_bias=False,
            quant=config.quant,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(self.act_fn(self.gate_up_proj.forward(hidden_states)))


class Glm4MoeRouterGate(BaseOP):
    def __init__(self, hidden_size: int, num_experts: int):
        self.weight = torch.empty(num_experts, hidden_size, dtype=torch.float32)
        self.e_score_correction_bias = torch.empty(num_experts, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(torch.float32), self.weight)


class Glm4MoeAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=_glm4_attention_bias(config),
            quant=config.quant,
        )
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=None,
            k_norm=None,
        )
        self.o_proj = LinearOProj(
            config.head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
            quant=config.quant,
        )

    @nvtx_annotate("MHA")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(hidden_states)
        if self.q_norm is not None and self.k_norm is not None:
            q, k, v = qkv.split(
                [self.attn.qo_attn_dim, self.attn.kv_attn_dim, self.attn.kv_attn_dim],
                dim=-1,
            )
            self.q_norm.forward_inplace(q.view(-1, self.attn.num_qo_heads, self.attn.head_dim))
            self.k_norm.forward_inplace(k.view(-1, self.attn.num_kv_heads, self.attn.head_dim))
            qkv = torch.cat([q, k, v], dim=-1)
        return self.o_proj.forward(self.attn.forward(qkv))


class Glm4MoE(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate = Glm4MoeRouterGate(config.hidden_size, config.num_experts)
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            activation="glm4_silu" if config.hidden_act == "silu" else config.hidden_act,
            quant=config.quant,
        )
        n_shared = _glm4_n_shared_experts(config)
        self.shared_experts = (
            Glm4MoeMLP(
                config,
                intermediate_size=config.moe_intermediate_size * n_shared,
            )
            if n_shared > 0
            else None
        )
        self.top_k = int(config.num_experts_per_tok)
        self.renormalize = bool(config.norm_topk_prob)
        self.num_expert_group = _glm4_n_group(config)
        self.topk_group = _glm4_topk_group(config)
        self.routed_scaling_factor = _glm4_routed_scaling_factor(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, original_shape[-1])
        shared_output = (
            self.shared_experts.forward(hidden_states) if self.shared_experts is not None else None
        )
        router_logits = self.gate.forward(hidden_states)
        topk = glm4_topk(
            router_logits,
            self.gate.e_score_correction_bias,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
        )
        dispatch_output = self.experts.dispatcher.dispatch(
            hidden_states,
            (topk.topk_weights, topk.topk_ids),
        )
        combine_input = self.experts._run_moe_core(dispatch_output)
        routed_output = self.experts.dispatcher.combine(combine_input)
        if self.experts.tp_size > 1:
            routed_output = self.experts._comm.all_reduce(routed_output)
        routed_output = routed_output * self.routed_scaling_factor
        if shared_output is not None:
            routed_output = routed_output + shared_output
        return routed_output.view(original_shape)


class Glm4MoeDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Glm4MoeAttention(config, layer_id)
        self.mlp = (
            Glm4MoeMLP(config, intermediate_size=config.intermediate_size)
            if layer_id < _glm4_first_dense_replace(config)
            else Glm4MoE(config)
        )
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm.forward(hidden_states, residual)
        hidden_states = self.self_attn.forward(hidden_states)
        hidden_states, residual = self.post_attention_layernorm.forward(
            hidden_states, residual
        )
        hidden_states = self.mlp.forward(hidden_states)
        return hidden_states, residual


class Glm4MoeModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Glm4MoeDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            hidden_states, residual = layer.forward(hidden_states, residual)
        return self.norm.forward(hidden_states, residual)[0]


class Glm4MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Glm4MoeModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        hidden_states = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden_states)


def __getattr__(name: str):
    if name == "Glm4AfdDenseRouterForCausalLM":
        from .glm4_moe_afd import Glm4AfdDenseRouterForCausalLM

        return Glm4AfdDenseRouterForCausalLM
    if name == "Glm4AfdExpertStage":
        from .glm4_moe_afd import Glm4AfdExpertStage

        return Glm4AfdExpertStage
    raise AttributeError(name)


__all__ = [
    "Glm4MoeForCausalLM",
    "Glm4AfdDenseRouterForCausalLM",
    "Glm4AfdExpertStage",
    "Glm4AfdTopK",
    "glm4_topk",
]
