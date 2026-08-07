from __future__ import annotations

import torch
from minisgl.layers import (
    BaseOP,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.models.afd import ModelStageState
from minisgl.utils import nvtx_annotate

from .config import ModelConfig
from .glm4_moe import (
    Glm4AfdTopK,
    Glm4MoeAttention,
    Glm4MoeMLP,
    Glm4MoeRouterGate,
    _glm4_first_dense_replace,
    _glm4_n_group,
    _glm4_n_shared_experts,
    _glm4_routed_scaling_factor,
    _glm4_topk_group,
    _remote_shared_expert_enabled,
    glm4_topk,
)


class _Glm4AfdMoEGateAndShared(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate = Glm4MoeRouterGate(config.hidden_size, config.num_experts)
        remote_shared = _remote_shared_expert_enabled()
        n_shared = _glm4_n_shared_experts(config)
        self.shared_experts = (
            Glm4MoeMLP(
                config,
                intermediate_size=config.moe_intermediate_size * n_shared,
            )
            if n_shared > 0 and not remote_shared
            else None
        )
        self.remote_shared_expert = bool(remote_shared and n_shared > 0)


class Glm4AfdAGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Glm4MoeAttention(config, layer_id)
        self.mlp = (
            Glm4MoeMLP(config, intermediate_size=config.intermediate_size)
            if layer_id < _glm4_first_dense_replace(config)
            else _Glm4AfdMoEGateAndShared(config)
        )
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.top_k = int(config.num_experts_per_tok)
        self.renormalize = bool(config.norm_topk_prob)
        self.num_expert_group = _glm4_n_group(config)
        self.topk_group = _glm4_topk_group(config)
        self.routed_scaling_factor = _glm4_routed_scaling_factor(config)
        self.first_k_dense_replace = _glm4_first_dense_replace(config)
        self._layer_id = layer_id

    def uses_remote_moe(self) -> bool:
        return self._layer_id >= self.first_k_dense_replace

    @nvtx_annotate("AFD_AG_Attn_{}", layer_id_field="_layer_id")
    def forward_attention(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> ModelStageState:
        hidden_states, residual = self.input_layernorm.forward(hidden_states, residual)
        hidden_states = self.self_attn.forward(hidden_states)
        hidden_states, residual = self.post_attention_layernorm.forward(
            hidden_states, residual
        )
        return ModelStageState(hidden_states=hidden_states, residual=residual)

    @nvtx_annotate("AFD_AG_Route_{}", layer_id_field="_layer_id")
    def route(self, hidden_states: torch.Tensor) -> Glm4AfdTopK:
        h = hidden_states.view(-1, hidden_states.shape[-1])
        if not self.uses_remote_moe():
            return Glm4AfdTopK(
                topk_ids=None,
                topk_weights=None,
                skip_moe=True,
                local_output=self.mlp.forward(h),
                dispatch_fp8=False,
            )
        shared_output = None
        if self.mlp.shared_experts is not None:
            shared_output = self.mlp.shared_experts.forward(h)
        router_logits = self.mlp.gate.forward(h)
        topk = glm4_topk(
            router_logits,
            self.mlp.gate.e_score_correction_bias,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
        )
        return topk._replace(
            shared_output=shared_output,
            routed_scaling_factor=self.routed_scaling_factor,
            dispatch_fp8=True,
        )


class Glm4AfdDenseRouterStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Glm4AfdAGLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> ModelStageState:
        return ModelStageState(self.embed_tokens.forward(input_ids), residual=None)

    def forward_attention(self, layer_id: int, h: torch.Tensor, residual=None) -> ModelStageState:
        return self.layers.op_list[layer_id].forward_attention(h, residual)

    def route(self, layer_id: int, h: torch.Tensor) -> Glm4AfdTopK:
        return self.layers.op_list[layer_id].route(h)

    def finalize_hidden(self, h: torch.Tensor, residual=None) -> torch.Tensor:
        return self.norm.forward(h, residual)[0]


class Glm4AfdDenseRouterForCausalLM(BaseOP):
    def __init__(self, config: ModelConfig):
        self.model = Glm4AfdDenseRouterStage(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward_attention(self, layer_id, h, residual=None):
        return self.model.forward_attention(layer_id, h, residual)

    def route(self, layer_id, h):
        return self.model.route(layer_id, h)

    def finalize_hidden(self, h, residual=None):
        return self.model.finalize_hidden(h, residual)

    def forward_lm_head(self, h):
        return self.lm_head.forward(h)


class _Glm4AfdExpertMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            activation="glm4_silu" if config.hidden_act == "silu" else config.hidden_act,
            quant=config.quant,
            a2a_backend="none",
        )
        n_shared = _glm4_n_shared_experts(config)
        self.shared_experts = (
            Glm4MoeMLP(
                config,
                intermediate_size=config.moe_intermediate_size * n_shared,
            )
            if _remote_shared_expert_enabled() and n_shared > 0
            else None
        )


class _Glm4AfdDenseSkipLayer(BaseOP):
    def forward(self, *args, **kwargs):
        raise RuntimeError("GLM dense layers do not run on AFD EG")


class Glm4AfdEGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.mlp = _Glm4AfdExpertMLP(config)
        self._layer_id = layer_id

    @nvtx_annotate("AFD_EG_Experts_{}", layer_id_field="_layer_id")
    def run_experts(self, dispatch_output) -> torch.Tensor:
        return self.mlp.experts._run_moe_core(dispatch_output).hidden_states


class _Glm4AfdExpertStageInner(BaseOP):
    def __init__(self, config: ModelConfig):
        first_dense = _glm4_first_dense_replace(config)
        self.layers = OPList(
            [
                _Glm4AfdDenseSkipLayer()
                if layer_id < first_dense
                else Glm4AfdEGLayer(config, layer_id)
                for layer_id in range(config.num_layers)
            ]
        )
        self.first_k_dense_replace = first_dense

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.layers.op_list[layer_id].run_experts(dispatch_output)


class Glm4AfdExpertStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.model = _Glm4AfdExpertStageInner(config)

    def dispatch_uses_fp8(self) -> bool:
        return True

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.model.run_experts(layer_id, dispatch_output)


__all__ = [
    "Glm4AfdAGLayer",
    "Glm4AfdDenseRouterStage",
    "Glm4AfdDenseRouterForCausalLM",
    "Glm4AfdEGLayer",
    "Glm4AfdExpertStage",
]
