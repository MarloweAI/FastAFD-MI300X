from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNormCrossHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class MiniMaxM2TopK(NamedTuple):
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    router_logits: torch.Tensor | None = None
    renormalize: bool = True
    deepep_topk: bool = False
    external_quant: bool = False


def minimax_m2_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    scoring_func: str,
    score_correction_bias: torch.Tensor | None,
    renormalize: bool,
) -> MiniMaxM2TopK:
    from minisgl.kernel import topk_gating

    logits = router_logits.contiguous()
    num_tokens = int(logits.shape[0])
    weights = torch.empty(
        (num_tokens, top_k),
        dtype=torch.float32,
        device=logits.device,
    )
    ids = torch.empty(
        (num_tokens, top_k),
        dtype=torch.int32,
        device=logits.device,
    )
    bias = None
    if score_correction_bias is not None:
        bias = score_correction_bias.contiguous()
    topk_gating(
        weights,
        ids,
        logits,
        scoring=scoring_func,
        bias=bias,
        renormalize=renormalize,
    )
    return MiniMaxM2TopK(
        topk_ids=ids,
        topk_weights=weights,
        renormalize=renormalize,
    )


class MiniMaxM2Attn(BaseOP):
    """MiniMax M2 attention: FP8 dense projections, cross-head Q/K RMSNorm,
    partial RoPE, then the standard paged attention backend."""

    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=False,
            quant=config.quant,
        )
        self.q_norm = RMSNormCrossHead(head_dim * config.num_qo_heads, eps=config.rms_norm_eps)
        self.k_norm = RMSNormCrossHead(head_dim * config.num_kv_heads, eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=None,
            k_norm=None,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
            quant=config.quant,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor, skip_o_proj_allreduce: bool = False) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        return self.forward_from_qkv(qkv, skip_o_proj_allreduce=skip_o_proj_allreduce)

    def forward_from_qkv(
        self,
        qkv: torch.Tensor,
        *,
        skip_o_proj_allreduce: bool = False,
    ) -> torch.Tensor:
        if (
            qkv.is_cuda
            and qkv.dtype in (torch.bfloat16, torch.float16)
            and self.q_norm.tp_size == 1
            and self.k_norm.tp_size == 1
            and self.q_norm.weight.dtype == qkv.dtype
            and self.k_norm.weight.dtype == qkv.dtype
        ):
            if self._try_fused_qk_norm_rope_store(qkv):
                o = self.attn.forward(
                    qkv,
                    qk_already_rotary_applied=True,
                    kv_store_merged=True,
                )
                return self.o_proj.forward(o, skip_allreduce=skip_o_proj_allreduce)

            from minisgl.kernel import minimax_qk_cross_head_norm_inplace

            minimax_qk_cross_head_norm_inplace(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                q_dim=self.attn.qo_attn_dim,
                kv_dim=self.attn.kv_attn_dim,
                eps=self.q_norm.eps,
            )
            o = self.attn.forward(qkv)
            return self.o_proj.forward(o, skip_allreduce=skip_o_proj_allreduce)
        q, k, v = qkv.split(
            [self.attn.qo_attn_dim, self.attn.kv_attn_dim, self.attn.kv_attn_dim],
            dim=-1,
        )
        del qkv
        q, k = RMSNormCrossHead.forward_qk(
            self.q_norm,
            self.k_norm,
            q.contiguous(),
            k.contiguous(),
        )
        qkv_normed = torch.cat([q, k, v], dim=-1)
        del q, k, v
        o = self.attn.forward(qkv_normed)
        return self.o_proj.forward(o, skip_allreduce=skip_o_proj_allreduce)

    def _try_fused_qk_norm_rope_store(self, qkv: torch.Tensor) -> bool:
        ctx = get_global_ctx()
        out_loc = getattr(ctx.batch, "out_loc", None)
        kvcache = getattr(ctx.attn_backend, "kvcache", None)
        positions = getattr(ctx.batch, "positions", None)
        if out_loc is None or positions is None or kvcache is None:
            return False
        if not positions.is_cuda or not out_loc.is_cuda:
            return False
        cos_sin_cache = self.attn.rotary.cos_sin_cache
        if not cos_sin_cache.is_cuda or cos_sin_cache.dtype != torch.float32:
            return False
        rotary_dim = int(cos_sin_cache.shape[-1])
        if (
            self.attn.head_dim != 128
            or rotary_dim <= 0
            or rotary_dim > self.attn.head_dim
            or rotary_dim % 2 != 0
        ):
            return False
        kv_dim = self.attn.kv_attn_dim
        try:
            k_cache = kvcache.k_cache(self.attn.layer_id)
            v_cache = kvcache.v_cache(self.attn.layer_id)
        except Exception:
            return False
        if (
            k_cache.dtype != qkv.dtype
            or v_cache.dtype != qkv.dtype
            or k_cache.numel() % kv_dim != 0
            or v_cache.numel() % kv_dim != 0
        ):
            return False

        from minisgl.kernel import minimax_qk_cross_head_norm_rope_store_inplace

        minimax_qk_cross_head_norm_rope_store_inplace(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            positions,
            cos_sin_cache,
            k_cache.view(-1, kv_dim),
            v_cache.view(-1, kv_dim),
            out_loc,
            q_dim=self.attn.qo_attn_dim,
            kv_dim=kv_dim,
            head_dim=self.attn.head_dim,
            eps=self.q_norm.eps,
        )
        return True


class MiniMaxM2MoE(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,
            quant=config.quant,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.gate.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self.e_score_correction_bias = (
            torch.empty(config.num_experts, dtype=torch.float32)
            if config.use_routing_bias
            else None
        )
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states.to(torch.float32))
        topk = minimax_m2_topk(
            router_logits,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
        )
        dispatch_output = self.experts.dispatcher.dispatch(
            hidden_states,
            (topk.topk_weights, topk.topk_ids),
        )
        combine_input = self.experts._run_moe_core(dispatch_output)
        final_hidden_states = self.experts.dispatcher.combine(combine_input)
        if self.experts.tp_size > 1:
            final_hidden_states = self.experts._comm.all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM2DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = MiniMaxM2Attn(config, layer_id)
        self.mlp = MiniMaxM2MoE(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class MiniMaxM2Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [MiniMaxM2DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class MiniMaxM2ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = MiniMaxM2Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = [
    "MiniMaxM2ForCausalLM",
    "MiniMaxM2TopK",
    "minimax_m2_topk",
]
