"""gpt-oss-120b decomposed for the AFD AG/EG architecture.

Mirrors qwen3_moe_afd.py: the AG (attention) stage does embed + attention + router +
norms + lm_head and never runs an expert; the EG stage runs only the grouped expert FFN
over dispatched tokens. The dispatch/combine comm is not here -- it lives in the runners.

Three things make gpt-oss different from every other model on this path, and all three are
silent if got wrong (no shape error, just wrong numbers):

1. **Expert biases.** Both expert projections have one. `gate_up_proj_bias` is sharded with
   the intermediate dim and belongs to the expert GEMM, so it rides to the EG rank and is
   applied there (plumbed through BaseMoeBackend.forward -> fused_experts_impl as w1_bias).

   `down_proj_bias` cannot be. It is full width and **replicated**, and its correct
   contribution is `sum_e w_e * bias_e` weighted by the ROUTER weights -- which exist only
   on the AG rank. Applying it on an expert rank would add it once per rank. So the AG side
   computes it and hands it to `afd_moe_output` via `shared_output`, which adds it exactly
   once after combine. That is the same placement the colocated path uses (`out + bias_out`
   after the all-reduce, GptOssMoE._forward_fused) and it reuses the hook built for
   DeepSeek-style shared experts rather than inventing a mechanism.

2. **Activation.** `gptoss_swiglu` is not a SiLU variant: gate/up are interleaved
   `[g0,u0,g1,u1,...]`, clamped at 7.0, alpha-scaled 1.702, and it computes `(up+1)*glu`.
   Passed via `MoELayer(activation=...)`, which the Triton runner already forwards.

3. **Attention sinks and per-layer sliding window.** Carried by `RopeAttn` exactly as in the
   colocated model, so `forward_attention` needs no special casing -- but the router has a
   bias and `norm_topk_prob` is forced True for this model (models/config.py).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple

import torch
from minisgl.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.layers.moe import MoELayer
from minisgl.models.afd import ModelStageState
from minisgl.utils import nvtx_annotate

from .utils import RopeAttn

if TYPE_CHECKING:
    from .config import ModelConfig


# NEGATIVE CONTROL for the correctness gate, deliberately wired as an env knob rather
# than a throwaway local edit so the result stays reproducible.
#
# The whole point: a PASS on T-10 means nothing unless the gate can FAIL when the term
# under test is broken. Setting this to 1 drops the router-weighted `down_proj_bias`
# contribution entirely. If T-10 still reports only a marginal divergence with the term
# missing, then T-10 at 16 prompts is BLIND to it and cannot be cited as evidence that
# the bias is applied correctly -- the direct MoE parity test becomes mandatory.
#
# Never set this outside that experiment: it makes the model silently wrong.
_DROP_DOWN_BIAS = os.environ.get("MINISGL_GPTOSS_AFD_DROP_DOWN_BIAS", "0") == "1"
if _DROP_DOWN_BIAS:
    # Printed at import in every AG worker, so the log PROVES the knob reached the actor.
    # Without this, a negative control that shows no change is ambiguous: term is
    # negligible, or the env var never propagated through Ray?
    print(
        "[gpt_oss_afd] NEGATIVE CONTROL ACTIVE: down_proj_bias term DROPPED "
        "(MINISGL_GPTOSS_AFD_DROP_DOWN_BIAS=1). The model is deliberately wrong.",
        flush=True,
    )

# route() runs 36x per decode step and costs 218 us/layer for ONE token
# (doc 28 §12.1). These split that; no-op unless MINISGL_PROFILE_DIR is set.
from minisgl.profiling import step_timer as _sub

class GptOssAfdTopK(NamedTuple):
    topk_ids: torch.Tensor | None      # [tokens, top_k]
    topk_weights: torch.Tensor | None  # [tokens, top_k] fp32
    router_logits: torch.Tensor | None = None
    renormalize: bool = True
    deepep_topk: bool = False
    dispatch_fp8: bool = False          # gpt-oss experts are bf16 here (MXFP4 dequantised)
    # Router-weighted down_proj bias, added once after combine. See module docstring.
    shared_output: torch.Tensor | None = None


class _GptOssAfdAGExperts(BaseOP):
    """Holds ONLY `down_proj_bias`, at its real checkpoint key.

    The AG rank needs this table to form `sum_e w_e * bias_e` (see the module docstring), and
    declaring it at the real checkpoint key lets the loader fill it. It does need one special
    case: `skip_expert_weights` drops EVERY key containing `.experts.` on the AG side
    (weight.py), so this one is exempted there explicitly. Nothing else of the expert weights
    lives here -- the table is replicated and ~26 MB across all 36 layers.
    """

    def __init__(self, config: ModelConfig):
        self.down_proj_bias = torch.empty(config.num_experts, config.hidden_size)


class _GptOssAfdRouter(BaseOP):
    """Router at the legacy `.mlp.router` path. gpt-oss's router HAS a bias."""

    def __init__(self, config: ModelConfig):
        self.router = LinearReplicated(config.hidden_size, config.num_experts, has_bias=True)
        self.experts = _GptOssAfdAGExperts(config)


class GptOssAfdAGLayer(BaseOP):
    """AG layer: attention (sinks + per-layer window) + router. No experts."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = RopeAttn(
            config,
            layer_id,
            has_qk_norm=False,
            has_attn_bias=config.attention_bias,
            has_o_proj_bias=config.attention_bias,
            has_sinks=True,
        )
        self.mlp = _GptOssAfdRouter(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.num_experts = config.num_experts
        self._layer_id = layer_id

    @nvtx_annotate("AFD_AG_Attn_{}", layer_id_field="_layer_id")
    def forward_attention(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> ModelStageState:
        h, residual = self.input_layernorm.forward(hidden_states, residual)
        h = self.self_attn.forward(h)
        h, residual = self.post_attention_layernorm.forward(h, residual)
        return ModelStageState(hidden_states=h, residual=residual)

    def _down_bias_term(self, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """sum_e w_e * down_proj_bias[e], per token.

        Same computation as GptOssMoE._down_bias_term: scatter the top-k weights into a
        dense [tokens, num_experts] matrix and contract against the bias table. Done here,
        on the AG rank, because the router weights are here and because the result must be
        added exactly once -- not once per expert rank.
        """
        bias = self.mlp.experts.down_proj_bias
        dense = torch.zeros(
            weights.shape[0], self.num_experts, dtype=weights.dtype, device=weights.device
        )
        dense.scatter_(1, ids.to(torch.int64), weights)
        return (dense @ bias.to(weights.dtype)).to(bias.dtype)

    @nvtx_annotate("AFD_AG_Route_{}", layer_id_field="_layer_id")
    def route(self, hidden_states: torch.Tensor) -> GptOssAfdTopK:
        h = hidden_states.view(-1, hidden_states.shape[-1])
        with _sub("r.router_gemm"):
            logits = self.mlp.router.forward(h)
        with _sub("r.topk"):
            weights, ids = torch.topk(logits, self.top_k, dim=-1)
        # gpt-oss softmaxes the top-k logits (config.py forces norm_topk_prob True), which
        # is softmax-after-select, NOT softmax-then-select. fused_topk does the latter.
        #
        # dtype=weights.dtype (bf16), NOT fp32. This is the router's own dtype and it is
        # what HF GptOssTopKRouter and the colocated path (gpt_oss.py:_route) both use.
        # fp32 here is *more* accurate but DIFFERENT: it moves every routing weight by
        # ~1e-3 relative, which is orders of magnitude above reduction-order noise and
        # feeds both the expert scaling and the down_proj_bias term below. The fp32
        # convention comes from fused_topk (moe/fused.py), which is the wrong reference
        # for a model whose router is defined in bf16.
        with _sub("r.softmax"):
            weights = torch.softmax(weights, dim=-1, dtype=weights.dtype)
        with _sub("r.ids_cast"):
            ids = ids.to(torch.int32)
        # Bias term first, in bf16, so it matches the colocated `dense @ down_proj_bias`
        # bit for bit; only then widen for the wire, which the M2N/DeepEP dispatch and
        # the grouped kernel both expect in fp32. bf16 -> fp32 is exact, so widening
        # after the softmax preserves colocated parity rather than reintroducing the gap.
        with _sub("r.down_bias"):
            shared_output = self._down_bias_term(weights, ids)
        if _DROP_DOWN_BIAS:
            # NEGATIVE CONTROL, not a runtime option. See the constant's comment.
            shared_output = None
        return GptOssAfdTopK(
            topk_ids=ids,
            topk_weights=weights.to(torch.float32),
            renormalize=self.renormalize,
            dispatch_fp8=False,
            shared_output=shared_output,
        )


class GptOssAfdDenseRouterStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [GptOssAfdAGLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> ModelStageState:
        return ModelStageState(
            hidden_states=self.embed_tokens.forward(input_ids), residual=None
        )

    def forward_attention(self, layer_id: int, h: torch.Tensor, residual=None) -> ModelStageState:
        return self.layers.op_list[layer_id].forward_attention(h, residual)

    def route(self, layer_id: int, h: torch.Tensor) -> GptOssAfdTopK:
        return self.layers.op_list[layer_id].route(h)

    def finalize_hidden(self, h: torch.Tensor, residual=None) -> torch.Tensor:
        return self.norm.forward(h, residual)[0]


class GptOssAfdDenseRouterForCausalLM(BaseOP):
    def __init__(self, config: ModelConfig):
        self.model = GptOssAfdDenseRouterStage(config)
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


# ================================ EG: experts only ================================


class _GptOssAfdExpertMLP(BaseOP):
    """Experts at the legacy `.mlp.experts` path (EG side has no router/attn)."""

    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            activation="gptoss_swiglu",
            quant=config.quant,
            a2a_backend="none",
            has_bias=True,
        )
        # Refuse the one configuration that produces SILENT GARBAGE rather than an error.
        #
        # With ep_size == 1 and tp_size > 1 the experts are TP-sharded on the intermediate
        # dim, so every EG rank computes a PARTIAL sum that has to be added across ranks.
        # MoELayer.forward does exactly that (`if self.tp_size > 1: all_reduce`), but the
        # AFD EG path calls `_run_moe_core` directly and so skips it -- and the dispatch is
        # expert-routed, which with a single EP group sends every token to one rank. The
        # result serves happily and answers "." to everything: measured 0/32 exact, 32 real
        # divergences at token 0, our token not even in HF's top-5.
        #
        # This is the "partial MoE TP/EP split" MoELayer.__init__ already says it does not
        # support; its own check only fires for ep_size > 1, so ep_size == 1 slipped past.
        # For AFD 1A+3F the right knob is MLP_EP=3 (the ragged 128/3 split is handled:
        # div_ceil gives 43 per rank and the loader tracks capacity separately).
        if self.experts.ep_size == 1 and self.experts.tp_size > 1:
            raise RuntimeError(
                f"gpt-oss AFD expert stage got ep_size=1 with tp_size="
                f"{self.experts.tp_size}: the experts would be TP-sharded on the "
                "intermediate dim, but the AFD EG path does not reduce partial sums "
                "across MLP ranks, so the output would be silently wrong. Set "
                "--afd-mlp-ep-size equal to --afd-mlp-tp-size (MLP_EP=MLP_TP)."
            )


class GptOssAfdEGLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.mlp = _GptOssAfdExpertMLP(config)
        self._layer_id = layer_id

    @nvtx_annotate("AFD_EG_Experts_{}", layer_id_field="_layer_id")
    def run_experts(self, dispatch_output) -> torch.Tensor:
        combine_input = self.mlp.experts._run_moe_core(dispatch_output)
        return combine_input.hidden_states


class _GptOssAfdExpertStageInner(BaseOP):
    def __init__(self, config: ModelConfig):
        self.layers = OPList(
            [GptOssAfdEGLayer(config, i) for i in range(config.num_layers)]
        )

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.layers.op_list[layer_id].run_experts(dispatch_output)


class GptOssAfdExpertStage(BaseOP):
    def __init__(self, config: ModelConfig):
        self.model = _GptOssAfdExpertStageInner(config)

    def dispatch_uses_fp8(self) -> bool:
        # MXFP4 experts are dequantised to bf16 at load (weight.py), so there is no fp8
        # dispatch path for this model on this stack.
        return False

    def run_experts(self, layer_id: int, dispatch_output) -> torch.Tensor:
        return self.model.run_experts(layer_id, dispatch_output)


__all__ = [
    "GptOssAfdTopK",
    "GptOssAfdAGLayer",
    "GptOssAfdDenseRouterStage",
    "GptOssAfdDenseRouterForCausalLM",
    "GptOssAfdEGLayer",
    "GptOssAfdExpertStage",
]
