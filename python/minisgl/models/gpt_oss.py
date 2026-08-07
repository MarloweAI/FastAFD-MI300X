"""openai/gpt-oss-120b.

Five things differ from every model already in this tree; see
dev_log/gpt_oss_120b/00_README.md for the full account.

  * attention **sinks** (a per-head logit in the softmax denominator) -- `triton_decode` /
    `triton_prefill` only, via `RopeAttn(has_sinks=True)`;
  * **alternating** sliding(128)/full attention, from `ModelConfig.layer_types`;
  * **interleaved** clamped SwiGLU in the experts, not split-half `silu_and_mul`;
  * **biases** on the router, both expert projections, and all four attention projections;
  * **MXFP4** expert weights, dequantised to BF16 at load (no FP4 datapath on gfx942).

The MoE has two implementations, selected by `MINISGL_GPTOSS_MOE`:

  * **`fused`** (default) -- grouped GEMM via `moe/fused.py`. `moe_align_block_size` buckets
    tokens per expert on device and sizes its outputs from `topk_ids.numel()` and constants
    only, so the launch grid is host-known while the real block count is read from
    `num_tokens_post_padded` inside the kernel. No host sync, and HIP-graph capturable.
  * **`loop`** -- the original per-expert loop. Kept because it is the implementation
    verified 8/8 against HF's own router+experts, so the fused path has something to be
    diffed against (dev_log/gpt_oss_120b/gptoss_moe_fused_parity.py). It is fp32-capable,
    which the grouped kernel is not.

The loop came first on purpose: writing it established the reference, and only then was the
grouped version added and checked against both it and HF. Its cost was measurable --
40-46% of MoE time in host synchronisation, plus no graph capture -- see
dev_log/gpt_oss_120b/gptoss_moe_cost.py.

Still outstanding: TP only (no EP), and no AFD transport integration.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
    gptoss_swiglu,
)
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .config import ModelConfig
from .utils import RopeAttn


class GptOssExpertWeights(BaseOP):
    """The four stacked expert tensors, laid out for `F.linear` (i.e. `x @ W.T`).

    HF stores `gate_up_proj` as `(E, H, 2I)` and `down_proj` as `(E, I, H)` and computes
    `x @ W`. This port uses `F.linear`, so both are held **transposed**. Nothing about the
    shapes catches a mix-up (gpt-oss's hidden and intermediate are both 2880), so the
    transpose is asserted numerically in gptoss_moe_parity.py instead.
    """

    def __init__(self, num_experts: int, hidden: int, inter_per_rank: int):
        # gate_up is column parallel: 2*inter is the output dim, and the interleaved
        # [g_0,u_0,g_1,u_1,...] packing means a contiguous row slice keeps (gate,up) pairs
        # together. down is row parallel: inter is contracted, so its bias is full width.
        from minisgl.models.weight import _mxfp4_packed_enabled

        self.packed = _mxfp4_packed_enabled()
        if self.packed:
            # MXFP4 kept 4-bit: one nibble per value along the contracted dim plus one e8m0
            # scale byte per 32 values. Attribute names match the checkpoint's own
            # `_blocks`/`_scales` keys, which is what lets the loader stay rename-free.
            if hidden % 32 != 0 or inter_per_rank % 32 != 0:
                raise ValueError(
                    f"MINISGL_MXFP4_PACKED needs hidden and per-rank intermediate to be "
                    f"multiples of the 32-wide MXFP4 group; got hidden={hidden}, "
                    f"inter_per_rank={inter_per_rank}"
                )
            self.gate_up_proj_blocks = torch.empty(
                num_experts, 2 * inter_per_rank, hidden // 2, dtype=torch.uint8
            )
            self.gate_up_proj_scales = torch.empty(
                num_experts, 2 * inter_per_rank, hidden // 32, dtype=torch.uint8
            )
            self.down_proj_blocks = torch.empty(
                num_experts, hidden, inter_per_rank // 2, dtype=torch.uint8
            )
            self.down_proj_scales = torch.empty(
                num_experts, hidden, inter_per_rank // 32, dtype=torch.uint8
            )
        else:
            self.gate_up_proj = torch.empty(num_experts, 2 * inter_per_rank, hidden)
            self.down_proj = torch.empty(num_experts, hidden, inter_per_rank)
        # Biases stay BF16 on both paths -- they are not quantised in the checkpoint.
        self.gate_up_proj_bias = torch.empty(num_experts, 2 * inter_per_rank)
        self.down_proj_bias = torch.empty(num_experts, hidden)

    def w1(self):
        """gate_up operand: a `(blocks, scales)` pair when packed, else the dense tensor."""
        if self.packed:
            return (self.gate_up_proj_blocks, self.gate_up_proj_scales)
        return self.gate_up_proj

    def w2(self):
        """down operand, same convention as `w1`."""
        if self.packed:
            return (self.down_proj_blocks, self.down_proj_scales)
        return self.down_proj

    def forward(self):  # never called; the MoE indexes these tensors directly
        raise NotImplementedError


class GptOssMoE(BaseOP):
    """Top-k MoE with per-expert biases and gpt-oss's clamped interleaved SwiGLU.

    Sharding is TP over the intermediate dimension. The interleaved gate/up layout is
    *convenient* here rather than awkward: because the packing is
    ``[g_0, u_0, g_1, u_1, ...]``, a contiguous slice of ``2 * intermediate_per_rank`` rows
    hands each rank complete ``(gate_i, up_i)`` pairs. It is the split-half layout that
    needs per-half slicing (hence `LinearColParallelMerged` taking a list of output sizes).
    """

    def __init__(self, config: ModelConfig):
        tp_info = get_tp_info()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self._tp_size = tp_info.size
        self._comm = DistributedCommunicator()
        self._swiglu_limit = float(config.model_extra.get("swiglu_limit", 7.0))
        # gpt-oss omits norm_topk_prob; ModelConfig forces it True because the router does
        # topk-then-softmax. If it is ever False here the routing weights would not sum to
        # 1 and the output would be quietly scaled down.
        if not config.norm_topk_prob:
            raise ValueError(
                "gpt-oss routing requires renormalised top-k weights (topk -> softmax); "
                "got norm_topk_prob=False"
            )

        # Per-rank intermediate. The packed path sizes itself from the same uniform whole-group
        # split the loader uses, because MXFP4 groups are 32 wide and `div_even` is not
        # group-aligned at tp 4/8 (2880/4 = 720 = 22.5 groups). Uniform-with-padding rather than
        # uneven, so every rank keeps the same shape -- see mxfp4_uniform_group_split. At tp 1/2 the
        # split is exact and this returns exactly what div_even did, so those paths are unchanged.
        from minisgl.models.weight import _mxfp4_packed_enabled, mxfp4_uniform_group_split

        if _mxfp4_packed_enabled():
            if config.moe_intermediate_size % 32 != 0:
                raise ValueError(
                    f"MINISGL_MXFP4_PACKED needs moe_intermediate_size divisible by the 32-wide "
                    f"MXFP4 group; got {config.moe_intermediate_size}"
                )
            per_groups, _ = mxfp4_uniform_group_split(
                config.moe_intermediate_size // 32, tp_info.size
            )
            inter = per_groups * 32
        else:
            inter = div_even(config.moe_intermediate_size, tp_info.size)
        self._inter_per_rank = inter
        # Router is replicated and biased. Its output must be identical on every rank, so
        # that the post-reduce `down_proj_bias` term below is too.
        self.router = LinearReplicated(config.hidden_size, config.num_experts, has_bias=True)
        # Nested under `experts` to mirror the checkpoint exactly
        # (`model.layers.N.mlp.experts.gate_up_proj*`). BaseOP.state_dict() derives keys
        # from attribute names, so matching the structure here is what lets the loader
        # avoid a rename rule.
        self.experts = GptOssExpertWeights(self.num_experts, config.hidden_size, inter)
        # "fused" = grouped GEMM (fast, sync-free, graph-capturable). "loop" = the
        # per-expert loop that was validated 8/8 against HF and is kept as the reference
        # the fused path is diffed against (dev_log/gpt_oss_120b/gptoss_moe_fused_parity.py).
        impl = os.environ.get("MINISGL_GPTOSS_MOE", "fused").strip().lower()
        if impl not in ("fused", "loop"):
            raise ValueError(
                f"MINISGL_GPTOSS_MOE must be 'fused' or 'loop', got {impl!r}"
            )
        self._impl = impl

    @nvtx_annotate("MoE")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._impl == "fused":
            return self._forward_fused(x)
        return self._forward_loop(x)

    def _route(self, x: torch.Tensor):
        """Router -> (top-k weights, top-k ids). Shared by both implementations.

        HF GptOssTopKRouter: topk over the raw logits, then softmax over just those k.
        Softmax is monotonic so the selection matches a softmax-first ordering, and
        softmax over the subset equals a renormalised full softmax. Kept in the
        router's own dtype to match HF exactly.
        """
        logits = self.router.forward(x)
        weights, ids = torch.topk(logits, self.top_k, dim=-1)
        return torch.softmax(weights, dim=-1, dtype=weights.dtype), ids

    def _down_bias_term(self, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """`sum_e w_e * down_proj_bias[e]`, as one scatter plus one GEMM.

        `down_proj_bias` is full width and replicated, so it must land exactly ONCE --
        after the TP all-reduce. That rules out folding it into the grouped kernel, whose
        output is a per-rank partial sum that gets reduced (it would be counted tp_size
        times, the trap from layers/linear.py and §10 of the dev log).

        Computing it as `scatter(w) @ down_bias` keeps it fixed-shape and sync-free, so
        the whole MoE stays HIP-graph capturable. Deliberately NOT special-cased to fold
        the bias into the kernel at tp_size==1: one code path cannot silently diverge
        between TP1 and TP4, which is exactly how the row-parallel bias bug hid.
        """
        dense = torch.zeros(
            weights.shape[0], self.num_experts, dtype=weights.dtype, device=weights.device
        )
        dense.scatter_(1, ids, weights)
        return dense @ self.experts.down_proj_bias

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Grouped-GEMM MoE: no host sync, fixed grid, graph-capturable.

        `moe_align_block_size` sorts tokens into per-expert blocks on device and sizes its
        outputs from `topk_ids.numel()` and constants only, so the launch grid is
        host-known while the real block count is read from `num_tokens_post_padded`
        inside the kernel. That is what removes the `torch.unique(...).tolist()` and
        `nonzero()` round trips the expert loop needed -- measured at 40-46% of MoE time
        in dev_log/gpt_oss_120b/gptoss_moe_cost.py -- and what makes graph capture legal.
        """
        from minisgl.moe.fused import fused_experts_impl

        if x.dtype not in (torch.bfloat16, torch.float16):
            # The grouped kernel has no fp32 path (see kernel/moe_impl.py). Say so rather
            # than silently computing the experts in fp16.
            raise ValueError(
                f"fused gpt-oss MoE requires bfloat16/float16 activations, got {x.dtype}; "
                "set MINISGL_GPTOSS_MOE=loop for an fp32-capable reference path."
            )
        weights, ids = self._route(x)
        bias_out = self._down_bias_term(weights, ids)
        # fused_experts_impl writes its result into `hidden_states` in place and returns
        # it; the bias term above is already computed, so mutating x here is safe.
        out = fused_experts_impl(
            x.contiguous(),
            self.experts.w1(),
            self.experts.w2(),
            weights,
            ids,
            activation="gptoss_swiglu",
            w1_bias=self.experts.gate_up_proj_bias,
            w2_bias=None,  # see _down_bias_term: must be added after the all-reduce
        )
        if self._tp_size > 1:
            out = self._comm.all_reduce(out)
        return out + bias_out

    def _forward_loop(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.experts, "packed", False):
            # The loop path indexes dense per-expert weights directly (F.linear on
            # gate_up_proj[e]); there is no dense tensor when the weights stay MXFP4. Refuse
            # rather than raise AttributeError from inside the loop.
            raise RuntimeError(
                "MINISGL_GPTOSS_MOE=loop is not supported with MINISGL_MXFP4_PACKED=1: the "
                "reference loop needs dequantised per-expert weights. Use the fused path, or "
                "unset MINISGL_MXFP4_PACKED to diff against the loop."
            )
        n_tok, hidden = x.shape
        logits = self.router.forward(x)
        # HF GptOssTopKRouter: topk over the raw logits, then softmax over just those k.
        # Softmax is monotonic so the selection matches a softmax-first ordering, and
        # softmax over the subset equals a renormalised full softmax. Kept in the router's
        # own dtype to match HF exactly.
        weights, ids = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1, dtype=weights.dtype)

        out = torch.zeros_like(x)
        # `down_proj_bias` is full width and replicated, and the routing weights are
        # identical on every rank, so `sum_e w_e * bias_e` must be added exactly ONCE --
        # after the all-reduce. Folding it into the per-expert loop would let the reduce
        # count it tp_size times, the same trap as the row-parallel bias in layers/linear.py.
        bias_out = torch.zeros_like(x)

        # Only experts that actually received a token. At decode bs=1 that is <= top_k (4),
        # not num_experts (128), which is what keeps this loop tolerable.
        for e in torch.unique(ids).tolist():
            rows, slot = (ids == e).nonzero(as_tuple=True)
            w_e = weights[rows, slot].unsqueeze(-1)
            gate_up = F.linear(
                x[rows], self.experts.gate_up_proj[e], self.experts.gate_up_proj_bias[e]
            )
            y = gptoss_swiglu(gate_up, limit=self._swiglu_limit)
            out.index_add_(0, rows, w_e * F.linear(y, self.experts.down_proj[e]))
            bias_out.index_add_(0, rows, w_e * self.experts.down_proj_bias[e])

        if self._tp_size > 1:
            out = self._comm.all_reduce(out)
        return (out + bias_out).view(n_tok, hidden)


class GptOssDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = RopeAttn(
            config,
            layer_id,
            has_attn_bias=config.attention_bias,
            has_o_proj_bias=config.attention_bias,
            has_sinks=True,
        )
        self.mlp = GptOssMoE(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        # `sink` is not passed here: RopeAttn falls back to its own loaded `self.sinks`.
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class GptOssModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [GptOssDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class GptOssForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = GptOssModel(config)
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


__all__ = ["GptOssForCausalLM"]
