"""Permutes bridging the M2N expanded layout to the BF16 Triton MoE runner.

Why this exists: on the AFD expert side `--afd-moe-runner-backend` defaults to
`deep_gemm`, and DeepGEMM is SM90/SM100-only (dev_log/qwen/02_dependency_inventory.md
sec D). Upstream registers permutes only for ("deepep_elastic", "deep_gemm"); the
Triton runners are registered against the "standard" format, whose layout is
different. This module supplies the missing pair so the Triton BF16 expert path can
serve AFD on gfx942. See dev_log/qwen/12_afd_wireup.md.

Layout translation
------------------
`RcclM2NAdapter.dispatch` returns rows **already expanded** — one row per
(token, expert) pair — **grouped by expert**, plus a per-expert prefix-sum count.
`TritonRunner` instead wants the "standard" shape: rows are tokens, and
`topk_output = (topk_weights, topk_ids)` says which experts each row goes to.

The two reconcile exactly when the expanded rows are presented as `top_k == 1`
tokens: row i is a token whose single expert is the one it was routed to. The
grouped GEMM inside `fused_experts_impl` then does precisely the per-expert
grouping the elastic layout was already in, and `moe_sum_reduce` over `topk=1` is a
no-op.

Expert ids are converted to **rank-local** indices and `expert_map` is overridden to
None, because the expert weights held on an EG rank cover only that rank's shard.
Using global ids would require the layer's expert map to be set up consistently by
the AFD weight loader; local ids make this path independent of that.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from minisgl.layers.moe.token_dispatcher.base import (
    CombineInputFormat,
    DispatchOutputFormat,
)
from minisgl.layers.moe.permute_registry import register_post_permute, register_pre_permute


class M2NTritonRunnerInput(NamedTuple):
    """What `TritonRunner.apply` reads: hidden_states, topk_output, expert-map override."""

    hidden_states: torch.Tensor
    topk_output: tuple[torch.Tensor, torch.Tensor]
    use_expert_map_override: bool
    expert_map_override: torch.Tensor | None
    global_num_experts_override: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.M2N_EXPANDED


class M2NCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    handle: object

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.M2N_EXPANDED


def _local_expert_ids(dispatch_output) -> tuple[torch.Tensor, int]:
    """Global expert ids -> rank-local ids, plus the local expert count."""
    layout = dispatch_output.layout
    experts_per_rank = layout.experts_per_union_rank
    if dispatch_output.is_eg:
        first_local = (dispatch_output.rank - layout.ag_size) * experts_per_rank
    else:
        # AG ranks own only dummy experts and receive nothing in practice; keep the
        # arithmetic well-defined rather than special-casing empty tensors.
        first_local = dispatch_output.rank * experts_per_rank
    ids = dispatch_output.topk_ids
    if ids is None:
        raise RuntimeError("M2N dispatch output is missing topk_ids (global expert ids)")
    local = ids.reshape(-1, 1).to(torch.int32) - int(first_local)
    return local, int(experts_per_rank)


@register_pre_permute("m2n_expanded", "triton")
def _pre_permute_m2n_to_triton(dispatch_output) -> M2NTritonRunnerInput:
    rows = dispatch_output.hidden_states
    local_ids, experts_per_rank = _local_expert_ids(dispatch_output)

    weights = dispatch_output.topk_weights
    if weights is None:
        weights = torch.ones(
            (rows.shape[0], 1), device=rows.device, dtype=torch.float32
        )
    else:
        weights = weights.reshape(-1, 1).to(torch.float32)

    if int(local_ids.shape[0]) != int(rows.shape[0]):
        raise RuntimeError(
            "M2N pre-permute row mismatch: "
            f"rows={rows.shape[0]} expert_ids={local_ids.shape[0]}"
        )
    return M2NTritonRunnerInput(
        hidden_states=rows,
        # top_k == 1: each expanded row goes to exactly one expert. The runner
        # applies these weights (MUL_ROUTED_WEIGHT), matching what the DeepGEMM
        # runner does, so `combine` stays a pure sum.
        topk_output=(weights, local_ids),
        use_expert_map_override=True,
        expert_map_override=None,          # weights hold only this rank's experts
        global_num_experts_override=experts_per_rank,
    )


@register_post_permute("triton", "m2n_expanded")
def _post_permute_triton_to_m2n(runner_output: torch.Tensor, dispatch_output):
    return M2NCombineInput(
        hidden_states=runner_output,
        handle=dispatch_output.handle,
    )


# AITER and the BF16/FP8 Triton runners consume the same top-k-one view of the
# expanded M:N rows; their compute kernels differ after this layout bridge.
register_pre_permute("m2n_expanded", "aiter")(_pre_permute_m2n_to_triton)
register_post_permute("aiter", "m2n_expanded")(_post_permute_triton_to_m2n)
register_pre_permute("m2n_expanded", "triton_fp8")(_pre_permute_m2n_to_triton)
register_post_permute("triton_fp8", "m2n_expanded")(_post_permute_triton_to_m2n)


__all__ = [
    "M2NTritonRunnerInput",
    "M2NCombineInput",
]
