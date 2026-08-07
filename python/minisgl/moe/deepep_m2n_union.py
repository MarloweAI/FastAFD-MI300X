from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepEPUnionM2NLayout:
    ag_size: int
    eg_size: int
    real_num_experts: int

    @property
    def union_size(self) -> int:
        return self.ag_size + self.eg_size

    @property
    def experts_per_union_rank(self) -> int:
        return (self.real_num_experts + self.eg_size - 1) // self.eg_size

    @property
    def dummy_expert_offset(self) -> int:
        return self.ag_size * self.experts_per_union_rank

    @property
    def deepep_num_experts(self) -> int:
        return self.union_size * self.experts_per_union_rank

    def union_rank_for_eg_rank(self, eg_rank: int) -> int:
        _validate_rank("eg_rank", eg_rank, self.eg_size)
        return self.ag_size + int(eg_rank)

    def expert_range_for_union_rank(self, union_rank: int) -> range:
        _validate_rank("union_rank", union_rank, self.union_size)
        start = int(union_rank) * self.experts_per_union_rank
        return range(start, start + self.experts_per_union_rank)

    def dummy_expert_range_for_ag_rank(self, ag_rank: int) -> range:
        _validate_rank("ag_rank", ag_rank, self.ag_size)
        return self.expert_range_for_union_rank(int(ag_rank))

    def real_expert_range_for_eg_rank(self, eg_rank: int) -> range:
        union_rank = self.union_rank_for_eg_rank(eg_rank)
        deepep_range = self.expert_range_for_union_rank(union_rank)
        start = min(self.real_num_experts, deepep_range.start - self.dummy_expert_offset)
        stop = min(self.real_num_experts, deepep_range.stop - self.dummy_expert_offset)
        return range(
            start,
            stop,
        )


def build_deepep_union_m2n_layout(
    *,
    ag_size: int,
    eg_size: int,
    real_num_experts: int,
) -> DeepEPUnionM2NLayout:
    _validate_layout(ag_size=ag_size, eg_size=eg_size, real_num_experts=real_num_experts)
    return DeepEPUnionM2NLayout(
        ag_size=int(ag_size),
        eg_size=int(eg_size),
        real_num_experts=int(real_num_experts),
    )


def remap_real_to_deepep_experts(
    real_topk_ids: torch.Tensor,
    layout: DeepEPUnionM2NLayout,
) -> torch.Tensor:
    if real_topk_ids.ndim != 2:
        raise ValueError(f"real_topk_ids must be 2D, got shape={tuple(real_topk_ids.shape)}")
    # No per-token bounds check here: it would require .item() (a D2H CPU sync) on
    # EVERY dispatch -> a per-MoE-layer stall. topk ids come from a router topk over
    # [0, num_experts) and are valid by construction.
    offset = int(layout.dummy_expert_offset)
    return torch.where(real_topk_ids >= 0, real_topk_ids + offset, real_topk_ids)


def _is_cuda_graph_capture_active(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def deterministic_real_topk_ids(
    *,
    ag_rank: int,
    num_tokens: int,
    top_k: int,
    layout: DeepEPUnionM2NLayout,
) -> torch.Tensor:
    _validate_rank("ag_rank", ag_rank, layout.ag_size)
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be nonnegative, got {num_tokens}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if top_k > layout.real_num_experts:
        raise ValueError(
            f"top_k={top_k} cannot exceed real_num_experts={layout.real_num_experts}"
        )
    if num_tokens == 0:
        return torch.empty((0, top_k), dtype=torch.int64)
    token_ids = torch.arange(num_tokens, dtype=torch.int64).view(-1, 1)
    slot_ids = torch.arange(top_k, dtype=torch.int64).view(1, -1)
    base = int(ag_rank) * int(num_tokens) * int(top_k)
    return (base + token_ids * int(top_k) + slot_ids) % layout.real_num_experts


def deterministic_hidden_states(
    *,
    rank: int,
    num_tokens: int,
    hidden_size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be nonnegative, got {num_tokens}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    values = torch.arange(
        int(num_tokens) * int(hidden_size),
        dtype=torch.float32,
        device=device,
    ).reshape(int(num_tokens), int(hidden_size))
    values = (values % 16) + float(int(rank) * 16)
    return values.to(dtype=dtype)


def _validate_layout(*, ag_size: int, eg_size: int, real_num_experts: int) -> None:
    if ag_size <= 0:
        raise ValueError(f"ag_size must be positive, got {ag_size}")
    if eg_size <= 0:
        raise ValueError(f"eg_size must be positive, got {eg_size}")
    if real_num_experts <= 0:
        raise ValueError(f"real_num_experts must be positive, got {real_num_experts}")


def _validate_rank(name: str, rank: int, size: int) -> None:
    if not 0 <= int(rank) < int(size):
        raise ValueError(f"{name} must be in [0, {size}), got {rank}")


__all__ = [
    "DeepEPUnionM2NLayout",
    "build_deepep_union_m2n_layout",
    "deterministic_hidden_states",
    "deterministic_real_topk_ids",
    "remap_real_to_deepep_experts",
]
