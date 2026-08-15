from __future__ import annotations

import pytest

from minisgl.afd_protocol import AfdTopology
from minisgl.moe.deepep_m2n_union import build_deepep_union_m2n_layout


FULL_NODE_SPLITS = tuple((attention, 8 - attention) for attention in range(1, 8))


@pytest.mark.parametrize(("attention", "experts"), FULL_NODE_SPLITS)
def test_all_mi355x_full_node_splits(attention: int, experts: int) -> None:
    topology = AfdTopology(
        attn_dp_size=attention,
        mlp_dp_size=experts,
        attn_tp_size=1,
        mlp_tp_size=1,
        ep_size=experts,
    )
    assert topology.total_workers == 8
    assert topology.total_attn_workers == attention
    assert topology.total_mlp_workers == experts
    assert topology.attn_fanin_per_mlp_dp == max(1, -(-attention // experts))
    assert topology.mlp_fanout_per_attn_dp == max(1, -(-experts // attention))


@pytest.mark.parametrize(("attention", "experts"), FULL_NODE_SPLITS)
def test_gptoss_expert_shards_cover_real_experts_once(
    attention: int, experts: int
) -> None:
    layout = build_deepep_union_m2n_layout(
        ag_size=attention,
        eg_size=experts,
        real_num_experts=128,
    )
    real_ids: list[int] = []
    for expert_rank in range(experts):
        real_ids.extend(layout.real_expert_range_for_eg_rank(expert_rank))
    assert real_ids == list(range(128))
    assert layout.experts_per_union_rank == -(-128 // experts)


def test_tp_divisibility_is_still_enforced() -> None:
    with pytest.raises(ValueError, match="attn_tp_size and mlp_tp_size"):
        AfdTopology(5, 3, 2, 3, 3)
