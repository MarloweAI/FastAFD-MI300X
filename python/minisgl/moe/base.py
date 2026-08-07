from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool,
        expert_map: torch.Tensor | None = None,
        global_num_experts: int | None = None,
        # Expert-projection biases. gpt-oss has them on both projections; every other
        # MoE model here has none, hence the None defaults. Without these the AFD path
        # could not express gpt-oss at all -- the biases loaded on the AG ranks and were
        # silently never applied (dev_log/gpt_oss_120b/24_afd_not_supported.md).
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
