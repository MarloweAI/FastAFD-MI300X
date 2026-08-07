from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch


class DispatchOutputFormat(Enum):
    STANDARD = "standard"
    DEEPEP_ELASTIC = "deepep_elastic"
    # Rows already expanded to one per (token, expert) pair and grouped by expert,
    # as produced by minisgl.moe.rccl_m2n_adapter on the AFD expert side. Distinct
    # from DEEPEP_ELASTIC so its permutes cannot be applied to a DeepEP handle,
    # whose recv_topk_ids/handle internals differ.
    M2N_EXPANDED = "m2n_expanded"


@runtime_checkable
class DispatchOutput(Protocol):
    hidden_states: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat: ...


class CombineInputFormat(Enum):
    STANDARD = "standard"
    DEEPEP_ELASTIC = "deepep_elastic"
    M2N_EXPANDED = "m2n_expanded"


@runtime_checkable
class CombineInput(Protocol):
    @property
    def format(self) -> CombineInputFormat: ...


class BaseDispatcher(ABC):
    @abstractmethod
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: Any
    ) -> DispatchOutput: ...

    @abstractmethod
    def combine(self, combine_input: CombineInput) -> torch.Tensor: ...
