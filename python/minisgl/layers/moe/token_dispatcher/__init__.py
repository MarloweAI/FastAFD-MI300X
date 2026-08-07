from __future__ import annotations

from .base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from .deepep import DeepEPDispatcher
from .standard import StandardCombineInput, StandardDispatcher, StandardDispatchOutput
from ..moe_runner.base import MoeA2ABackend, MoeRunnerConfig


def create_moe_dispatcher(
    a2a_backend: str,
    moe_runner_config: MoeRunnerConfig,
    dp_size: int = 1,
) -> BaseDispatcher:
    backend = MoeA2ABackend(a2a_backend)
    if backend.is_none():
        return StandardDispatcher(
            moe_runner_config=moe_runner_config,
            dp_size=dp_size,
        )
    if backend.is_deepep():
        # DeepEP is still useful with moe-dp size 1 when EP equals dense TP:
        # it gives the FP8/DeepGEMM path one dispatcher contract and lets the
        # fused topk kernel handle padded rows.  If EP is not enabled, keep the
        # no-A2A standard path.
        if int(moe_runner_config.ep_size) < int(moe_runner_config.tp_size):
            return StandardDispatcher(
                moe_runner_config=moe_runner_config,
                dp_size=dp_size,
            )
        return DeepEPDispatcher(moe_runner_config=moe_runner_config)
    raise NotImplementedError(f"Unsupported MoE a2a backend: {a2a_backend!r}")


__all__ = [
    "BaseDispatcher",
    "CombineInput",
    "CombineInputFormat",
    "DispatchOutput",
    "DispatchOutputFormat",
    "DeepEPDispatcher",
    "StandardCombineInput",
    "StandardDispatcher",
    "StandardDispatchOutput",
    "create_moe_dispatcher",
]
