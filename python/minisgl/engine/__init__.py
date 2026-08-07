from __future__ import annotations

from importlib import import_module

from .config import EngineConfig
from .sample import BatchSamplingArgs

__all__ = [
    "Engine",
    "EngineConfig",
    "ForwardOutput",
    "BatchSamplingArgs",
]

_LAZY_EXPORTS = {
    "Engine": ".engine",
    "ForwardOutput": ".engine",
    "_init_tp_communication": ".engine",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"_init_tp_communication"})
