from __future__ import annotations

from importlib import import_module

from .config import SchedulerConfig

__all__ = ["Scheduler", "SchedulerConfig"]

_LAZY_EXPORTS = {
    "Scheduler": ".scheduler",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
