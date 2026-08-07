from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .cpu_trace import enter_nvtx_cpu_trace, exit_nvtx_cpu_trace

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


@contextmanager
def nvtx_range(name: str):
    import torch.cuda.nvtx as nvtx

    record = enter_nvtx_cpu_trace(name)
    try:
        with nvtx.range(name):
            yield
    except BaseException as exc:
        exit_nvtx_cpu_trace(record, exc)
        raise
    else:
        exit_nvtx_cpu_trace(record)


def nvtx_label(name: str, /, **fields: object) -> str:
    if not fields:
        return name
    suffix = "".join(f"[{key}={value}]" for key, value in fields.items() if value is not None)
    return f"{name}{suffix}"


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx_range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
