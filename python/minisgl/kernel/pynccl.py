from __future__ import annotations

import functools
import os
import site
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...
        @abstractmethod
        def get_async_error(self) -> int: ...
        @abstractmethod
        def get_async_error_string(self) -> str: ...
        @abstractmethod
        def abort(self) -> None: ...

else:
    PyNCCLCommunicator = Any


def _is_hip() -> bool:
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


@functools.cache
def _nccl_ldflags() -> list[str]:
    """Link flags for the collectives library: NCCL on CUDA, RCCL on ROCm.

    RCCL is API-compatible for everything pynccl.cu uses, including the host-side
    ncclCommWindowRegister/ncclMemAlloc calls. (RCCL does NOT provide the
    device-side nccl_device.h API -- that is what blocks DeepEP, not this.)
    """
    flags = []
    lib_dirs: list[Path] = []
    hip = _is_hip()

    if hip:
        rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        lib_dirs.append(rocm / "lib")
        # torch's ROCm wheels bundle their own librccl.so; prefer whichever is found
        for root in site.getsitepackages():
            lib_dirs.append(Path(root) / "torch" / "lib")
    else:
        for root in site.getsitepackages():
            lib_dirs.append(Path(root) / "nvidia" / "nccl" / "lib")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        lib_dirs.append(Path(conda_prefix) / "lib")

    seen: set[Path] = set()
    for lib_dir in lib_dirs:
        if lib_dir.exists() and lib_dir not in seen:
            flags.append(f"-L{lib_dir}")
            seen.add(lib_dir)

    flags.append("-lrccl" if hip else "-lnccl")
    return flags


@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=_nccl_ldflags())


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            group=tp_cpu_group,
            group_src=0,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            group=tp_cpu_group,
            group_src=0,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
