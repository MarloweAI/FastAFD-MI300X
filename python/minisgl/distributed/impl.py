from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        if x.shape[0] % tp_size != 0:
            raise ValueError(
                "reduce_scatter requires x.shape[0] % world_size == 0; "
                f"got x.shape[0]={x.shape[0]} world_size={tp_size}"
            )
        shape = list(x.shape)
        shape[0] //= tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "PyNCCLDistributedImpl.reduce_scatter is not implemented; "
            "use TorchDistributedImpl instead"
        )


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].reduce_scatter(x)


def get_active_pynccl_communicator() -> PyNCCLCommunicator | None:
    for plugin in reversed(DistributedCommunicator.plugins):
        if isinstance(plugin, PyNCCLDistributedImpl):
            return plugin.comm
    return None


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    for plugin in reversed(DistributedCommunicator.plugins):
        close = getattr(plugin, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception:
            pass
    DistributedCommunicator.plugins = [TorchDistributedImpl()]
