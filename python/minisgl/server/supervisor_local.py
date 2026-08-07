from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

from .supervisor import (
    _format_failures,
    _run_scheduler,
    _shutdown_processes,
    _spawn_tokenizer_processes,
    _wait_for_acks,
)

if TYPE_CHECKING:
    from .args import ServerArgs


def _spawn_scheduler_processes(server_args: ServerArgs, ack_queue: mp.Queue[str]) -> list[mp.Process]:
    world_size = server_args.tp_info.size
    processes: list[mp.Process] = []

    for i in range(world_size):
        new_args = replace(
            server_args,
            tp_info=DistributedInfo(i, world_size),
        )
        proc = mp.Process(
            target=_run_scheduler,
            args=(new_args, ack_queue),
            daemon=False,
            name=f"minisgl-TP{i}-scheduler",
        )
        proc.start()
        processes.append(proc)

    return processes


class LocalBackendSupervisor:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.logger = init_logger(__name__, "backend-supervisor")
        self.processes: list[mp.Process] = []
        self.ack_queue: mp.Queue[str] | None = None

    def start(self) -> None:
        if self.processes:
            raise RuntimeError("Backend supervisor has already started")

        mp.set_start_method("spawn", force=True)
        self.ack_queue = mp.Queue()
        scheduler_processes = _spawn_scheduler_processes(self.server_args, self.ack_queue)
        tokenizer_processes = _spawn_tokenizer_processes(self.server_args, self.ack_queue)
        self.processes = scheduler_processes + tokenizer_processes

        try:
            assert self.ack_queue is not None
            _wait_for_acks(
                expected_acks=self.server_args.tp_info.size + self.server_args.num_tokenizer + 1,
                ack_queue=self.ack_queue,
                processes=self.processes,
                logger=self.logger,
            )
        except Exception:
            self.shutdown()
            raise

    def get_failures(self) -> list[str]:
        if not self.processes:
            return []
        return _format_failures(self.processes)

    def shutdown(self) -> None:
        if self.processes:
            _shutdown_processes(self.processes)
            self.processes = []
        if self.ack_queue is not None:
            self.ack_queue.close()
            self.ack_queue = None
