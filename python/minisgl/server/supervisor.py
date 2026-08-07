"""
Shared supervisor utilities and backend supervisor factory.

Backend-specific supervisors live in dedicated modules:
  - supervisor_local.py  →  LocalBackendSupervisor
  - supervisor_ray.py    →  RayBackendSupervisor
  - supervisor_afd.py    →  AfdBackendSupervisor
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
from queue import Empty
from typing import TYPE_CHECKING, Any

from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


# ---------------------------------------------------------------------------
# Scheduler subprocess entry point (shared by Local and Ray backends)
# ---------------------------------------------------------------------------

def _run_scheduler(args: ServerArgs, ack_queue: Any) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        ack_queue.put(f"Scheduler rank {args.tp_info.rank} is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


# ---------------------------------------------------------------------------
# Tokenizer processes (shared by all backends)
# ---------------------------------------------------------------------------

def _spawn_tokenizer_processes(server_args: ServerArgs, ack_queue: mp.Queue[str]) -> list[mp.Process]:
    from minisgl.tokenizer import tokenize_worker

    num_tokenizers = server_args.num_tokenizer
    processes: list[mp.Process] = []
    proc = mp.Process(
        target=tokenize_worker,
        kwargs={
            "tokenizer_path": server_args.model_path,
            "addr": server_args.zmq_detokenizer_addr,
            "backend_addr": server_args.zmq_backend_addr,
            "frontend_addr": server_args.zmq_frontend_addr,
            "local_bs": 1,
            "create": server_args.tokenizer_create_addr,
            "tokenizer_id": num_tokenizers,
            "ack_queue": ack_queue,
        },
        daemon=False,
        name="minisgl-detokenizer-0",
    )
    proc.start()
    processes.append(proc)

    for i in range(num_tokenizers):
        proc = mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_tokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": i,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name=f"minisgl-tokenizer-{i}",
        )
        proc.start()
        processes.append(proc)

    return processes


# ---------------------------------------------------------------------------
# Process lifecycle helpers (shared by all backends)
# ---------------------------------------------------------------------------

def _format_failures(processes: list[mp.Process]) -> list[str]:
    failures = []
    for proc in processes:
        if proc.exitcode not in (None, 0):
            failures.append(f"{proc.name}(pid={proc.pid}, exitcode={proc.exitcode})")
    return failures


def _shutdown_processes(processes: list[mp.Process], timeout_s: float = 5.0) -> None:
    alive = [proc for proc in processes if proc.is_alive()]
    for proc in alive:
        if proc.pid is None:
            continue
        try:
            os.kill(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    for proc in alive:
        proc.join(timeout=timeout_s)

    alive = [proc for proc in processes if proc.is_alive()]
    for proc in alive:
        proc.terminate()

    for proc in alive:
        proc.join(timeout=timeout_s)

    alive = [proc for proc in processes if proc.is_alive()]
    for proc in alive:
        proc.kill()
        proc.join(timeout=1.0)


def _wait_for_acks(
    *,
    expected_acks: int,
    ack_queue: mp.Queue[str],
    processes: list[mp.Process],
    logger,
) -> None:
    received_acks = 0
    while received_acks < expected_acks:
        try:
            logger.info(ack_queue.get(timeout=5))
            received_acks += 1
            continue
        except Empty:
            pass

        failures = _format_failures(processes)
        if failures:
            raise RuntimeError("Backend subprocess exited before ready: " + ", ".join(failures))


def _send_exit_msg(server_args: ServerArgs) -> None:
    from minisgl.message import BaseBackendMsg, ExitMsg
    from minisgl.utils import ZmqPushQueue

    queue = ZmqPushQueue(server_args.zmq_backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    try:
        queue.put(ExitMsg())
    finally:
        queue.stop()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend_supervisor(server_args: ServerArgs):
    if server_args.mode == "afd-serve":
        from .supervisor_afd import AfdBackendSupervisor
        return AfdBackendSupervisor(server_args)
    if server_args.launch_backend == "ray":
        from .supervisor_ray import RayBackendSupervisor
        return RayBackendSupervisor(server_args)
    from .supervisor_local import LocalBackendSupervisor
    return LocalBackendSupervisor(server_args)
