from __future__ import annotations

from typing import Any

import torch


def capture_cuda_graph(
    *,
    device: torch.device,
    engine_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    extra_streams: tuple[torch.cuda.Stream, ...] = (),
    overlap_comm: bool,
    pool,
    fn,
) -> tuple[torch.cuda.CUDAGraph, Any]:
    streams = [comm_stream]
    for stream in extra_streams:
        if all(stream is not existing for existing in streams):
            streams.append(stream)

    torch.cuda.synchronize(device)
    if overlap_comm:
        for stream in streams:
            stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(engine_stream):
        with torch.cuda.graph(graph, pool=pool, stream=engine_stream):
            if overlap_comm:
                for stream in streams:
                    stream.wait_stream(engine_stream)
            fn()
            if overlap_comm:
                for stream in streams:
                    engine_stream.wait_stream(stream)

    engine_stream.synchronize()
    if overlap_comm:
        for stream in streams:
            stream.synchronize()
    return graph, graph.pool() if pool is None else pool


def clone_graph_output(obj):
    """Copy a replay's result out of the graph's own memory.

    A captured graph writes its outputs into buffers owned by its private memory pool, and the next
    replay of the SAME graph overwrites them. The AG/EG pipelines then hand those buffers to
    `dispatch`/`combine`, which run on a *lane* stream while the replay ran on the engine stream, so
    with `afd_num_mb >= 2` more steps are in flight and a buffer gets overwritten while a previous
    step's lane work is still reading it.

    That is the num_mb=2 + graphs crash of docs 42/43. Under `AMD_SERIALIZE_KERNEL=3` the fault
    lands on `rccl_m2n_adapter.py:329` -- `topk_ids.to(...)`, the first read of the route graph's
    retained output -- rather than on the later `tolist()` where the async report surfaced it. Two
    earlier hypotheses (shared memory pool, shared RCCL communicator) were refuted before this
    diagnostic was run; it named the site in one attempt.

    Cloning allocates fresh memory per call from the caching allocator, so no two steps share a
    buffer. The copies are small at decode (topk ids/weights are (T, K); attention and expert
    outputs are (T, hidden) with T <= 32), which is why this is affordable per layer.
    """
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # NamedTuple: rebuild by field
        return type(obj)(*(clone_graph_output(x) for x in obj))
    if isinstance(obj, (tuple, list)):
        return type(obj)(clone_graph_output(x) for x in obj)
    return obj
