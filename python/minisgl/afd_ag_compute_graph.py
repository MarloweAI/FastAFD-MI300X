"""Per-layer AG compute graphs: capture compute, leave the M2N collectives eager.

Why this is not the existing `AfdAgDecodeGraphRunner`
----------------------------------------------------
That runner captures the WHOLE AG step, `_run_ag_pipeline_body` included, so its graph contains
the M2N dispatch/combine. That cannot be captured today for two independent reasons
(dev_log/gpt_oss_120b/28_afd_m2n_runs_on_gloo.md §14.1): the transport uses boolean-mask
indexing and `.tolist()` D2H syncs, both capture-illegal; and RCCL requires every rank of a
communicator to capture together, so AG-only capture of a graph holding M2N collectives is a
capture-*time* mismatch, not a replay-time one.

This sidesteps both by capturing only regions with **no M2N collective inside**. Measured, the
AG's compute is 29.2 ms of a 76 ms decode step (38%), and it is almost entirely host-side launch
overhead: `route` issues ~8 tiny kernels for 218 us/layer on a single token, where casting FOUR
int64s to int32 costs 15.6 us because the per-launch floor on this host is ~16 us (doc 28 §12.3).

Scope, and why it is staged
--------------------------
**Stage 1 (this file): `route` only.** `route` is a pure function of the hidden states and static
weights -- router GEMM, top-k, softmax, and the `down_proj_bias` contraction. No per-step
metadata, so a captured graph stays valid across steps. 7.9 ms of the 76 ms step (10.4%).

**Stage 2 (NOT done): add `forward_attention`.** This is the larger prize -- `c.attn` is 21.3 ms
(28%) -- but attention reads per-step metadata (page table, sequence lengths, output locations).
A graph bakes in those addresses, so replay is only correct if the runtime writes the current
step's metadata into the *same* buffers, which is exactly what
`AfdAgDecodeGraphRunner._prepare_capture` exists to do. Capturing attention therefore has to
hook that machinery rather than allocate its own buffers; doing it naively would silently read
stale KV positions, which is far worse than being slow. Left for a focused change.

Mechanism
---------
A graph bakes in addresses, and `launch_compute` rebinds `state.hidden[mb]` to a fresh tensor
every layer, so the captured body reads a **static input buffer** that the caller copies into.
Outputs need no static buffers: the Python objects created during capture own storage from the
graph pool, and each replay rewrites that same storage -- so retaining the capture-time `topk`
is what makes the model-specific output type work without this module knowing the model.

One graph per (bucket, mb, layer): weights differ per layer, so 36 per bucket per microbatch.
Intermediates come from one shared pool, so the marginal cost is graph metadata, not activations.

Capture happens at WARMUP, never inline during a live step
----------------------------------------------------------
The first version captured lazily, on the last layer of a real step. That works at bs=1 and
FAILS under concurrency: with overlapping requests there is M2N work in flight on the lane
streams while capture holds the engine stream, and capture died mid-way with

    capture FAILED bs=6 layer=29 HIP error: operation failed due to a previous error during
    capture

followed by the AG worker erroring out and the RCCL heartbeat monitor tripping, because a
36-graph capture stalls the stream long enough for the peer stage to notice. Sequential T-10
never saw it: with one request in flight there is nothing to corrupt.

So capture now runs once, at warmup, before the coordinator sends any step -- the same point
`warmup_afd_decode_graphs` uses. `route` is **collective-free** (the router is replicated and
the down-bias is a local matmul), so unlike the whole-step graphs this needs no cross-rank
lockstep at all: each AG rank captures on its own.

Buckets come from `MINISGL_AFD_AG_ROUTE_GRAPH_BS` since `decode_graph_bs` is empty whenever
`GRAPH_MAX_BS=0`, which is the AFD default. A running batch that misses every captured bucket
simply runs eager.

ON by default; `MINISGL_AFD_AG_ROUTE_GRAPH=0` disables it. The default was flipped once the
paired A/B/A gate (T-16, `dev_log/probes/selfconsistency_gate.py`) could actually rank it:
over 128 prompts / 4096 tokens, two eager runs of the same binary agree on only **0.8479** of
token positions, while graphs-vs-eager agree on **0.8864** -- so the candidate differs from the
baseline LESS than the baseline differs from itself. See doc 28 §17.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from .afd_support import log_line
from .cuda_graph_utils import clone_graph_output

__all__ = ["AfdAgRouteGraphs", "ag_route_graph_enabled", "AfdAgAttnGraphs", "ag_attn_graph_enabled"]


def ag_route_graph_enabled() -> bool:
    return os.environ.get("MINISGL_AFD_AG_ROUTE_GRAPH", "1") not in ("0", "false", "")


class AfdAgRouteGraphs:
    """Captures `model.route(layer, hidden)` per (bucket, mb, layer)."""

    def __init__(
        self,
        *,
        runtime_state: Any,
        model: Any,
        num_layers: int,
        hidden_size: int,
        num_mb: int,
        log_path: str,
    ) -> None:
        self.state = runtime_state
        self.model = model
        self.device = runtime_state.device
        self.dtype = runtime_state.dtype
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.num_mb = max(1, int(num_mb))
        self.log_path = log_path
        # Memory pool PER MICROBATCH, not one for all captures. A private pool lets HIP reuse
        # blocks between graphs on the assumption they never run simultaneously -- true across
        # buckets and layers within a lane, since those replay sequentially, but FALSE across `mb`:
        # with num_mb>=2 the lanes have their own streams and replay concurrently.
        #
        # NECESSARY BUT NOT SUFFICIENT: tried as a fix for the num_mb=2 crash (doc 42); the crash
        # PERSISTED. Kept because concurrent replay from a shared pool is unsafe regardless.
        self._pools: dict[int, Any] = {}
        self._clone_outputs = self.num_mb > 1
        self._entries: dict[tuple[int, int], dict[str, Any]] = {}
        self._failed: set[tuple[int, int]] = set()
        # Capture is keyed on the OBSERVED batch size, and the running batch fluctuates as
        # requests retire -- a B=32 run captured bs=1, 7 and 8. Left unbounded a long-lived
        # server would accumulate 36 graphs per distinct size seen, so cap the number of
        # buckets and fall back to eager past it rather than growing without limit.
        self.max_buckets = int(os.environ.get("MINISGL_AFD_AG_ROUTE_GRAPH_MAX_BUCKETS", "8"))

    def _key(self, bs: int, mb: int) -> tuple[int, int]:
        return (int(bs), int(mb))

    def usable(self, bs: int, mb: int) -> bool:
        return self._key(bs, mb) in self._entries

    def tried(self, bs: int, mb: int) -> bool:
        k = self._key(bs, mb)
        return k in self._entries or k in self._failed

    def warmup(self) -> None:
        """Capture every configured bucket at a quiescent point. Best effort, never raises."""
        raw = os.environ.get("MINISGL_AFD_AG_ROUTE_GRAPH_BS", "1,2,4,8,16,32")
        try:
            bs_list = sorted({int(x) for x in raw.split(",") if x.strip()})
        except ValueError:
            log_line(self.log_path, f"[ag_route_graph] bad BS list {raw!r}; skipping", flush=True)
            return
        for bs in bs_list:
            if bs <= 0:
                continue
            for mb in range(self.num_mb):
                # Random rather than zeros: an all-equal router input makes top-k arbitrary and
                # can select a different Triton config than replay will want.
                seed = torch.randn(
                    (bs, self.hidden_size), dtype=self.dtype, device=self.device
                )
                self.capture(bs=bs, mb=mb, hidden=seed)
        log_line(
            self.log_path,
            f"[ag_route_graph] warmup done: {len(self._entries)} bucket(s) captured, "
            f"{len(self._failed)} failed, requested={bs_list}",
            flush=True,
        )

    def capture(self, *, bs: int, mb: int, hidden: torch.Tensor) -> bool:
        """Capture `route` for every layer at this (bs, mb). Best effort.

        `hidden` seeds the static input buffer so capture runs on realistic values rather than
        zeros -- a degenerate router input can select a different Triton config than replay
        wants, and top-k over all-equal logits is not representative.
        """
        from .cuda_graph_utils import capture_cuda_graph

        k = self._key(bs, mb)
        if self.tried(bs, mb):
            return self.usable(bs, mb)
        if len(self._entries) >= self.max_buckets:
            self._failed.add(k)
            log_line(
                self.log_path,
                f"[ag_route_graph] bucket cap reached ({self.max_buckets}); bs={bs} mb={mb} "
                f"stays eager. Raise MINISGL_AFD_AG_ROUTE_GRAPH_MAX_BUCKETS if intended.",
                flush=True,
            )
            return False

        buf_in = torch.zeros((bs, self.hidden_size), dtype=self.dtype, device=self.device)
        n = min(int(hidden.shape[0]), bs)
        if n:
            buf_in[:n].copy_(hidden[:n])

        graphs: list[Any] = []
        packets: list[Any] = []
        try:
            # hipBLASLt initialises lazily on first use, and that init is ILLEGAL inside a
            # capture -- it aborts with
            #     Hip error: 'operation not permitted when stream is capturing'(900)
            #     at hipblaslt.cpp:171
            # and takes the worker down. The earlier lazy-capture version never hit it only
            # because by then the router had already run eagerly hundreds of times. Capturing
            # at warmup makes capture the FIRST router call, so every layer's kernels must be
            # forced into existence eagerly first.
            with torch.cuda.stream(self.state.stream):
                for layer in range(self.num_layers):
                    self.model.route(layer, buf_in)
            torch.cuda.synchronize(self.device)

            for layer in range(self.num_layers):
                out: dict[str, Any] = {}

                def body(_layer: int = layer, _out: dict[str, Any] = out) -> None:
                    _out["topk"] = self.model.route(_layer, buf_in)

                graph, self._pools[mb] = capture_cuda_graph(
                    device=self.device,
                    engine_stream=self.state.stream,
                    comm_stream=self.state.stream,
                    overlap_comm=False,
                    pool=self._pools.get(mb),
                    fn=body,
                )
                if "topk" not in out:
                    raise RuntimeError("route() produced no output during capture")
                graphs.append(graph)
                packets.append(out["topk"])
        except Exception as exc:
            # Never let a capture failure break serving: eager remains correct, and a bucket
            # that failed is recorded so it is not retried every step.
            self._failed.add(k)
            log_line(
                self.log_path,
                f"[ag_route_graph] capture FAILED bs={bs} mb={mb} layer={len(graphs)} "
                f"{type(exc).__name__}: {exc} -- eager fallback for this bucket",
                flush=True,
            )
            return False

        self._entries[k] = {"graphs": graphs, "packets": packets, "buf_in": buf_in}
        log_line(
            self.log_path,
            f"[ag_route_graph] captured bs={bs} mb={mb} layers={len(graphs)} "
            f"hidden={self.hidden_size}",
            flush=True,
        )
        return True

    def replay(self, *, bs: int, mb: int, layer: int, hidden: torch.Tensor) -> Any:
        """Copy `hidden` into the static input, replay, and return the retained topk.

        The returned object's tensors are the graph's own storage, refreshed by this replay.
        The caller must consume them before the next replay of the same (bs, mb, layer).
        """
        entry = self._entries[self._key(bs, mb)]
        entry["buf_in"].copy_(hidden)
        entry["graphs"][layer].replay()
        # With more than one microbatch the caller's consumption (dispatch/combine on a LANE
        # stream) outlives this replay, so the graph's own buffer must not be handed out: the next
        # step's replay would overwrite it mid-read. That is the docs 42/43 crash. At num_mb=1 the
        # schedule is strictly serial and the existing path is validated, so it is left untouched.
        out = entry["packets"][layer]
        return clone_graph_output(out) if self._clone_outputs else out


class AfdAgAttnGraphs:
    """Stage 2: captures `forward_attention(layer, hidden, residual)` per (bucket, mb, layer).

    This is the bigger prize than `route` -- `c.attn` is 20.7 ms of a 69.8 ms decode step (29.6%)
    against `route`'s 1.7 ms -- and it is harder for one reason: **attention reads per-step
    metadata** (page table, sequence lengths, output locations) whose addresses a graph bakes in.
    Replay is only correct if the current step's metadata has been written into those same buffers
    first. Done naively it silently reads stale KV positions, which is far worse than being slow.

    The attention backend already has the protocol for exactly this, used by the whole-step
    `AfdAgDecodeGraphRunner`:

        init_capture_graph(max_seq_len, bs_list)   once per backend
        prepare_for_capture(capture_batch)         before capturing
        prepare_for_replay(batch)                  each step, copies the REAL metadata into the
                                                   fixed buffers the captured graph reads

    Crucially `batch.attn_metadata` is **per step, not per layer** (`triton_decode.forward` reads
    `batch.attn_metadata`), so one `prepare_for_replay` per step serves all 36 layer graphs. That is
    what makes per-layer capture affordable: 1 metadata refill + 36 cheap replays, instead of 36 of
    each.

    Unlike `route`, attention DOES contain a collective -- the TP all-reduce over the AG ranks. That
    is fine and is not the M2N lockstep problem: both AG ranks capture together here, and it is
    exactly what the colocated path already captures via pynccl. What must stay outside the graph is
    the M2N dispatch/combine, and it does.

    ON by default; `MINISGL_AFD_AG_ATTN_GRAPH=0` disables it. §22.6 kept it off on two grounds --
    the gain was only 1.10x, and T-16 put it at the floor's edge in the unfavourable direction
    (pooled z=+1.63). Both have since changed. Re-tested against the sync-free-plan baseline
    (doc 28 §24) it lands *inside* the floor and marginally better than it (z=-0.17), and the
    combined gain with the sync-free plan is 1.30x at B=1. Two independent T-16 experiments, both
    non-significant with OPPOSITE signs, is what a true effect of ~zero looks like.
    """

    def __init__(
        self,
        *,
        runtime_state: Any,
        model: Any,
        num_layers: int,
        hidden_size: int,
        num_mb: int,
        log_path: str,
    ) -> None:
        self.state = runtime_state
        self.model = model
        self.device = runtime_state.device
        self.dtype = runtime_state.dtype
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.num_mb = max(1, int(num_mb))
        self.log_path = log_path
        # Memory pool PER MICROBATCH, not one for all captures. A private pool lets HIP reuse
        # blocks between graphs on the assumption they never run simultaneously -- true across
        # buckets and layers within a lane, since those replay sequentially, but FALSE across `mb`:
        # with num_mb>=2 the lanes have their own streams and replay concurrently.
        #
        # NECESSARY BUT NOT SUFFICIENT: tried as a fix for the num_mb=2 crash (doc 42); the crash
        # PERSISTED. Kept because concurrent replay from a shared pool is unsafe regardless.
        self._pools: dict[int, Any] = {}
        self._clone_outputs = self.num_mb > 1
        self._entries: dict[tuple[int, int], dict[str, Any]] = {}
        self._failed: set[tuple[int, int]] = set()
        # PER BACKEND INDEX, not a single flag. `_capture` selects
        # `state.attn_backends[mb % len(...)]`, and there is one backend per microbatch because each
        # wrapper owns mutable decode metadata. A scalar flag was set by the first (bs, mb=0)
        # capture, so every mb>=1 backend NEVER received `init_capture_graph` -- and that call is the
        # whole safety mechanism: it sizes the metadata buffers for the largest bucket and sets
        # `_frozen`, which is what turns a later reallocation into an assert instead of silent
        # use-after-free. Unfrozen, `_ensure_capacity` reallocates as the warmup sweep walks
        # 1,2,4,8,16,32, so every mb>=1 graph except the last bucket's baked a freed address.
        #
        # Invisible at num_mb=1, where only mb=0 exists and the scalar is accidentally correct.
        # Confirmed by log count: "metadata buffers frozen at N slots" appears exactly ONCE per AG
        # rank in every num_mb=2 run on record (and `info_rank0` does not deduplicate, so one line
        # is one call).
        self._backend_inited: set[int] = set()
        self.max_buckets = int(os.environ.get("MINISGL_AFD_AG_ATTN_GRAPH_MAX_BUCKETS", "8"))
        self._pending_bs: tuple[int, ...] = ()

    def _key(self, bs: int, mb: int) -> tuple[int, int]:
        return (int(bs), int(mb))

    def usable(self, bs: int, mb: int) -> bool:
        return self._key(bs, mb) in self._entries

    # ---------------------------------------------------------------- warmup
    def warmup(self, *, bs_list: tuple[int, ...], dummy_req: Any) -> None:
        """Capture every configured bucket at a quiescent point. Best effort, never raises."""
        from minisgl.core import Batch

        if dummy_req is None:
            log_line(self.log_path, "[ag_attn_graph] no dummy req; skipping", flush=True)
            return
        wanted = sorted({int(b) for b in bs_list if int(b) > 0})
        # The backend freezes its metadata slots at init_capture_graph and cannot grow them later
        # ("batch 32 exceeds the 1 slots frozen at graph capture. Growing the buffers now would
        # move them and every captured graph would read freed memory"). So it must see EVERY
        # bucket up front, not one bucket at a time -- passing only the first froze 1 slot and
        # every larger bucket then failed.
        self._pending_bs = tuple(wanted)
        for bs in wanted:
            for mb in range(self.num_mb):
                self._capture(bs=bs, mb=mb, dummy_req=dummy_req, Batch=Batch)
        log_line(
            self.log_path,
            f"[ag_attn_graph] warmup done: {len(self._entries)} bucket(s) captured, "
            f"{len(self._failed)} failed",
            flush=True,
        )

    def _capture(self, *, bs: int, mb: int, dummy_req: Any, Batch: Any) -> bool:
        from minisgl.core import get_global_ctx

        from .cuda_graph_utils import capture_cuda_graph

        k = self._key(bs, mb)
        if k in self._entries or k in self._failed:
            return k in self._entries
        if len(self._entries) >= self.max_buckets:
            self._failed.add(k)
            return False

        backend_idx = mb % len(self.state.attn_backends)
        backend = self.state.attn_backends[backend_idx]
        ctx = get_global_ctx()
        h = torch.zeros((bs, self.hidden_size), dtype=self.dtype, device=self.device)
        r = torch.zeros((bs, self.hidden_size), dtype=self.dtype, device=self.device)
        ids = torch.zeros((bs,), dtype=torch.int32, device=self.device)
        pos = torch.zeros((bs,), dtype=torch.int32, device=self.device)
        out_loc = torch.zeros((bs,), dtype=torch.int32, device=self.device)

        graphs: list[Any] = []
        outs: list[Any] = []
        try:
            if backend_idx not in self._backend_inited:
                backend.init_capture_graph(
                    max_seq_len=self.state.max_seq_len,
                    bs_list=list(self._pending_bs or (bs,)),
                )
                self._backend_inited.add(backend_idx)

            reqs = [dummy_req] * bs
            cap = Batch(reqs=reqs, phase="decode")
            cap.padded_reqs = reqs
            cap.input_ids, cap.positions, cap.out_loc = ids, pos, out_loc
            backend.prepare_for_capture(cap)

            prev_backend = ctx.attn_backend
            ctx.attn_backend = backend
            try:
                # Force every lazily-initialised library into existence before capturing:
                # hipBLASLt aborts if its first use is inside a capture (doc 28 §16.3), and
                # Triton autotune would too. One eager pass over all layers does it.
                with ctx.forward_batch(cap):
                    with torch.cuda.stream(self.state.stream):
                        for layer in range(self.num_layers):
                            self.model.forward_attention(layer, h, r)
                torch.cuda.synchronize(self.device)

                for layer in range(self.num_layers):
                    out: dict[str, Any] = {}

                    def body(_layer: int = layer, _out: dict[str, Any] = out) -> None:
                        with ctx.forward_batch(cap):
                            st = self.model.forward_attention(_layer, h, r)
                            # Write back so layer L+1 reads what L produced. copy_ is a
                            # self-copy when the model updated in place, which is legal.
                            h.copy_(st.hidden_states)
                            r.copy_(st.residual)
                            _out["st"] = st

                    graph, self._pools[mb] = capture_cuda_graph(
                        device=self.device,
                        engine_stream=self.state.stream,
                        comm_stream=self.state.stream,
                        overlap_comm=False,
                        pool=self._pools.get(mb),
                        fn=body,
                    )
                    if "st" not in out:
                        raise RuntimeError("forward_attention produced no output during capture")
                    graphs.append(graph)
                    outs.append(out["st"])
            finally:
                ctx.attn_backend = prev_backend
        except Exception as exc:
            self._failed.add(k)
            log_line(
                self.log_path,
                f"[ag_attn_graph] capture FAILED bs={bs} mb={mb} layer={len(graphs)} "
                f"{type(exc).__name__}: {exc} -- eager fallback for this bucket",
                flush=True,
            )
            return False

        self._entries[k] = {
            "graphs": graphs, "outs": outs, "h": h, "r": r,
            "cap": cap, "backend": backend,
            "ids": ids, "pos": pos, "out_loc": out_loc,
        }
        log_line(
            self.log_path,
            f"[ag_attn_graph] captured bs={bs} mb={mb} layers={len(graphs)}",
            flush=True,
        )
        return True

    # ----------------------------------------------------------------- replay
    def prepare_step(self, *, bs: int, mb: int, batch: Any) -> None:
        """Refill the captured metadata buffers from THIS step's batch. Once per step.

        Skipping this, or pointing it at the captured batch's own metadata, makes the graph replay
        against the warmup layout every step -- the exact silent-wrong-output failure the whole-step
        runner documents at afd_attention_worker.py's `replay`.
        """
        e = self._entries[self._key(bs, mb)]
        cap, backend = e["cap"], e["backend"]
        e["ids"].copy_(batch.input_ids[:bs])
        e["pos"].copy_(batch.positions[:bs])
        e["out_loc"].copy_(batch.out_loc[:bs])
        cap.reqs = batch.reqs
        cap.padded_reqs = getattr(batch, "padded_reqs", batch.reqs)
        cap.attn_metadata = batch.attn_metadata
        backend.prepare_for_replay(cap)

    def replay(
        self, *, bs: int, mb: int, layer: int, hidden: torch.Tensor, residual: Any
    ) -> Any:
        """Seed the static buffers, replay one layer, and return its retained output."""
        e = self._entries[self._key(bs, mb)]
        e["h"].copy_(hidden)
        if residual is None:
            e["r"].zero_()
        else:
            e["r"].copy_(residual)
        e["graphs"][layer].replay()
        # With more than one microbatch the caller's consumption (dispatch/combine on a LANE
        # stream) outlives this replay, so the graph's own buffer must not be handed out: the next
        # step's replay would overwrite it mid-read. That is the docs 42/43 crash. At num_mb=1 the
        # schedule is strictly serial and the existing path is validated, so it is left untouched.
        out = e["outs"][layer]
        return clone_graph_output(out) if self._clone_outputs else out


def ag_attn_graph_enabled() -> bool:
    return os.environ.get("MINISGL_AFD_AG_ATTN_GRAPH", "1") not in ("0", "false", "")
