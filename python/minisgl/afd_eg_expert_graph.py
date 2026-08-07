"""HIP graph capture of the EG expert stage, to remove host dispatch cost.

Why
---
Doc 38 measured the EG expert region: **782 launches per decode step, 2.76 ms of MoE device time,
and a 13.34 ms host bucket.** So ~10.5 ms is the host issuing ~20 small eager ops per layer one at a
time, each costing ~20 us of Python -> dispatcher -> HIP while the device finishes the kernel in
2-8 us and waits. The GEMM itself is at its bandwidth floor (2.19 ms measured against a 1.85 ms
weight-reading floor), so there is nothing to win inside the kernels -- only in how they are issued.

This is deliberately NOT the treatment docs 35/36 applied to `d.plan` and `d.group`. Those were ~15
launches of tensor bookkeeping that collapsed into one Triton kernel. The expert path is ~20
*different* operations including two library GEMMs, an alignment kernel, a sort and an activation;
there is no single kernel to write, so the fix is to stop paying per-launch host cost at all.

It is also not the M2N graph capture that doc 37 deprioritised. That target's buckets hold *waiting*
for the peer (15.27 ms of peer wait against 1.17 ms of own work) and a graph cannot remove a wait.
This target's bucket holds *issuing*, which is exactly what a replay collapses. Whether a bucket
holds waiting or issuing is the discriminator, and both were measured.

Shape strategy
--------------
`run_experts` needs a fixed row count to capture, and `num_recv` varies per layer per step -- it
comes from the counts `tolist`. Graphs are therefore keyed on the **exact** `num_recv`, with a
bucket cap and eager fallback past it, mirroring `AfdAgRouteGraphs`.

Capacity padding would make `num_recv` constant and cut the graph count, but it is **deliberately
not used**: padding rows carry expert id -1, and `moe_align_block_size` is not known to tolerate -1
in `topk_ids` (the eager path never shows it one, because `num_recv` counts only real rows). That is
the same class of assumption that made fixed-shape M2N emit 20-logit garbage while passing every
component probe, and it has never been localised. Exact keying needs no such assumption.

At decode B=1 the range is small: 2 AG ranks x 4 slots spread over 2 EG ranks, so `num_recv` lands
in [0, 8]. `num_recv == 0` stays eager -- the runner already early-returns on an empty batch.

Traps this module has to respect, each learned the hard way (doc 28, doc 35 §4):
  * capture at **warmup**, never inline on a live step -- inline capture corrupts under concurrency;
  * run an **eager pre-pass over every layer first** -- hipBLASLt initialises lazily and that init is
    illegal inside a capture (`hipblaslt.cpp:171`), which aborts the worker;
  * **retain the capture-time output object**, so replay returns the graph's own refreshed storage;
  * never let a capture failure break serving -- eager stays correct.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from .afd_support import log_line
from .cuda_graph_utils import clone_graph_output


def eg_expert_graph_enabled() -> bool:
    """ON by default after measuring: `expert` 12.65 -> 2.88 ms, decode ITL 40.76 -> 33.40 (1.22x),
    32/32 greedy prompts bit-identical on the running server, 288 graphs captured with 0 failures
    and no measurable startup cost (32 s to ready against 33 s baseline). Set to 0 for eager."""
    return os.environ.get("MINISGL_AFD_EG_EXPERT_GRAPH", "1") not in ("0", "false", "")


class AfdEgExpertGraphs:
    """Captures `expert_stage.run_experts(layer, dispatch_output)` per (num_recv, mb, layer)."""

    def __init__(
        self,
        *,
        runtime_state: Any,
        expert_stage: Any,
        adapter: Any,
        num_layers: int,
        hidden_size: int,
        num_mb: int,
        log_path: str,
    ) -> None:
        self.state = runtime_state
        self.expert_stage = expert_stage
        self.adapter = adapter
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
        # NECESSARY BUT NOT SUFFICIENT: this was tried as a fix for the num_mb=2 crash (doc 42) and
        # the crash PERSISTED, so pool sharing was not the cause. Kept because concurrent replay
        # from a shared pool is unsafe regardless. See doc 43 for what is still unexplained.
        self._pools: dict[int, Any] = {}
        self._clone_outputs = self.num_mb > 1
        self._entries: dict[tuple[int, int], dict[str, Any]] = {}
        self._failed: set[tuple[int, int]] = set()
        self.max_buckets = int(os.environ.get("MINISGL_AFD_EG_EXPERT_GRAPH_MAX_BUCKETS", "12"))
        self.local_experts = int(adapter.experts_per_eg_rank)
        self.first_local = (int(adapter.rank) - int(adapter.layout.ag_size)) * self.local_experts

    def _key(self, n: int, mb: int) -> tuple[int, int]:
        return (int(n), int(mb))

    def usable(self, n: int, mb: int) -> bool:
        return self._key(n, mb) in self._entries

    def _make_disp(self, buf: dict[str, torch.Tensor]) -> Any:
        """A dispatch output backed by the static buffers.

        `handle=None` is safe on this path: the post-permute stores it into an `M2NCombineInput`
        without dereferencing it, and `run_experts` returns only `.hidden_states`. Combine uses the
        REAL handle from the real dispatch, never this one.
        """
        from minisgl.moe.rccl_m2n_adapter import RcclM2NDispatchOutput

        return RcclM2NDispatchOutput(
            hidden_states=buf["hidden"],
            handle=None,
            topk_ids=buf["topk_ids"],
            topk_weights=buf["topk_weights"],
            recv_count=buf["recv_count"],
            layout=self.adapter.layout,
            rank=int(self.adapter.rank),
            is_ag=False,
            is_eg=True,
        )

    def warmup(self) -> None:
        """Capture every configured row count at a quiescent point. Best effort, never raises."""
        raw = os.environ.get("MINISGL_AFD_EG_EXPERT_GRAPH_ROWS", "1,2,3,4,5,6,7,8")
        try:
            rows = sorted({int(x) for x in raw.split(",") if x.strip()})
        except ValueError:
            log_line(self.log_path, f"[eg_expert_graph] bad ROWS {raw!r}; skipping", flush=True)
            return
        for n in rows:
            if n <= 0:
                continue
            for mb in range(self.num_mb):
                self.capture(n=n, mb=mb)
        log_line(
            self.log_path,
            f"[eg_expert_graph] warmup done: {len(self._entries)} captured, "
            f"{len(self._failed)} failed, requested={rows}",
            flush=True,
        )

    def capture(self, *, n: int, mb: int) -> bool:
        from .cuda_graph_utils import capture_cuda_graph

        k = self._key(n, mb)
        if k in self._entries or k in self._failed:
            return k in self._entries
        if len(self._entries) >= self.max_buckets:
            self._failed.add(k)
            log_line(
                self.log_path,
                f"[eg_expert_graph] bucket cap reached ({self.max_buckets}); n={n} mb={mb} stays "
                "eager. Raise MINISGL_AFD_EG_EXPERT_GRAPH_MAX_BUCKETS if intended.",
                flush=True,
            )
            return False

        # Realistic seeds, not zeros: expert ids spread over the local range so the alignment
        # kernel and the grouped GEMM see a representative distribution, and a degenerate input
        # cannot select a Triton config that replay will not want.
        g = torch.Generator(device="cpu").manual_seed(1234 + n)
        ids = (
            self.first_local
            + torch.randint(0, max(self.local_experts, 1), (n,), generator=g)
        ).to(self.device, torch.int64)
        buf = {
            "hidden": torch.randn(
                (n, self.hidden_size), generator=g
            ).to(self.device, self.dtype),
            "topk_ids": ids,
            "topk_weights": torch.rand((n,), generator=g).to(self.device, torch.float32),
            "recv_count": torch.zeros(
                self.local_experts, device=self.device, dtype=torch.int32
            ),
        }
        disp = self._make_disp(buf)

        graphs: list[Any] = []
        outs: list[Any] = []
        try:
            # hipBLASLt initialises lazily and that init is ILLEGAL inside a capture -- it aborts
            # with "operation not permitted when stream is capturing" at hipblaslt.cpp:171 and
            # takes the worker down. Capture at warmup makes this the FIRST expert call, so force
            # every layer's kernels into existence eagerly first.
            with torch.cuda.stream(self.state.stream):
                for layer in range(self.num_layers):
                    self.expert_stage.run_experts(layer, disp)
            torch.cuda.synchronize(self.device)

            for layer in range(self.num_layers):
                out: dict[str, Any] = {}

                def body(_layer: int = layer, _out: dict[str, Any] = out) -> None:
                    _out["y"] = self.expert_stage.run_experts(_layer, disp)

                graph, self._pools[mb] = capture_cuda_graph(
                    device=self.device,
                    engine_stream=self.state.stream,
                    comm_stream=self.state.stream,
                    overlap_comm=False,
                    pool=self._pools.get(mb),
                    fn=body,
                )
                if "y" not in out:
                    raise RuntimeError("run_experts produced no output during capture")
                graphs.append(graph)
                outs.append(out["y"])
        except Exception as exc:
            self._failed.add(k)
            log_line(
                self.log_path,
                f"[eg_expert_graph] capture FAILED n={n} mb={mb} layer={len(graphs)} "
                f"{type(exc).__name__}: {exc} -- eager fallback for this bucket",
                flush=True,
            )
            return False

        self._entries[k] = {"graphs": graphs, "outs": outs, "buf": buf}
        log_line(
            self.log_path,
            f"[eg_expert_graph] captured n={n} mb={mb} layers={len(graphs)}",
            flush=True,
        )
        return True

    def replay(self, *, n: int, mb: int, layer: int, dispatch_output: Any) -> torch.Tensor:
        """Copy the real dispatch into the static buffers, replay, return the retained output.

        The returned tensor is the graph's own storage, refreshed by this replay. It must be
        consumed before the same (n, mb, layer) replays again -- true under the serial schedule,
        since combine for this layer is issued before the next step reaches it, and
        `launch_combine` calls `record_stream` so the cross-stream lifetime is handled.
        """
        entry = self._entries[self._key(n, mb)]
        buf = entry["buf"]
        buf["hidden"].copy_(dispatch_output.hidden_states)
        buf["topk_ids"].copy_(dispatch_output.topk_ids)
        if dispatch_output.topk_weights is not None:
            buf["topk_weights"].copy_(dispatch_output.topk_weights.reshape(-1))
        buf["recv_count"].copy_(dispatch_output.recv_count)
        entry["graphs"][layer].replay()
        # With more than one microbatch the caller's consumption (dispatch/combine on a LANE
        # stream) outlives this replay, so the graph's own buffer must not be handed out: the next
        # step's replay would overwrite it mid-read. That is the docs 42/43 crash. At num_mb=1 the
        # schedule is strictly serial and the existing path is validated, so it is left untouched.
        out = entry["outs"][layer]
        return clone_graph_output(out) if self._clone_outputs else out
