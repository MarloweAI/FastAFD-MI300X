"""Per-kernel profiling shared by the colocated engine and the AFD workers.

Extracted from engine/engine.py::_maybe_profile so the AFD workers can use it: AFD had **no**
profiling hooks at all, which is why doc 26 §3 ended up attributing a step's cost by subtraction
and getting it wrong by ~5x. A worker that cannot be profiled will eventually be guessed about.

Two lessons are baked in here; both cost real debugging time and neither is obvious:

1. **Filter on `evt.device_type == DeviceType.CUDA`.** torch.profiler reports the same
   `self_device_time_total` on the host-side aten op AND on the kernel it launched, so summing
   everything with device time > 0 double-counts by exactly 2x. Verified on a bare 2048**3
   matmul: `aten::mm` (CPU) and `Cijk_*` (CUDA) each reported 0.496 ms.

2. **Collectives are reported twice among CUDA events** -- once as the `nccl:*` annotation, once as
   the `rcclGenericKernel` implementing it -- so `nccl:*` rows are excluded from `kernel_ms` and
   reported separately as `annotation_ms`. Summing both inflated an AFD AG step from ~43 to ~76 ms.

3. **`TIME_ONLY` exists to tell a real launch gap from roctracer overhead.** A profiled capture
   once left 22.8 ms "unaccounted" over 871 launches (26 us each) and that was read as launch
   gaps; with the profiler detached the wall time matched the kernel sum to 0.1 ms, i.e. the
   gaps were the tracer. Always confirm a gap claim with a TIME_ONLY run.

Env knobs (all shared with the colocated path):
    MINISGL_PROFILE_DIR        where to write; also the enable switch
    MINISGL_PROFILE_PHASE      which phase to match: prefill | decode | both
    MINISGL_PROFILE_SKIP       matching regions to skip first (default 2), past Triton autotune
    MINISGL_PROFILE_STEPS      how many to capture (default 1)
    MINISGL_PROFILE_MIN_TOK    only count regions with at least this many tokens
    MINISGL_PROFILE_TIME_ONLY  no profiler; just report wall time (see lesson 2)
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any, Dict

import torch

_STATE: Dict[str, Dict[str, int]] = {}


def _state(tag: str) -> Dict[str, int]:
    return _STATE.setdefault(tag, {"seen": 0, "written": 0})


def profile_enabled() -> bool:
    return bool(os.environ.get("MINISGL_PROFILE_DIR"))


@contextlib.contextmanager
def profile_region(
    *,
    tag: str,
    phase: str,
    n_tok: int,
    device: torch.device | None = None,
    rank: Any = 0,
    extra: Dict[str, Any] | None = None,
):
    """Profile one region (a forward pass, or one AFD worker's step).

    `tag` namespaces the skip/steps counters so an AG rank and an EG rank in the same process
    tree do not consume each other's budget. Output goes to
    `prof_{tag}_{phase}_ntok{n_tok}_rank{rank}.json`.
    """
    out_dir = os.environ.get("MINISGL_PROFILE_DIR")
    if not out_dir:
        yield
        return

    def _trace(reason: str) -> None:
        """Log the skip/capture DECISION, once per (tag, reason).

        A region that silently declines to fire is indistinguishable from one that is never
        called, and that ambiguity cost a whole debugging round on the AFD side (doc 26 §8.3).
        Cheap because it is once per reason, not once per step.
        """
        key = f"__traced_{reason}"
        if _state(tag).get(key):
            return
        _state(tag)[key] = 1
        print(
            f"[profiling] tag={tag} phase={phase!r} n_tok={n_tok} -> {reason} "
            f"(want={os.environ.get('MINISGL_PROFILE_PHASE', 'decode')!r} "
            f"skip={os.environ.get('MINISGL_PROFILE_SKIP', '2')} "
            f"seen={_state(tag)['seen']})",
            flush=True,
        )

    want = os.environ.get("MINISGL_PROFILE_PHASE", "decode")
    if want not in ("both", phase):
        _trace("SKIP:phase-mismatch")
        yield
        return
    if n_tok < int(os.environ.get("MINISGL_PROFILE_MIN_TOK", "0")):
        _trace("SKIP:below-min-tok")
        yield
        return

    st = _state(tag)
    skip = int(os.environ.get("MINISGL_PROFILE_SKIP", "2"))
    steps = int(os.environ.get("MINISGL_PROFILE_STEPS", "1"))
    st["seen"] += 1
    if st["seen"] <= skip:
        _trace("SKIP:warmup")
        yield
        return
    if st["written"] >= steps:
        _trace("SKIP:budget-spent")
        yield
        return
    _trace("CAPTURE")

    os.makedirs(out_dir, exist_ok=True)

    if os.environ.get("MINISGL_PROFILE_TIME_ONLY"):
        # Lesson 2: no tracer attached, so this wall time is the ground truth a profiled
        # capture's kernel-sum must be checked against before any "gap" is claimed.
        if device is not None:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        yield
        if device is not None:
            torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - t0) * 1e3
        st["written"] += 1
        path = os.path.join(out_dir, f"time_{tag}_{phase}_ntok{n_tok}_rank{rank}.json")
        with open(path, "w") as f:
            json.dump(
                {"tag": tag, "phase": phase, "n_tok": n_tok, "rank": rank,
                 "wall_ms": wall_ms, "profiler": False, **(extra or {})}, f, indent=1)
        return

    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    if device is not None:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        yield
        if device is not None:
            torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - t0) * 1e3

    # Raw histogram BEFORE any filtering. Without this, an empty result cannot be told apart:
    # "the profiler saw nothing at all" and "it saw events but none were attributed to the
    # device" have completely different causes, and doc 26 §8.2 could not distinguish them.
    hist: Dict[str, int] = {}
    n_events = 0
    for evt in prof.key_averages():
        n_events += 1
        key = str(evt.device_type)
        hist[key] = hist.get(key, 0) + 1
        if evt.self_device_time_total > 0:
            hist[key + "|self_dev>0"] = hist.get(key + "|self_dev>0", 0) + 1

    rows = []
    annotation_ms = 0.0
    for evt in prof.key_averages():
        # Lesson 1: device-side events only, or every kernel is counted twice.
        if evt.device_type != DeviceType.CUDA:
            continue
        if evt.self_device_time_total <= 0:
            continue
        # Lesson 3, and it is a SECOND double-count distinct from lesson 1. Collectives appear
        # TWICE among CUDA events: once as the `nccl:*` annotation and once as the
        # `rcclGenericKernel` that implements it. On an AFD AG step that was 32.4 ms of
        # `nccl:all_to_all` alongside 33.2 ms of `rcclGenericKernel` -- the same work, inflating
        # kernel_ms from ~43 ms to ~76 ms. The call counts differ (180 vs 216) because the
        # annotation covers only the all_to_alls while the kernel row also carries the 36 TP
        # all-reduces, so they cannot simply be paired off; the annotation is the one to drop.
        if evt.key.startswith("nccl:"):
            annotation_ms += float(evt.self_device_time_total)
            rows.append({
                "name": evt.key, "count": int(evt.count),
                "self_device_us": float(evt.self_device_time_total),
                "annotation": True,
            })
            continue
        rows.append({
            "name": evt.key,
            "count": int(evt.count),
            "self_device_us": float(evt.self_device_time_total),
        })
    rows.sort(key=lambda r: -r["self_device_us"])
    # Annotations are kept in `rows` for visibility but excluded from the total.
    kernel_ms = sum(
        r["self_device_us"] for r in rows if not r.get("annotation")
    ) / 1e3

    st["written"] += 1
    path = os.path.join(out_dir, f"prof_{tag}_{phase}_ntok{n_tok}_rank{rank}.json")
    with open(path, "w") as f:
        json.dump({
            "tag": tag, "phase": phase, "n_tok": n_tok, "rank": rank,
            "wall_ms": wall_ms,
            "kernel_ms": kernel_ms,
            # wall - kernel is only a GAP if a TIME_ONLY run agrees the wall is real.
            "unaccounted_ms": wall_ms - kernel_ms,
            "num_launches": sum(r["count"] for r in rows if not r.get("annotation")),
            "annotation_ms": round(annotation_ms / 1e3, 4),
            "n_events_total": n_events,
            "device_type_histogram": hist,
            "rows": rows[:60],
            **(extra or {}),
        }, f, indent=1)





# ---------------------------------------------------------------------------
# Sub-step attribution
# ---------------------------------------------------------------------------
# torch.profiler yields NOTHING on the AFD path -- no files, and the driver left blocked in
# ray.get -- while TIME_ONLY works on both stages (doc 28 §8.3). That is doc 26 §8.2's
# "Defect A" resurfacing for Ray actors, and it means the per-kernel breakdown that would
# split the AFD step is unavailable there.
#
# These timers are the cheap substitute. They are HOST-side and deliberately do NOT
# synchronize: under the serial afd_num_mb=1 schedule the collectives block the host, so
# host wall time per call is a faithful decomposition of where the step goes -- and idle
# time waiting for the peer stage is precisely what needs attributing. Adding syncs would
# both perturb the schedule and measure the wrong thing.
#
# Every call site must be safe with profiling OFF: `step_timer` returns a null context and
# `step_flush` is a no-op. Argument lists are evaluated before the callee runs, so cheap
# literals only at call sites (cf. the EG hook's `self.rank` crash).

_SUB: Dict[str, Dict[str, float]] = {}
_SUB_STATE: Dict[str, int] = {}


@contextlib.contextmanager
def step_timer(bucket: str):
    """Accumulate host wall time into `bucket` for the current step."""
    if not os.environ.get("MINISGL_PROFILE_DIR"):
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        d = _SUB.setdefault("cur", {})
        dt = (time.perf_counter() - t0) * 1e3
        d[bucket] = d.get(bucket, 0.0) + dt
        d[bucket + ".n"] = d.get(bucket + ".n", 0.0) + 1


def step_flush(*, tag: str, phase: str, rank: Any, total_ms: float,
               extra: Dict[str, Any] | None = None) -> None:
    """Write one step's bucket totals, then reset. Honours SKIP/STEPS/PHASE like
    profile_region so the captured step is the same warmed step."""
    out_dir = os.environ.get("MINISGL_PROFILE_DIR")
    if not out_dir:
        return
    buckets = _SUB.pop("cur", None)
    if not buckets:
        return
    want = os.environ.get("MINISGL_PROFILE_PHASE", "decode")
    if want != "both" and phase != want:
        return
    key = f"{tag}:{rank}"
    seen = _SUB_STATE.get(key, 0) + 1
    _SUB_STATE[key] = seen
    if seen <= int(os.environ.get("MINISGL_PROFILE_SKIP", "2")):
        return
    written = _SUB_STATE.get(key + ":w", 0)
    if written >= int(os.environ.get("MINISGL_PROFILE_STEPS", "1")):
        return
    _SUB_STATE[key + ":w"] = written + 1
    # One file PER STEP. These are host wall times at ~100 us granularity, so a single step is
    # not a measurement -- the spread across steps is part of the answer. The path used to be
    # fixed per (tag, phase, rank), so MINISGL_PROFILE_STEPS=8 silently overwrote seven steps
    # and returned only the eighth while looking like it had averaged them.
    named = {k: round(v, 4) for k, v in buckets.items() if not k.endswith(".n")}
    counts = {k[:-2]: int(v) for k, v in buckets.items() if k.endswith(".n")}
    # A bucket named "a.b" is NESTED inside bucket "a", so it must not be added again --
    # otherwise the residual goes negative and reads like a measurement error. The first
    # capture with nested compute buckets reported residual = -28.7 ms (-37.8%) for exactly
    # this reason. Only top-level buckets (no dot) count toward `accounted`.
    accounted = sum(v for k, v in named.items() if "." not in k)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sub_{tag}_{phase}_rank{rank}_s{written}.json")
    with open(path, "w") as f:
        json.dump({
            "tag": tag, "phase": phase, "rank": rank, "step": written,
            "total_ms": round(total_ms, 4),
            "buckets_ms": named,
            "call_counts": counts,
            "accounted_ms": round(accounted, 4),
            # total - accounted is the part of the step OUTSIDE the instrumented calls.
            "residual_ms": round(total_ms - accounted, 4),
            **(extra or {}),
        }, f, indent=1)




# ---------------------------------------------------------------------------
# Sync scanning
# ---------------------------------------------------------------------------
# Two of the three biggest AFD wins this session were HIDDEN host syncs in code that reads as pure
# device work: boolean mask indexing (which calls nonzero() internally) and `torch.bincount` (which
# reads the input's max on the host). Neither announces itself at the call site, and inspection
# "checked" both lines and missed them. `torch.cuda.set_sync_debug_mode("warn")` finds every one in a
# single run -- but its warning text names only the C++ origin, so the Python stack has to be captured
# separately to be of any use.
#
# Gated by MINISGL_SYNC_SCAN=1 and applied to one region per tag, because the mode warns on every
# synchronising op and is far too noisy to leave on.

_SYNC_SCAN_DONE: Dict[str, bool] = {}


@contextlib.contextmanager
def sync_scan(*, tag: str, log: Any = None):
    """Report the Python call site of every synchronising op inside this region, once per tag."""
    if not os.environ.get("MINISGL_SYNC_SCAN") or _SYNC_SCAN_DONE.get(tag):
        yield
        return
    _SYNC_SCAN_DONE[tag] = True

    def _emit(msg: str) -> None:
        # A Ray actor's stdout is not a reliable channel -- the first attempt at scanning the AFD
        # pipeline bodies produced no output at all, and the hooks and env forwarding were both
        # correct, so the lines were simply lost. Write to a file when one is named.
        path = os.environ.get("MINISGL_SYNC_SCAN_FILE")
        if path:
            try:
                with open(path, "a") as fh:
                    fh.write(msg + "\n")
                return
            except Exception:
                pass
        if log is not None:
            try:
                log(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    # Unconditional, so "region never entered" is distinguishable from "entered, found nothing".
    _emit(f"[sync_scan] tag={tag} ENTER")

    import traceback
    import warnings

    hits: list[str] = []

    def _show(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
        # Walk out to the first frame in THIS repo. Matching on "minisgl" alone is wrong -- the conda
        # env is called minisgl-rocm7, so every frame matches and the report names warnings.py.
        for fr in reversed(traceback.extract_stack()[:-1]):
            f = fr.filename
            if "/python/minisgl/" in f and "profiling.py" not in f:
                hits.append(f"{f.split('/python/minisgl/')[-1]}:{fr.lineno}  {(fr.line or '').strip()[:80]}")
                return
        hits.append("(no in-repo frame)")

    old_show, old_filters = warnings.showwarning, warnings.filters[:]
    warnings.showwarning = _show
    warnings.simplefilter("always")
    try:
        torch.cuda.set_sync_debug_mode("warn")
    except Exception:
        warnings.showwarning = old_show
        yield
        return
    try:
        yield
    finally:
        try:
            torch.cuda.set_sync_debug_mode("default")
        except Exception:
            pass
        warnings.showwarning = old_show
        warnings.filters[:] = old_filters
        from collections import Counter
        lines = [f"[sync_scan] tag={tag}: {len(hits)} synchronising op(s) in one region"]
        for site, n in Counter(hits).most_common():
            lines.append(f"[sync_scan]   x{n}  {site}")
        _emit("\n".join(lines))


__all__ = [
    "profile_region", "profile_enabled", "step_timer", "step_flush", "sync_scan",
]
