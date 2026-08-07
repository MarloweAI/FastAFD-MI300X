"""M2N attention→expert transport over torch.distributed collectives (RCCL on ROCm).

Stage A of the DeepEP replacement — see dev_log/qwen/11_m2n_contract.md for the contract
this satisfies and dev_log/qwen/10_afd_transport_options.md for why DeepEP itself cannot
be ported (it needs the NCCL 2.28 *device* API, which RCCL does not provide, plus
SM90 TMA/mbarrier, which CDNA3 does not have).

**Deliberately the slow path.** It is host-orchestrated: computing all-to-all split
sizes requires the per-destination counts on the CPU, so every dispatch pays a D2H
sync. Its purpose is to be obviously correct and to serve as the diff target for the
symmetric-memory + Triton fast path (Stage B), exactly as `attention/torch_ref.py`
does for attention.

Design notes
------------
* The DeepEP adapter pads the expert space so attention ranks own dummy experts,
  purely to satisfy DeepEP's symmetric-group requirement. This transport routes
  N→M directly and does not need the padding, but it keeps the same
  `DeepEPUnionM2NLayout` so `dummy_expert_offset` / `experts_per_union_rank`
  arithmetic stays comparable and callers that pre-remapped ids still work.
* Rows are grouped by **global expert id** on arrival, because the grouped expert
  GEMM consumes `recv_count` as a per-expert prefix sum.
* `handle.handle` is a small shim carrying the two attributes the static helpers in
  the contract reach for, so `num_recv_tokens` / `valid_expanded_slots` keep working
  without touching their call sites.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

import torch
import torch.distributed as dist

from minisgl.layers.moe.token_dispatcher.base import DispatchOutputFormat

from .deepep_m2n_union import (
    DeepEPUnionM2NLayout,
    build_deepep_union_m2n_layout,
    remap_real_to_deepep_experts,
)

# Imported for its side effect: registers the m2n_expanded <-> triton permute pair
# that MoELayer._run_moe_core looks up by format. Without this the EG side would
# fall back to the deep_gemm-only registration and fail on gfx942.
from . import m2n_permute as _m2n_permute  # noqa: F401
from . import m2n_plan_triton as _fused_plan

# Sub-step timers. The transport is now 72% of the AFD decode step (doc 28 §22.3), and the
# "~0.70 ms of the 0.87 ms round trip is bookkeeping" figure was obtained BY SUBTRACTION -- the
# error mode this log keeps repeating. These measure it directly. Dotted names so the accounting
# in profiling.py treats them as nested inside the dispatch/combine buckets.
from minisgl.profiling import step_timer as _sub

# Fuse dispatch's payload/expert/weight collectives into one packed uint8 all_to_all.
#
# DEFAULT OFF: measured and it LOSES. Removing 2 of dispatch's 4 collectives per layer made
# the step ~2% worse everywhere (decode B=1 83.07 vs 81.35 ms, B=8 95.04 vs 93.20, prefill
# 8K 821 vs 784, 32K 3144 vs 2989). Bit-exactness is proven -- T-14c,
# dev_log/probes/_m2n_fuse_parity_worker.py -- so this is purely a cost result: the pack and
# unpack copies cost more than the saved launches.
#
# The useful part is what it falsifies. Dispatch's ~1.02 ms/layer is NOT dominated by
# per-call latency, so collective COUNT is not the lever. What the bucket mostly contains is
# rendezvous: the AG blocks until the EG reaches the matching collective, and the EG cannot
# get there until its expert GEMM finishes. That is why the isolated microbenchmark -- same
# adapter code, same 4 collectives, all ranks always ready -- reports 0.843 ms/layer while
# the server sees 1.21 ms. Kept behind the flag so the next person does not re-derive it.
_FUSE_DISPATCH = os.environ.get("MINISGL_M2N_FUSE_DISPATCH", "0") not in ("0", "false", "")

# Fixed-shape capacity-padded dispatch (doc 26 §7, prerequisites in §9; costed in doc 28 §25).
# Every buffer size becomes a host-known constant, so the counts collective and the last `tolist`
# sync both disappear -- equal-split `all_to_all_single` needs no split sizes at all.
#
# Capacity C = bucket * K per destination, which makes overflow IMPOSSIBLE rather than unlikely (all
# T*K slots could target one rank), so no overflow check is needed and therefore no sync to perform
# one. The bucket MUST come from the per-call `num_max_dispatch_tokens_per_rank`, i.e.
# `plan.dispatch_bucket`: the construction-time value is the PREFILL sizing (8192), which would ask
# for 755 MB of staging. doc 26 §9.1 verified both eager paths take plan.dispatch_bucket from a
# single coordinator local, so AG and EG derive the same C without communicating -- and a capacity
# disagreement is silent corruption, not an error.
#
# Size-gated: prefill stays on the dynamic path, which is also where the payoff is not (transport is
# 43% of the decode step and ~1% of prefill).
_FIXED_SHAPE = os.environ.get("MINISGL_M2N_FIXED_SHAPE", "0") not in ("0", "false", "")
_FIXED_SHAPE_MAX_BYTES = int(os.environ.get("MINISGL_M2N_FIXED_SHAPE_MAX_BYTES", str(64 << 20)))
# Diagnostic only. `d.tolist_sync` blocks for TWO things and their sum was being read as the
# cost of the sync: (a) draining every kernel already queued on the stream -- on AG that
# includes the captured attention/route graph replays, launched async and not yet waited for
# -- and (b) the D2H read plus host round trip. Only (b) is what removing the sync recovers;
# (a) has to happen regardless and would just be waited for somewhere else. Splitting them
# decides whether fixed-shape M2N is worth ~1.4x or nearly nothing.
_DRAIN_SPLIT = os.environ.get("MINISGL_M2N_DRAIN_SPLIT", "0") not in ("0", "false", "")
# Diagnostic only, and PERTURBING (see the call site at the end of `combine`). Splits the carried-over
# peer rendezvous out of `d.drain`, which doc 34 §2 could not separate.
_COMBINE_DRAIN = os.environ.get("MINISGL_M2N_COMBINE_DRAIN", "0") not in ("0", "false", "")
# Fused Triton send-plan build (doc 34 §3: `d.plan` was 17% of the AFD decode step and is pure
# launch overhead). Bit-identical to the eager build by construction; verified three ways before
# this default was flipped ON, because a component probe agreeing with the eager path is exactly
# the evidence fixed-shape M2N also had and it was still wrong on the server:
#   1. element-wise vs a transcription        dev_log/probes/plan_build_bench.py
#   2. element-wise vs THE ADAPTER's own path, through dispatch+combine, with scattered unrouted
#      slots                                  dev_log/probes/_m2n_fused_plan_parity_worker.py
#   3. 32/32 greedy prompts bit-identical on the RUNNING server, fused vs eager
#                                             dev_log/probes/server_ab_identical.py
# Set to 0 to fall back to the eager build, which stays in the tree as the reference.
_FUSED_PLAN = os.environ.get("MINISGL_M2N_FUSED_PLAN", "1") not in ("0", "false", "")
# Same treatment for the EG receive-side grouping (doc 35 §6: 5.68 ms, 13% of the EG step).
# ON after the same three gates: element-wise vs a transcription, element-wise vs the adapter's own
# path through dispatch+combine on 4 ranks, and 32/32 greedy prompts bit-identical on the running
# server. Measured 5.68 -> 2.78 ms; see doc 36.
_FUSED_GROUP = os.environ.get("MINISGL_M2N_FUSED_GROUP", "1") not in ("0", "false", "")


def _counts_no_sync(key: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Histogram of `key` over `n_bins`, WITHOUT synchronising.

    `torch.bincount` synchronises -- twice per call on ROCm, confirmed with
    `torch.cuda.set_sync_debug_mode("warn")`: it reads the input's max on the host to size its
    output even when `minlength` already covers the range. That made doc 28 §23's "sync-free plan
    build" a misnomer: three syncs were removed and this one was introduced in their place, so the
    build still blocked twice per layer per stage.

    `index_add_` into a pre-sized buffer computes the same thing and never leaves the device.
    """
    out = torch.zeros(n_bins, device=key.device, dtype=torch.int64)
    out.index_add_(0, key, torch.ones_like(key, dtype=torch.int64))
    return out


class _LazyNegOnes:
    """A `(rows, 3)` int64 tensor of -1, materialised only if something actually reads it.

    The rccl M2N path never reads `recv_src_metadata`; only the DeepEP adapter and
    `kernel/deepep_moe.py` do. Filling it eagerly cost one kernel launch per layer per forward
    for a tensor with no consumer, which at ~16 us of host time per launch is ~0.6 ms per decode
    step. `__getattr__` forwards to the real tensor so any genuine reader still works.
    """

    __slots__ = ("_rows", "_device", "_t")

    def __init__(self, rows: int, device: Any) -> None:
        self._rows = int(rows)
        self._device = device
        self._t: torch.Tensor | None = None

    def _materialize(self) -> torch.Tensor:
        if self._t is None:
            self._t = torch.full((self._rows, 3), -1, device=self._device, dtype=torch.int64)
        return self._t

    def __getattr__(self, item: str) -> Any:
        return getattr(self._materialize(), item)

    def __getitem__(self, item: Any) -> Any:
        return self._materialize()[item]


@dataclass
class _RuntimeHandleShim:
    """Mimics the DeepEP runtime handle for the two static helpers in the contract.

    dev_log/qwen/11_m2n_contract.md sec 3: `num_recv_tokens` reads
    `psum_num_recv_tokens_per_scaleup_rank[-1]` and `valid_expanded_slots` reads
    `recv_src_metadata[:n][:, 2:]`. Providing them here keeps those helpers, and
    every call site, unchanged.
    """

    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    psum_num_recv_tokens_per_expert: torch.Tensor
    recv_src_metadata: torch.Tensor


@dataclass
class RcclM2NHandle:
    """Everything `combine` needs to invert the dispatch."""

    handle: _RuntimeHandleShim
    # send side (populated on AG ranks)
    send_counts: list[int]
    recv_counts: list[int]
    slot_order: torch.Tensor          # permutation applied before sending
    num_local_tokens: int
    top_k: int
    valid_slot_mask: torch.Tensor     # (T*K,) bool, which expanded slots were routed
    topk_weights_flat: torch.Tensor | None
    # recv side (populated on EG ranks)
    expert_order: torch.Tensor        # permutation that grouped arrivals by expert
    num_recv_rows: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device
    # Set only by the fixed-shape path, so combine knows the reverse exchange is equal-split and
    # that padding rows must be zeroed before the reduction.
    fixed_cap: int | None = None
    fixed_dst: torch.Tensor | None = None   # (T*K,) staging row per source slot, invalid clamped to 0
    fixed_ok: torch.Tensor | None = None    # (T*K,) which source slots were real


class RcclM2NDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    handle: Any
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    recv_count: torch.Tensor
    layout: DeepEPUnionM2NLayout
    rank: int
    is_ag: bool
    is_eg: bool
    hidden_states_scale: torch.Tensor | None = None

    @property
    def format(self) -> DispatchOutputFormat:
        # Own format tag: the expanded+grouped layout differs from DeepEP's handle,
        # so its permutes must not be applied here. See moe/m2n_permute.py.
        return DispatchOutputFormat.M2N_EXPANDED


class RcclM2NAdapter:
    """Drop-in replacement for `DeepEPM2NAdapter` built on torch.distributed."""

    def __init__(
        self,
        *,
        group: Any,
        ag_size: int,
        eg_size: int,
        real_num_experts: int,
        hidden_size: int,
        top_k: int,
        num_max_dispatch_tokens_per_rank: int,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("RcclM2NAdapter requires torch.distributed initialization")
        self.group = group
        self.layout = build_deepep_union_m2n_layout(
            ag_size=int(ag_size),
            eg_size=int(eg_size),
            real_num_experts=int(real_num_experts),
        )
        self.rank = int(dist.get_rank(group=group))
        self.world_size = int(dist.get_world_size(group=group))
        if self.world_size != self.layout.union_size:
            raise RuntimeError(
                "RcclM2NAdapter group size mismatch: "
                f"group={self.world_size} layout={self.layout.union_size}"
            )
        self.is_ag = self.rank < self.layout.ag_size
        self.is_eg = not self.is_ag
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.num_max_dispatch_tokens_per_rank = int(num_max_dispatch_tokens_per_rank)
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        self._log_callback = log
        self._destroyed = False

        # experts are sharded contiguously across EG ranks
        self.experts_per_eg_rank = self.layout.experts_per_union_rank
        self._emit_log(
            f"RcclM2NAdapter rank={self.rank} role={self.role} "
            f"ag={self.layout.ag_size} eg={self.layout.eg_size} "
            f"real_experts={self.layout.real_num_experts} "
            f"experts_per_eg_rank={self.experts_per_eg_rank} hidden={self.hidden_size} "
            f"top_k={self.top_k} (host-orchestrated all_to_all; correctness path)"
        )

    # ------------------------------------------------------------------ properties
    @property
    def role(self) -> str:
        return "ag" if self.is_ag else "eg"

    @property
    def topk_idx_dtype(self) -> torch.dtype:
        return torch.int64

    # -------------------------------------------------------------------- dispatch
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        *,
        hidden_states_scale: torch.Tensor | None = None,
        expert_alignment: int | None = None,
        num_max_dispatch_tokens_per_rank: int | None = None,
        do_cpu_sync: bool = False,
        router_logits: torch.Tensor | None = None,
        renormalize: bool = False,
        topk_ids_are_deepep: bool = False,
    ) -> RcclM2NDispatchOutput:
        self._check_not_destroyed()
        del expert_alignment, do_cpu_sync  # this path always syncs; see module docstring
        if hidden_states_scale is not None:
            raise NotImplementedError(
                "RcclM2NAdapter is a BF16 correctness path; FP8 dispatch scales are "
                "not supported yet (dev_log/qwen/11_m2n_contract.md Stage A)."
            )
        if router_logits is not None:
            raise NotImplementedError(
                "RcclM2NAdapter expects the caller to supply topk_ids/topk_weights; "
                "fused routing from router_logits is not implemented."
            )
        del renormalize

        device = hidden_states.device
        dtype = hidden_states.dtype
        num_tokens = int(hidden_states.shape[0])

        if self.is_eg and num_tokens != 0:
            raise RuntimeError(
                f"EG ranks must not source tokens: rank={self.rank} tokens={num_tokens}"
            )

        world = self.world_size
        K = self.top_k

        # ---- build the send plan -------------------------------------------
        _plan_ctx = _sub("d.plan")
        _plan_ctx.__enter__()
        if self.is_ag and num_tokens > 0:
            if topk_ids is None:
                raise RuntimeError("AG dispatch requires topk_ids")
            ids = topk_ids.to(device=device, dtype=torch.int64).reshape(num_tokens, K)
            if topk_ids_are_deepep:
                # undo the dummy-expert offset the caller already applied
                ids = torch.where(ids >= 0, ids - self.layout.dummy_expert_offset, ids)
            flat_ids = ids.reshape(-1)                                  # (T*K,)
            valid = flat_ids >= 0
            # THREE host-blocking operations used to live in this block, and together they were
            # 59% of the whole AFD decode step -- 38.15 ms of 64.54, i.e. 1.06 ms per layer to
            # plan FOUR slots at decode B=1 (doc 28 §23). Two were not even visibly syncs:
            #
            #   flat_ids[valid]        boolean indexing calls nonzero() internally -> D2H sync
            #   dest[valid]            same
            #   valid.sum().item()     explicit D2H sync
            #
            # Each blocks the host until the GPU drains, so the plan build was absorbing the wait
            # for the attention compute and the previous layer's combine. `torch.where` and a
            # bincount over the sort key do the same work with no sync, and the one host value
            # genuinely needed -- how many slots are valid -- falls out of the counts `tolist()`
            # that the split sizes require anyway, a few lines below.
            #
            # Removing the boolean indexing also removes the data-dependent shapes that make graph
            # capture illegal here, so this is a step toward capturing the transport as well.
            # `w_flat` is computed ONCE here and reused for both the permuted `send_weight` and the
            # unpermuted `topk_weights_flat`. It used to be built twice from `topk_weights` -- the
            # second cast was a whole extra launch per layer for an identical tensor.
            w_flat = (
                topk_weights.to(device=device, dtype=torch.float32).reshape(-1)
                if topk_weights is not None
                else torch.ones(flat_ids.numel(), device=device, dtype=torch.float32)
            )
            if _FUSED_PLAN and _fused_plan.can_fuse(flat_ids.numel()):
                (
                    key, order, send_counts_t, send_payload, send_expert, send_weight, valid,
                ) = _fused_plan.build_plan_fused(
                    flat_ids, w_flat, hidden_states,
                    ag_size=self.layout.ag_size,
                    experts_per_eg_rank=self.experts_per_eg_rank,
                    world=world, top_k=K,
                )
            else:
                safe_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
                dest = torch.where(
                    valid,
                    self.layout.ag_size
                    + torch.div(safe_ids, self.experts_per_eg_rank, rounding_mode="floor"),
                    torch.full_like(flat_ids, -1),
                )
                # Sort key sends invalid slots to the end, so the valid ones form a contiguous
                # prefix and the tail can simply be sliced off once its length is known.
                key = torch.where(valid, dest, torch.full_like(dest, world))
                order = torch.argsort(key, stable=True)          # FULL (T*K,), invalid last
                send_counts_t = _counts_no_sync(key, world + 1)[:world]

                token_of_slot = torch.div(order, K, rounding_mode="floor")
                send_payload = hidden_states.index_select(0, token_of_slot).contiguous()
                send_expert = flat_ids.index_select(0, order).contiguous()
                send_weight = w_flat.index_select(0, order).contiguous()
            # Still full length here; trimmed to the valid prefix after the counts sync.
            _trim_to_valid = True
            valid_slot_mask = valid
            topk_weights_flat = w_flat if topk_weights is not None else None
        else:
            order = torch.empty(0, device=device, dtype=torch.int64)
            send_counts_t = torch.zeros(world, device=device, dtype=torch.int64)
            send_payload = hidden_states.new_empty((0, self.hidden_size))
            send_expert = torch.empty(0, device=device, dtype=torch.int64)
            send_weight = torch.empty(0, device=device, dtype=torch.float32)
            valid_slot_mask = torch.zeros(0, device=device, dtype=torch.bool)
            topk_weights_flat = None
            _trim_to_valid = False

        _plan_ctx.__exit__(None, None, None)

        # ---- fixed-shape path: no counts exchange, no split sizes, no sync --
        _bucket = int(
            num_max_dispatch_tokens_per_rank
            if num_max_dispatch_tokens_per_rank
            else self.num_max_dispatch_tokens_per_rank
        )
        _cap = _bucket * K
        _fx_bytes = world * _cap * (self.hidden_size * dtype.itemsize + 12)
        if _FIXED_SHAPE and _fx_bytes <= _FIXED_SHAPE_MAX_BYTES:
            if self.is_ag and num_tokens > _bucket:
                raise RuntimeError(
                    f"fixed-shape M2N: {num_tokens} tokens exceeds bucket {_bucket}; capacity "
                    "C = bucket*K is only overflow-proof while tokens <= bucket"
                )
            with _sub("d.fixed"):
                return self._dispatch_fixed_shape(
                    hidden_states=hidden_states, dtype=dtype, device=device, world=world, K=K,
                    cap=_cap, num_tokens=num_tokens,
                    order=order if self.is_ag and num_tokens > 0 else None,
                    key=key if self.is_ag and num_tokens > 0 else None,
                    flat_ids=flat_ids if self.is_ag and num_tokens > 0 else None,
                    w_flat=(
                        w_flat if (self.is_ag and num_tokens > 0 and topk_weights is not None)
                        else None
                    ),
                    valid_slot_mask=valid_slot_mask,
                    topk_weights_flat=topk_weights_flat,
                )

        # ---- exchange counts, then the payload -----------------------------
        # all_to_all on the count vector tells each rank how much it will receive.
        recv_counts_t = torch.empty_like(send_counts_t)
        with _sub("d.counts_a2a"):
            dist.all_to_all_single(recv_counts_t, send_counts_t, group=self.group)
        # An explicit synchronize immediately before a call that already synchronizes adds no
        # work -- the tolist below would have drained the queue anyway -- so this splits the
        # bucket exactly instead of perturbing the schedule.
        if _DRAIN_SPLIT:
            with _sub("d.drain"):
                torch.cuda.synchronize()
        # The one unavoidable host sync on this path: all_to_all_single needs
        # python-int split sizes.
        with _sub("d.tolist_sync"):
            send_counts = send_counts_t.tolist()
            recv_counts = recv_counts_t.tolist()
        num_recv = int(sum(recv_counts))
        if _trim_to_valid:
            # `sum(send_counts)` IS the valid-slot count, so no extra sync is needed for it.
            n_valid = int(sum(send_counts))
            order = order[:n_valid]
            send_payload = send_payload[:n_valid]
            send_expert = send_expert[:n_valid]
            send_weight = send_weight[:n_valid]

        # Payload, expert id and weight all move with the SAME split sizes, so they can
        # travel as one packed buffer instead of three collectives. That matters because the
        # cost here is per-call latency, not bytes: at decode B=1 a payload row is 5.7 KB, and
        # the measured dispatch cost is ~1.02 ms/layer against an isolated round trip of
        # 0.843 ms -- four calls' worth of latency (doc 28 §9.2, §9.3).
        #
        # Packing is via a uint8 view, so it is BIT-EXACT: no dtype is narrowed. Packing into
        # spare bf16 columns would be cheaper still but would have to round the fp32 weights
        # and could not represent expert ids above 256 exactly.
        if _FUSE_DISPATCH:
            H = self.hidden_size
            item = dtype.itemsize
            w_bytes = H * item          # hidden row
            e_off = w_bytes             # int64 expert id
            k_off = e_off + 8           # float32 weight
            row_bytes = k_off + 4
            n_send = int(send_payload.shape[0])

            send_buf = torch.empty((n_send, row_bytes), device=device, dtype=torch.uint8)
            if n_send > 0:
                send_buf[:, :w_bytes] = send_payload.view(torch.uint8).view(n_send, w_bytes)
                send_buf[:, e_off:k_off] = send_expert.view(torch.uint8).view(n_send, 8)
                send_buf[:, k_off:] = send_weight.view(torch.uint8).view(n_send, 4)

            recv_buf = torch.empty((num_recv, row_bytes), device=device, dtype=torch.uint8)
            dist.all_to_all_single(
                recv_buf.view(-1), send_buf.view(-1),
                output_split_sizes=[c * row_bytes for c in recv_counts],
                input_split_sizes=[c * row_bytes for c in send_counts],
                group=self.group,
            )
            # A column slice is strided, and `.contiguous()` is NOT enough to normalise it:
            # when the leading dim is 1, PyTorch already calls (1, 8) with stride (5772, 1)
            # contiguous, so `.contiguous()` is a no-op and the dtype view then fails with
            # "stride(0) must be divisible by 8 to view Byte as Long ... but got 5772".
            # Decode at B=1 routes exactly that shape, so this is the common case, not a
            # corner. `.reshape(-1)` forces the copy and yields a genuinely packed buffer.
            def _unpack(lo: int, hi: int, out_dtype: torch.dtype) -> torch.Tensor:
                return recv_buf[:, lo:hi].reshape(-1).contiguous().view(out_dtype)

            recv_payload = _unpack(0, w_bytes, dtype).view(num_recv, H)
            recv_expert = _unpack(e_off, k_off, torch.int64).view(num_recv)
            recv_weight = _unpack(k_off, row_bytes, torch.float32).view(num_recv)
        else:
            with _sub("d.payload_a2a"):
                recv_payload = hidden_states.new_empty((num_recv, self.hidden_size))
                dist.all_to_all_single(
                    recv_payload, send_payload,
                    output_split_sizes=recv_counts, input_split_sizes=send_counts,
                    group=self.group,
                )
                recv_expert = torch.empty(num_recv, device=device, dtype=torch.int64)
                dist.all_to_all_single(
                    recv_expert, send_expert,
                    output_split_sizes=recv_counts, input_split_sizes=send_counts,
                    group=self.group,
                )
                recv_weight = torch.empty(num_recv, device=device, dtype=torch.float32)
                dist.all_to_all_single(
                    recv_weight, send_weight,
                    output_split_sizes=recv_counts, input_split_sizes=send_counts,
                    group=self.group,
                )

        # ---- group arrivals by expert (grouped GEMM needs contiguous experts)
        _grp = _sub("d.group")
        _grp.__enter__()
        local_experts = self.experts_per_eg_rank
        first_local = (
            (self.rank - self.layout.ag_size) * local_experts if self.is_eg else 0
        )
        # `is_eg` guard is deliberate, not defensive padding. On the non-EG branch the eager path
        # does `local_idx = recv_expert_sorted` and then `clamp_` IN PLACE, so it mutates
        # `recv_expert_sorted` to the clamped values; the fused kernel stores the unclamped ids.
        # AG ranks receive nothing in dispatch so the branch should never run, but "should never"
        # is not a reason to introduce a silent behavioural difference on it.
        if (
            _FUSED_GROUP and self.is_eg and num_recv > 0
            and _fused_plan.can_fuse_group(num_recv)
        ):
            (
                expert_order, recv_payload, recv_expert_sorted, recv_weight,
                _counts_per_expert, recv_count,
            ) = _fused_plan.build_group_fused(
                recv_expert, recv_weight, recv_payload,
                first_local=first_local, local_experts=local_experts,
            )
        else:
            if num_recv > 0:
                expert_order = torch.argsort(recv_expert, stable=True)
                recv_payload = recv_payload.index_select(0, expert_order).contiguous()
                recv_expert_sorted = recv_expert.index_select(0, expert_order)
                recv_weight = recv_weight.index_select(0, expert_order).contiguous()
            else:
                expert_order = torch.empty(0, device=device, dtype=torch.int64)
                recv_expert_sorted = recv_expert

            # recv_count: inclusive prefix sum of rows per LOCAL expert
            local_idx = recv_expert_sorted - first_local if self.is_eg else recv_expert_sorted
            counts_per_expert = (
                _counts_no_sync(
                    local_idx.clamp_(min=0, max=max(local_experts - 1, 0)), local_experts
                )
                if num_recv > 0
                else torch.zeros(local_experts, device=device, dtype=torch.int64)
            )
            recv_count = torch.cumsum(counts_per_expert, dim=0).to(torch.int32)

        _grp.__exit__(None, None, None)

        # ---- handle ---------------------------------------------------------
        # `recv_counts_t` is ALREADY on the device. Building psum from the python list
        # `recv_counts` meant device -> host (`.tolist()` above) -> host -> device
        # (`torch.tensor(..., device=device)`) every layer, every forward: an avoidable H2D
        # copy on the critical path, 36 per decode step. Cumsum the device tensor directly.
        psum_rank = torch.cumsum(recv_counts_t, dim=0)
        # recv_src_metadata columns [2:] hold expanded slot ids in the DeepEP handle. NOTHING
        # reads it on this path -- `valid_expanded_slots` is only consumed by the DeepEP adapter
        # and kernel/deepep_moe.py -- so the (num_recv, 3) fill was a kernel per layer producing
        # a tensor no one looks at. Allocate it lazily instead, and keep the -1 fill (rather than
        # empty) so that if some future caller does read it, it reads sentinels rather than
        # uninitialised memory.
        recv_src_metadata = _LazyNegOnes(num_recv, device)
        shim = _RuntimeHandleShim(
            psum_num_recv_tokens_per_scaleup_rank=psum_rank,
            psum_num_recv_tokens_per_expert=recv_count,
            recv_src_metadata=recv_src_metadata,
        )
        handle = RcclM2NHandle(
            handle=shim,
            send_counts=send_counts,
            recv_counts=recv_counts,
            slot_order=order,
            num_local_tokens=num_tokens,
            top_k=K,
            valid_slot_mask=valid_slot_mask,
            topk_weights_flat=topk_weights_flat,
            expert_order=expert_order,
            num_recv_rows=num_recv,
            hidden_size=self.hidden_size,
            dtype=dtype,
            device=device,
        )
        return RcclM2NDispatchOutput(
            hidden_states=recv_payload,
            handle=handle,
            topk_ids=recv_expert_sorted,
            topk_weights=recv_weight,
            recv_count=recv_count,
            layout=self.layout,
            rank=self.rank,
            is_ag=self.is_ag,
            is_eg=self.is_eg,
        )

    # --------------------------------------------------------------------- combine
    def combine(
        self,
        expert_output: torch.Tensor,
        dispatch_output: RcclM2NDispatchOutput | Any,
        *,
        topk_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_not_destroyed()
        if topk_weights is not None:
            raise RuntimeError(
                "RcclM2NAdapter.combine must not receive topk_weights; the expert path "
                "applies expanded top-k weights before combine"
            )
        handle: RcclM2NHandle = (
            dispatch_output.handle
            if isinstance(dispatch_output, RcclM2NDispatchOutput)
            else dispatch_output
        )
        device, dtype = handle.device, handle.dtype
        H = handle.hidden_size

        # Undo the expert grouping so rows line up with the order they arrived in,
        # which is the order the sender laid them out.
        rows = expert_output
        if handle.num_recv_rows > 0:
            if int(rows.shape[0]) != handle.num_recv_rows:
                raise RuntimeError(
                    "combine row count mismatch: "
                    f"expert_output={tuple(rows.shape)} expected {handle.num_recv_rows}"
                )
            with _sub("k.inverse"):
                inverse = torch.empty_like(handle.expert_order)
                inverse.scatter_(
                    0, handle.expert_order,
                    torch.arange(handle.num_recv_rows, device=device, dtype=torch.int64),
                )
                rows = rows.index_select(0, inverse).contiguous()
        rows = rows.to(dtype)

        # Reverse all-to-all: send counts and recv counts swap roles.
        with _sub("k.reverse_a2a"):
            back = torch.empty(
                (int(sum(handle.send_counts)), H), device=device, dtype=dtype
            )
            if handle.fixed_cap is not None:
                # Equal split both ways, so no split sizes -- the same reason dispatch needed no
                # counts exchange. Padding rows hold uninitialised memory (the grouped GEMM sorts
                # them past the last expert's range and never writes them), but they are discarded
                # by the validity mask in the reduction below rather than zeroed here: `torch.where`
                # SELECTS, so it is safe even if the garbage is NaN, and one masked select beats an
                # extra full-buffer write.
                dist.all_to_all_single(back, rows.contiguous(), group=self.group)
            else:
                dist.all_to_all_single(
                    back, rows.contiguous(),
                    output_split_sizes=handle.send_counts,
                    input_split_sizes=handle.recv_counts,
                    group=self.group,
                )

        # Reduce the top-k contributions back onto the originating tokens.
        #
        # NOTE: do NOT apply topk_weights here. Per the contract
        # (dev_log/qwen/11_m2n_contract.md sec 3, and the guard at the top of this method)
        # the expert path applies the expanded top-k weights BEFORE combine, so
        # combine is a pure sum. Weighting again cost a factor of ~mean(w) = 1/top_k;
        # the T-14a combine check caught it as a uniform rel_err of 0.87 with
        # top_k=8, i.e. exactly 1 - 1/8.
        out = torch.zeros(
            (handle.num_local_tokens, H), device=device, dtype=torch.float32
        )
        if handle.num_local_tokens > 0 and back.shape[0] > 0:
            with _sub("k.reduce"):
                if handle.fixed_cap is not None:
                    # `back` is the staging buffer in destination-block order, and `fixed_dst` records
                    # where each of the T*K source slots was placed, so gathering by it returns each
                    # slot's own result. Invalid slots are zeroed rather than skipped, so the reduce
                    # runs over a FIXED T*K rows and contributes nothing for them.
                    src = back.index_select(0, handle.fixed_dst)
                    src = torch.where(
                        handle.fixed_ok[:, None], src, torch.zeros((), device=device, dtype=src.dtype)
                    )
                    token_of_slot = torch.div(
                        handle.slot_order, handle.top_k, rounding_mode="floor"
                    )
                    out.index_add_(0, token_of_slot, src.to(torch.float32))
                else:
                    token_of_slot = torch.div(
                        handle.slot_order, handle.top_k, rounding_mode="floor"
                    )
                    out.index_add_(0, token_of_slot, back.to(torch.float32))
        # DIAGNOSTIC ONLY (MINISGL_M2N_COMBINE_DRAIN=1). Doc 34 §2 left `d.drain` confounded: a
        # synchronize waits on this rank's whole stream, which at the next layer's tolist still
        # holds THIS layer's combine collectives -- so part of that 14 ms is peer rendezvous carried
        # forward, not own compute. Draining here attributes the carried-over part to `k.drain` and
        # leaves `d.drain` holding only the next layer's own attention/route/plan.
        #
        # Unlike the drain split in dispatch, this one DOES perturb: today the host runs ahead and
        # issues the next layer's plan build while the combine collectives are still in flight, and
        # this stops that. Compare step totals against a run without it before reading the split.
        if _COMBINE_DRAIN:
            with _sub("k.drain"):
                torch.cuda.synchronize()
        return out.to(dtype)

    # ---------------------------------------------------------- fixed-shape
    def _dispatch_fixed_shape(
        self, *, hidden_states, dtype, device, world, K, cap, num_tokens,
        order, key, flat_ids, w_flat, valid_slot_mask, topk_weights_flat,
    ) -> "RcclM2NDispatchOutput":
        """Capacity-padded dispatch. Every shape is a host-known constant.

        Layout: one `(world * cap, ...)` staging buffer, destination `d` owning rows
        `[d*cap, (d+1)*cap)`. Because the blocks are equal and contiguous in destination order, a
        plain `all_to_all_single` with **no split sizes** does the routing -- which is what removes
        the counts collective and the `tolist` sync together.

        Padding rows carry expert id **-1**, so validity travels with the data and the receiver needs
        no separate mask. Invalid source slots are written to one extra dump row that is then dropped,
        which keeps the scatter over a fixed `T*K` index set and avoids the boolean indexing that
        would reintroduce a sync.
        """
        H = self.hidden_size
        n_slots = world * cap
        local_experts = self.experts_per_eg_rank

        # ---- pack the send side (AG only; EG sources nothing) ----
        send_payload = hidden_states.new_zeros((n_slots + 1, H))
        send_expert = torch.full((n_slots + 1,), -1, device=device, dtype=torch.int64)
        send_weight = torch.zeros((n_slots + 1,), device=device, dtype=torch.float32)
        fixed_dst = torch.empty(0, device=device, dtype=torch.int64)
        fixed_ok = torch.zeros(0, device=device, dtype=torch.bool)
        if order is not None:
            key_sorted = key.index_select(0, order)
            cnt = _counts_no_sync(key, world + 1)[:world]
            offs = torch.cumsum(cnt, dim=0) - cnt                      # exclusive prefix
            pos = torch.arange(key_sorted.numel(), device=device, dtype=torch.int64)
            ok = key_sorted < world
            # Position within the destination block; garbage for invalid slots, which are routed to
            # the dump row instead of being masked out (masking would need nonzero() -> sync).
            within = pos - offs.index_select(0, torch.where(ok, key_sorted, torch.zeros_like(key_sorted)))
            dst = torch.where(
                ok, key_sorted * cap + within, torch.full_like(within, n_slots)
            )
            # For combine's gather, invalid slots must index INSIDE `back` (which has n_slots rows,
            # without the dump row), so clamp them to 0 and rely on `fixed_ok` to zero them.
            fixed_dst = torch.where(ok, dst, torch.zeros_like(dst))
            fixed_ok = ok
            tok = torch.div(order, K, rounding_mode="floor")
            send_payload.index_copy_(0, dst, hidden_states.index_select(0, tok))
            send_expert.index_copy_(0, dst, flat_ids.index_select(0, order))
            send_weight.index_copy_(
                0, dst,
                (w_flat.index_select(0, order) if w_flat is not None
                 else torch.ones_like(dst, dtype=torch.float32)),
            )

        # ---- equal-split exchange: no split sizes, therefore no host value needed ----
        recv_payload = hidden_states.new_empty((n_slots, H))
        recv_expert = torch.empty((n_slots,), device=device, dtype=torch.int64)
        recv_weight = torch.empty((n_slots,), device=device, dtype=torch.float32)
        with _sub("d.payload_a2a"):
            dist.all_to_all_single(recv_payload, send_payload[:n_slots], group=self.group)
            dist.all_to_all_single(recv_expert, send_expert[:n_slots], group=self.group)
            dist.all_to_all_single(recv_weight, send_weight[:n_slots], group=self.group)

        # ---- group by expert, padding last ----
        with _sub("d.group"):
            first_local = (self.rank - self.layout.ag_size) * local_experts if self.is_eg else 0
            local_idx = recv_expert - first_local
            valid_rx = recv_expert >= 0
            # Padding sorts after every real expert, so it falls outside the counted range and the
            # grouped GEMM never touches it.
            gkey = torch.where(valid_rx, local_idx, torch.full_like(local_idx, local_experts))
            expert_order = torch.argsort(gkey, stable=True)
            recv_payload = recv_payload.index_select(0, expert_order).contiguous()
            recv_expert_sorted = recv_expert.index_select(0, expert_order)
            recv_weight = recv_weight.index_select(0, expert_order).contiguous()
            counts_per_expert = _counts_no_sync(gkey, local_experts + 1)[:local_experts]
            recv_count = torch.cumsum(counts_per_expert, dim=0).to(torch.int32)

        # `send_counts`/`recv_counts` are now CONSTANTS: every rank sends and receives exactly `cap`
        # rows per peer. combine's reverse exchange is equal-split for the same reason.
        counts = [cap] * world
        psum_rank = torch.cumsum(
            torch.full((world,), cap, device=device, dtype=torch.int64), dim=0
        )
        handle = RcclM2NHandle(
            handle=_RuntimeHandleShim(
                psum_num_recv_tokens_per_scaleup_rank=psum_rank,
                psum_num_recv_tokens_per_expert=recv_count,
                recv_src_metadata=_LazyNegOnes(n_slots, device),
            ),
            send_counts=counts, recv_counts=counts,
            slot_order=order if order is not None else torch.empty(0, device=device, dtype=torch.int64),
            num_local_tokens=num_tokens, top_k=K,
            valid_slot_mask=valid_slot_mask, topk_weights_flat=topk_weights_flat,
            expert_order=expert_order, num_recv_rows=n_slots,
            hidden_size=H, dtype=dtype, device=device,
        )
        handle.fixed_cap = cap  # marks the reverse path; see combine
        handle.fixed_dst = fixed_dst
        handle.fixed_ok = fixed_ok
        return RcclM2NDispatchOutput(
            hidden_states=recv_payload, handle=handle,
            topk_ids=recv_expert_sorted, topk_weights=recv_weight,
            recv_count=recv_count, layout=self.layout, rank=self.rank,
            is_ag=self.is_ag, is_eg=self.is_eg,
        )

    # ------------------------------------------------------------- static helpers
    @staticmethod
    def num_recv_tokens(dispatch_output: RcclM2NDispatchOutput) -> int:
        psum = dispatch_output.handle.handle.psum_num_recv_tokens_per_scaleup_rank
        if psum.numel() == 0:
            return 0
        return int(psum[-1].item())

    @staticmethod
    def valid_expanded_slots(dispatch_output: RcclM2NDispatchOutput) -> torch.Tensor:
        metadata = dispatch_output.handle.handle.recv_src_metadata
        if metadata.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=metadata.device)
        slots = metadata[:, 2:].reshape(-1)
        return slots[slots >= 0].to(dtype=torch.long)

    # -------------------------------------------------------------------- teardown
    def destroy(self) -> None:
        self._destroyed = True

    close = destroy

    # --------------------------------------------------------------------- helpers
    def route_to_deepep_topk(
        self,
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused router top-k, returning ids in the PADDED ("deepep") expert space.

        Signature and return contract must match `DeepEPM2NAdapter`: the caller
        (`afd_attention_worker.materialize_deepep_topk`) sets `deepee_topk=True` on
        the result and dispatch is then told `topk_ids_are_deepep=True`, so the ids
        must carry `dummy_expert_offset`. dispatch() subtracts it again.

        Unlike the DeepEP version this does not use `topk_softmax_group_local` with a
        padded expert map -- it takes the plain top-k over the real expert space and
        shifts, which is equivalent and avoids depending on the group-local kernel.
        """
        local_tokens = int(hidden_states.shape[0])
        if router_logits.ndim != 2:
            raise ValueError(
                f"router_logits must be 2D, got shape={tuple(router_logits.shape)}"
            )
        expected = (local_tokens, self.layout.real_num_experts)
        if tuple(router_logits.shape) != expected:
            raise ValueError(
                "router_logits shape mismatch: "
                f"got={tuple(router_logits.shape)} expected={expected}"
            )
        if router_logits.device != hidden_states.device:
            raise ValueError(
                "router_logits and hidden_states must be on the same device: "
                f"{router_logits.device} vs {hidden_states.device}"
            )

        topk_weights = torch.empty(
            (local_tokens, self.top_k), dtype=torch.float32, device=hidden_states.device
        )
        real_topk_ids = torch.empty(
            (local_tokens, self.top_k), dtype=torch.int32, device=hidden_states.device
        )
        if local_tokens > 0:
            from minisgl.kernel import topk_softmax

            topk_softmax(
                topk_weights,
                real_topk_ids,
                router_logits.float().contiguous(),
                bool(renormalize),
            )
        deepep_topk_ids = remap_real_to_deepep_experts(
            real_topk_ids.to(torch.int64), self.layout
        ).to(self.topk_idx_dtype)
        return deepep_topk_ids, topk_weights

    def _check_not_destroyed(self) -> None:
        if self._destroyed:
            raise RuntimeError("RcclM2NAdapter has been destroyed")

    def _emit_log(self, message: str) -> None:
        if self._log_callback is not None:
            self._log_callback(message)


__all__ = [
    "RcclM2NAdapter",
    "RcclM2NDispatchOutput",
    "RcclM2NHandle",
]
