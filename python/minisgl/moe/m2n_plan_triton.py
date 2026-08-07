"""Fused send-plan build for the M2N dispatch: ~15 kernel launches collapsed into one.

Why this exists
---------------
Doc 34 §3 measured the AG send-plan build (`d.plan`) at **8.20 ms of a 47.7 ms AFD decode step --
17%, 228 us per layer** -- to plan `T*K = 4` slots at decode B=1. None of that is compute. The
eager build issues roughly twenty kernels (`ge`, `zeros_like`, two `where`, `div`, `add`,
`full_like`, a stable `argsort`, `zeros`, `ones_like`, `index_add_`, another `div`, three
`index_select`) over four-element tensors, and on gfx942 the per-launch floor is ~16 us.

Isolated on one GPU (`dev_log/probes/plan_build_bench.py`, T*K=4):

    eager (ships today)     114.6 us
    fused (this module)      39.2 us    2.93x
    HIP graph replay         67.6 us    1.70x

**Fusion beats graph capture here**, which is worth stating because the opposite was assumed: a
replay still replays twenty kernels, just with a cheaper launch each, while this issues one.

Stability is a correctness requirement, not a nicety
----------------------------------------------------
The shipping path uses `torch.argsort(key, stable=True)`, and `send_expert` / `send_weight` /
`send_payload` are gathered with that exact permutation while `send_counts` is a histogram of the
same key. The receiving EG rank un-permutes with `inverse`. So the sort must be stable in the same
sense, not merely a valid permutation of each bucket. A counting sort is stable when each element's
destination is

    (number of elements with a strictly smaller key) + (number of EARLIER elements with the same key)

which is what `start + rank` below computes. That makes the output bit-identical to `argsort(
stable=True)`, verified element-wise against the eager path in the benchmark rather than assumed.

Scope
-----
The rank computation is O(n^2) within a single Triton program, so this is a **decode-shaped**
optimisation. `FUSED_MAX_SLOTS` caps it and the caller must fall back to the eager build above that;
at `T*K = 256` the fused advantage is already gone (1.09x), and prefill's `T*K` runs to 32768.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - ROCm builds ship Triton, but do not hard-require it
    HAVE_TRITON = False

# One program, O(n^2) rank computation. Above this the eager path is both correct and faster.
FUSED_MAX_SLOTS = 512


if HAVE_TRITON:

    @triton.jit
    def _plan_kernel(
        ids_ptr, w_ptr,                                   # in:  (n,) int64, (n,) float32
        key_ptr, order_ptr, tok_ptr, cnt_ptr,             # out: plan
        expert_ptr, weight_ptr, valid_ptr,                # out: permuted payload metadata
        n,
        AG: tl.constexpr, EPER: tl.constexpr, WORLD: tl.constexpr, K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        ids = tl.load(ids_ptr + offs, mask=m, other=-1)
        valid = ids >= 0
        # Invalid (unrouted/padding) slots take key = WORLD so they sort to the very end and the
        # valid ones form a contiguous prefix the caller can slice once the length is known.
        dest = tl.where(valid, AG + ids // EPER, -1)
        key = tl.where(valid, dest, WORLD)

        for b in tl.static_range(WORLD):
            tl.store(cnt_ptr + b, tl.sum(tl.where(m & (key == b), 1, 0).to(tl.int64)))

        same_earlier = (key[:, None] == key[None, :]) & (offs[:, None] > offs[None, :])
        rank = tl.sum(tl.where(same_earlier & m[:, None] & m[None, :], 1, 0).to(tl.int64), axis=1)
        smaller = (key[None, :] < key[:, None]) & m[None, :]
        start = tl.sum(tl.where(smaller, 1, 0).to(tl.int64), axis=1)
        pos = start + rank

        tl.store(key_ptr + offs, key, mask=m)
        tl.store(valid_ptr + offs, valid, mask=m)
        tl.store(order_ptr + pos, offs.to(tl.int64), mask=m)
        tl.store(tok_ptr + pos, (offs // K).to(tl.int64), mask=m)
        tl.store(expert_ptr + pos, ids, mask=m)
        tl.store(weight_ptr + pos, tl.load(w_ptr + offs, mask=m, other=0.0), mask=m)


def can_fuse(n_slots: int) -> bool:
    return HAVE_TRITON and 0 < n_slots <= FUSED_MAX_SLOTS


# ---------------------------------------------------------------------------
# Receive-side grouping (EG). Same shape of problem as the send plan: ~12 launches over a handful
# of rows, sitting between the payload landing and the grouped expert GEMM. Doc 35 §6 measured it
# at 5.68 ms of a 43.73 ms EG step. Everything fuses except the payload gather, which moves
# num_recv * hidden * 2 bytes and is a genuine copy.
#
# The row cap is LOWER than the send plan's: measured 2.42x at 128 rows but **0.75x at 256**, where
# the O(n^2) rank computation overtakes ~12 launches. Decode B=1 gives ~4 rows per EG rank, but
# B=64 gives ~256 -- right past the crossover -- so the cap is load-bearing, not a formality.
FUSED_MAX_GROUP_ROWS = 128


def can_fuse_group(n_rows: int) -> bool:
    return HAVE_TRITON and 0 < n_rows <= FUSED_MAX_GROUP_ROWS


if HAVE_TRITON:

    @triton.jit
    def _group_kernel(
        exp_ptr, w_ptr,
        order_ptr, exp_sorted_ptr, w_sorted_ptr, cnt_ptr, recv_cnt_ptr,
        n,
        FIRST: tl.constexpr, NE: tl.constexpr, BLOCK: tl.constexpr, EBLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        # `other` must sort AFTER every real key, so masked lanes can never displace a real row.
        e = tl.load(exp_ptr + offs, mask=m, other=2**30)

        same_earlier = (e[:, None] == e[None, :]) & (offs[:, None] > offs[None, :])
        rank = tl.sum(tl.where(same_earlier & m[:, None] & m[None, :], 1, 0).to(tl.int64), axis=1)
        smaller = (e[None, :] < e[:, None]) & m[None, :]
        start = tl.sum(tl.where(smaller, 1, 0).to(tl.int64), axis=1)
        pos = start + rank

        tl.store(order_ptr + pos, offs.to(tl.int64), mask=m)
        tl.store(exp_sorted_ptr + pos, e, mask=m)
        tl.store(w_sorted_ptr + pos, tl.load(w_ptr + offs, mask=m, other=0.0), mask=m)

        # Histogram on the CLAMPED local index and its inclusive prefix sum, matching the eager
        # path exactly. Clamping is monotonic so it cannot disturb the ordering established above.
        li = tl.minimum(tl.maximum(e - FIRST, 0), NE - 1)
        bins = tl.arange(0, EBLOCK)
        counts = tl.sum(tl.where((bins[:, None] == li[None, :]) & m[None, :], 1, 0).to(tl.int64),
                        axis=1)
        bmask = bins < NE
        tl.store(cnt_ptr + bins, counts, mask=bmask)
        tl.store(recv_cnt_ptr + bins,
                 tl.cumsum(tl.where(bmask, counts, 0)).to(tl.int32), mask=bmask)


def build_group_fused(
    recv_expert: torch.Tensor,    # (n,) int64, global expert ids as they arrived
    recv_weight: torch.Tensor,    # (n,) float32
    recv_payload: torch.Tensor,   # (n, H)
    *, first_local: int, local_experts: int,
) -> tuple[torch.Tensor, ...]:
    """Returns (expert_order, payload, expert_sorted, weight, counts, recv_count).

    Bit-identical to the eager grouping in `RcclM2NAdapter.dispatch`.
    """
    n = recv_expert.numel()
    dev = recv_expert.device
    order = torch.empty(n, device=dev, dtype=torch.int64)
    exp_sorted = torch.empty(n, device=dev, dtype=torch.int64)
    weight = torch.empty(n, device=dev, dtype=torch.float32)
    counts = torch.empty(local_experts, device=dev, dtype=torch.int64)
    recv_count = torch.empty(local_experts, device=dev, dtype=torch.int32)
    _group_kernel[(1,)](
        recv_expert, recv_weight, order, exp_sorted, weight, counts, recv_count, n,
        FIRST=first_local, NE=local_experts,
        BLOCK=triton.next_power_of_2(max(n, 8)),
        EBLOCK=triton.next_power_of_2(local_experts),
    )
    payload = recv_payload.index_select(0, order).contiguous()
    return order, payload, exp_sorted, weight, counts, recv_count


def warmup(device, *, ag_size: int, experts_per_eg_rank: int, world: int, top_k: int) -> int:
    """Compile the kernel for every BLOCK it can be launched with, before serving starts.

    `BLOCK` is `next_power_of_2(T*K)`, so each distinct running batch size can trigger a fresh
    Triton compilation. Left lazy, the FIRST request pays it: measured **25.2 s** for a 48-token
    completion against 1.55 s once warm. That cost is invisible to steady-state profiling -- the
    sub-step capture skips the first 8 steps and read a healthy 42.79 ms while the user-visible
    first token was catastrophic -- so it has to be warmed explicitly rather than noticed.

    Returns the number of variants compiled. AG ranks only; EG ranks source no tokens and never
    take this path.
    """
    if not HAVE_TRITON:
        return 0
    n = 8
    count = 0
    while n <= FUSED_MAX_SLOTS:
        # CEIL: `tok` holds `offs // top_k` for `offs < n`, so the payload needs
        # `ceil(n / top_k)` rows. Floor division would put index_select out of bounds whenever
        # top_k does not divide the power-of-two BLOCK (e.g. n=8, top_k=3 -> tok reaches 2).
        tokens = max(1, -(-n // top_k))
        build_plan_fused(
            torch.zeros(n, device=device, dtype=torch.int64),
            torch.zeros(n, device=device, dtype=torch.float32),
            torch.zeros((tokens, 1), device=device, dtype=torch.bfloat16),
            ag_size=ag_size, experts_per_eg_rank=experts_per_eg_rank,
            world=world, top_k=top_k,
        )
        count += 1
        n *= 2
    torch.cuda.synchronize()
    return count


def warmup_group(device, *, first_local: int, local_experts: int) -> int:
    """Same as `warmup`, for the EG grouping kernel. EG ranks only.

    Both kernels JIT per `BLOCK`, and an unwarmed kernel puts multiple seconds on a user's first
    request while steady-state profiling reports nothing wrong (doc 35 §4).
    """
    if not HAVE_TRITON:
        return 0
    n = 8
    count = 0
    while n <= FUSED_MAX_GROUP_ROWS:
        build_group_fused(
            torch.full((n,), first_local, device=device, dtype=torch.int64),
            torch.zeros(n, device=device, dtype=torch.float32),
            torch.zeros((n, 1), device=device, dtype=torch.bfloat16),
            first_local=first_local, local_experts=local_experts,
        )
        count += 1
        n *= 2
    torch.cuda.synchronize()
    return count


def build_plan_fused(
    flat_ids: torch.Tensor,      # (T*K,) int64, negative = unrouted
    w_flat: torch.Tensor,        # (T*K,) float32
    hidden_states: torch.Tensor, # (T, H)
    *, ag_size: int, experts_per_eg_rank: int, world: int, top_k: int,
) -> tuple[torch.Tensor, ...]:
    """Returns (key, order, send_counts, send_payload, send_expert, send_weight, valid).

    Bit-identical to the eager build in `RcclM2NAdapter.dispatch`; see module docstring.
    """
    n = flat_ids.numel()
    dev = flat_ids.device
    key = torch.empty(n, device=dev, dtype=torch.int64)
    order = torch.empty(n, device=dev, dtype=torch.int64)
    tok = torch.empty(n, device=dev, dtype=torch.int64)
    expert = torch.empty(n, device=dev, dtype=torch.int64)
    weight = torch.empty(n, device=dev, dtype=torch.float32)
    valid = torch.empty(n, device=dev, dtype=torch.bool)
    counts = torch.zeros(world, device=dev, dtype=torch.int64)
    _plan_kernel[(1,)](
        flat_ids, w_flat, key, order, tok, counts, expert, weight, valid, n,
        AG=ag_size, EPER=experts_per_eg_rank, WORLD=world, K=top_k,
        BLOCK=triton.next_power_of_2(max(n, 8)),
    )
    send_payload = hidden_states.index_select(0, tok).contiguous()
    return key, order, counts, send_payload, expert, weight, valid
