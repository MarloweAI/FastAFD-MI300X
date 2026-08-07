"""HIP-graph-capturable paged decode attention in Triton.

Why this exists
---------------
`torch_ref` is correct but cannot be graph-captured: its forward loops over requests
in Python using host-side offsets, so lengths and shapes would be baked into the graph
(see that module's docstring). With capture disabled, colocated decode measured
**1.93x launch-bound** -- CPU 16.1 ms/step against GPU 8.4 ms/step, across 1146 kernel
launches (dev_log/qwen/14_performance.md). Graphs are worth up to that whole factor, and this
backend is the enabler; the attention arithmetic itself was only 1.6-3.4% of GPU time,
so speed is *not* the goal here -- capturability is.

What makes it capturable
------------------------
Graph capture bakes in the **grid dimensions and the kernel's pointer/scalar arguments**,
not a kernel's internal control flow. So:

* The KV loop bound is read from a **device** tensor (`seq_lens`) at kernel runtime. A
  data-dependent trip count inside the kernel is invisible to the graph.
* Every buffer the kernel touches is at a **fixed address**: the KV pool and the global
  `page_table` are allocated once in the Context, and the per-batch metadata
  (`table_rows`, `seq_lens`) lives in buffers this backend allocates once and then
  overwrites in place.
* There is no host sync, no `.item()`, no `.tolist()`, and no ragged `torch.cat` on the
  forward path -- the ragged gather `torch_ref` does per request is replaced by indexing
  the global page table with a row index.

Because `prepare_metadata()` runs every step *before* `Engine.forward_batch()` (see
`scheduler.py:_prepare_batch`), and it writes into the very buffers the graph captured,
`prepare_for_replay()` has nothing left to do.

Split-K over the sequence
-------------------------
The first version launched one program per `(request, kv_head)` and no more. That is
`bs * kv_head_local` workgroups -- **16 at bs=16/TP4, on a 304-CU GPU** -- each walking
the whole KV serially. It was fine at the 128-token contexts doc 16 measured (attention
was 0.2% of GPU time) and catastrophic at 8192: 785 us/layer, **86 GB/s**, and
`bs=16` took the *same* time as `bs=1`, which is the signature of pure serialization
rather than a bandwidth limit. That made attention ~18% of a TP4 long-context decode
step.

`_paged_decode_split_kernel` + `_decode_reduce_kernel` add a third grid dimension over
the sequence and merge the partials by log-sum-exp: **15.8x** at bs=16/TP4/8k and
**18.0x** at 32k. The split count comes from `_splits_for(bs)`, which depends only on
host-known values so the grid stays static for HIP graph capture -- it cannot look at
the runtime sequence length, so splits landing past the end of a short sequence store
`l = 0` and the reducer skips them.

Measured eagerly, split-K *loses* at 128 tokens (0.41x) because it pays a second kernel
launch. Inside a graph that launch is free and it wins in both regimes -- bs=1 short
context went 143.8 -> 157.6 tok/s. So this is only a good default *because* capture
works; see dev_log/19.

Deliberate limitations
----------------------
* **Decode only** (one query token per request). Pair it with a prefill backend:
  `--attention-backend torch_ref,triton_decode`.
* `page_size` must be 1, matching `torch_ref` and the FlashInfer backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
import triton
import triton.language as tl
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)


def _build_layer_windows(config) -> "callable":
    """Return `layer_id -> attention window` (0 = full attention).

    gpt-oss sets `layer_types` to an alternating list of `sliding_attention` /
    `full_attention` and a single `sliding_window`. Models without `layer_types`
    (Qwen3, Llama, ...) are full attention everywhere, which keeps `WINDOW=0` and the
    kernel identical to before this was added.
    """
    layer_types = getattr(config, "layer_types", None)
    window = int(getattr(config, "sliding_window", 0) or 0)
    if not layer_types or window <= 0:
        return lambda _layer_id: 0
    types = list(layer_types)

    def lookup(layer_id: int) -> int:
        if 0 <= layer_id < len(types) and types[layer_id] == "sliding_attention":
            return window
        return 0

    return lookup


@triton.jit
def _paged_decode_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    page_table_ptr, table_rows_ptr, seq_lens_ptr,
    q_s_tok, q_s_head,
    kv_s_slot, kv_s_head,
    pt_s_row,
    o_s_tok, o_s_head,
    sink_ptr,
    scale,
    GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SINK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """One program per (request, kv_head); online-softmax over that request's KV.

    `BLOCK_M` is the GQA group padded up to the MFMA tile height (16 on CDNA3), so the
    `GQA`-wide group can go through `tl.dot`. Rows beyond `GQA` are loaded as zero and
    never stored.
    """
    b = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1).to(tl.int64)

    # Both come from device memory -- that is what keeps this capturable.
    seq_len = tl.load(seq_lens_ptr + b)
    row = tl.load(table_rows_ptr + b).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < GQA
    offs_d = tl.arange(0, HEAD_DIM)
    q_head = kvh * GQA + offs_m

    q = tl.load(
        q_ptr + b * q_s_tok + q_head[:, None] * q_s_head + offs_d[None, :],
        mask=mask_m[:, None], other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    if WINDOW > 0:
        win_lo = tl.maximum(0, seq_len - WINDOW)
    else:
        win_lo = 0
    for start in range(win_lo, seq_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = (offs_n < seq_len) & (offs_n >= win_lo)
        slots = tl.load(page_table_ptr + row * pt_s_row + offs_n, mask=mask_n, other=0)
        kv_off = (
            slots.to(tl.int64)[:, None] * kv_s_slot
            + kvh * kv_s_head
            + offs_d[None, :]
        )
        # `.to(q.dtype)` dequantises an FP8 cache, no-op for bf16. Must precede any use of
        # k.dtype as the compute dtype (see `p.to(k.dtype)` below).
        k = tl.load(k_ptr + kv_off, mask=mask_n[:, None], other=0.0).to(q.dtype)
        v = tl.load(v_ptr + kv_off, mask=mask_n[:, None], other=0.0).to(q.dtype)

        # fp32 accumulate, then scale -- matches SDPA, which divides by sqrt(E) after
        # the matmul rather than pre-scaling q.
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(k.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Attention sink (gpt-oss): a per-head logit that joins the softmax DENOMINATOR but
    # not the numerator. HF concatenates it to the logits, softmaxes, then drops the last
    # column -- equivalently, add exp(sink - m) to l after the KV loop.
    if HAS_SINK:
        sink = tl.load(sink_ptr + q_head, mask=mask_m, other=float("-inf"))
        m_s = tl.maximum(m_i, sink)
        l_i = l_i * tl.exp(m_i - m_s) + tl.exp(sink - m_s)
        acc = acc * tl.exp(m_i - m_s)[:, None]
        m_i = m_s

    # seq_len == 0 would leave l_i at zero; a padded slot must not produce NaN and
    # poison the graph's output buffer.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + b * o_s_tok + q_head[:, None] * o_s_head + offs_d[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


@triton.jit
def _paged_decode_split_kernel(
    q_ptr, k_ptr, v_ptr,
    acc_ptr, m_ptr, l_ptr,
    page_table_ptr, table_rows_ptr, seq_lens_ptr,
    q_s_tok, q_s_head,
    kv_s_slot, kv_s_head,
    pt_s_row,
    a_s_tok, a_s_head, a_s_split,
    ml_s_tok, ml_s_head,
    sink_ptr,
    scale,
    GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """Flash-decoding split-K: program (b, kv_head, split) owns one slice of the KV.

    Emits per-split partial accumulator, max and sum so `_decode_reduce_kernel` can
    merge them with the usual log-sum-exp rescale. This exists because the one-program
    -per-(request, kv_head) version left only `bs * kv_heads` workgroups on a 304-CU
    GPU -- 16 at TP4 -- and 8192-token contexts then walked 128 KV blocks serially at
    86 GB/s. See dev_log/19_long_context_attention.md.
    """
    b = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2).to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + b)
    row = tl.load(table_rows_ptr + b).to(tl.int64)

    # Contiguous slice per split, rounded up to whole BLOCK_N tiles so each split
    # starts tile-aligned. Splits past the end of the sequence do no work and store
    # l = 0, which the reducer skips.
    # Sliding-window layers (gpt-oss `sliding_attention`) see only the last WINDOW
    # keys. WINDOW == 0 means full attention, which is Qwen3 and gpt-oss's
    # `full_attention` layers.
    if WINDOW > 0:
        win_lo = tl.maximum(0, seq_len - WINDOW)
    else:
        win_lo = 0
    span = seq_len - win_lo
    n_blocks = (span + BLOCK_N - 1) // BLOCK_N
    blocks_per_split = (n_blocks + SPLITS - 1) // SPLITS
    lo = win_lo + split * blocks_per_split * BLOCK_N
    hi = tl.minimum(lo + blocks_per_split * BLOCK_N, seq_len)

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < GQA
    offs_d = tl.arange(0, HEAD_DIM)
    q_head = kvh * GQA + offs_m
    q = tl.load(
        q_ptr + b * q_s_tok + q_head[:, None] * q_s_head + offs_d[None, :],
        mask=mask_m[:, None], other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start in range(lo, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < hi
        slots = tl.load(page_table_ptr + row * pt_s_row + offs_n, mask=mask_n, other=0)
        kv_off = (
            slots.to(tl.int64)[:, None] * kv_s_slot + kvh * kv_s_head + offs_d[None, :]
        )
        # `.to(q.dtype)` dequantises an FP8 cache, no-op for bf16. Must precede any use of
        # k.dtype as the compute dtype (see `p.to(k.dtype)` below).
        k = tl.load(k_ptr + kv_off, mask=mask_n[:, None], other=0.0).to(q.dtype)
        v = tl.load(v_ptr + kv_off, mask=mask_n[:, None], other=0.0).to(q.dtype)
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(k.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Attention sink: a per-head logit that joins the softmax DENOMINATOR but not the
    # numerator (HF `eager_attention_forward` concatenates it, softmaxes, then drops the
    # last column). Fold it into `l` on split 0 only -- adding it on every split would
    # count it SPLITS times once the reducer sums them.
    if HAS_SINK:
        if split == 0:
            sink = tl.load(sink_ptr + q_head, mask=mask_m, other=float("-inf"))
            m_s = tl.maximum(m_i, sink)
            l_i = l_i * tl.exp(m_i - m_s) + tl.exp(sink - m_s)
            acc = acc * tl.exp(m_i - m_s)[:, None]
            m_i = m_s

    a_off = b * a_s_tok + q_head[:, None] * a_s_head + split * a_s_split + offs_d[None, :]
    tl.store(acc_ptr + a_off, acc, mask=mask_m[:, None])
    ml_off = b * ml_s_tok + q_head * ml_s_head + split
    tl.store(m_ptr + ml_off, m_i, mask=mask_m)
    tl.store(l_ptr + ml_off, l_i, mask=mask_m)


@triton.jit
def _decode_reduce_kernel(
    acc_ptr, m_ptr, l_ptr, o_ptr,
    a_s_tok, a_s_head, a_s_split,
    ml_s_tok, ml_s_head,
    o_s_tok, o_s_head,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Merge the per-split partials for one (request, q_head)."""
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    offs_s = tl.arange(0, SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)

    ml_off = b * ml_s_tok + h * ml_s_head + offs_s
    m = tl.load(m_ptr + ml_off)
    l = tl.load(l_ptr + ml_off)
    # An empty split stored l == 0 and m == -inf; exclude it so it cannot contribute
    # a NaN through exp(-inf - m_max).
    live = l > 0.0
    m = tl.where(live, m, float("-inf"))
    m_max = tl.max(m, 0)
    scale = tl.where(live, tl.exp(m - m_max), 0.0)
    l_tot = tl.sum(l * scale, 0)

    acc = tl.load(
        acc_ptr + b * a_s_tok + h * a_s_head + offs_s[:, None] * a_s_split + offs_d[None, :]
    )
    out = tl.sum(acc * scale[:, None], 0) / tl.where(l_tot == 0.0, 1.0, l_tot)
    tl.store(o_ptr + b * o_s_tok + h * o_s_head + offs_d, out.to(o_ptr.dtype.element_ty))


@dataclass
class TritonDecodeMetadata(BaseAttnMetadata):
    # fmt: off
    table_rows:   torch.Tensor   # (bs,) device int32 -- row of the global page_table
    seq_lens:     torch.Tensor   # (bs,) device int32 -- KV length per request
    last_indices: torch.Tensor   # (bs,) device int32 -- arange; decode emits 1 token each
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # Consumed by the LM head (layers/embedding.py) *inside* the captured graph, so
        # this must be a persistent device tensor. For decode it is a constant arange.
        return self.last_indices[:bs]


class TritonDecodeBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device

        tp_size = get_tp_info().size
        self.qo_head_local = div_even(config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.gqa_group = self.qo_head_local // self.kv_head_local
        self.scale = 1.0 / (self.head_dim ** 0.5)

        page_size = get_global_ctx().page_size
        if page_size != 1:
            raise NotImplementedError(
                f"triton_decode requires page_size=1, got {page_size}"
            )

        # MFMA on CDNA3 wants a tile height of 16; pad the GQA group up to it.
        self.block_m = max(16, triton.next_power_of_2(self.gqa_group))
        self.block_n = 64
        # gpt-oss alternates `sliding_attention` (window 128) and `full_attention` per
        # layer, so the window is per-layer, not per-model. Qwen3 has no layer_types and
        # gets window 0 (full attention) everywhere.
        self._layer_window = _build_layer_windows(config)
        # Split-K sizing. Aim for ~2 workgroups per CU so the machine is full even at
        # bs=1; cap the split count so short contexts do not launch a swarm of
        # do-nothing programs plus an oversized reduction.
        cu_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        # 4x rather than 2x CUs: swept in dev_log/probes/attn_knob_sweep.py, and splits=32
        # (the cap) is optimal or within 2% at every shape measured. The 2x target picked
        # 16 at TP1/ISL 8192/bs=16, where 32 is 1.10x faster; oversubscribing costs
        # nothing measurable even at bs=1/ctx=128 (44.1 us at splits=1 vs 45 at splits=32),
        # because splits past the end of a sequence exit immediately.
        self._target_programs = 4 * int(cu_count)
        self._max_splits = 32
        # MINISGL_DECODE_SPLITS pins the split count. Its purpose is measurement: the
        # only way to get attention's *in-server* cost is to change attention alone and
        # diff end-to-end latency, because the isolated-kernel benchmark overpredicts by
        # ~3x (dev_log/19 sec 2) and decode_profile.py cannot separate decode from
        # prefill at long ISL. Not a tuning knob for production.
        override = os.environ.get("MINISGL_DECODE_SPLITS")
        self._splits_override = int(override) if override else None
        if self._splits_override is not None:
            logger.warning_rank0(
                "triton_decode: MINISGL_DECODE_SPLITS=%d pins the split count "
                "(measurement override)", self._splits_override,
            )

        # Persistent metadata buffers. Sized lazily on first use to cover the largest
        # batch seen, since the scheduler's cap is not visible here. Reallocating would
        # invalidate a captured graph, so once capture has happened the size is frozen
        # and `_ensure_capacity` refuses to grow (see the assert there).
        self._cap = 0
        self._frozen = False
        self.table_rows: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.last_indices: torch.Tensor | None = None
        self._host: torch.Tensor | None = None
        self._captured_bs: List[int] = []

        logger.info_rank0(
            "triton_decode attention: qo_heads=%d kv_heads=%d head_dim=%d gqa=%d "
            "BLOCK_M=%d BLOCK_N=%d -- HIP-graph capturable",
            self.qo_head_local, self.kv_head_local, self.head_dim,
            self.gqa_group, self.block_m, self.block_n,
        )

    # ------------------------------------------------------------------ buffers
    def _ensure_capacity(self, bs: int) -> None:
        if bs <= self._cap:
            return
        assert not self._frozen, (
            f"triton_decode: batch {bs} exceeds the {self._cap} slots frozen at graph "
            "capture. Growing the buffers now would move them and every captured graph "
            "would read freed memory. Raise --cuda-graph-max-bs (or lower "
            "--max-running-requests) so capture covers the largest decode batch."
        )
        self._cap = bs
        self.table_rows = torch.zeros(bs, dtype=torch.int32, device=self.device)
        self.seq_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
        self.last_indices = torch.arange(bs, dtype=torch.int32, device=self.device)
        # Pinned staging buffer: one non-blocking H2D per step instead of two.
        self._host = torch.zeros((2, bs), dtype=torch.int32, pin_memory=True)

    def _write_metadata(self, batch: Batch) -> TritonDecodeMetadata:
        reqs = batch.padded_reqs
        bs = len(reqs)
        self._ensure_capacity(bs)
        assert self.table_rows is not None and self.seq_lens is not None
        assert self.last_indices is not None and self._host is not None

        host = self._host
        for i, req in enumerate(reqs):
            host[0, i] = req.table_idx
            host[1, i] = req.device_len
        # In-place into the persistent buffers: this is exactly what makes replay work
        # without a separate prepare_for_replay step.
        self.table_rows[:bs].copy_(host[0, :bs], non_blocking=True)
        self.seq_lens[:bs].copy_(host[1, :bs], non_blocking=True)

        return TritonDecodeMetadata(
            table_rows=self.table_rows[:bs],
            seq_lens=self.seq_lens[:bs],
            last_indices=self.last_indices,
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch,
        *, sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TritonDecodeMetadata), (
            "triton_decode is a decode-only backend; use it as the decode half of a "
            "hybrid spec, e.g. --attention-backend torch_ref,triton_decode"
        )

        if not batch.afd_kv_store_merged:
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        flat = (-1, self.kv_head_local, self.head_dim)
        k_cache = self.kvcache.k_cache(layer_id).view(*flat)
        v_cache = self.kvcache.v_cache(layer_id).view(*flat)
        page_table = get_global_ctx().page_table

        bs = q.shape[0]
        out = torch.empty_like(q)
        splits = self._splits_for(bs)
        window = self._layer_window(layer_id)
        has_sink = sink is not None
        sink_arg = sink if has_sink else q  # a real pointer even when unused

        if splits == 1:
            _paged_decode_kernel[(bs, self.kv_head_local)](
                q, k_cache, v_cache, out,
                page_table, metadata.table_rows, metadata.seq_lens,
                q.stride(0), q.stride(1),
                k_cache.stride(0), k_cache.stride(1),
                page_table.stride(0),
                out.stride(0), out.stride(1),
                sink_arg,
                self.scale,
                GQA=self.gqa_group,
                BLOCK_M=self.block_m,
                HEAD_DIM=self.head_dim,
                BLOCK_N=self.block_n,
                HAS_SINK=has_sink,
                WINDOW=window,
                num_warps=4,
            )
            return out

        # Split-K. Partials are fp32 and shaped (bs, qo_heads, splits, head_dim); at
        # bs=16/TP4/32 splits that is 16*8*32*128*4 = 2 MiB, negligible next to the KV
        # it saves re-walking. Allocated per call so graph capture takes them from the
        # graph's private pool at fixed addresses.
        acc = torch.empty(
            (bs, self.qo_head_local, splits, self.head_dim),
            dtype=torch.float32, device=q.device,
        )
        m = torch.empty((bs, self.qo_head_local, splits), dtype=torch.float32, device=q.device)
        l = torch.empty_like(m)
        _paged_decode_split_kernel[(bs, self.kv_head_local, splits)](
            q, k_cache, v_cache, acc, m, l,
            page_table, metadata.table_rows, metadata.seq_lens,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1),
            page_table.stride(0),
            acc.stride(0), acc.stride(1), acc.stride(2),
            m.stride(0), m.stride(1),
            sink_arg,
            self.scale,
            GQA=self.gqa_group,
            BLOCK_M=self.block_m,
            HEAD_DIM=self.head_dim,
            BLOCK_N=self.block_n,
            SPLITS=splits,
            HAS_SINK=has_sink,
            WINDOW=window,
            num_warps=4,
        )
        _decode_reduce_kernel[(bs, self.qo_head_local)](
            acc, m, l, out,
            acc.stride(0), acc.stride(1), acc.stride(2),
            m.stride(0), m.stride(1),
            out.stride(0), out.stride(1),
            HEADS=self.qo_head_local,
            HEAD_DIM=self.head_dim,
            SPLITS=splits,
            num_warps=4,
        )
        return out

    def _splits_for(self, bs: int) -> int:
        """How many sequence splits to launch, so the grid fills the GPU.

        `bs * kv_head_local` alone is tiny -- 16 workgroups at bs=16/TP4 on 304 CUs,
        which measured 86 GB/s and made attention 57% of a long-context decode step
        (dev_log/19_long_context_attention.md). Splits must be a host-side constant so
        the grid stays static for HIP graph capture, so this depends only on `bs`, not
        on the runtime sequence length; splits that fall past the end of a short
        sequence simply do no work.
        """
        if self._splits_override is not None:
            return self._splits_override
        base = max(1, bs * self.kv_head_local)
        want = max(1, self._target_programs // base)
        return int(min(self._max_splits, triton.next_power_of_2(want)))

    # ------------------------------------------------------------ metadata prep
    def prepare_metadata(self, batch: Batch) -> None:
        assert batch.is_decode, "triton_decode received a prefill batch"
        batch.attn_metadata = self._write_metadata(batch)

    # -------------------------------------------------------------- graph hooks
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # Allocate for the largest graph batch and freeze: after this point the buffer
        # addresses are baked into captured graphs.
        self._ensure_capacity(max(bs_list))
        self._frozen = True
        self._captured_bs = sorted(bs_list)
        logger.info_rank0(
            "triton_decode: metadata buffers frozen at %d slots for graph sizes %s",
            self._cap, self._captured_bs,
        )

    def prepare_for_capture(self, batch: Batch) -> None:
        # Dummy values are fine; the captured kernel only needs valid *addresses*.
        # Real values arrive every step via prepare_metadata(), which writes the same
        # buffers in place.
        batch.attn_metadata = self._write_metadata(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        # Intentionally empty. `prepare_metadata()` already wrote this step's
        # table_rows/seq_lens into the persistent buffers the graph captured, and
        # `scheduler._prepare_batch` calls it before `Engine.forward_batch`. Writing
        # again here would be a redundant H2D copy per step.
        return None
