"""Paged flash-attention prefill in Triton, for gfx942.

Why: `torch_ref` prefill measured **163 TFLOPS ≈ 12% of bf16 peak**, and TTFT is 24–65%
of wall-clock time in the 4-card matrix (dev_log/23 §5). It is the last un-optimised part
of the colocated path. It works like this:

    for each request:                      # Python loop, O(bs) launches per layer
        slots = indices[k_lo:k_hi]
        k_i   = k_cache[slots]             # MATERIALISES this request's whole KV
        SDPA(q_i, k_i, v_i, mask)

So per layer it pays `bs` kernel launches, and it allocates and writes `sum(Lk)` rows of
K and V before reading them once. This module replaces that with **one launch per layer**
that streams the paged KV in blocks and never materialises it.

## The correctness trap, restated

The causal mask must be **bottom-right aligned**: the `Lq` query rows are the *last* `Lq`
positions of an `Lk`-long sequence, so query `i` may attend to key positions
`0 .. (Lk - Lq) + i`. PyTorch's `is_causal=True` is top-left aligned and is correct only
when `Lq == Lk` — it silently corrupts every partial-cache-hit prefill, which is exactly
what a radix cache produces (dev_log/09). `probes/triton_prefill_parity.py` asserts both
that this matches bottom-right **and** that it differs from top-left, so a regression
fails loudly rather than quietly.

## Not graph-captured

`GraphRunner` only captures decode (`can_use_cuda_graph` requires `batch.is_decode`), so
this path has no static-shape constraint and can size its grid from host-known lengths.
That is why it may use `max(extend_len)` for the grid where `triton_decode` may not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

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

# Launch parameters, module-level so probes read the SAME values the backend ships.
# gptoss_attn_parity.py used to hardcode BLOCK_M=64 and so reported 23/23 on a config that
# was no longer in use the moment these were retuned to 128 -- it validated a kernel nobody
# ran, and missed a NaN that 128 introduced (see PREFILL_BLOCK_M note below). A tuning
# constant and its gate must not be able to drift apart.
PREFILL_BLOCK_M = 128
PREFILL_BLOCK_N = 64
PREFILL_NUM_WARPS = 4
PREFILL_NUM_STAGES = 1


@triton.jit
def _paged_prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    page_table_ptr, table_rows_ptr, q_start_ptr, q_len_ptr, k_len_ptr,
    q_s_tok, q_s_head,
    kv_s_slot, kv_s_head,
    pt_s_row,
    o_s_tok, o_s_head,
    sink_ptr,
    scale,
    GQA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SINK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """One program per (query block, qo_head, request). Online softmax over paged KV."""
    m_blk = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)

    q_len = tl.load(q_len_ptr + b)
    # Wave-uniform: m_blk and b are program ids, so every lane agrees. This is NOT the
    # wave64 hazard from dev_log/qwen/wave64_fix.md, where the predicate varied per lane.
    if m_blk * BLOCK_M >= q_len:
        return

    k_len = tl.load(k_len_ptr + b)
    q_off = tl.load(q_start_ptr + b).to(tl.int64)
    row = tl.load(table_rows_ptr + b).to(tl.int64)
    kvh = h // GQA

    offs_m = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < q_len
    q = tl.load(
        q_ptr + (q_off + offs_m)[:, None] * q_s_tok + h * q_s_head + offs_d[None, :],
        mask=mask_m[:, None], other=0.0,
    )

    # Finite sentinel, NOT -inf. If a row's first iterated key block is entirely masked, the
    # running max stays at the initial value and `alpha = exp(m_i - m_new)` becomes
    # exp(-inf - -inf) = exp(nan) = nan, which poisons `acc` for the rest of the loop.
    #
    # That is reachable whenever BLOCK_M > BLOCK_N on a sliding-window layer: `lo` below is
    # derived from the block's FIRST row, so with BLOCK_M rows spanning more positions than one
    # key block covers, the block's last rows can have their whole window past `lo + BLOCK_N`.
    # A finite sentinel makes alpha = exp(0) = 1 and p = exp(-inf - sentinel) = 0, so an
    # all-masked block contributes nothing instead of nan; fully-masked rows then fall through
    # to the `l_i == 0` guard below and yield 0.
    m_i = tl.full((BLOCK_M,), -1e38, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Bottom-right alignment. `cached` is how many keys precede the first query row.
    cached = k_len - q_len
    # The largest key index any row in this block may see; blocks past it are all-masked
    # and skipped entirely rather than iterated and thrown away.
    hi = tl.minimum(cached + m_blk * BLOCK_M + BLOCK_M, k_len)
    # Sliding-window layers: row i sees keys in (pos_i - WINDOW, pos_i], so the lowest key
    # ANY row in this block needs is pos_of_first_row - WINDOW + 1. WINDOW == 0 is full
    # attention (Qwen3, and gpt-oss's `full_attention` layers) and starts at 0.
    if WINDOW > 0:
        lo = tl.maximum(0, cached + m_blk * BLOCK_M - WINDOW + 1)
    else:
        lo = 0

    for start in range(lo, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        in_range = offs_n < k_len
        slots = tl.load(page_table_ptr + row * pt_s_row + offs_n, mask=in_range, other=0)
        kv_off = (
            slots.to(tl.int64)[:, None] * kv_s_slot + kvh * kv_s_head + offs_d[None, :]
        )
        # `.to(q.dtype)` dequantises an FP8 cache and is a no-op for a bf16 one. It must come
        # immediately after the load: `p.to(k.dtype)` below uses k's dtype as a stand-in for
        # the compute dtype, so with a raw fp8 k that would round the softmax probabilities
        # to fp8 -- 3 mantissa bits on values in [0,1] -- and quietly wreck the PV product.
        k = tl.load(k_ptr + kv_off, mask=in_range[:, None], other=0.0).to(q.dtype)
        v = tl.load(v_ptr + kv_off, mask=in_range[:, None], other=0.0).to(q.dtype)

        qk = tl.dot(q, tl.trans(k)) * scale
        pos = cached + offs_m[:, None]
        causal = offs_n[None, :] <= pos
        if WINDOW > 0:
            # Per-row lower edge, not a per-block one: within a block the rows have
            # different positions, so a single scalar bound would over- or under-mask.
            causal = causal & (offs_n[None, :] > pos - WINDOW)
        qk = tl.where(in_range[None, :] & causal, qk, float("-inf"))

        # Block 0 always contains key 0, which every row may see (cached + i >= 0), so
        # m_i is finite from the first iteration and exp(-inf - -inf) never arises.
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(k.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Attention sink: joins the softmax denominator only (HF concatenates it, softmaxes,
    # then drops the last column). One scalar per q head, so no mask needed here.
    if HAS_SINK:
        sink = tl.load(sink_ptr + h)
        m_s = tl.maximum(m_i, sink)
        l_i = l_i * tl.exp(m_i - m_s) + tl.exp(sink - m_s)
        acc = acc * tl.exp(m_i - m_s)[:, None]
        m_i = m_s

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + (q_off + offs_m)[:, None] * o_s_tok + h * o_s_head + offs_d[None, :],
        acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None],
    )


@dataclass
class TritonPrefillMetadata(BaseAttnMetadata):
    # fmt: off
    q_start:      torch.Tensor   # (bs,)   device int32, offset of each request's queries
    q_len:        torch.Tensor   # (bs,)   device int32, extend_len
    k_len:        torch.Tensor   # (bs,)   device int32, device_len
    table_rows:   torch.Tensor   # (bs,)   device int32, row in the global page_table
    last_indices: torch.Tensor   # (bs,)   device int32, last query row per request
    max_q_len:    int            # host, for the grid
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class TritonPrefillBackend(BaseAttnBackend):
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
                f"triton_prefill requires page_size=1, got {page_size}"
            )

        # Retuned on gfx942; see dev_log/gpt_oss_120b/19_prefill_attention_tile.md.
        #
        # `num_stages=1` is the dominant factor and it is NOT head_dim specific, which is the
        # opposite of what was expected. Software-pipelining the K/V loop double-buffers both
        # tiles in LDS, and on this Triton/ROCm build that costs enough occupancy to more than
        # cancel the latency hiding. Measured at M=8192, 18 full + 18 sliding layers:
        #
        #                        default 64/64/w4/s2   this 128/64/w4/s1
        #   gpt-oss  head_dim 64        43.9 ms             19.9 ms   2.20x
        #   Qwen3-30B head_dim 128      46.4 ms             22.2 ms   2.09x
        #
        # So the old default was ~2.1x off at BOTH shapes, including the head_dim 128 one it
        # was originally swept for. Changing num_stages alone is 1.5-1.8x of that.
        #
        # BLOCK_M=128 rather than the 256 that wins by 4% at M=8192: prefill chunk sizes vary
        # (remainder chunks, short prompts, mixed-length batches) and 256 is 1.5x WORSE at
        # M=2048. This config is best or tied at M=256/1024/2048 and within 4% at 8192.
        self.block_m = PREFILL_BLOCK_M
        self.block_n = PREFILL_BLOCK_N
        self.num_warps = PREFILL_NUM_WARPS
        self.num_stages = PREFILL_NUM_STAGES
        from .triton_decode import _build_layer_windows
        self._layer_window = _build_layer_windows(config)
        logger.info_rank0(
            "triton_prefill attention: qo_heads=%d kv_heads=%d head_dim=%d gqa=%d "
            "BLOCK_M=%d BLOCK_N=%d warps=%d stages=%d -- paged, one launch per layer, "
            "bottom-right causal",
            self.qo_head_local, self.kv_head_local, self.head_dim,
            self.gqa_group, self.block_m, self.block_n, self.num_warps, self.num_stages,
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch,
        *, sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TritonPrefillMetadata), (
            "triton_prefill is a prefill-only backend; pair it with a decode backend, "
            "e.g. --attention-backend triton_prefill,triton_decode"
        )

        # The current tokens' K/V must be in the pool before we read it: the causal
        # window includes them.
        if not batch.afd_kv_store_merged:
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        flat = (-1, self.kv_head_local, self.head_dim)
        k_cache = self.kvcache.k_cache(layer_id).view(*flat)
        v_cache = self.kvcache.v_cache(layer_id).view(*flat)
        page_table = get_global_ctx().page_table

        out = torch.empty_like(q)
        bs = metadata.q_len.numel()
        grid = (triton.cdiv(metadata.max_q_len, self.block_m), self.qo_head_local, bs)
        _paged_prefill_kernel[grid](
            q, k_cache, v_cache, out,
            page_table, metadata.table_rows,
            metadata.q_start, metadata.q_len, metadata.k_len,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1),
            page_table.stride(0),
            out.stride(0), out.stride(1),
            sink if sink is not None else q,
            self.scale,
            GQA=self.gqa_group,
            HEAD_DIM=self.head_dim,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            HAS_SINK=sink is not None,
            WINDOW=self._layer_window(layer_id),
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return out

    # ------------------------------------------------------------ metadata prep
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        q_lens, k_lens, rows, starts = [], [], [], []
        off = 0
        for req in reqs:
            starts.append(off)
            q_lens.append(req.extend_len)
            k_lens.append(req.device_len)
            rows.append(req.table_idx)
            off += req.extend_len

        kw = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        host = torch.empty((4, len(reqs)), **kw)
        host[0] = torch.tensor(starts, dtype=torch.int32)
        host[1] = torch.tensor(q_lens, dtype=torch.int32)
        host[2] = torch.tensor(k_lens, dtype=torch.int32)
        host[3] = torch.tensor(rows, dtype=torch.int32)
        dev = host.to(self.device, non_blocking=True)

        # Last query row of each request, for the LM head.
        last = torch.tensor(
            [s + n - 1 for s, n in zip(starts, q_lens)], dtype=torch.int32, **{"device": "cpu"}
        ).to(self.device, non_blocking=True)

        batch.attn_metadata = TritonPrefillMetadata(
            q_start=dev[0], q_len=dev[1], k_len=dev[2], table_rows=dev[3],
            last_indices=last,
            max_q_len=max(q_lens) if q_lens else 0,
        )

    # -------------------------------------------------------------- graph hooks
    _MSG = (
        "triton_prefill is a prefill backend and is never graph-captured: GraphRunner "
        "only captures decode batches. Pair it with a capturable decode backend."
    )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError(self._MSG)

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError(self._MSG)

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError(self._MSG)
