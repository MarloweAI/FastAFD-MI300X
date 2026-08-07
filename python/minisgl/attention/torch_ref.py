"""Reference attention backend built on `torch.scaled_dot_product_attention`.

Purpose: unblock the end-to-end path on hardware with no FlashInfer/TRT-LLM
build (gfx942). It is a *correctness* backend — deliberately simple, obviously
right, and slow. It exists so the rest of the stack (scheduler, KV cache, MoE,
sampling) can be validated before any tuned kernel work starts, and so the
tuned backend has something to be diffed against.

Registered as `"torch_ref"`; see `minisgl.attention.__init__`.

What it does
------------
Per request, gather that request's KV slots out of the paged pool into a
contiguous tensor and call SDPA:

    q_i     = q[cu_q[i] : cu_q[i+1]]                    (Lq, qo_heads, D)
    slots   = indices[cu_k[i] : cu_k[i+1]]              (Lk,)
    k_i     = k_cache.view(-1, kv_heads, D)[slots]      (Lk, kv_heads, D)

Masking is the one genuinely easy thing to get wrong. The Lq query rows are the
*last* Lq positions of an Lk-long sequence, so the causal mask must be
**bottom-right aligned**: query j (0-based within the extend) may attend to key
positions 0 .. cached_len + j, where cached_len = Lk - Lq. `is_causal=True` in
PyTorch is top-left aligned and would be wrong whenever Lq != Lk — i.e. every
cache-hit prefill. An explicit mask is built instead. Decode (Lq == 1) needs no
mask at all, since every key is visible.

Limitations (intentional, for this milestone)
---------------------------------------------
* **No HIP/CUDA graph capture.** The forward is a Python loop over requests
  driven by host-side offsets, which cannot be captured. `init_capture_graph`
  raises with an actionable message; run with `--cuda-graph-max-bs 0`.
* Materializes each request's KV, so memory and bandwidth scale with context
  length rather than being streamed. O(bs) kernel launches per layer.
* `page_size` must be 1, matching the FlashInfer backend's own restriction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)

# torch>=2.5 does GQA natively inside SDPA; older versions need an explicit
# key/value head expansion.
_SDPA_HAS_GQA = "enable_gqa" in F.scaled_dot_product_attention.__doc__ if (
    F.scaled_dot_product_attention.__doc__
) else False


@dataclass
class TorchRefMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:  torch.Tensor   # (bs+1,) host, cumulative query lengths
    cu_seqlens_k_cpu:  torch.Tensor   # (bs+1,) host, cumulative KV lengths
    cu_seqlens_q_gpu:  torch.Tensor   # (bs+1,) device, for get_last_indices()
    indices:           torch.Tensor   # (sum Lk,) device, concatenated KV slots
    seq_lens_cpu:      torch.Tensor   # (bs,) host, per-request Lk
    num_qo_heads:      int
    num_kv_heads:      int
    head_dim:          int
    is_decode:         bool
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class TorchRefBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device

        tp_size = get_tp_info().size
        self.qo_head_local = div_even(config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.gqa_group = self.qo_head_local // self.kv_head_local

        page_size = get_global_ctx().page_size
        if page_size != 1:
            raise NotImplementedError(
                f"torch_ref backend requires page_size=1, got {page_size}"
            )

        logger.info_rank0(
            "torch_ref attention backend: qo_heads=%d kv_heads=%d head_dim=%d gqa=%d "
            "sdpa_native_gqa=%s. Reference implementation — correctness first, no graph capture.",
            self.qo_head_local,
            self.kv_head_local,
            self.head_dim,
            self.gqa_group,
            _SDPA_HAS_GQA,
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch,
        *, sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `sink` is gpt-oss's attention sink. This backend cannot represent it, so refuse
        # rather than silently dropping it -- a dropped sink changes the softmax
        # denominator and would degrade output without any error.
        if sink is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement attention sinks; use "
                "triton_prefill/triton_decode for gpt-oss."
            )
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchRefMetadata)

        if not batch.afd_kv_store_merged:
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        flat = (-1, self.kv_head_local, self.head_dim)
        k_cache = self.kvcache.k_cache(layer_id).view(*flat)
        v_cache = self.kvcache.v_cache(layer_id).view(*flat)

        cu_q = metadata.cu_seqlens_q_cpu.tolist()
        cu_k = metadata.cu_seqlens_k_cpu.tolist()
        out = torch.empty_like(q)

        for i in range(len(cu_q) - 1):
            q_lo, q_hi = cu_q[i], cu_q[i + 1]
            k_lo, k_hi = cu_k[i], cu_k[i + 1]
            len_q, len_k = q_hi - q_lo, k_hi - k_lo
            if len_q == 0 or len_k == 0:
                continue

            slots = metadata.indices[k_lo:k_hi].long()
            # (1, heads, seq, dim) for SDPA
            q_i = q[q_lo:q_hi].transpose(0, 1).unsqueeze(0)
            k_i = k_cache[slots].transpose(0, 1).unsqueeze(0)
            v_i = v_cache[slots].transpose(0, 1).unsqueeze(0)

            attn_mask = None
            if len_q > 1:
                # Bottom-right aligned causal mask: the len_q query rows are the
                # LAST len_q positions of a len_k sequence.
                cached = len_k - len_q
                q_pos = torch.arange(len_q, device=q.device).unsqueeze(1) + cached
                k_pos = torch.arange(len_k, device=q.device).unsqueeze(0)
                attn_mask = k_pos <= q_pos

            o_i = self._sdpa(q_i, k_i, v_i, attn_mask)
            out[q_lo:q_hi] = o_i.squeeze(0).transpose(0, 1)

        return out

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """SDPA with GQA, using the native path when available."""
        if self.gqa_group == 1:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if _SDPA_HAS_GQA:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=True
            )
        k = k.repeat_interleave(self.gqa_group, dim=1)
        v = v.repeat_interleave(self.gqa_group, dim=1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    # ------------------------------------------------------------ metadata prep
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max(seqlens_q) == 1:  # pure decode
            cu_seqlens_q_cpu = torch.arange(0, len(reqs) + 1, **CPU_KWARGS)
        elif all(length == 0 for length in cached_lens):  # prefill, no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # extend prefill with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = get_global_ctx().page_table
        batch.attn_metadata = TorchRefMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(self.device, non_blocking=True),
            indices=torch.cat(
                [page_table[req.table_idx, : req.device_len] for req in reqs]
            ),
            seq_lens_cpu=torch.tensor(seqlens_k, **CPU_KWARGS),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.head_dim,
            is_decode=batch.is_decode,
        )

    # -------------------------------------------------------------- graph hooks
    _GRAPH_MSG = (
        "torch_ref is a reference backend and cannot be HIP/CUDA-graph captured: its "
        "forward loops over requests using host-side offsets, so the shapes and lengths "
        "would be baked into the graph and replays would silently read the wrong KV. "
        "Re-run with `--cuda-graph-max-bs 0` to disable capture, or use a graph-capable "
        "backend."
    )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError(self._GRAPH_MSG)

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError(self._GRAPH_MSG)

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError(self._GRAPH_MSG)
