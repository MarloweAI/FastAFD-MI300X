from __future__ import annotations

import os

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from .base import BaseKVCachePool

logger = init_logger(__name__)

# MINISGL_KV_RANGE_CHECK=1 reports real K/V magnitudes per layer on the first forward.
# FP8 is a float format, so its relative precision does not depend on scale -- a scale factor
# buys nothing unless values leave the representable band. What DOES depend on magnitude is
# accuracy downstream: qk grows with |K|, so a fixed relative error in K becomes a larger
# absolute error in the logits, which exp() amplifies. Measured in
# gptoss_fp8_kv_parity.py: attention output error goes 2.1e-2 -> 5.1e-1 as |K| goes 1.7 -> 169.
# Hence this: whether fp8 KV is safe for a given model is an empirical question about |K|.
_KV_RANGE_CHECK = bool(os.environ.get("MINISGL_KV_RANGE_CHECK"))
_RANGE_SEEN: set[int] = set()


def _report_kv_range(k, v, layer_id: int, k_dtype, v_dtype) -> None:
    if layer_id in _RANGE_SEEN:
        return
    _RANGE_SEEN.add(layer_id)
    kf, vf = k.float(), v.float()
    # RMS as well as max: the softmax logit error depends on K's typical magnitude through the
    # dot product, not on a single outlier element, so max alone would mis-rank the risk.
    parts = [f"|K|max={kf.abs().max().item():.3f} rms={kf.pow(2).mean().sqrt().item():.3f}",
             f"|V|max={vf.abs().max().item():.3f} rms={vf.pow(2).mean().sqrt().item():.3f}"]
    for tag, t, dt in (("k", kf, k_dtype), ("v", vf, v_dtype)):
        fi = torch.finfo(dt)
        if fi.max < 1e30:
            parts.append(f"clip_{tag}={int((t.abs() > fi.max).sum().item())}/{fi.max:g}")
    logger.info_rank0("KV range layer %2d: %s", layer_id, "  ".join(parts))


class MHAKVCache(BaseKVCachePool):
    """Concrete paged MHA key-value cache.

    Stores keys and values in a single contiguous
    [2, num_layers, num_pages, page_size, local_kv_heads, head_dim] buffer.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        v_dtype: torch.dtype | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        v_dtype = v_dtype or dtype
        shape = (num_layers, num_pages, page_size, local_kv_heads, head_dim)
        if v_dtype == dtype:
            # One allocation, K and V as views: keeps the original contiguity.
            self._kv_buffer = torch.empty((2, *shape), device=device, dtype=dtype)
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
        else:
            # Split precisions (e.g. K bf16 + V fp8) cannot share a buffer. Worth the second
            # allocation because K and V are NOT equally quantisable: K feeds the softmax, so
            # its error is exponentiated, while V enters the output linearly. See
            # dev_log/gpt_oss_120b/21_fp8_kv_cache.md §3.
            self._kv_buffer = None
            self._k_buffer = torch.empty(shape, device=device, dtype=dtype)
            self._v_buffer = torch.empty(shape, device=device, dtype=v_dtype)
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        if _KV_RANGE_CHECK:
            _report_kv_range(k, v, layer_id, self._k_buffer.dtype, self._v_buffer.dtype)

        # FP8 (or any narrower) cache: quantise before the copy so store_cache stays a pure
        # byte move. Casting here rather than inside the store kernel keeps that kernel
        # dtype-agnostic -- it is JIT-keyed on row *bytes*, which already differ for fp8
        # (128 vs 256 per row at head_dim 64), so it needs no change at all. K and V are cast
        # independently because they may have different dtypes.
        if k.dtype != self._k_buffer.dtype:
            k = k.to(self._k_buffer.dtype)
        if v.dtype != self._v_buffer.dtype:
            v = v.to(self._v_buffer.dtype)

        kc = self._k_buffer[layer_id].view(self._storage_shape)
        vc = self._v_buffer[layer_id].view(self._storage_shape)
        if kc.dtype == vc.dtype:
            store_cache(k_cache=kc, v_cache=vc, indices=out_loc, k=k, v=v)
        else:
            # store_cache derives ONE element_size from k_cache and applies it to both sides
            # (kernel/store.py:40), so a split-precision pair would copy the wrong byte count
            # for V. Issuing it once per side with the pair duplicated keeps the existing
            # JIT kernel unchanged; the cost is writing each side twice to the same addresses,
            # ~113 MB of extra writes on an 8k prefill chunk (~0.03 ms) and under 1 MB at
            # decode. Not worth a bespoke kernel for an experimental precision mode.
            store_cache(k_cache=kc, v_cache=kc, indices=out_loc, k=k, v=k)
            store_cache(k_cache=vc, v_cache=vc, indices=out_loc, k=v, v=v)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._k_buffer.dtype

    @property
    def v_dtype(self) -> torch.dtype:
        return self._v_buffer.dtype
