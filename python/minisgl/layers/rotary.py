from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
        attention_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.attention_scaling = attention_scaling
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        if attention_scaling != 1.0:
            # YaRN's "mscale". HF multiplies cos/sin by this factor *before* applying the
            # rotation (see `_compute_yarn_parameters`' `attention_factor`), so folding it
            # into the precomputed cache is exact and costs nothing at runtime. Note the
            # rotation is then no longer norm-preserving -- that is intended: q and k are
            # each scaled, so the qk logit is scaled by attention_scaling**2.
            cos = cos * attention_scaling
            sin = sin * attention_scaling
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        from ._ops_backend import get_ops

        self.apply_rope_with_cos_sin_cache_inplace = (
            get_ops().apply_rope_with_cos_sin_cache_inplace
        )

    @property
    def cos_sin_cache(self) -> torch.Tensor:
        return self._cos_sin_cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            # Transcribed from HF `transformers.modeling_rope_utils._compute_yarn_parameters`
            # and verified against it in dev_log/gpt_oss_120b/gptoss_rope_parity.py. Three
            # things here were previously wrong and every one of them failed *silently*:
            #   1. `truncate` was ignored (always floor/ceil). gpt-oss sets it False.
            #   2. `high` was clamped to rotary_dim//2 - 1 instead of HF's rotary_dim - 1.
            #   3. `attention_factor` was missing entirely -- a 1.35x scale on every
            #      cos/sin at gpt-oss's factor=32.
            # Measured before the fix, at gpt-oss's config: 76% max relative error on
            # inv_freq and 34.7% on cos/sin.
            factor: float = rope_scaling["factor"]
            # HF uses `or`, so an explicit 0 falls back to the default.
            beta_fast: float = rope_scaling.get("beta_fast") or 32.0
            beta_slow: float = rope_scaling.get("beta_slow") or 1.0
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]
            truncate: bool = rope_scaling.get("truncate", True)

            attention_factor = rope_scaling.get("attention_factor")
            if attention_factor is None:
                mscale = rope_scaling.get("mscale")
                mscale_all_dim = rope_scaling.get("mscale_all_dim")

                def _get_mscale(scale: float, m: float = 1.0) -> float:
                    if scale <= 1.0:
                        return 1.0
                    return 0.1 * m * math.log(scale) + 1.0

                if mscale and mscale_all_dim:
                    # DeepSeek-style split mscale.
                    attention_factor = float(
                        _get_mscale(factor, mscale) / _get_mscale(factor, mscale_all_dim)
                    )
                else:
                    attention_factor = _get_mscale(factor)

            def _find_correction_dim(num_rotations: float) -> float:
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = _find_correction_dim(beta_fast)
            high = _find_correction_dim(beta_slow)
            if truncate:
                low, high = math.floor(low), math.ceil(high)
            low, high = max(low, 0), min(high, rotary_dim - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                lo, hi = low, high
                if lo == hi:
                    hi += 0.001  # HF: prevent a zero-width ramp
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - lo) / (hi - lo),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                post_process,
                attention_scaling=float(attention_factor),
            )

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
