"""MXFP4 → BF16 dequantisation for gpt-oss expert weights.

gpt-oss ships its MoE experts as MXFP4: `*_blocks` (uint8, two 4-bit values per byte) plus
`*_scales` (uint8, an E8M0 shared exponent per 32-value block). `quantization_config` is
`{"quant_method": "mxfp4"}`.

**This dequantises rather than computing in FP4, and that is not a shortcut** — native
MXFP4/MXFP6 tensor-core math arrived with **CDNA 4 (gfx950, MI350X)**. On gfx942/CDNA3 there
is no FP4 datapath at all, so any MXFP4 model must be widened before the GEMM. The AITER
"MXFP4 a16w4" path on this generation is likewise a mixed-precision dequant-then-GEMM, not
native FP4 (dev_log/qwen/05_model_selection.md §2).

Consequence worth stating plainly: dequantising to BF16 turns gpt-oss-120b's 61 GB checkpoint
into **241 GB** of resident weights, i.e. ~60 GB/card at TP4. It fits, but the memory
advantage that makes MXFP4 attractive does not survive on this hardware.

The conversion itself is **exact**: every FP4 E2M1 code is representable in BF16, and the
E8M0 scale is applied with `ldexp`, so there is no rounding to reason about. Verified
bit-exact against HF's `convert_moe_packed_tensors` in
`dev_log/gpt_oss_120b/gptoss_ops_parity.py`.
"""

from __future__ import annotations

import torch

# FP4 E2M1, indexed by the 4-bit code. Bit 3 is the sign, so the table is the positive
# ladder followed by its negation -- the same order HF's FP4_VALUES uses.
FP4_VALUES: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

__all__ = ["FP4_VALUES", "dequant_mxfp4", "is_mxfp4"]


def is_mxfp4(quant_method: str | None) -> bool:
    return bool(quant_method) and "mxfp4" in str(quant_method).lower()


def dequant_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 512,
) -> torch.Tensor:
    """Unpack MXFP4 `(blocks, scales)` to `dtype`.

    `blocks` is `(..., G, B)` uint8 holding `2*B` FP4 values per row-group; `scales` is
    `(..., G)` uint8 E8M0. Returns `(..., G*B*2)` transposed on the last two dims, matching
    HF's layout so the expert GEMM sees weights the same way `Mxfp4GptOssExperts` would.

    Chunked over rows because the intermediate index tensors are int64 and 120b's experts
    are large: unchunked, `gate_up_proj` alone would materialise tens of GB of indices.
    """
    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError(
            f"expected uint8 blocks/scales, got {blocks.dtype}/{scales.dtype}"
        )
    # HF's reference ends with `.transpose(1, 2)`, which requires the dequantised
    # tensor to be >=3-D, i.e. blocks must be 4-D. Every real gpt-oss tensor is:
    # gate_up_proj_blocks is (experts, 5760, 90, 16), down_proj_blocks is
    # (experts, 2880, 90, 16). Fail with that explanation rather than an IndexError
    # from the transpose.
    if blocks.dim() != 4:
        raise ValueError(
            f"expected 4-D blocks (experts, out_features, G, B), got {tuple(blocks.shape)}. "
            "The trailing transpose that matches HF's layout needs two prefix dims."
        )
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(
            f"blocks.shape[:-1]={tuple(blocks.shape[:-1])} != scales.shape="
            f"{tuple(scales.shape)}"
        )

    # E8M0 bias is 127, so the stored byte is exponent+127.
    exp = scales.to(torch.int32) - 127
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix, G, B = blocks.shape
    rows = 1
    for d in prefix:
        rows *= d
    rows *= G

    blk = blocks.reshape(rows, B)
    exp = exp.reshape(rows, 1)
    out = torch.empty(rows, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows)
        b = blk[r0:r1]
        sub = out[r0:r1]
        # Low nibble goes to EVEN lanes, high nibble to ODD. Getting this backwards
        # produces a plausible-looking tensor with the wrong values, so it is asserted
        # bit-exact against HF in the probe rather than eyeballed.
        sub[:, 0::2] = lut[(b & 0x0F).to(torch.long)]
        sub[:, 1::2] = lut[(b >> 4).to(torch.long)]
        torch.ldexp(sub, exp[r0:r1], out=sub)

    out = out.reshape(*prefix, G, B * 2).view(*prefix, G * B * 2)
    return out.transpose(1, 2).contiguous()
