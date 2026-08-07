from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def _jit_moe_align_module() -> Module:
    from .utils import load_jit

    return load_jit(
        "moe_align",
        cuda_files=["moe_align.cu"],
        cuda_wrappers=[("launch", "MoeAlignKernel::run")],
    )


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert topk_ids.is_cuda, "topk_ids must be on CUDA"
    assert topk_ids.is_contiguous(), "topk_ids must be contiguous"
    assert topk_ids.dtype in (torch.int32, torch.int64), "topk_ids must be int32 or int64"
    assert block_size > 0, "block_size must be positive"
    assert num_experts > 0, "num_experts must be positive"

    num_slots = num_experts + 1
    if topk_ids.numel() == 0:
        empty = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
        zero = torch.zeros((1,), dtype=torch.int32, device=topk_ids.device)
        return empty, empty, zero

    if topk_ids.numel() < num_slots:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + num_slots * (block_size - 1)

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_slots + 1,), dtype=torch.int32, device=topk_ids.device)

    module = _jit_moe_align_module()
    module.launch(
        topk_ids,
        num_slots,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad
