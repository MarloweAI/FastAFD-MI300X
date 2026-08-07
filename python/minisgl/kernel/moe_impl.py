from typing import Any, Dict

import torch


def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
    bias: torch.Tensor | None = None,
) -> None:
    import triton
    import triton.language as tl

    from .triton.fused_moe import fused_moe_kernel

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    # `compute_type` selects the Triton store dtype. There is no fp32 branch, so an
    # fp32 activation used to fall through to tl.float16 and compute the whole MoE in
    # half precision -- silently, with relative errors measured at 1e2-1e4 (see
    # dev_log/gpt_oss_120b/00_README.md §14). Refuse instead of degrading.
    if compute_type == torch.bfloat16:
        dtype = tl.bfloat16
    elif compute_type == torch.float16:
        dtype = tl.float16
    else:
        raise ValueError(
            f"fused MoE kernel supports bfloat16/float16 activations, got {compute_type}. "
            "It has no fp32 path; running one would silently compute in fp16."
        )
    if bias is not None:
        # (num_experts, N), indexed by the block's expert id inside the kernel.
        if bias.dim() != 2 or bias.shape[1] != B.shape[1]:
            raise ValueError(
                f"bias must be (num_experts, N={B.shape[1]}), got {tuple(bias.shape)}"
            )
        # The expert count must match too, and checking only N is not enough -- it is the
        # hole that lets an EP sharding bug through SILENTLY. Under EP the weight is
        # sliced to the local experts but the bias keeps its full width, so an unsharded
        # bias still satisfies the N check; the kernel then indexes bias[local_id] where
        # local_id means global_id - start, and every rank past 0 reads the wrong
        # expert's bias. No shape error, no NaN, just wrong numbers of exactly the size
        # that flips a near-tie. See models/weight.py:_gptoss_shard_override.
        if bias.shape[0] != B.shape[0]:
            raise ValueError(
                f"bias has {bias.shape[0]} experts but the weight has {B.shape[0]}; "
                "under EP both must be sliced to the same local expert range"
            )
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        bias,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        bias.stride(0) if bias is not None else 0,
        HAS_BIAS=bias is not None,  # type: ignore
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        **config,
    )


import functools


@functools.cache
def _get_moe_sum_cuda_module():
    from .utils import load_jit
    return load_jit(
        "moe_sum",
        cuda_files=["moe_sum.cu"],
        cuda_wrappers=[("launch", "MoeSumKernel::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def moe_sum_reduce_cuda(input: torch.Tensor, output: torch.Tensor) -> None:
    """CUDA fused split-K reduce for MoE: out[m, h] = sum_k input[m, k, h].

    Block-per-token, 256 thr/block, uint4 (8 bf16) vec loads + fp32 acc.
    Templated on TOPK ∈ {2, 4, 6, 8, 16} (compile-time unroll), generic
    fallback otherwise. Requires h % 8 == 0 and bf16/fp16 input.
    """
    assert input.is_contiguous() and output.is_contiguous()
    assert input.dim() == 3 and output.dim() == 2
    m, topk, h = input.shape
    assert output.shape == (m, h)
    assert input.dtype == output.dtype
    assert h % 8 == 0, f"hidden={h} must be divisible by 8 (uint4 vec)"
    if m == 0 or topk == 0 or h == 0:
        return
    _get_moe_sum_cuda_module().launch(output, input)


def fused_moe_kernel_mxfp4_triton(
    A: torch.Tensor,
    blocks: torch.Tensor,        # (E, N, K/2) uint8, packed MXFP4
    scales: torch.Tensor,        # (E, N, K/32) uint8, e8m0
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
    bias: torch.Tensor | None = None,
) -> None:
    """Launch `fused_moe_kernel_mxfp4`. Mirrors `fused_moe_kernel_triton`'s guards.

    Kept as a separate entry point rather than a flag on the BF16 wrapper because the operand
    *shapes* differ (packed bytes and a scale tensor instead of one weight), and a single wrapper
    would have to infer which it was handed.
    """
    import triton
    import triton.language as tl

    from .triton.fused_moe import fused_moe_kernel_mxfp4

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    K = blocks.shape[2] * 2                       # logical contracted size
    N = blocks.shape[1]
    BK = config["BLOCK_SIZE_K"]
    # A ragged K tail IS allowed, and must be: gpt-oss-120b has K=2880 (gate_up) and K=1440 (down),
    # and gfx942 forces BK=128 at BM=16, which divides neither. What actually has to hold is that no
    # 32-element MXFP4 group is ever split across the tail boundary -- i.e. K % 32 == 0. An earlier
    # version required BK | K instead, which refused every real gpt-oss shape.
    if K % 32 != 0:
        raise ValueError(
            f"packed MoE kernel needs K to be a multiple of 32 (MXFP4 group size), got K={K}; "
            f"otherwise a masked tail would split a group and duplicate its scale"
        )
    if BK % 32 != 0:
        raise ValueError(f"BLOCK_SIZE_K must be a multiple of 32 (MXFP4 group), got {BK}")
    # BK=64 with BM=16 does not lower on Triton 3.5.1 / gfx942 (doc 48 §1). Refuse rather than
    # emit "LLVM Translation failed ... unrealized_conversion_cast" from deep inside Triton.
    if config["BLOCK_SIZE_M"] <= 16 and BK < 128:
        raise ValueError(
            f"packed MoE kernel needs BLOCK_SIZE_K >= 128 when BLOCK_SIZE_M <= 16 "
            f"(got M={config['BLOCK_SIZE_M']}, K={BK}); BK=64 fails to lower on gfx942"
        )
    if C.dim() not in (2, 3):
        raise ValueError(f"C must be (M*top_k, N) or (M, top_k, N), got {tuple(C.shape)}")
    if C.shape[-1] != N:
        raise ValueError(f"C last dim {C.shape[-1]} must equal N={N}")
    if scales.shape[2] * 32 != K:
        raise ValueError(
            f"scales imply K={scales.shape[2] * 32} but blocks imply K={K}"
        )

    # Same refusal as the BF16 wrapper: there is no fp32 path, and falling through to fp16 once
    # computed the whole MoE in half precision silently (00_README.md §14).
    if compute_type == torch.bfloat16:
        dtype = tl.bfloat16
    elif compute_type == torch.float16:
        dtype = tl.float16
    else:
        raise ValueError(
            f"packed MoE kernel supports bfloat16/float16 activations, got {compute_type}."
        )
    if bias is not None:
        if bias.dim() != 2 or bias.shape[1] != N:
            raise ValueError(f"bias must be (num_experts, N={N}), got {tuple(bias.shape)}")
        if bias.shape[0] != blocks.shape[0]:
            raise ValueError(
                f"bias has {bias.shape[0]} experts but weights have {blocks.shape[0]}; under EP "
                "an unsharded bias satisfies the N check and then indexes the wrong expert"
            )

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fused_moe_kernel_mxfp4[grid](
        A, blocks, scales, C, topk_weights,
        bias if bias is not None else A,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, sorted_token_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1),
        blocks.stride(0), blocks.stride(1),
        scales.stride(0), scales.stride(1),
        # C is (M, top_k, N) from `fused_experts_impl` but (M*top_k, N) when the kernel is driven
        # directly by a probe. The kernel addresses rows flatly by `offs_token`, so it wants the
        # last two strides in both cases -- the BF16 wrapper hardcodes `C.stride(1), C.stride(2)`
        # for the same reason. Using stride(0)/stride(1) here read a 3-D C with the batch stride
        # and walked off the allocation (illegal access, not a wrong answer).
        C.stride(-2), C.stride(-1),
        bias.stride(0) if bias is not None else 0,
        even_Ks=(K % BK == 0),
        HAS_BIAS=bias is not None,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=dtype,
        **config,
    )
