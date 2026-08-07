import functools
from typing import Dict, Tuple

import torch
from minisgl.kernel import moe_align_block_size as kernel_moe_align_block_size
from minisgl.kernel import topk_softmax as kernel_topk_softmax
from minisgl.moe import BaseMoeBackend


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    kernel_topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)[:, None]
        valid = indices < num_token_non_padded
        topk_ids = torch.where(valid, topk_ids, torch.full_like(topk_ids, -1))
        topk_weights = torch.where(valid, topk_weights, torch.zeros_like(topk_weights))
    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    return kernel_moe_align_block_size(topk_ids, block_size, num_experts)


# Above this many tokens the grouped MoE is compute-bound and wants a much larger tile
# than the decode-shaped default. Measured crossover on gfx942 (M=1024 still prefers
# 64x64, M=2048 prefers the big tile); see dev_log/gpt_oss_120b/16_projection_vs_silicon.md
# §6 and gptoss_moe_tile_crossover.py.
_BIG_TILE_MIN_M = 2048


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None = None,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    elif M >= _BIG_TILE_MIN_M and dtype is None:
        # Prefill. The 64x64x32 tile above reaches only ~0.34 of the plain dense bf16
        # GEMM rate at prefill M, which accounted for the whole 2.7-2.9x gap between
        # measured and projected gpt-oss-120b TTFT (§5). The fix is tile AREA, not K
        # depth: sweeping MxNxKxwarps showed BLOCK_SIZE_K is nearly neutral (K=128 is
        # worse everywhere) while going to 128x256 with 8 waves is, over the current
        # default at M=8192 -- one full prefill chunk, since max_extend_tokens is 8192:
        #
        #   gpt-oss-120b TP4 (N=720,  K=2880, topk=4)  2.52 -> 1.74 ms   1.45x
        #   Qwen3-30B    TP4 (N=192,  K=2048, topk=8)  1.30 -> 0.95 ms   1.37x
        #   Qwen3-30B    TP1 (N=768,  K=2048, topk=8)  4.14 -> 2.55 ms   1.63x
        #
        # It is a win at all three shapes, so this is not conditioned on N -- but it is
        # not the per-shape optimum either: 128x128 is ~7% better at Qwen TP4 (a 256-wide
        # tile covers its 384-wide gate_up with 2 tiles, wasting a third of the lanes) and
        # 256x256 is ~13% better at Qwen TP1. Those margins sit at the run-to-run noise
        # floor observed here (~10%), which is why one entry is preferred to three.
        #
        # `dtype is None` keeps the FP8 path (fused_fp8.py, which passes
        # dtype="fp8_w8a8") on the old tile: it was not measured here. FP8 would
        # tolerate it -- that kernel clamps BLOCK_SIZE_K to <= min(block_shape) and 64
        # still divides the 128 quant group -- but an unmeasured perf change to another
        # path is not something to make silently.
        config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 2,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
    dtype: str | None = None,
    block_shape: Tuple[int, int] | None = None,
) -> Dict[str, int]:
    E, _, N = w2_shape
    config = get_default_config(M, E, N, w1_shape[2], top_k, dtype=dtype)
    return config


PackedWeight = tuple  # (blocks uint8 (E, N, K/2), scales uint8 (E, N, K/32))


def _is_packed(w) -> bool:
    """A packed MXFP4 weight arrives as a `(blocks, scales)` pair instead of one dense tensor."""
    return isinstance(w, tuple)


def _logical_shape(w) -> tuple[int, int, int]:
    """`(E, N_out, K_in)` for either a dense weight or a packed `(blocks, scales)` pair.

    Everything downstream -- the hidden-size assert, the config lookup, the intermediate cache
    widths -- is written against the *logical* shape, so it must not see K/2.
    """
    if _is_packed(w):
        blocks, scales = w
        if blocks.dim() != 3 or scales.dim() != 3:
            raise ValueError(
                f"packed MoE weight needs 3-D blocks and scales, got {blocks.dim()}-D / "
                f"{scales.dim()}-D"
            )
        E, N, k_bytes = blocks.shape
        K = k_bytes * 2
        if tuple(scales.shape[:2]) != (E, N):
            raise ValueError(
                f"packed scales {tuple(scales.shape)} do not agree with blocks "
                f"{tuple(blocks.shape)} on (experts, out_features)"
            )
        if scales.shape[2] * 32 != K:
            raise ValueError(
                f"packed scales imply K={scales.shape[2] * 32} but blocks imply K={K}"
            )
        return E, N, K
    return tuple(w.shape)  # type: ignore[return-value]


def _assert_contiguous(w, name: str) -> None:
    if _is_packed(w):
        blocks, scales = w
        assert blocks.is_contiguous(), f"{name} blocks must be contiguous"
        assert scales.is_contiguous(), f"{name} scales must be contiguous"
    else:
        assert w.is_contiguous(), f"{name} must be contiguous"


# Packed-MXFP4 tile config, keyed by the number of ROWS the grouped GEMM sees (tokens * top_k).
# Measured by `dev_log/probes/moe_packed_tile_sweep.py` and cross-checked by an independent pass
# over four finalists; see doc 52. `tl.dot_scaled` on packed weights wants a wider N tile than the
# BF16 table's BN=32, and that single change is most of the win:
#
#   speedup vs today's BF16 MoE, this entry, tokens = 1 / 8 / 32 / 64 / 128
#     gate_up  1.00x / 1.69x / 1.79x / 1.63x / 1.79x
#     down     0.78x / 1.45x / 1.76x / 1.86x / 1.91x
#
# Only decode-ish row counts are covered. Prefill (rows in the tens of thousands) is deliberately
# left on the BF16 table's big tile: it was not measured for the packed kernel, and doc 51's
# regression came precisely from shipping an unmeasured config.
#
# Prefill (rows in the thousands) has its OWN entry, added after measuring it rather than assuming
# the BF16 big tile would do. On the BF16 tile with BK forced to 128 the packed kernel runs at only
# 0.50-0.56x of BF16 at prefill -- a ~1.8x slowdown that the decode-only tuning hid, and that shows
# up as worse TTFT. The culprit is `num_warps=8`: at 4 warps the same tile reaches 0.83-0.99x.
#
# Note what that still says: **even tuned, packed prefill does not reach BF16.** Prefill is
# compute-dense, so the in-kernel e2m1 upcast is no longer free the way it is when decode is
# bandwidth- and launch-bound. This entry recovers most of the loss; it does not erase it.
_PACKED_TILE_TABLE: list[tuple[int, dict]] = [
    (
        1024,
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        },
    ),
    (
        1 << 30,
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        },
    ),
]


def _packed_config(config: dict, n_rows: int) -> dict:
    """The packed kernel's own tile config and hard constraints.

    `BLOCK_SIZE_K` must be a multiple of 32 and **>= 128, unconditionally**. Doc 48 §1 observed
    `BK=64` failing to lower at `BM=16` and I briefly inferred the rule was conditional on BM --
    it is not. With `BK=64` the prefill tile (`BM=128, BN=256`) fails identically, at the same
    `builtin.unrealized_conversion_cast` on the packed load. `tl.dot_scaled` simply will not lower
    `BK=64` on this Triton/gfx942, whatever BM is. The BF16 table emits BK=32/64, so this is
    enforced here rather than hoped for.

    `n_rows` is `tokens * top_k`, which is what the grouped GEMM's M actually is -- not the token
    count. The returned dict is used for BOTH the aligner and the kernel (see the call site), so a
    `BLOCK_SIZE_M` chosen here cannot desynchronise them.
    """
    out = dict(config)
    for max_rows, override in _PACKED_TILE_TABLE:
        if n_rows <= max_rows:
            out.update(override)
            break
    bk = out.get("BLOCK_SIZE_K", 128)
    out["BLOCK_SIZE_K"] = max(128, bk - bk % 32)
    return out


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    global_num_experts: int | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from minisgl.kernel import fused_moe_kernel_triton, moe_sum_reduce_cuda
    from minisgl.kernel.moe_impl import fused_moe_kernel_mxfp4_triton
    from minisgl.layers import gelu_and_mul, glm4_silu_and_mul, gptoss_swiglu, silu_and_mul

    padded_size = 0
    # Both GEMMs are dispatched independently, so a half-converted model (packed gate_up, dense
    # down) would run and produce plausible-looking garbage. Resolve each separately and let the
    # per-GEMM call sites decide, but validate the shapes through one logical path.
    w1_packed, w2_packed = _is_packed(w1), _is_packed(w2)
    w1_shape, w2_shape = _logical_shape(w1), _logical_shape(w2)
    assert hidden_states.shape[1] == w1_shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    _assert_contiguous(w1, "Expert weights1")
    _assert_contiguous(w2, "Expert weights2")
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    if expert_map is not None and global_num_experts is None:
        raise RuntimeError("global_num_experts is required when expert_map is provided")
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1_shape
    M = num_tokens
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1_shape,
        (w2_shape[0], w2_shape[1], w2_shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2_shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if expert_map is not None:
        # EP: remote experts are skipped by the kernel and would otherwise leave
        # stale cache rows that are later summed.
        cache.zero_()
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2_shape[1]].view(
        (M, topk_ids.shape[1], w2_shape[1]),
    )
    compute_type = hidden_states.dtype

    out_hidden_states = hidden_states
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    begin_token_idx, end_token_idx = 0, num_tokens

    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = get_config_func(tokens_num)

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    # Resolve the packed config BEFORE the aligner, not at the call site. `moe_align_block_size`
    # blocks tokens by `BLOCK_SIZE_M` and the kernel indexes those blocks with its own
    # `BLOCK_SIZE_M`; if the two ever disagree the kernel reads the wrong rows and returns wrong
    # numbers with no error. Applying the override here keeps one config for both by construction,
    # so tuning BLOCK_SIZE_M cannot reintroduce the mismatch.
    if w1_packed or w2_packed:
        config = _packed_config(config, tokens_num * topk_ids.shape[1])

    num_experts_for_align = global_num_experts if expert_map is not None else E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], num_experts_for_align
    )
    if expert_map is not None:
        valid = (expert_ids >= 0) & (expert_ids < global_num_experts)
        safe_ids = torch.where(valid, expert_ids, torch.zeros_like(expert_ids))
        expert_ids = torch.where(
            valid,
            expert_map[safe_ids.to(torch.long)],
            torch.full_like(expert_ids, -1),
        )

    def grouped_gemm(w, packed, a, out, mul_routed_weight, top_k, bias):
        """One grouped GEMM, dense or packed. Both call sites go through here so a packed w1 with a
        dense w2 cannot silently take two different code paths."""
        if packed:
            blocks, scales = w
            fused_moe_kernel_mxfp4_triton(
                a, blocks, scales, out,
                curr_topk_weights, curr_topk_ids,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                mul_routed_weight, top_k, config,
                compute_type=compute_type, bias=bias,
            )
        else:
            fused_moe_kernel_triton(
                a, w, out,
                curr_topk_weights, curr_topk_ids,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                mul_routed_weight, top_k, config,
                compute_type=compute_type, bias=bias,
            )

    grouped_gemm(
        w1, w1_packed, curr_hidden_states, intermediate_cache1,
        apply_router_weight_on_input, topk_ids.shape[1], w1_bias,
    )
    FN_MAP = {
        "silu": silu_and_mul,
        "gelu": gelu_and_mul,
        "glm4_silu": glm4_silu_and_mul,
        # gpt-oss: INTERLEAVED gate/up, clamped, alpha-scaled, (up+1)*glu. Consumes the
        # same (.., 2*I) buffer and writes (.., I), so it drops in here -- but it is a
        # different function from silu_and_mul, not a variant. See layers/_triton_ops.py.
        "gptoss_swiglu": gptoss_swiglu,
    }
    if activation not in FN_MAP:
        raise ValueError(f"unsupported MoE activation {activation!r}; have {sorted(FN_MAP)}")
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    if expert_map is not None:
        # intermediate_cache1 and intermediate_cache3 alias the same cache with
        # different row widths. Re-zero before down-proj so skipped remote rows
        # stay zero.
        intermediate_cache3.zero_()
    grouped_gemm(
        w2, w2_packed, intermediate_cache2, intermediate_cache3,
        not apply_router_weight_on_input, 1, w2_bias,
    )

    moe_sum_reduce_cuda(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        expert_map: torch.Tensor | None = None,
        global_num_experts: int | None = None,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )
