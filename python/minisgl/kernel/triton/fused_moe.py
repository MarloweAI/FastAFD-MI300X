import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    bias_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias_e,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts < 0:
        return
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.

        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        # Per-expert bias over the N (output) dim. `offs_bn` is already masked by
        # `% N`, so out-of-range lanes alias a valid column and are dropped by
        # `c_mask` at the store, exactly as the B operand relies on.
        bias = tl.load(bias_ptr + off_experts * stride_bias_e + offs_bn)
        accumulator += bias[None, :].to(tl.float32)

    # Order matters: HF computes (y @ W + b) * routing_weight, so the bias goes in
    # BEFORE the routed-weight multiply. Swapping these silently rescales the bias.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_mxfp4(
    # Pointers to matrices
    a_ptr,
    blk_ptr,                 # packed MXFP4 values, per expert (N, K/2) uint8
    scl_ptr,                 # e8m0 scales,        per expert (N, K/32) uint8
    c_ptr,
    topk_weights_ptr,
    bias_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions -- K is the LOGICAL contracted size, not the byte count
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_blk_e,
    stride_blk_n,            # bytes per output row = K/2
    stride_scl_e,
    stride_scl_n,            # scales per output row = K/32
    stride_cm,
    stride_cn,
    stride_bias_e,
    even_Ks: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """`fused_moe_kernel` with the expert weights left PACKED in MXFP4.

    Same contract as the BF16 kernel -- same sorted_token_ids / expert_ids blocking, same bias and
    routed-weight order -- with `tl.dot` replaced by `tl.dot_scaled`, so the 4-bit values are upcast
    inside the MFMA pipeline and never materialise as BF16 in HBM. That is what keeps the weights at
    ~16 GB/card instead of ~58 (doc 47), which is the point: KV capacity, not GEMM speed.

    Three layout facts, each of which fails silently if broken (doc 48 §1 took three attempts):

      * `rhs` must present the PACKED K axis FIRST -- `(BLOCK_K/2, BLOCK_N)`. A transposed rhs fails
        to compile with "Reduction dimension should pack the same number of elements", which is the
        friendly case; getting the strides wrong instead compiles and computes nonsense.
      * `rhs_scale` must put the scaled dimension LAST -- `(BLOCK_N, BLOCK_K/32)`.
      * **`BLOCK_SIZE_K` must be >= 128 when `BLOCK_SIZE_M` is 16.** `BK=64` with `BM=16` does not
        lower on Triton 3.5.1 / gfx942 (`LLVM Translation failed ... unrealized_conversion_cast`).

    `even_Ks` masks a ragged K tail. An earlier version of this kernel refused one, on the reasoning
    that "a partial group would need its scale duplicated and there is no correct masked form of
    that". That was too strong, and it made the kernel **unusable on gpt-oss-120b**, whose K is 2880
    (gate_up) and 1440 (down) -- neither divisible by the BLOCK_SIZE_K=128 that gfx942 forces at
    BM=16. The real requirement is weaker: MXFP4 groups are 32 wide, so as long as **K is a multiple
    of 32** every tail contains a whole number of groups and no group is ever split. The tail is then
    masked at group granularity (`k_rem // 32` scales, `k_rem // 2` value bytes) and zero-filled;
    e2m1 zeros contribute nothing to the dot regardless of what scale byte pairs with them.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts < 0:
        return

    # Packed operands. Rows of `blk`/`scl` are output channels (N); the contracted dim runs along
    # the row, so the K axis is the CONTIGUOUS one and the transpose into "K first" is free of a
    # separate permute -- it is just which index goes on which side of the load.
    offs_kb = tl.arange(0, BLOCK_SIZE_K // 2)
    offs_ks = tl.arange(0, BLOCK_SIZE_K // 32)
    blk_ptrs = (blk_ptr + off_experts * stride_blk_e
                + offs_bn[None, :] * stride_blk_n + offs_kb[:, None])
    scl_ptrs = (scl_ptr + off_experts * stride_scl_e
                + offs_bn[:, None] * stride_scl_n + offs_ks[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(blk_ptrs)
            s = tl.load(scl_ptrs)
        else:
            # k_rem is a multiple of 32 because the wrapper requires K % 32 == 0, so the tail is a
            # whole number of MXFP4 groups and `k_rem // 2` / `k_rem // 32` are both exact.
            k_rem = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
            b = tl.load(blk_ptrs, mask=offs_kb[:, None] < (k_rem // 2), other=0)
            s = tl.load(scl_ptrs, mask=offs_ks[None, :] < (k_rem // 32), other=0)
        accumulator = tl.dot_scaled(a, None, "bf16", b, s, "e2m1", acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        blk_ptrs += BLOCK_SIZE_K // 2
        scl_ptrs += BLOCK_SIZE_K // 32

    if HAS_BIAS:
        bias = tl.load(bias_ptr + off_experts * stride_bias_e + offs_bn)
        accumulator += bias[None, :].to(tl.float32)
    # Same order as the BF16 kernel: bias BEFORE the routed weight, or the bias is rescaled.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=token_mask[:, None] & (offs_cn[None, :] < N))
