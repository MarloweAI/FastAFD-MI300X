from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info, try_get_ep_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight, init_logger
from tqdm import tqdm

from .fp8_utils import dequant_fp8_block

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj", ".w1", ".w3"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]
_FP8_SCALE_SUFFIX = ".weight_scale_inv"
_FP8_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale")
_SCALE_RENAME_SUFFIX = "_scale"

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
    ".w1": (".gate_up_proj", ("gate", "up")),
    ".w3": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
    ".w1": "gate",
    ".w3": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_GLM4_LAYER_PATTERN = re.compile(r"^model\.layers\.(?P<idx>\d+)\.")
_RENAME_SUBSTRINGS = {
    ".block_sparse_moe.": ".mlp.",
    ".w2": ".down_proj",
}

logger = init_logger(__name__)


def _config_field(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _is_glm4_moe_config(config: Any) -> bool:
    archs = _config_field(config, "architectures", default=[]) or []
    return "Glm4MoeForCausalLM" in archs or _config_field(
        config, "model_type", default=""
    ) == "glm4_moe"


def _glm4_layer_index_for_key(key: str) -> int | None:
    match = _GLM4_LAYER_PATTERN.match(key.removeprefix("language_model."))
    return int(match.group("idx")) if match is not None else None


def _glm4_is_expected_nextn_key(key: str, config: Any) -> bool:
    layer_idx = _glm4_layer_index_for_key(key)
    if layer_idx is None:
        return False
    num_layers = int(_config_field(config, "num_layers", "num_hidden_layers", default=0) or 0)
    num_nextn = int(_config_field(config, "num_nextn_predict_layers", default=0) or 0)
    return num_nextn > 0 and num_layers <= layer_idx < num_layers + num_nextn


def _scale_suffix_for_name(name: str) -> str | None:
    for suffix in _FP8_SCALE_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def _shard_tensor(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    num_kv_heads: int,
    *,
    skip_tp_shard: bool = False,
    fp8_block_size: Optional[Tuple[int, int]] = None,
):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if skip_tp_shard:
        return value
    if fp8_block_size is not None and (
        key.endswith(".weight_scale") or key.endswith(_SCALE_RENAME_SUFFIX)
    ):
        return _shard_fp8_scale_tensor(key, value, r, n, num_kv_heads, fp8_block_size)
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        if key.endswith(".bias"):
            # A row-parallel bias (o_proj, down_proj) is full width on every rank and is
            # added AFTER the all-reduce (layers/linear.py `_defer_bias`), so it must not
            # be sharded. chunk(dim=1) on a 1-D bias would raise anyway; returning it
            # unsharded is the correct behaviour, not just the non-crashing one.
            return value
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _slice_fp8_scale_dim(
    value: torch.Tensor,
    *,
    dim: int,
    r: int,
    n: int,
    block_size: int,
) -> torch.Tensor:
    padded_weight_dim = value.shape[dim] * int(block_size)
    weight_start = r * padded_weight_dim // n
    weight_end = (r + 1) * padded_weight_dim // n
    scale_start = weight_start // int(block_size)
    scale_end = div_ceil(weight_end, int(block_size))
    index = [slice(None)] * value.dim()
    index[dim] = slice(scale_start, scale_end)
    return value[tuple(index)].clone()


def _shard_fp8_scale_tensor(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    num_kv_heads: int,
    block_size: Tuple[int, int],
) -> torch.Tensor:
    block_out, block_in = block_size
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            blocks_per_head = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[
                head_idx * blocks_per_head : (head_idx + 1) * blocks_per_head
            ].clone()
        return _slice_fp8_scale_dim(value, dim=0, r=r, n=n, block_size=block_out)
    if any(key.count(sub) for sub in _SPLIT_DIM_1):
        return _slice_fp8_scale_dim(value, dim=1, r=r, n=n, block_size=block_in)
    return value


def _shard_fp8_channel_scale_tensor(
    key: str,
    value: torch.Tensor,
    r: int,
    n: int,
    num_kv_heads: int,
    *,
    skip_tp_shard: bool = False,
) -> torch.Tensor:
    if skip_tp_shard:
        return value
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            rows_per_head = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[
                head_idx * rows_per_head : (head_idx + 1) * rows_per_head
            ].clone()
        return value.chunk(n, dim=0)[r].clone()
    # Per-channel FP8 scales are per output row. Row-parallel weights shard
    # input columns, so their row scales are replicated.
    return value


def _find_paired_scale_key(
    *,
    normalized_weight_name: str,
    raw_weight_name: str,
    shard_keys: set[str],
) -> str | None:
    if not normalized_weight_name.endswith(".weight"):
        return None
    normalized_base = normalized_weight_name.removesuffix(".weight")
    raw_base = raw_weight_name.removesuffix(".weight")
    for suffix in _FP8_SCALE_SUFFIXES:
        raw_candidate = raw_base + suffix
        if raw_candidate in shard_keys:
            return raw_candidate
        normalized_candidate = normalized_base + suffix
        if normalized_candidate in shard_keys:
            return normalized_candidate
    return None


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def _rename_checkpoint_key(name: str) -> str:
    for old, new in _RENAME_SUBSTRINGS.items():
        if old in name:
            name = name.replace(old, new)
    return name


@dataclass(frozen=True)
class _ExpertPartition:
    start_idx: int
    end_idx: int
    capacity: int


def _get_local_expert_partition(num_experts: int) -> _ExpertPartition | None:
    """Return locally-owned global expert range when full EP is enabled."""
    ep_info = try_get_ep_info()
    if ep_info is None or ep_info.size == 1:
        return None
    local_num_experts = (int(num_experts) + int(ep_info.size) - 1) // int(ep_info.size)
    start = int(ep_info.rank) * local_num_experts
    end = min(int(num_experts), start + local_num_experts)
    return _ExpertPartition(start_idx=start, end_idx=end, capacity=local_num_experts)


def _pad_experts_to_capacity(t: torch.Tensor, part: _ExpertPartition) -> torch.Tensor:
    """Pad a sliced expert-dim tensor up to `part.capacity` rows.

    When num_experts does not divide the EP size the last rank owns FEWER experts than the
    others (128 over 3 ranks -> 43/43/42), but MoELayer allocates div_ceil == capacity rows
    on EVERY rank so the collectives stay uniform. A slice of the real range is therefore
    short on that rank, and the expert worker asserts on the shape mismatch
    (afd_expert_worker.py: `assert param.shape == item.shape`) -- which surfaces as a
    startup HANG, because the exception is raised inside a Ray actor and never propagates.

    gpt-oss is the model that hits this: its expert tensors arrive pre-stacked over the
    expert dim, so they take a single slice here. Per-expert checkpoints (`experts.{i}.`)
    never do -- each expert is written into its own slot and the tail slot just stays
    whatever the buffer was allocated with.

    The pad rows are unreachable: _build_expert_map only maps the real global range, so no
    routed token ever indexes them. Zeros rather than empty anyway -- a NaN parked in an
    unreachable slot is the kind of thing that stops being unreachable after a refactor.
    """
    n = t.shape[0]
    if n == part.capacity:
        return t
    if n > part.capacity:
        raise ValueError(
            f"expert slice has {n} rows, more than capacity {part.capacity}"
        )
    pad = torch.zeros((part.capacity - n, *t.shape[1:]), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0)


def _is_gptoss_config(config: Any) -> bool:
    archs = _config_field(config, "architectures", default=[]) or []
    return "GptOssForCausalLM" in archs or _config_field(
        config, "model_type", default=""
    ) == "gpt_oss"


def _mxfp4_packed_enabled() -> bool:
    """Whether to keep MXFP4 expert weights packed instead of dequantising at load.

    Off by default. When on, `MoELayer` must also be on the packed path -- both read the same
    env var, so they cannot disagree, but a stale allocation would surface as a shape assert in
    `load_state_dict` rather than as wrong numbers.
    """
    return os.environ.get("MINISGL_MXFP4_PACKED", "0") not in ("", "0", "false", "False")


def mxfp4_group_split(num_groups: int, n: int) -> list[int]:
    """Split `num_groups` MXFP4 groups over `n` ranks, **whole groups only**, largest-first.

    Row-parallel sharding does not require equal `K` per rank -- each rank computes a partial sum
    over its own slice of the contracted dim and the all-reduce adds them -- so an *uneven* split is
    legal, and an uneven split on group boundaries avoids ever cutting a scale group.

    gpt-oss needs this: `I = 2880` is 90 groups, and `90 = 2 x 45` has only one factor of 2, so an
    even element split is group-aligned at tp 2 but not at tp 4 (22.5) or tp 8 (11.25).

        >>> mxfp4_group_split(90, 4)
        [23, 23, 22, 22]                  # -> I per rank 736, 736, 704, 704

    The corresponding `gate_up_proj` split is `2 * 32 * groups` per rank, which is always even, so
    interleaved (gate_i, up_i) pairs are never torn.
    """
    if n <= 0 or num_groups < n:
        raise ValueError(f"cannot split {num_groups} MXFP4 groups over {n} ranks")
    base, rem = divmod(num_groups, n)
    return [base + (1 if r < rem else 0) for r in range(n)]


def _group_slice(split: list[int], r: int) -> tuple[int, int]:
    start = sum(split[:r])
    return start, start + split[r]


def _pad_groups(t: torch.Tensor, target: int, *, dim: int, fill: int) -> torch.Tensor:
    """Pad `t` along `dim` up to `target` with `fill`, so every rank holds the same shape.

    `fill` differs by tensor on purpose. Packed VALUES pad with 0: e2m1 zero contributes nothing to
    the dot. SCALES pad with 127 (e8m0 for 2^0) and **never 0xFF**, which is the e8m0 NaN encoding --
    `0 * NaN` is NaN, so a NaN pad scale would poison the accumulator even though the value is zero.
    """
    have = t.shape[dim]
    if have == target:
        return t
    if have > target:
        raise ValueError(f"slice of {have} exceeds uniform target {target} on dim {dim}")
    shape = list(t.shape)
    shape[dim] = target - have
    pad = torch.full(shape, fill, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def mxfp4_uniform_group_split(num_groups: int, n: int) -> tuple[int, list[tuple[int, int]]]:
    """Split `num_groups` MXFP4 groups over `n` ranks so **every rank holds the same count**,
    zero-padding the ranks that run out of real groups.

    Returns `(groups_per_rank, [(start, real_count), ...])`.

        >>> mxfp4_uniform_group_split(90, 4)
        (23, [(0, 23), (23, 23), (46, 23), (69, 21)])
        >>> mxfp4_uniform_group_split(90, 2)          # exact -- no padding at all
        (45, [(0, 45), (45, 45)])

    **Why uniform rather than uneven.** `mxfp4_group_split` gives a legal *uneven* partition
    (90 -> 23,23,22,22 -> I per rank 736/736/704/704), and row-parallel sharding tolerates unequal
    `K`. But unequal per-rank shapes are a liability everywhere else in this stack: CUDA graph
    capture, static buffer sizing and the AFD M2N transport all assume rank symmetry, and a shape
    that differs by rank is exactly the kind of asymmetry that fails silently rather than loudly.

    **It costs nothing in time.** The MoE is followed by an all-reduce, so the step waits for the
    slowest rank either way: with the uneven split that rank does 736, and with padding every rank
    does 736. Same critical path, uniform shapes. The cost is memory -- `n * groups_per_rank` group
    slots instead of `num_groups`, i.e. 2/90 = 2.2% at tp 4 and 6/90 = 6.7% at tp 8.

    **Padding must be numerically inert**, and it is, twice over. `gate_up_proj`'s padded output
    channels carry zero weights and zero bias, so their activation is `swiglu(0, 0) = 0`; and
    `down_proj`'s padded contracted rows carry zero weights, so they contribute nothing regardless.
    Padded *scales* are set to 127 (2^0) rather than 0xFF, because 0xFF is the e8m0 NaN encoding and
    `0 * NaN` is NaN, not 0.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    per = div_ceil(num_groups, n)
    out: list[tuple[int, int]] = []
    for r in range(n):
        start = min(r * per, num_groups)
        out.append((start, max(0, min(per, num_groups - start))))
    return per, out


def _gptoss_expert_weight_packed(
    name: str, blocks: torch.Tensor, scales: torch.Tensor, r: int, n: int
) -> Tuple[str, Tuple[torch.Tensor, torch.Tensor]]:
    """MXFP4 expert weight -> TP/EP-sharded, still PACKED (blocks, scales).

    Why: the InferenceX reference keeps MXFP4 resident on gfx942 — vLLM logs
    `Model loading took 17.83 GiB` per rank at TP4, ~16 GB of 4-bit weights where a BF16
    expansion would be ~58 (doc 47 §3). We dequantise at load, which costs ~3.3x model memory
    and therefore KV capacity (112.41 GiB against their 160.15), and that difference makes our
    numbers not directly comparable to theirs. This is the loader half of closing that gap; the
    consumer half is a `dot_scaled` GEMM (doc 48 §6 step 3).

    Layout, from `dequant_mxfp4`: `blocks` is `(E, out_features, G, B)` with `G = K/32`, `B = 16`,
    and the dequantised result is `(E, K, out_features)`. So:

      * `gate_up_proj`  out_features = 2I = 5760, K = H = 2880
      * `down_proj`     out_features = H  = 2880, K = I = 2880

    **The sharding axes differ, and only one of them is packable in general.** `gate_up_proj` is
    column-parallel: it shards `out_features`, which is `blocks` axis 1 — no constraint beyond
    keeping gate/up pairs together. `down_proj` is row-parallel: it shards the CONTRACTED dim,
    which lives inside `(G, B)`, so a shard must cut on a **32-element MXFP4 group boundary**.
    `I = 2880` gives 90 groups, and `90 = 2 x 45` has only one factor of 2, so an *even element* split
    is group-aligned at tp 2 but not at tp 4 (22.5) or tp 8 (11.25). Rather than refuse those sizes,
    both weights are sharded by `mxfp4_group_split`, an **uneven whole-group** split (90 ->
    23,23,22,22) applied consistently to gate_up and down. That is legal because row-parallel
    sharding does not require equal `K` per rank -- each rank sums over its own slice and the
    all-reduce combines them.

    **This means the packed partition differs from the BF16 partition at tp 4 and 8** (736/736/704/704
    against 720 each). They are alternative valid partitions, not interchangeable per layer, so a
    model must use one loader throughout. It also requires the MoE layer to tolerate an unequal
    per-rank intermediate size, which is the next thing to wire (doc 48 step 3).

    The EP path has no such problem: it shards the expert dim only.
    """
    part = _get_local_expert_partition(blocks.shape[0])
    if part is not None:
        # EP: expert-dim shard, intermediate stays FULL, so packing is unaffected.
        if not name.endswith((".gate_up_proj", ".down_proj")):
            raise ValueError(f"unexpected MXFP4 expert tensor: {name}")
        b = _pad_experts_to_capacity(blocks[part.start_idx : part.end_idx], part)
        s = _pad_experts_to_capacity(scales[part.start_idx : part.end_idx], part)
        return name, (b.contiguous(), s.contiguous())

    if n == 1:
        return name, (blocks.contiguous(), scales.contiguous())

    # Both weights are sharded by the SAME group-aligned split of the intermediate `I`, because
    # `gate_up_proj`'s output IS `down_proj`'s input: the per-rank intermediate size must agree.
    # The split is uneven when `I/32` does not divide by `n` (gpt-oss at tp 4 and 8), which is
    # legal for row-parallel and is what keeps every scale group whole.
    if name.endswith(".gate_up_proj"):
        # out_features = 2I. Groups here run along K = H and are NOT the sharded axis, so the
        # split is applied in units of `2 * 32` output elements per group of intermediate.
        n_groups = blocks.shape[1] // 64            # 2I / (2*32)
        per, parts = mxfp4_uniform_group_split(n_groups, n)
        g0, real = parts[r]
        b = _pad_groups(blocks[:, g0 * 64 : (g0 + real) * 64], per * 64, dim=1, fill=0)
        s = _pad_groups(scales[:, g0 * 64 : (g0 + real) * 64], per * 64, dim=1, fill=127)
    elif name.endswith(".down_proj"):
        # The contracted dim IS the group axis: slice G, never elements.
        per, parts = mxfp4_uniform_group_split(blocks.shape[2], n)
        g0, real = parts[r]
        b = _pad_groups(blocks[:, :, g0 : g0 + real], per, dim=2, fill=0)
        s = _pad_groups(scales[:, :, g0 : g0 + real], per, dim=2, fill=127)
    else:
        raise ValueError(f"unexpected MXFP4 expert tensor: {name}")
    return name, (b.contiguous(), s.contiguous())


def _gptoss_expert_weight(
    name: str, blocks: torch.Tensor, scales: torch.Tensor, r: int, n: int
) -> Tuple[str, torch.Tensor]:
    """MXFP4 expert weight -> TP-sharded BF16 in `F.linear` layout.

    `dequant_mxfp4` reproduces HF's layout: `gate_up_proj` is `(E, H, 2I)` and `down_proj`
    is `(E, I, H)`, both intended for `x @ W`. This port uses `F.linear` (`x @ W.T`), so
    each is transposed on the way in. gpt-oss's hidden and intermediate are both 2880, so
    a transpose mix-up would NOT show up as a shape error -- it is pinned numerically in
    dev_log/gpt_oss_120b/gptoss_moe_parity.py.

    Sharding happens BEFORE the transpose+contiguous, so the peak is one dequantised
    tensor plus one shard rather than two full tensors (`gate_up_proj` alone is 4.25 GB
    per layer in BF16).
    """
    from minisgl.kernel.mxfp4 import dequant_mxfp4

    deq = dequant_mxfp4(blocks, scales, dtype=torch.bfloat16)

    # EP and TP shard DIFFERENT axes, and gpt-oss reaches here on a path that used to know
    # only about TP. Under EP, MoELayer shards the expert dim and keeps the intermediate
    # FULL (layer.py: intermediate_size_per_partition = intermediate_size when ep_size > 1),
    # so slicing the intermediate as well would hand the rank a tensor of the wrong shape on
    # both axes. Qwen never hit this because its checkpoint stores `experts.{i}.` separately
    # and goes through the pre-merge machinery; gpt-oss's experts arrive pre-stacked with an
    # expert dim and bypass it. See dev_log/gpt_oss_120b/25_afd_gpt_oss_status.md §3.
    part = _get_local_expert_partition(deq.shape[0])
    if part is not None:
        deq = _pad_experts_to_capacity(deq[part.start_idx : part.end_idx], part)
        if not name.endswith((".gate_up_proj", ".down_proj")):
            raise ValueError(f"unexpected MXFP4 expert tensor: {name}")
        return name, deq.transpose(1, 2).contiguous()

    if name.endswith(".gate_up_proj"):
        # (E, H, 2I) -- shard the interleaved 2I output dim. A contiguous slice keeps each
        # (gate_i, up_i) pair on one rank; see models/gpt_oss.py. Splitting a pair would
        # pair a gate with the wrong up, silently.
        per = deq.shape[2] // n
        if per % 2 != 0:
            raise ValueError(
                f"{name}: 2*intermediate={deq.shape[2]} does not split into even "
                f"chunks over tp_size={n}; a gate/up pair would be torn apart"
            )
        deq = deq[:, :, r * per : (r + 1) * per]
    elif name.endswith(".down_proj"):
        # (E, I, H) -- shard the contracted I dim.
        per = deq.shape[1] // n
        deq = deq[:, r * per : (r + 1) * per, :]
    else:
        raise ValueError(f"unexpected MXFP4 expert tensor: {name}")
    return name, deq.transpose(1, 2).contiguous()


def _gptoss_shard_override(
    name: str, value: torch.Tensor, r: int, n: int
) -> torch.Tensor | None:
    """Sharding for gpt-oss tensors the generic rules would get wrong. None = not ours."""
    # Expert biases follow the expert weights: under EP the expert dim is sharded and the
    # intermediate stays full. Checked before the n == 1 early return, because EP can be
    # enabled while dense tp_size is 1 (AFD 1A+3F runs MLP_EP=1, but 2A+2F runs MLP_EP=2).
    if name.endswith((".experts.gate_up_proj_bias", ".experts.down_proj_bias")):
        part = _get_local_expert_partition(value.shape[0])
        if part is not None:
            return _pad_experts_to_capacity(
                value[part.start_idx : part.end_idx], part
            ).clone()
    if n == 1:
        return value if _is_gptoss_special_key(name) else None
    if name.endswith(".experts.gate_up_proj_bias"):
        # Column-parallel bias over the interleaved 2I dim. It MUST use the same partition as
        # `gate_up_proj` itself: the bias is indexed by output channel, so a different split pairs
        # each channel with another channel's bias. Under the packed loader that partition is the
        # uniform whole-group one (padded), not the even element split -- at tp 4 they disagree
        # (1472 against 1440), which would be a silent numeric error wherever the shapes happened
        # to line up. Padding is zero so the padded channels contribute nothing.
        if _mxfp4_packed_enabled():
            n_groups = value.shape[1] // 64
            per, parts = mxfp4_uniform_group_split(n_groups, n)
            g0, real = parts[r]
            sliced = value[:, g0 * 64 : (g0 + real) * 64]
            return _pad_groups(sliced, per * 64, dim=1, fill=0).contiguous()
        per = value.shape[1] // n
        if per % 2 != 0:
            raise ValueError(f"{name}: 2*intermediate does not split evenly over {n}")
        return value[:, r * per : (r + 1) * per].clone()
    if name.endswith(".experts.down_proj_bias"):
        # Row-parallel: full width on every rank, added after the reduce.
        return value
    if name.endswith(".self_attn.sinks"):
        # One logit per QUERY head, so it shards exactly like the q heads.
        return value.chunk(n, dim=0)[r].clone()
    return None


def _is_gptoss_special_key(name: str) -> bool:
    return name.endswith(
        (".experts.gate_up_proj_bias", ".experts.down_proj_bias", ".self_attn.sinks")
    )


def _is_expert_key(name: str) -> bool:
    return _EXPERT_PATTERN.match(name) is not None


def _is_remote_shared_expert_key(name: str, config: Any) -> bool:
    if not _is_glm4_moe_config(config):
        return False
    if int(_config_field(config, "n_shared_experts", default=0) or 0) <= 0:
        return False
    if os.environ.get("MINISGL_AFD_MOE_BACKEND") != "megamoe_m2n":
        return False
    return ".shared_experts." in name


def load_weight(
    model_path: str,
    device: torch.device,
    *,
    skip_expert_weights: bool = False,
    skip_non_expert_weights: bool = False,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer.

    afd AG/EG role split: skip_expert_weights loads only the dense/attention/
    router/embed/lm_head weights (AG side); skip_non_expert_weights loads only the
    MoE expert weights (EG side)."""
    from .config import ModelConfig

    if skip_expert_weights and skip_non_expert_weights:
        raise ValueError("Cannot skip both expert and non-expert weights")

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()
    local_expert_partition = (
        _get_local_expert_partition(config.num_experts) if config.is_moe else None
    )
    local_num_experts = (
        local_expert_partition.capacity
        if local_expert_partition is not None
        else config.num_experts
    )
    real_local_num_experts = (
        local_expert_partition.end_idx - local_expert_partition.start_idx
        if local_expert_partition is not None
        else local_num_experts
    )
    fp8_block_size: Optional[Tuple[int, int]] = (
        config.quant.weight_block_size
        if config.quant is not None and config.quant.method == "fp8"
        else None
    )
    fp8_channel_scale = config.quant is not None and config.quant.method == "fp8_channel"
    is_gptoss = _is_gptoss_config(cached_load_hf_config(model_path))
    logger.info(
        f"Loading weights from {len(files)} safetensors files"
        + (f" (FP8 block={fp8_block_size})" if fp8_block_size else "")
        + (" (FP8 channel)" if fp8_channel_scale else "")
    )

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            shard_keys = set(f.keys())
            for raw_name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if raw_name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                name = raw_name.removeprefix("language_model.")
                if _is_glm4_moe_config(config) and _glm4_is_expected_nextn_key(name, config):
                    continue

                scale_suffix = _scale_suffix_for_name(name)
                is_scale_tensor = scale_suffix is not None
                paired_scale_name = _find_paired_scale_key(
                    normalized_weight_name=name,
                    raw_weight_name=raw_name,
                    shard_keys=shard_keys,
                )
                has_paired_scale = (
                    paired_scale_name is not None
                    and paired_scale_name in shard_keys
                )

                if is_scale_tensor:
                    assert scale_suffix is not None
                    base_weight_name = name.removesuffix(scale_suffix)
                    if fp8_block_size is None and not fp8_channel_scale:
                        continue
                    name = (
                        base_weight_name + _SCALE_RENAME_SUFFIX
                        if _is_expert_key(base_weight_name)
                        else base_weight_name + ".weight_scale"
                    )
                name = _rename_checkpoint_key(name)

                if is_gptoss:
                    # gpt-oss packs its experts as ONE tensor per layer with the expert
                    # index inside (`mlp.experts.gate_up_proj_blocks`), not as
                    # `experts.{i}.`, so _EXPERT_PATTERN never matches and none of the
                    # stacking/merging machinery below applies. MXFP4 also arrives as a
                    # (blocks, scales) pair that must be consumed together.
                    if name.endswith("_scales"):
                        continue  # consumed alongside the paired _blocks tensor
                    if name.endswith("_blocks"):
                        if skip_expert_weights:
                            continue
                        scales_raw = raw_name.replace("_blocks", "_scales")
                        if scales_raw not in shard_keys:
                            raise RuntimeError(
                                f"{raw_name}: MXFP4 blocks without a paired "
                                f"{scales_raw} in the same shard; the loader cannot "
                                "dequantise across files."
                            )
                        base = name.removesuffix("_blocks")
                        if _mxfp4_packed_enabled():
                            # Keep the weights 4-bit. Two entries are yielded instead of one
                            # because MoELayer holds blocks and scales as separate tensors --
                            # BaseOP.load_state_dict matches one state-dict key per tensor
                            # attribute, so a tuple has nowhere to land. The names are gpt-oss's
                            # own checkpoint names, so nothing downstream needs renaming.
                            _, (blk, scl) = _gptoss_expert_weight_packed(
                                base,
                                f.get_tensor(raw_name),
                                f.get_tensor(scales_raw),
                                tp_info.rank,
                                tp_info.size,
                            )
                            # (E, N, G, B) -> (E, N, K/2): the kernel indexes packed bytes flatly
                            # along the row. Both are contiguous here, so this is a view.
                            E_, N_ = blk.shape[0], blk.shape[1]
                            yield f"{base}_blocks", blk.reshape(E_, N_, -1)
                            yield f"{base}_scales", scl.reshape(E_, N_, -1)
                            continue
                        yield _gptoss_expert_weight(
                            base,
                            f.get_tensor(raw_name),
                            f.get_tensor(scales_raw),
                            tp_info.rank,
                            tp_info.size,
                        )
                        continue
                    is_gptoss_expert = ".experts." in name
                    # `down_proj_bias` is the one tensor in the expert namespace the AG side
                    # genuinely needs: its contribution is sum_e w_e * bias_e over the router
                    # weights, so it has to be applied on the attention rank after combine
                    # (models/gpt_oss_afd.py). It is replicated and tiny -- 128 x 2880 bf16 per
                    # layer, ~26 MB for all 36 -- so keeping a copy on the AG side is free.
                    # The EG side still gets it via the skip_non_expert_weights branch below.
                    ag_needs_expert_key = name.endswith(".experts.down_proj_bias")
                    if skip_expert_weights and is_gptoss_expert and not ag_needs_expert_key:
                        continue
                    if skip_non_expert_weights and not is_gptoss_expert:
                        continue
                    if _is_gptoss_special_key(name):
                        yield name, _gptoss_shard_override(
                            name, f.get_tensor(raw_name), tp_info.rank, tp_info.size
                        )
                        continue

                pre_merge_expert_info = (
                    _get_expert_stack_info(name) if config.is_moe else None
                )
                is_remote_shared_expert = _is_remote_shared_expert_key(name, config)
                # afd AG/EG role split: drop the other role's weights.
                if skip_expert_weights and (
                    pre_merge_expert_info is not None or is_remote_shared_expert
                ):
                    continue
                if skip_non_expert_weights and (
                    pre_merge_expert_info is None and not is_remote_shared_expert
                ):
                    continue
                if local_expert_partition is not None and pre_merge_expert_info is not None:
                    _, expert_idx = pre_merge_expert_info
                    if not (
                        local_expert_partition.start_idx
                        <= expert_idx
                        < local_expert_partition.end_idx
                    ):
                        continue
                raw = f.get_tensor(raw_name)
                if fp8_channel_scale and is_scale_tensor:
                    tensor = _shard_fp8_channel_scale_tensor(
                        name,
                        raw,
                        tp_info.rank,
                        tp_info.size,
                        config.num_kv_heads,
                        skip_tp_shard=(
                            local_expert_partition is not None
                            and pre_merge_expert_info is not None
                        ),
                    )
                else:
                    tensor = _shard_tensor(
                        name,
                        raw,
                        tp_info.rank,
                        tp_info.size,
                        config.num_kv_heads,
                        skip_tp_shard=(
                            local_expert_partition is not None
                            and pre_merge_expert_info is not None
                        ),
                        fp8_block_size=fp8_block_size,
                    )
                del raw

                if (
                    fp8_block_size is not None
                    and has_paired_scale
                    and not _is_expert_key(name)
                    and tensor.dtype != torch.float8_e4m3fn
                ):
                    assert paired_scale_name is not None
                    scale_full = f.get_tensor(paired_scale_name)
                    scale_sharded = _shard_tensor(
                        name,
                        scale_full,
                        tp_info.rank,
                        tp_info.size,
                        config.num_kv_heads,
                        skip_tp_shard=False,
                        fp8_block_size=fp8_block_size,
                    )
                    del scale_full
                    tensor = dequant_fp8_block(tensor, scale_sharded, fp8_block_size)
                    del scale_sharded

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    if local_expert_partition is not None:
                        if not (
                            local_expert_partition.start_idx
                            <= expert_idx
                            < local_expert_partition.end_idx
                        ):
                            continue
                        expert_idx = expert_idx - local_expert_partition.start_idx
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != real_local_num_experts:
                        continue
                    if local_num_experts == real_local_num_experts:
                        experts = [slots[idx] for idx in range(local_num_experts)]
                    else:
                        template = next(iter(slots.values()))
                        zero = torch.zeros_like(template)
                        experts = [slots.get(idx, zero) for idx in range(local_num_experts)]
                    del expert_buf[packed_key]
                    stacked = torch.stack(experts, dim=0)
                    if packed_key.endswith(_SCALE_RENAME_SUFFIX):
                        stacked = stacked.to(torch.float32)
                    yield packed_key, stacked
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
    logger.info("Finished loading weights")
