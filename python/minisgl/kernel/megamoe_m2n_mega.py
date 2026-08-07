"""Split MegaMoE M2N: the DeepGEMM mega kernel split into AG/EG role kernels.

This wraps the vendored DeepGEMM extension's split mega-MoE entry points:

- ``mega_moe_m2n_ag``: AG/attention half — one kernel that only dispatch-sends
  (token-topk index + count push; EG ranks pull the FP8 payload) and
  combine-receives (waits for EG write-back, reduces top-k, writes ``y``).
- ``mega_moe_m2n_eg``: EG/MLP half — one persistent kernel that pulls FP8
  tokens from AG ranks over NVLink, runs the fused FP8xFP4 L1/L2 grouped GEMM
  pipeline (SwiGLU + top-k weights fused in the L1 epilogue), and pushes BF16
  expert outputs into the source AG ranks' combine buffers.

Topology: one union process group of ``ag_size + eg_size`` ranks, AG first.
EG rank ``g`` owns experts ``[g * E_local, (g + 1) * E_local)``.  The
symmetric buffer is allocated with ``torch.distributed._symmetric_memory``
(NVLink/MNNVL fabric), identical layout on every rank.
"""

from __future__ import annotations

import types
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from . import deepgemm as _dg


def _ext():
    return _dg._load_extension()


def get_token_alignment() -> int:
    return int(_ext().get_token_alignment_for_mega_moe())


def align(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y) * int(y)


class MegaMoEM2NSymmBuffer:
    """Union-symmetric buffer for the split M2N mega kernels.

    AG ranks use the ``x``/``x_sf``/``topk_idx``/``topk_weights`` input views
    (copy inputs in before every AG launch) and receive combine output via the
    kernel's ``y``.  EG ranks use the L1/L2 pool views implicitly.
    """

    def __init__(self, group: dist.ProcessGroup, *,
                 ag_size: int, eg_size: int,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.ag_size = int(ag_size)
        self.eg_size = int(eg_size)
        if group.size() != self.ag_size + self.eg_size:
            raise RuntimeError(
                f"group size {group.size()} != ag_size + eg_size = {self.ag_size + self.eg_size}"
            )
        self.num_experts = int(num_experts)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.num_max_tokens_per_rank = align(num_max_tokens_per_rank, get_token_alignment())

        num_bytes, num_max_pool_tokens, slice_input_buffers = (
            _ext().get_symm_buffer_size_for_mega_moe_m2n(
                self.ag_size, self.eg_size, self.num_experts,
                self.num_max_tokens_per_rank, self.num_topk,
                self.hidden, self.intermediate_hidden,
            )
        )
        self.num_max_pool_tokens = int(num_max_pool_tokens)
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(int(num_bytes), dtype=torch.int8, device="cuda")
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        # Seed the phase-0 NVLink barrier signal with a completed +1 round:
        # the kernels defer the cleanup-barrier WAIT into the next launch, so
        # the very first wait must observe a finished phase.  Workspace head:
        # [0:16) grid-sync counters, [16:20) barrier counter, [20:24) signal
        # phase 0, [24:28) signal phase 1.
        self.buffer[20:24].view(torch.int32).fill_(group.size())
        dist.barrier(group=group)
        torch.cuda.synchronize()

        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    @property
    def buffer_ptrs(self) -> list[int]:
        return list(self.handle.buffer_ptrs)

    def destroy(self) -> None:
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_acts = None
        self.l1_acts_sf = None
        self.l2_acts = None
        self.l2_acts_sf = None


def mega_moe_m2n_ag(y: torch.Tensor,
                    hidden_states: torch.Tensor,
                    topk_ids: Optional[torch.Tensor],
                    topk_weights: Optional[torch.Tensor],
                    sym_buffer: MegaMoEM2NSymmBuffer,
                    *, rank: int,
                    num_sms: int = 0,
                    gate_weight: Optional[torch.Tensor] = None,
                    renormalize: bool = True,
                    phase: int = 0,
                    external_quant: bool = False,
                    direct_combine: bool = False,
                    input_num_topk: Optional[int] = None,
                    num_routed_experts: Optional[int] = None,
                    shared_topk_weight: float = 0.0,
                    debug_timings: Optional[torch.Tensor] = None) -> None:
    """AG half: fused per-32 FP8 quant + dispatch-send + combine-recv.

    ``phase`` splits the kernel: 0 = fused (default), 1 = dispatch-only
    (quant + pushes, exits without waiting so its SMs go back to attention
    during the EG round trip), 2 = combine-only (tag-2 wait + top-k reduce +
    cleanup).  A phase-1 launch MUST be followed by exactly one phase-2
    launch before the next phase-1 on the same buffer.

    One kernel does everything: quantizes ``hidden_states`` (bf16
    [num_tokens, hidden]) into the symmetric buffer (e4m3 + ceil-UE8M0 per-32
    scales), stages ``topk_weights`` for the EG pull, pushes route metadata,
    waits for the expert write-back and reduces top-k into ``y`` (bf16).
    ``topk_ids`` may be int32 or int64 (read in place, not copied).

    If ``gate_weight`` (bf16 [num_experts, hidden]) is given, the ROUTER runs
    inside the kernel too (GEMV + softmax + top-k + optional renormalize) and
    ``topk_ids``/``topk_weights`` must be None.  Intended for decode-sized
    batches (a warp-per-token GEMV; use the host gate for prefill).

    ``num_sms`` limits the persistent grid (0 = all SMs).  The kernel spends
    most of its time spin-waiting for the EG combine push, so a small grid
    keeps the GPU free for concurrent compute (e.g. the other microbatch's
    attention).
    """
    _ext().mega_moe_m2n_ag(
        y,
        hidden_states, topk_ids, topk_weights,
        gate_weight, bool(renormalize),
        sym_buffer.buffer,
        sym_buffer.buffer_ptrs, int(rank),
        sym_buffer.ag_size, sym_buffer.eg_size,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        sym_buffer.intermediate_hidden,
        int(num_sms),
        int(phase),
        bool(external_quant),
        bool(direct_combine),
        int(input_num_topk if input_num_topk is not None else sym_buffer.num_topk),
        int(num_routed_experts if num_routed_experts is not None else sym_buffer.num_experts),
        float(shared_topk_weight),
        debug_timings,
    )


def make_m2n_weight_descs(l1_weights: Tuple[torch.Tensor, torch.Tensor],
                          l2_weights: Tuple[torch.Tensor, torch.Tensor],
                          *, num_experts_per_rank: int,
                          hidden: int, intermediate_hidden: int) -> torch.Tensor:
    """Build the per-layer weight tensormap blob (uint8 [4, 128], CPU) for the
    persistent EG kernel.  Config-independent, so it is built once at weight
    registration."""
    return _ext().make_m2n_weight_descs(
        l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1],
        int(num_experts_per_rank), int(hidden), int(intermediate_hidden),
    )


def mega_moe_m2n_eg(weight_descs: torch.Tensor,
                    l1_weight_ptrs: torch.Tensor,
                    sym_buffer: MegaMoEM2NSymmBuffer,
                    *, rank: int,
                    expected_num_tokens_per_rank: int,
                    num_experts_per_rank: int,
                    l1_weights_nbytes: int,
                    num_layers: int = 1,
                    use_fp8_weights: bool = True,
                    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                    activation_clamp: Optional[float] = None,
                    fast_math: bool = True,
                    num_sms: int = 0,
                    num_prefetch_bytes: int = 0,
                    sym_buffer1: Optional[MegaMoEM2NSymmBuffer] = None,
                    sym_buffer2: Optional[MegaMoEM2NSymmBuffer] = None,
                    sym_buffer3: Optional[MegaMoEM2NSymmBuffer] = None,
                    debug_timings: Optional[torch.Tensor] = None) -> None:
    """EG half: dispatch-recv + FP8 expert MLP + combine-send.

    ``weight_descs`` is the device uint8 [num_layers, 4, 128] tensormap blob
    (built via :func:`make_m2n_weight_descs`); ``l1_weight_ptrs`` the device
    int64 [num_layers] L1-weight base pointers (for the L2 prefetch).  With
    ``num_layers > 1`` ONE persistent kernel serves the whole step (per-launch
    smem/TMEM init amortized); with ``sym_buffer1`` it also serves both
    microbatch lanes per layer at full device width.
    """
    _ext().mega_moe_m2n_eg(
        weight_descs, l1_weight_ptrs,
        int(num_layers),
        int(num_experts_per_rank),
        sym_buffer.hidden, sym_buffer.intermediate_hidden,
        int(l1_weights_nbytes),
        bool(use_fp8_weights),
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer1.buffer if sym_buffer1 is not None else None,
        sym_buffer2.buffer if sym_buffer2 is not None else None,
        sym_buffer3.buffer if sym_buffer3 is not None else None,
        sym_buffer.buffer_ptrs,
        sym_buffer1.buffer_ptrs if sym_buffer1 is not None else sym_buffer.buffer_ptrs,
        sym_buffer2.buffer_ptrs if sym_buffer2 is not None else sym_buffer.buffer_ptrs,
        sym_buffer3.buffer_ptrs if sym_buffer3 is not None else sym_buffer.buffer_ptrs,
        int(rank),
        sym_buffer.ag_size, sym_buffer.eg_size,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        int(expected_num_tokens_per_rank),
        activation_clamp,
        bool(fast_math),
        int(num_sms),
        int(num_prefetch_bytes),
        debug_timings,
    )


# ---------------------------------------------------------------------------
# FP4 weight utilities (ported from upstream DeepGEMM `deep_gemm.utils.math`
# and `deep_gemm.mega`)
# ---------------------------------------------------------------------------
def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                              device=x.device, dtype=torch.float)
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def per_token_cast_to_fp4(x: torch.Tensor, *, use_ue8m0: bool = True, gran_k: int = 32,
                          use_packed_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    assert n % 2 == 0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = _dg.ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, :n // 2].contiguous(), \
        (_dg.pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf)


def dequant_fp4(packed: torch.Tensor, sf: torch.Tensor, *, gran_k: int = 32) -> torch.Tensor:
    """Dequantize packed FP4 e2m1 with float UE8M0 scales back to fp32."""
    m, n2 = packed.shape
    n = n2 * 2
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x = _dequantize_from_fp4_e2m1(unpacked)
    scale = torch.repeat_interleave(sf.float(), gran_k, dim=1)[:, :n]
    return x * scale


def requant_qwen_fp8_weights_per32(
    w_fp8: torch.Tensor,
    block_scales: torch.Tensor,
    *,
    block: int = 128,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Qwen FP8 weights -> mega-kernel FP8 weights with per-32 UE8M0 scales.

    Input: ``w_fp8`` [E, N, K] e4m3 with float 128x128 ``block_scales``
    [E, ceil(N/128), ceil(K/128)].  The block scales are rounded UP to UE8M0
    and the weights are rescaled by ``s / s_ue8m0`` (one fp8 round trip), so
    ``w' * sf == w * s`` up to e4m3 rounding.  Returns the requantized weights
    and the DeepGEMM-layout packed UE8M0 SF for the (1, 32) recipe.
    """
    assert w_fp8.dtype == torch.float8_e4m3fn and w_fp8.dim() == 3
    e, n, k = w_fp8.shape
    assert k % 32 == 0
    s = block_scales.float()
    s_u = _dg.ceil_to_ue8m0(s)
    ratio = torch.where(s_u > 0, s / s_u, torch.zeros_like(s))
    if inplace and not w_fp8.is_contiguous():
        raise RuntimeError("in-place MegaMoE FP8 weight requant requires contiguous weights")
    w_new = w_fp8 if inplace else torch.empty_like(w_fp8, memory_format=torch.contiguous_format)
    for expert in range(e):
        ratio_e = torch.repeat_interleave(ratio[expert], block, dim=0)[:n]
        ratio_e = torch.repeat_interleave(ratio_e, block, dim=1)[:, :k]
        w_new[expert].copy_(
            (w_fp8[expert].float() * ratio_e).to(torch.float8_e4m3fn)
        )
        del ratio_e
    sf32 = torch.repeat_interleave(s_u, block, dim=1)[:, :n]
    sf32 = torch.repeat_interleave(sf32, block // 32, dim=2)[:, :, : k // 32].contiguous()
    sf_packed = _dg.transform_sf_into_required_layout(sf32, n, k, (1, 32), e)
    return w_new, sf_packed


def dequant_fp8_per32(w_fp8: torch.Tensor, sf32: torch.Tensor) -> torch.Tensor:
    """Dequantize [N, K] fp8 with per-32 float scales [N, K/32] (reference)."""
    return w_fp8.float() * torch.repeat_interleave(sf32.float(), 32, dim=1)


def cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """[E, N, K] bf16 -> packed FP4 weights + DeepGEMM-layout UE8M0 SF."""
    num_groups, n, k = bf16_weights.shape
    w = torch.empty((num_groups, n, k // 2), device="cuda", dtype=torch.int8)
    w_sf = torch.empty((num_groups, n, k // 32), device="cuda", dtype=torch.float)
    for i in range(num_groups):
        w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
    w_sf_t = _dg.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
    return w, w_sf_t


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    out = torch.empty_like(t)
    out_view = out.reshape(g, half // gran, 2, gran, *rest)
    out_view[:, :, 0].copy_(gate)
    out_view[:, :, 1].copy_(up)
    return out


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return (l1_w, l1_sf), l2_transformed


__all__ = [
    "MegaMoEM2NSymmBuffer",
    "cast_weights_to_fp4",
    "dequant_fp4",
    "dequant_fp8_per32",
    "get_token_alignment",
    "mega_moe_m2n_ag",
    "mega_moe_m2n_eg",
    "per_token_cast_to_fp4",
    "requant_qwen_fp8_weights_per32",
    "transform_weights_for_mega_moe",
]
