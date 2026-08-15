#!/usr/bin/env python3
"""Numerically probe the native gfx950 AITER GPT-OSS conversion.

This is deliberately a standalone diagnostic rather than a unit test: it launches
the CK kernel and therefore needs the pinned ROCm image and one MI355X GPU.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from minisgl.kernel.mxfp4 import dequant_mxfp4
from minisgl.layers.moe.moe_runner.aiter import (
    _deinterleave_gate_up_rows,
    _prepare_aiter_mxfp4_weights,
)


def _dense_weight(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    experts, output, packed_k = blocks.shape
    return dequant_mxfp4(
        blocks.reshape(experts, output, packed_k // 16, 16), scales,
        dtype=torch.bfloat16,
    ).transpose(1, 2).contiguous()


def _gptoss_activation(interleaved: torch.Tensor) -> torch.Tensor:
    gate = interleaved[..., 0::2].clamp(max=7.0)
    up = interleaved[..., 1::2].clamp(min=-7.0, max=7.0)
    return (up + 1.0) * gate * torch.sigmoid(1.702 * gate)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> str:
    actual = actual.float()
    expected = expected.float()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    max_abs = (actual - expected).abs().max().item()
    mean_abs = (actual - expected).abs().mean().item()
    return (
        f"cos={cosine:.7f} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"actual_max={actual.abs().max().item():.6g} expected_max={expected.abs().max().item():.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2880)
    parser.add_argument("--intermediate", type=int, default=2880)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--checkpoint-shard")
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda")
    e, h, i = 1, args.hidden, args.intermediate
    if args.checkpoint_shard:
        from safetensors import safe_open

        prefix = "model.layers.0.mlp.experts."
        with safe_open(args.checkpoint_shard, framework="pt", device="cpu") as handle:
            w1 = handle.get_tensor(prefix + "gate_up_proj_blocks")[:1]
            s1 = handle.get_tensor(prefix + "gate_up_proj_scales")[:1]
            b1 = handle.get_tensor(prefix + "gate_up_proj_bias")[:1]
            w2 = handle.get_tensor(prefix + "down_proj_blocks")[:1]
            s2 = handle.get_tensor(prefix + "down_proj_scales")[:1]
        h = int(w1.shape[-2] * w1.shape[-1] * 2)
        i = int(w2.shape[-2] * w2.shape[-1] * 2)
        w1 = w1.reshape(e, 2 * i, h // 2).to(device)
        s1 = s1.reshape(e, 2 * i, h // 32).to(device)
        b1 = b1.reshape(e, 2 * i).to(device)
        w2 = w2.reshape(e, h, i // 2).to(device)
        s2 = s2.reshape(e, h, i // 32).to(device)
    else:
        # Use all FP4 codes, but keep E8M0 exponents near unity so errors stay readable.
        w1 = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, device=device)
        s1 = torch.randint(124, 128, (e, 2 * i, h // 32), dtype=torch.uint8, device=device)
        b1 = torch.randn((e, 2 * i), dtype=torch.float32, device=device) * 0.01
        w2 = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8, device=device)
        s2 = torch.randint(124, 128, (e, h, i // 32), dtype=torch.uint8, device=device)
    x = torch.randn((args.tokens, h), dtype=torch.bfloat16, device=device) * 0.02

    dense_w1 = _dense_weight(w1, s1)
    dense_w2 = _dense_weight(w2, s2)
    gate_up = F.linear(x, dense_w1[0], b1[0].to(torch.bfloat16))
    expected = F.linear(_gptoss_activation(gate_up), dense_w2[0])

    # AITER's own unshuffled PyTorch oracle distinguishes checkpoint/dequant
    # interpretation errors from CK shuffle/kernel errors.
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, torch_moe_stage1, torch_moe_stage2

    ids = torch.zeros((args.tokens, 1), dtype=torch.int32, device=device)
    weights = torch.ones((args.tokens, 1), dtype=torch.float32, device=device)
    fp4 = torch.float4_e2m1fn_x2
    oracle_w1 = _deinterleave_gate_up_rows(w1).view(fp4)
    oracle_s1 = _deinterleave_gate_up_rows(s1)
    oracle_b1 = _deinterleave_gate_up_rows(b1)
    oracle_w2 = w2.view(fp4)
    stage1 = torch_moe_stage1(
        x, oracle_w1, oracle_w2, weights, ids,
        dtype=torch.bfloat16, activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32, w1_scale=oracle_s1,
        w1_bias=oracle_b1,
    )
    oracle = torch_moe_stage2(
        stage1, oracle_w1, oracle_w2, weights, ids,
        dtype=torch.bfloat16, quant_type=QuantType.per_1x32,
        w2_scale=s2,
    )
    print(f"aiter-torch-oracle {_metrics(oracle, expected)}")

    layer = SimpleNamespace(
        _is_mxfp4_packed=True,
        gate_up_proj_blocks=w1,
        gate_up_proj_scales=s1,
        gate_up_proj_bias=b1,
        down_proj_blocks=w2,
        down_proj_scales=s2,
    )
    _prepare_aiter_mxfp4_weights(layer)

    # Diff against the conversion in the pinned vLLM image. This comparison is
    # intentionally bitwise; both paths ultimately call the same AITER shufflers.
    try:
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            Mxfp4MoeBackend,
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        )

        ph = h + int(layer._aiter_hidden_pad)
        pi = i + int(layer._aiter_intermediate_pad)
        vw1 = torch.zeros((e, 2 * pi, ph // 2), dtype=torch.uint8, device=device)
        vs1 = torch.full((e, 2 * pi, ph // 32), 127, dtype=torch.uint8, device=device)
        vb1 = torch.zeros((e, 2 * pi), dtype=torch.float32, device=device)
        vw2 = torch.zeros((e, ph, pi // 2), dtype=torch.uint8, device=device)
        vs2 = torch.full((e, ph, pi // 32), 127, dtype=torch.uint8, device=device)
        vw1[:, : 2 * i, : h // 2] = w1
        vs1[:, : 2 * i, : h // 32] = s1
        vb1[:, : 2 * i] = b1
        vw2[:, :h, : i // 2] = w2
        vs2[:, :h, : i // 32] = s2
        converted = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            Mxfp4MoeBackend.AITER_MXFP4_BF16,
            SimpleNamespace(), vw1, vw2, vs1, vs2, vb1, None,
        )
        ours = (
            layer.gate_up_proj_blocks,
            layer.down_proj_blocks,
            layer.gate_up_proj_scales,
            layer.down_proj_scales,
            layer.gate_up_proj_bias,
        )
        names = ("w1", "w2", "s1", "s2", "b1")
        for name, actual_tensor, vllm_tensor in zip(names, ours, converted[:5]):
            if actual_tensor.dtype == torch.float4_e2m1fn_x2:
                actual_tensor = actual_tensor.view(torch.uint8)
                vllm_tensor = vllm_tensor.view(torch.uint8)
            print(
                f"vllm-convert-{name}: equal={torch.equal(actual_tensor, vllm_tensor)} "
                f"shape={tuple(actual_tensor.shape)}/{tuple(vllm_tensor.shape)}"
            )
    except ImportError as exc:
        print(f"vllm conversion comparison unavailable: {exc}")

    hp = int(layer._aiter_hidden_pad)
    ip = int(layer._aiter_intermediate_pad)
    kernel_x = x.new_zeros((x.shape[0], h + hp))
    kernel_x[:, :h] = x
    candidates = [(hp, ip), (0, 0), (hp // 128 * 128, ip // 64 * 64 * 2)]
    seen: set[tuple[bool, int, int]] = set()
    for shuffled in (True, False):
        layer.gate_up_proj_blocks.is_shuffled = shuffled
        layer.down_proj_blocks.is_shuffled = shuffled
        for hidden_pad, intermediate_pad in candidates:
            if (shuffled, hidden_pad, intermediate_pad) in seen:
                continue
            seen.add((shuffled, hidden_pad, intermediate_pad))
            actual = fused_moe(
                hidden_states=kernel_x,
                w1=layer.gate_up_proj_blocks,
                w2=layer.down_proj_blocks,
                topk_weight=weights,
                topk_ids=ids,
                activation=ActivationType.Swiglu,
                quant_type=QuantType.per_1x32,
                w1_scale=layer.gate_up_proj_scales,
                w2_scale=layer.down_proj_scales,
                dtype=x.dtype,
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
                bias1=layer.gate_up_proj_bias,
                bias2=None,
            )[:, :h]
            torch.cuda.synchronize()
            print(
                f"shuffled={shuffled} pad=({hidden_pad},{intermediate_pad}) "
                f"{_metrics(actual, expected)}"
            )


if __name__ == "__main__":
    main()
