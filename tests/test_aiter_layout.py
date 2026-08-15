from __future__ import annotations

import torch

from minisgl.layers.moe.moe_runner.aiter import (
    _deinterleave_gate_up_rows,
    _pad_gptoss_for_aiter,
)


def test_deinterleave_gate_up_rows() -> None:
    value = torch.tensor([[[0], [10], [1], [11], [2], [12]]])
    actual = _deinterleave_gate_up_rows(value)
    expected = torch.tensor([[[0], [1], [2], [10], [11], [12]]])
    torch.testing.assert_close(actual, expected)


def test_gptoss_2880_is_padded_without_moving_real_values() -> None:
    w1 = torch.zeros((1, 2 * 2880, 2880 // 2), dtype=torch.uint8)
    w1[:, :2880].fill_(17)
    w1[:, 2880:].fill_(29)
    s1 = torch.ones((1, 2 * 2880, 2880 // 32), dtype=torch.uint8)
    b1 = torch.arange(2 * 2880).reshape(1, -1)
    w2 = torch.ones((1, 2880, 2880 // 2), dtype=torch.uint8)
    s2 = torch.ones((1, 2880, 2880 // 32), dtype=torch.uint8)
    pw1, ps1, pb1, pw2, ps2, hidden_pad, intermediate_pad = _pad_gptoss_for_aiter(
        w1, s1, b1, w2, s2
    )
    assert pw1.shape == (1, 6144, 1536)
    assert ps1.shape == (1, 6144, 96)
    assert pw2.shape == (1, 3072, 1536)
    assert ps2.shape == (1, 3072, 96)
    assert (hidden_pad, intermediate_pad) == (192, 192)
    torch.testing.assert_close(pw1[:, :2880, :1440], w1[:, :2880])
    torch.testing.assert_close(pw1[:, 3072:5952, :1440], w1[:, 2880:])
    torch.testing.assert_close(pb1[:, :2880], b1[:, :2880])
    torch.testing.assert_close(pb1[:, 3072:5952], b1[:, 2880:])
