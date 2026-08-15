#!/usr/bin/env python3
"""Fail-fast environment and backend manifest for MI355X runs."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess

import torch


def main() -> int:
    if not torch.version.hip:
        raise SystemExit("FAIL: PyTorch is not a ROCm build")
    if torch.cuda.device_count() < 1:
        raise SystemExit("FAIL: no allocated AMD GPUs are visible")
    arches = [
        str(torch.cuda.get_device_properties(index).gcnArchName)
        for index in range(torch.cuda.device_count())
    ]
    if any(not arch.startswith("gfx950") for arch in arches):
        raise SystemExit(f"FAIL: MI355X gfx950 is required, found {arches}")

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    assert callable(fused_moe)
    assert callable(shuffle_weight_a16w4)
    assert callable(shuffle_scale_a16w4)
    assert hasattr(ActivationType, "Swiglu")
    assert hasattr(QuantType, "per_1x32")

    try:
        source_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unknown"
    try:
        aiter_version = importlib.metadata.version("amd-aiter")
    except importlib.metadata.PackageNotFoundError:
        aiter_version = "unknown"

    print(
        json.dumps(
            {
                "source_revision": source_revision,
                "host": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "rocm": torch.version.hip,
                "aiter": aiter_version,
                "visible_gpus": torch.cuda.device_count(),
                "gpu_arches": arches,
                "mxfp4_backend": "aiter_ck_a16w4",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
