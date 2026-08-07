#!/usr/bin/env python3
"""Fail-fast checks for a copied FastAFD MI300X runtime."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys


REQUIRED_MODULES = (
    "torch",
    "triton",
    "tvm_ffi",
    "ray",
    "transformers",
    "fastapi",
    "uvicorn",
    "zmq",
    "msgpack",
    "pydantic",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    missing: list[str] = []
    versions: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # dependency import errors need their original detail
            missing.append(f"{name} ({exc})")
            continue
        versions.append(f"{name}={getattr(module, '__version__', 'present')}")
    if missing:
        fail("missing/broken Python dependencies: " + ", ".join(missing))

    import torch

    if torch.version.hip is None:
        fail(f"torch {torch.__version__} is not a ROCm build")
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        fail("hipcc is not on PATH")
    if not torch.cuda.is_available():
        fail("ROCm torch cannot see a GPU; check /dev/kfd permissions and device visibility")

    devices: list[str] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, "gcnArchName", "unknown")
        devices.append(f"{index}:{props.name}:{arch}")
    if not any("gfx942" in device for device in devices):
        fail("no gfx942/MI300X device found: " + ", ".join(devices))

    import minisgl
    from minisgl.models.gpt_oss import GptOssForCausalLM  # noqa: F401
    from minisgl.server import launch_server  # noqa: F401

    print(f"python={platform.python_version()} arch={platform.machine()}")
    print(f"torch={torch.__version__} hip={torch.version.hip} hipcc={hipcc}")
    print("devices=" + ", ".join(devices))
    print("dependencies=" + " ".join(versions))
    package_file = getattr(minisgl, "__file__", None)
    package_path = os.path.dirname(package_file) if package_file else next(iter(minisgl.__path__))
    print(f"minisgl={package_path}")


if __name__ == "__main__":
    main()
