#!/usr/bin/env python3
"""Portable entry point for the port's original serving benchmark."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


model = os.environ.get("MODEL")
if not model:
    raise SystemExit("MODEL must point to the local Hugging Face model snapshot")
model_path = Path(model).expanduser()
if not model_path.is_dir():
    raise SystemExit(f"MODEL directory does not exist: {model_path}")
os.environ["MODEL"] = str(model_path)

implementation = Path(__file__).with_name("_serve_bench_impl.py")
if not implementation.exists():
    implementation = Path(__file__).parents[1] / "dev_log/probes/serve_bench.py"
namespace = runpy.run_path(str(implementation))
raise SystemExit(namespace["main"]())
