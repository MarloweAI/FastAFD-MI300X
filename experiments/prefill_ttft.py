#!/usr/bin/env python3
"""Portable entry point for the port's unique-prompt prefill benchmark."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


model = os.environ.get("MODEL")
if not model:
    raise SystemExit("MODEL must point to the local Hugging Face model snapshot")

implementation = Path(__file__).with_name("_prefill_ttft_impl.py")
if not implementation.exists():
    implementation = Path(__file__).parents[1] / "dev_log/gpt_oss_120b/gptoss_prefill_ttft.py"
namespace = runpy.run_path(str(implementation))
namespace["MODEL"] = model
raise SystemExit(namespace["main"]())
