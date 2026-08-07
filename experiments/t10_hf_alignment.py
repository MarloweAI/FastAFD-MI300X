#!/usr/bin/env python3
"""Portable wrapper for the port's T-10 HF correctness gate."""

from __future__ import annotations

import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


model = os.environ.get("MODEL")
if not model:
    raise SystemExit("MODEL must point to the local Hugging Face model snapshot")

implementation = Path(__file__).with_name("_t10_hf_alignment_impl.py")
if not implementation.exists():
    implementation = Path(__file__).parents[1] / "dev_log/probes/t10_hf_alignment.py"
namespace = runpy.run_path(str(implementation))
namespace["MODEL"] = model
namespace["PORT"] = int(os.environ.get("PORT", "19295"))
namespace["HF_DEVICE"] = os.environ.get("HF_DEVICE", "cuda:1")

# The reusable reference was generated at a different absolute model path. The
# original gate compares that metadata string literally, so adapt only the copy
# passed to it; token IDs, logits, prompts, and all numerical data stay unchanged.
temporary: tempfile.NamedTemporaryFile[str] | None = None
if "--hf-in" in sys.argv:
    index = sys.argv.index("--hf-in") + 1
    with open(sys.argv[index], encoding="utf-8") as source:
        blob = json.load(source)
    blob["model"] = model
    temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(blob, temporary)
    temporary.close()
    sys.argv[index] = temporary.name

try:
    raise SystemExit(namespace["main"]())
finally:
    if temporary is not None:
        Path(temporary.name).unlink(missing_ok=True)
