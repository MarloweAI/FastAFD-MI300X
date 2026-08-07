from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from minisgl.layers import BaseOP


# ---------------------------------------------------------------------------
# AFD inter-node data types
# ---------------------------------------------------------------------------

@dataclass
class ModelStageState:
    hidden_states: torch.Tensor
    residual: torch.Tensor | None


# ---------------------------------------------------------------------------
# Weight extraction utility
# ---------------------------------------------------------------------------

def extract_stage_state_dict(
    stage_model: BaseOP,
    full_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    expected_keys = stage_model.state_dict().keys()
    missing = [key for key in expected_keys if key not in full_state_dict]
    if missing:
        raise RuntimeError(f"Missing keys for AFD model stage: {missing[:8]}")
    return {key: full_state_dict[key] for key in expected_keys}


__all__ = [
    "ModelStageState",
    "extract_stage_state_dict",
]
