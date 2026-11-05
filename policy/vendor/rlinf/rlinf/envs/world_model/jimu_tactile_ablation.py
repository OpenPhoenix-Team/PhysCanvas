"""Pure observation transforms used by Jimu tactile ablation audits."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def zero_generated_tactile_for_policy(
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Clone one observation and zero only Pi0.5's tactile values.

    The four-sensor modality mask remains enabled, which distinguishes the
    ablation from removing or masking the tactile modality.  The input object
    is not mutated, so the WM's internal multimodal state remains unchanged.
    """

    result = copy.deepcopy(observation)
    tactile = np.asarray(result["observation.tactile"])
    if tactile.shape != (4, 700, 3):
        raise ValueError(f"Expected tactile [4,700,3], got {tactile.shape}")
    result["observation.tactile"] = np.zeros_like(tactile)
    mask = np.asarray(result["observation.tactile_mask"], dtype=np.bool_)
    if mask.shape != (4,):
        raise ValueError(f"Expected tactile mask [4], got {mask.shape}")
    result["observation.tactile_mask"] = np.ones_like(mask, dtype=np.bool_)
    result.setdefault("wm", {})[
        "policy_tactile_source"
    ] = "ablation_all_zero_values_mask_enabled"
    return result


def mask_generated_tactile_for_policy(
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Clone an observation and remove tactile from Pi0.5's attention.

    Values are zeroed for an unambiguous payload and all four mask bits are
    disabled. The WM-owned observation is not mutated, so this changes only
    the next Pi0.5 replan and not the WM's internal multimodal state.
    """

    result = copy.deepcopy(observation)
    tactile = np.asarray(result["observation.tactile"])
    if tactile.shape != (4, 700, 3):
        raise ValueError(f"Expected tactile [4,700,3], got {tactile.shape}")
    result["observation.tactile"] = np.zeros_like(tactile)
    mask = np.asarray(result["observation.tactile_mask"], dtype=np.bool_)
    if mask.shape != (4,):
        raise ValueError(f"Expected tactile mask [4], got {mask.shape}")
    result["observation.tactile_mask"] = np.zeros_like(mask, dtype=np.bool_)
    result.setdefault("wm", {})[
        "policy_tactile_source"
    ] = "ablation_tactile_mask_disabled"
    return result
