# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wire-format validation for the Jimu Pi0.5/world-model integration.

The PPU service returns RGB frames as three 224x224 views concatenated along
the image width.  Keeping the conversion here makes the environment and reward
adapter share one, testable interpretation of the payload.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

ACTION_HORIZON = 16
WM_SUB_FRAMES = 8
ACTION_DIM = 14
TACTILE_SHAPE = (4, 700, 3)
TACTILE_GRID_HEIGHT = 35
TACTILE_GRID_WIDTH = 20
TACTILE_FLOW_LINEAR_MAX_DISP = 0.08
VIEW_ORDER = ("base", "left_wrist", "right_wrist")
WM_IMAGE_KEYS = (
    "observation.images.base_0",
    "observation.images.left_wrist_0",
    "observation.images.right_wrist_0",
)


def validate_action_batch(actions: Any) -> np.ndarray:
    """Return contiguous float32 actions with shape ``[B,16,14]``."""

    value = np.asarray(actions, dtype=np.float32)
    expected_tail = (ACTION_HORIZON, ACTION_DIM)
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_tail:
        raise ValueError(
            f"Expected action batch [B,{ACTION_HORIZON},{ACTION_DIM}], got {value.shape}"
        )
    if value.shape[0] <= 0:
        raise ValueError("Action batch must contain at least one rollout")
    if not np.isfinite(value).all():
        raise ValueError("Action batch contains NaN or Inf")
    return np.ascontiguousarray(value)


def validate_wm_action_batch(actions: Any) -> np.ndarray:
    """Return contiguous float32 actions for one 8-frame WM call."""

    value = np.asarray(actions, dtype=np.float32)
    expected_tail = (WM_SUB_FRAMES, ACTION_DIM)
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_tail:
        raise ValueError(
            f"Expected WM action batch [B,{WM_SUB_FRAMES},{ACTION_DIM}], "
            f"got {value.shape}"
        )
    if value.shape[0] <= 0:
        raise ValueError("WM action batch must contain at least one rollout")
    if not np.isfinite(value).all():
        raise ValueError("WM action batch contains NaN or Inf")
    return np.ascontiguousarray(value)


def split_action_horizon(actions: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split one policy chunk into the two consecutive WM action chunks."""

    value = validate_action_batch(actions)
    first = np.ascontiguousarray(value[:, :WM_SUB_FRAMES])
    second = np.ascontiguousarray(value[:, WM_SUB_FRAMES:])
    if first.shape[1] != WM_SUB_FRAMES or second.shape[1] != WM_SUB_FRAMES:
        raise AssertionError("16-step action split must produce two 8-step chunks")
    return first, second


def validate_panel_video(
    video: Any,
    *,
    expected_frames: int = ACTION_HORIZON,
) -> np.ndarray | torch.Tensor:
    """Validate a ``[B,T,H,3W,3]`` uint8 RGB panel video."""

    if expected_frames <= 0:
        raise ValueError("expected_frames must be positive")

    if torch.is_tensor(video):
        value = video
        if value.ndim != 5:
            raise ValueError(
                f"Expected panel video [B,T,H,W,C], got {tuple(value.shape)}"
            )
        if value.shape[1] != expected_frames or value.shape[-1] != 3:
            raise ValueError(
                f"Expected T={expected_frames} and C=3, got {tuple(value.shape)}"
            )
        height, width = int(value.shape[2]), int(value.shape[3])
        if width != len(VIEW_ORDER) * height:
            raise ValueError(
                "Expected three square views concatenated horizontally; "
                f"received panel {height}x{width}"
            )
        if value.dtype != torch.uint8:
            if value.is_floating_point() and value.numel():
                lo, hi = float(value.min()), float(value.max())
                if lo < 0.0 or hi > 255.0:
                    raise ValueError(
                        f"Floating RGB payload is outside [0,255]: [{lo},{hi}]"
                    )
            value = value.clamp(0, 255).to(torch.uint8)
        return value.contiguous()

    value = np.asarray(video)
    if value.ndim != 5:
        raise ValueError(f"Expected panel video [B,T,H,W,C], got {value.shape}")
    if value.shape[1] != expected_frames or value.shape[-1] != 3:
        raise ValueError(f"Expected T={expected_frames} and C=3, got {value.shape}")
    height, width = int(value.shape[2]), int(value.shape[3])
    if width != len(VIEW_ORDER) * height:
        raise ValueError(
            "Expected three square views concatenated horizontally; "
            f"received panel {height}x{width}"
        )
    if value.dtype != np.uint8:
        if np.issubdtype(value.dtype, np.floating) and value.size:
            lo, hi = float(value.min()), float(value.max())
            if lo < 0.0 or hi > 255.0:
                raise ValueError(
                    f"Floating RGB payload is outside [0,255]: [{lo},{hi}]"
                )
        value = np.clip(value, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def panel_video_to_views(
    video: Any,
    *,
    expected_frames: int = ACTION_HORIZON,
) -> np.ndarray | torch.Tensor:
    """Convert ``[B,T,H,3W,3]`` to ``[B,T,V,C,H,W]`` in fixed view order."""

    panels = validate_panel_video(video, expected_frames=expected_frames)
    view_width = panels.shape[2]
    views = [
        panels[:, :, :, i * view_width : (i + 1) * view_width, :]
        for i in range(len(VIEW_ORDER))
    ]
    if torch.is_tensor(panels):
        stacked = torch.stack(views, dim=2)  # [B,T,V,H,W,C]
        return stacked.permute(0, 1, 2, 5, 3, 4).contiguous()
    stacked = np.stack(views, axis=2)  # [B,T,V,H,W,C]
    return np.ascontiguousarray(stacked.transpose(0, 1, 2, 5, 3, 4))


def stack_policy_observations(
    observations: Sequence[dict[str, Any]],
    *,
    prompt: str,
    policy_chunk_index: int | None = None,
) -> dict[str, Any]:
    """Stack WM CHW observations into the standardized RLinf policy payload."""

    if not observations:
        raise ValueError("WM response contained no observations")

    images: list[np.ndarray] = []
    for key in WM_IMAGE_KEYS:
        per_env = []
        for index, observation in enumerate(observations):
            if key not in observation:
                raise KeyError(f"Observation {index} is missing {key!r}")
            image = np.asarray(observation[key], dtype=np.uint8)
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"{key} must be CHW uint8, got {image.shape}")
            per_env.append(image)
        images.append(np.ascontiguousarray(np.stack(per_env, axis=0)))

    states = []
    tactiles = []
    tactile_masks = []
    for index, observation in enumerate(observations):
        state = np.asarray(observation.get("observation.state"), dtype=np.float32)
        if state.shape != (ACTION_DIM,):
            raise ValueError(
                f"Observation {index} state must have shape ({ACTION_DIM},), got {state.shape}"
            )
        states.append(state)
        tactile = np.asarray(observation.get("observation.tactile"), dtype=np.float32)
        if tactile.shape != TACTILE_SHAPE:
            raise ValueError(
                f"Observation {index} tactile must have shape {TACTILE_SHAPE}, "
                f"got {tactile.shape}"
            )
        tactile_mask = np.asarray(
            observation.get("observation.tactile_mask"), dtype=np.bool_
        )
        if tactile_mask.shape != (TACTILE_SHAPE[0],):
            raise ValueError(
                f"Observation {index} tactile_mask must have shape "
                f"({TACTILE_SHAPE[0]},), got {tactile_mask.shape}"
            )
        if not np.isfinite(tactile).all():
            raise ValueError(f"Observation {index} tactile contains NaN or Inf")
        tactiles.append(tactile)
        tactile_masks.append(tactile_mask)

    batch_size = len(observations)
    result = {
        # Rollout workers merge observations from multiple EnvWorkers with
        # torch.cat. Returning tensors here preserves the batch dimension;
        # NumPy arrays would be wrapped in a Python list by the generic merger.
        "main_images": torch.from_numpy(images[0]),
        "wrist_images": torch.from_numpy(images[1]),
        "extra_view_images": torch.from_numpy(images[2]),
        "states": torch.from_numpy(
            np.ascontiguousarray(np.stack(states, axis=0))
        ),
        "tactiles": torch.from_numpy(
            np.ascontiguousarray(np.stack(tactiles, axis=0))
        ),
        "tactile_masks": torch.from_numpy(
            np.ascontiguousarray(np.stack(tactile_masks, axis=0))
        ),
        "task_descriptions": [prompt] * batch_size,
    }
    if policy_chunk_index is not None:
        if policy_chunk_index < 0:
            raise ValueError("policy_chunk_index must be non-negative")
        result["policy_chunk_indices"] = torch.full(
            (batch_size,), int(policy_chunk_index), dtype=torch.long
        )
    return result


def validate_group_layout(num_envs: int, group_size: int) -> int:
    if num_envs <= 0 or group_size <= 1:
        raise ValueError(
            "num_envs must be positive and GRPO group_size must exceed one"
        )
    if num_envs % group_size:
        raise ValueError(
            f"num_envs={num_envs} is not divisible by group_size={group_size}"
        )
    return num_envs // group_size
