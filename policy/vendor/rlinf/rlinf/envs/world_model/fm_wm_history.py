# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""History conversion for the 30 Hz FM policy and 10 Hz Jimu WM."""

from __future__ import annotations

import numpy as np
import torch


FM_HISTORY_STEPS = 8
FM_EXECUTED_STEPS = 48
FM_DOWNSAMPLE_RATIO = 3
FM_STATE_DIM = 14
FM_TACTILE_SHAPE = (35, 20, 12)
WM_TACTILE_SHAPE = (4, 700, 3)


def validate_fm_reset_history(
    state_history: np.ndarray | torch.Tensor,
    tactile_history: np.ndarray | torch.Tensor,
) -> None:
    """Require real, aligned eight-frame reset histories for FM rollouts.

    This intentionally refuses to synthesize an FM history by repeating a
    single Pi0.5 reset frame. The policy was trained with temporal state and
    tactile context, so silently fabricating those inputs changes its policy.
    """

    if tuple(state_history.shape) != (FM_HISTORY_STEPS, FM_STATE_DIM):
        raise ValueError(
            "FM reset state history must be [8,14], "
            f"got {tuple(state_history.shape)}"
        )
    if tuple(tactile_history.shape) != (FM_HISTORY_STEPS, *FM_TACTILE_SHAPE):
        raise ValueError(
            "FM reset tactile history must be [8,35,20,12], "
            f"got {tuple(tactile_history.shape)}"
        )


def wm_tactile_to_fm_grid(
    tactile: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Convert four 35x20 tactile sensors to FM's channel-last grid.

    Input is ``[...,4,700,3]`` in sensor-major order. Output is
    ``[...,35,20,12]`` with the four three-channel sensors concatenated in
    the final dimension.
    """

    if tuple(tactile.shape[-3:]) != WM_TACTILE_SHAPE:
        raise ValueError(
            "WM tactile must end in [4,700,3], "
            f"got {tuple(tactile.shape)}"
        )
    prefix = tuple(tactile.shape[:-3])
    if torch.is_tensor(tactile):
        return (
            tactile.reshape(*prefix, 4, 35, 20, 3)
            .movedim(-4, -2)
            .reshape(*prefix, *FM_TACTILE_SHAPE)
            .contiguous()
        )
    return np.ascontiguousarray(
        np.moveaxis(
            np.asarray(tactile).reshape(*prefix, 4, 35, 20, 3), -4, -2
        ).reshape(*prefix, *FM_TACTILE_SHAPE)
    )


def interpolate_wm_tactile_10hz_to_30hz(
    previous: np.ndarray | torch.Tensor,
    anchors: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Interpolate 16 future 10 Hz anchors into 48 future 30 Hz frames.

    ``previous`` is the tactile observation at time zero and each item in
    ``anchors`` is the endpoint of the next 100 ms WM interval. Samples at
    one-third, two-thirds and the endpoint are produced for every interval.
    """

    if tuple(previous.shape) != FM_TACTILE_SHAPE:
        raise ValueError(
            f"previous tactile must be {FM_TACTILE_SHAPE}, "
            f"got {tuple(previous.shape)}"
        )
    if anchors.ndim != 4 or tuple(anchors.shape[1:]) != FM_TACTILE_SHAPE:
        raise ValueError(
            "anchors must be [T,35,20,12], "
            f"got {tuple(anchors.shape)}"
        )
    if type(previous) is not type(anchors):
        raise TypeError("previous and anchors must use the same array type")

    if torch.is_tensor(anchors):
        start = torch.cat([previous.unsqueeze(0), anchors[:-1]], dim=0)
        alpha = torch.tensor(
            (1.0 / 3.0, 2.0 / 3.0, 1.0),
            device=anchors.device,
            dtype=anchors.dtype,
        ).reshape(1, 3, 1, 1, 1)
        frames = start[:, None] + (anchors - start)[:, None] * alpha
        return frames.reshape(-1, *FM_TACTILE_SHAPE).contiguous()

    start = np.concatenate([previous[None], anchors[:-1]], axis=0)
    alpha = np.asarray(
        (1.0 / 3.0, 2.0 / 3.0, 1.0), dtype=anchors.dtype
    ).reshape(1, 3, 1, 1, 1)
    frames = start[:, None] + (anchors - start)[:, None] * alpha
    return np.ascontiguousarray(frames.reshape(-1, *FM_TACTILE_SHAPE))


def next_fm_histories(
    executed_actions: np.ndarray | torch.Tensor,
    previous_tactile: np.ndarray | torch.Tensor,
    wm_tactile_anchors: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Build the next FM state/tactile histories after one 1.6 s plan."""

    if executed_actions.ndim != 2 or tuple(executed_actions.shape) != (48, 14):
        raise ValueError(
            "executed actions must be the full [48,14] 30 Hz trajectory, "
            f"got {tuple(executed_actions.shape)}"
        )
    tactile_30hz = interpolate_wm_tactile_10hz_to_30hz(
        previous_tactile, wm_tactile_anchors
    )
    if int(tactile_30hz.shape[0]) != int(executed_actions.shape[0]):
        raise ValueError(
            "FM state and tactile timebases differ: "
            f"state={executed_actions.shape[0]} tactile={tactile_30hz.shape[0]}"
        )
    return (
        executed_actions[-FM_HISTORY_STEPS:].copy()
        if isinstance(executed_actions, np.ndarray)
        else executed_actions[-FM_HISTORY_STEPS:].clone(),
        tactile_30hz[-FM_HISTORY_STEPS:],
    )


def select_wm_action_endpoints(
    executed_actions: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Map ``[B,48,14]`` at 30 Hz to endpoints ``[B,16,14]`` at 10 Hz."""

    expected = (FM_EXECUTED_STEPS, FM_STATE_DIM)
    if executed_actions.ndim != 3 or tuple(executed_actions.shape[1:]) != expected:
        raise ValueError(
            f"FM executed actions must be [B,{expected[0]},{expected[1]}], "
            f"got {tuple(executed_actions.shape)}"
        )
    if torch.is_tensor(executed_actions):
        indices = torch.arange(
            FM_DOWNSAMPLE_RATIO - 1,
            FM_EXECUTED_STEPS,
            FM_DOWNSAMPLE_RATIO,
            device=executed_actions.device,
        )
        return executed_actions.index_select(1, indices).contiguous()
    result = np.asarray(executed_actions)[
        :, FM_DOWNSAMPLE_RATIO - 1 : FM_EXECUTED_STEPS : FM_DOWNSAMPLE_RATIO
    ]
    return np.ascontiguousarray(result)
