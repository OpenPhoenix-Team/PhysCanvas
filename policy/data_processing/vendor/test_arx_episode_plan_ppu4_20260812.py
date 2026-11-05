from __future__ import annotations

import copy

import numpy as np

from arx_episode_plan_ppu4_20260812 import compute_task_aware_trim_bounds
from arx_episode_plan_ppu4_20260812 import clean_joint_stream
from arx_episode_plan_ppu4_20260812 import swap_robot_side_dicts


def _ramp(length: int, start: int, end: int, scale: float = 1.0) -> np.ndarray:
    values = np.zeros((length, 7), dtype=np.float32)
    values[start:end, 0] = np.linspace(0.0, scale, end - start, dtype=np.float32)
    values[end:, 0] = scale
    return values


def test_right_only_uses_true_right_arm() -> None:
    length = 240
    state = np.concatenate([np.zeros((length, 7), np.float32), _ramp(length, 70, 170)], axis=1)
    trim = compute_task_aware_trim_bounds(state, arm_mode="right")
    assert trim.right.found
    assert not trim.left.found
    assert 0 < trim.trim_start < 100
    assert 150 < trim.trim_end < length


def test_bimanual_uses_union() -> None:
    length = 300
    left = _ramp(length, 45, 145)
    right = _ramp(length, 125, 245)
    state = np.concatenate([left, right], axis=1)
    trim = compute_task_aware_trim_bounds(state, arm_mode="bimanual")
    assert trim.left.found and trim.right.found
    assert trim.trim_start < 70
    assert trim.trim_end > 220


def test_bimanual_ignores_inactive_arm() -> None:
    length = 240
    state = np.concatenate([np.zeros((length, 7), np.float32), _ramp(length, 70, 170)], axis=1)
    trim = compute_task_aware_trim_bounds(state, arm_mode="bimanual")
    assert not trim.left.found and trim.right.found
    assert trim.reason == "trimmed"


def test_swap_rebinds_complete_side_dict_without_aliasing() -> None:
    raw = {
        "left": {"joint": [[1.0]], "eef": [[2.0]], "timestamps": [10]},
        "right": {"joint": [[3.0]], "eef": [[4.0]], "timestamps": [20]},
    }
    original = copy.deepcopy(raw)
    fixed = swap_robot_side_dicts(raw, True)
    assert fixed["left"] == original["right"]
    assert fixed["right"] == original["left"]
    fixed["left"]["joint"][0][0] = 999.0
    assert raw == original


def test_no_swap_returns_original_mapping() -> None:
    raw = {"left": {"joint": [1]}, "right": {"joint": [2]}}
    assert swap_robot_side_dicts(raw, False) is raw


def test_duplicate_robot_timestamps_are_allowed(tmp_path) -> None:
    side = {
        "joint": [np.zeros(7, np.float32), np.ones(7, np.float32), np.full(7, 2.0, np.float32)],
        "timestamps": [1000, 1000, 1033],
    }
    joints, timestamps, dropped = clean_joint_stream(side, tmp_path, "right")
    assert joints.shape == (3, 7)
    assert timestamps.tolist() == [1000, 1000, 1033]
    assert dropped == 0
