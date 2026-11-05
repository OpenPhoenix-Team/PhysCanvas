from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from convert_arx2_packing_phone_to_lerobot_v21_10hz_codex import CAMERAS


MASTER_CAMERA = "base_0"
DEFAULT_ALIGNMENT_SLOP_MS = 50.0
DEFAULT_TRIM_WINDOW = 15
DEFAULT_TRIM_MIN_ACTIVE_RUN = 10
DEFAULT_TRIM_MARGIN = 10
DEFAULT_TRIM_MIN_LENGTH = 60


@dataclass(frozen=True)
class ArmMotion:
    arm: str
    found: bool
    first_active_diff: int | None
    last_active_diff: int | None
    threshold: float | None
    baseline: float | None


@dataclass(frozen=True)
class TaskAwareTrim:
    original_length: int
    trim_start: int
    trim_end: int
    dropped_start: int
    dropped_end: int
    trimmed_length: int
    reason: str
    arm_mode: str
    left: ArmMotion
    right: ArmMotion


@dataclass(frozen=True)
class EpisodePlan:
    episode_dir: Path
    timestamps_ms: np.ndarray
    camera_indices: dict[str, np.ndarray]
    states: np.ndarray
    actions: np.ndarray
    dropped_robot_samples: dict[str, int]
    swap_robot_sides: bool
    arm_mode: str
    trim: TaskAwareTrim
    pre_crop_start_ms: int
    pre_crop_end_ms: int
    trim_start_ms: int
    trim_end_ms: int
    high_rate_timestamps_ms: np.ndarray


def nearest_indices(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps for nearest-neighbor alignment")
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_left = np.abs(targets - timestamps[left]) <= np.abs(timestamps[right] - targets)
    return np.where(choose_left, left, right).astype(np.int64)


def swap_robot_side_dicts(robot_raw: dict[str, Any], enabled: bool) -> dict[str, Any]:
    if not enabled:
        return robot_raw
    if "left" not in robot_raw or "right" not in robot_raw:
        raise KeyError("robot_state_dict.pkl does not contain both left and right")
    fixed = dict(robot_raw)
    fixed["left"] = copy.deepcopy(robot_raw["right"])
    fixed["right"] = copy.deepcopy(robot_raw["left"])
    return fixed


def clean_joint_stream(robot_side: dict[str, Any], episode_dir: Path, side: str) -> tuple[np.ndarray, np.ndarray, int]:
    joints = robot_side["joint"]
    timestamps = robot_side["timestamps"]
    clean_joints: list[np.ndarray] = []
    clean_timestamps: list[int] = []
    dropped = 0
    for joint, timestamp in zip(joints, timestamps, strict=False):
        arr = np.asarray(joint, dtype=np.float32) if joint is not None else None
        if arr is None or arr.shape != (7,) or not np.all(np.isfinite(arr)):
            dropped += 1
            continue
        clean_joints.append(arr)
        clean_timestamps.append(int(timestamp))
    if len(clean_joints) < 2:
        raise ValueError(f"{episode_dir}: {side} has only {len(clean_joints)} valid joint samples")
    ts = np.asarray(clean_timestamps, dtype=np.int64)
    # Some validated ARX recordings contain adjacent samples with the same
    # millisecond timestamp. searchsorted-based nearest alignment supports
    # non-decreasing timestamps; only a true time reversal is invalid.
    if np.any(np.diff(ts) < 0):
        raise ValueError(f"{episode_dir}: {side} robot timestamps move backwards")
    return np.stack(clean_joints).astype(np.float32), ts, dropped


def _first_run(active: np.ndarray, run: int) -> int | None:
    if len(active) < run:
        return None
    hits = np.convolve(active.astype(np.int32), np.ones(run, dtype=np.int32), mode="valid") >= run
    indices = np.flatnonzero(hits)
    return None if len(indices) == 0 else int(indices[0])


def _last_run(active: np.ndarray, run: int) -> int | None:
    if len(active) < run:
        return None
    hits = np.convolve(active.astype(np.int32), np.ones(run, dtype=np.int32), mode="valid") >= run
    indices = np.flatnonzero(hits)
    return None if len(indices) == 0 else int(indices[-1] + run - 1)


def detect_arm_motion(state: np.ndarray, arm: str, window: int, min_active_run: int) -> ArmMotion:
    if len(state) < min_active_run + 2:
        return ArmMotion(arm, False, None, None, None, None)
    diff = np.abs(np.diff(np.asarray(state, dtype=np.float32), axis=0))
    scale = np.maximum(np.percentile(diff, 95, axis=0), 1e-4)
    motion = np.mean(diff / scale, axis=1)
    window = max(1, int(window))
    if window > 1:
        motion = np.convolve(motion, np.ones(window, dtype=np.float32) / float(window), mode="same")
    baseline = float(np.percentile(motion, 15))
    threshold = max(0.05, baseline * 3.0)
    active = motion > threshold
    first = _first_run(active, int(min_active_run))
    last = _last_run(active, int(min_active_run))
    return ArmMotion(arm, first is not None and last is not None, first, last, threshold, baseline)


def compute_task_aware_trim_bounds(
    state: np.ndarray,
    *,
    arm_mode: str,
    window: int = DEFAULT_TRIM_WINDOW,
    min_active_run: int = DEFAULT_TRIM_MIN_ACTIVE_RUN,
    margin: int = DEFAULT_TRIM_MARGIN,
    min_length: int = DEFAULT_TRIM_MIN_LENGTH,
) -> TaskAwareTrim:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 14:
        raise ValueError(f"Expected [T,14] state, got {state.shape}")
    if arm_mode not in {"left", "right", "bimanual"}:
        raise ValueError(f"Unsupported arm_mode={arm_mode}")
    n = int(len(state))
    left_motion = detect_arm_motion(state[:, :7], "left", window, min_active_run)
    right_motion = detect_arm_motion(state[:, 7:], "right", window, min_active_run)

    if n < max(int(min_length), int(min_active_run) + 2):
        return TaskAwareTrim(n, 0, n, 0, 0, n, "too_short", arm_mode, left_motion, right_motion)

    selected: list[ArmMotion]
    if arm_mode == "left":
        selected = [left_motion]
    elif arm_mode == "right":
        selected = [right_motion]
    else:
        selected = [left_motion, right_motion]
    active = [item for item in selected if item.found]
    if not active:
        return TaskAwareTrim(n, 0, n, 0, 0, n, "no_active_run", arm_mode, left_motion, right_motion)

    first = min(int(item.first_active_diff) for item in active if item.first_active_diff is not None)
    last = max(int(item.last_active_diff) for item in active if item.last_active_diff is not None)
    start = max(0, first - int(margin))
    end = min(n, last + 2 + int(margin))
    if end - start < int(min_length):
        return TaskAwareTrim(n, 0, n, 0, 0, n, "trim_too_short", arm_mode, left_motion, right_motion)
    reason = "trimmed" if start > 0 or end < n else "unchanged"
    return TaskAwareTrim(n, start, end, start, n - end, end - start, reason, arm_mode, left_motion, right_motion)


def _load_camera_timestamps(episode_dir: Path) -> dict[str, np.ndarray]:
    with (episode_dir / "camera" / "timestamp.pkl").open("rb") as stream:
        raw = pickle.load(stream)
    result: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        if camera not in raw:
            raise KeyError(f"{episode_dir}: missing camera timestamps for {camera}")
        timestamps = np.asarray(raw[camera], dtype=np.int64)
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{episode_dir}: invalid {camera} timestamps")
        result[camera] = timestamps
    return result


def _filter_alignment_by_slop(
    targets: np.ndarray,
    camera_ts: dict[str, np.ndarray],
    left_ts: np.ndarray,
    right_ts: np.ndarray,
    slop_ms: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    camera_indices = {camera: nearest_indices(camera_ts[camera], targets) for camera in CAMERAS}
    left_indices = nearest_indices(left_ts, targets)
    right_indices = nearest_indices(right_ts, targets)
    skews = [np.abs(camera_ts[camera][camera_indices[camera]] - targets) for camera in CAMERAS]
    skews.extend([np.abs(left_ts[left_indices] - targets), np.abs(right_ts[right_indices] - targets)])
    keep = np.max(np.stack(skews, axis=0), axis=0) <= float(slop_ms)
    kept_targets = targets[keep]
    if len(kept_targets) < 3:
        raise ValueError(f"too few aligned frames after {slop_ms} ms slop filtering ({len(kept_targets)})")
    return (
        kept_targets,
        {camera: indices[keep] for camera, indices in camera_indices.items()},
        left_indices[keep],
        right_indices[keep],
    )


def make_episode_plan(
    episode_dir: Path,
    *,
    target_hz: int,
    swap_robot_sides: bool,
    arm_mode: str,
    alignment_slop_ms: float = DEFAULT_ALIGNMENT_SLOP_MS,
    trim_window: int = DEFAULT_TRIM_WINDOW,
    trim_min_active_run: int = DEFAULT_TRIM_MIN_ACTIVE_RUN,
    trim_margin: int = DEFAULT_TRIM_MARGIN,
    trim_min_length: int = DEFAULT_TRIM_MIN_LENGTH,
) -> EpisodePlan:
    episode_dir = Path(episode_dir)
    camera_ts = _load_camera_timestamps(episode_dir)
    with (episode_dir / "robot" / "robot_state_dict.pkl").open("rb") as stream:
        robot_raw = pickle.load(stream)
    robot = swap_robot_side_dicts(robot_raw, swap_robot_sides)
    left_joint, left_ts, left_dropped = clean_joint_stream(robot["left"], episode_dir, "left")
    right_joint, right_ts, right_dropped = clean_joint_stream(robot["right"], episode_dir, "right")

    common_start = max(
        *(int(camera_ts[camera][0]) for camera in CAMERAS),
        int(left_ts[0]),
        int(right_ts[0]),
    )
    common_end = min(
        *(int(camera_ts[camera][-1]) for camera in CAMERAS),
        int(left_ts[-1]),
        int(right_ts[-1]),
    )
    if common_end <= common_start:
        raise ValueError(f"{episode_dir}: no common robot/camera time window")

    master = camera_ts[MASTER_CAMERA]
    high_targets = master[(master >= common_start) & (master <= common_end)]
    high_targets, _, high_left_idx, high_right_idx = _filter_alignment_by_slop(
        high_targets, camera_ts, left_ts, right_ts, alignment_slop_ms
    )
    high_state = np.concatenate([left_joint[high_left_idx], right_joint[high_right_idx]], axis=1).astype(np.float32)
    trim = compute_task_aware_trim_bounds(
        high_state,
        arm_mode=arm_mode,
        window=trim_window,
        min_active_run=trim_min_active_run,
        margin=trim_margin,
        min_length=trim_min_length,
    )
    trimmed_high_ts = high_targets[trim.trim_start : trim.trim_end]
    if len(trimmed_high_ts) < 3:
        raise ValueError(f"{episode_dir}: too few frames after task-aware trim")

    step_ms = int(round(1000 / int(target_hz)))
    regular_targets = np.arange(
        int(trimmed_high_ts[0]), int(trimmed_high_ts[-1]) + 1, step_ms, dtype=np.int64
    )
    output_ts, camera_indices, left_idx, right_idx = _filter_alignment_by_slop(
        regular_targets, camera_ts, left_ts, right_ts, alignment_slop_ms
    )
    aligned_states = np.concatenate([left_joint[left_idx], right_joint[right_idx]], axis=1).astype(np.float32)
    if len(aligned_states) < 3:
        raise ValueError(f"{episode_dir}: too few aligned {target_hz} Hz frames")

    return EpisodePlan(
        episode_dir=episode_dir,
        timestamps_ms=output_ts[:-1],
        camera_indices={camera: indices[:-1] for camera, indices in camera_indices.items()},
        states=aligned_states[:-1],
        actions=aligned_states[1:],
        dropped_robot_samples={"left": left_dropped, "right": right_dropped},
        swap_robot_sides=swap_robot_sides,
        arm_mode=arm_mode,
        trim=trim,
        pre_crop_start_ms=int(high_targets[0]),
        pre_crop_end_ms=int(high_targets[-1]),
        trim_start_ms=int(trimmed_high_ts[0]),
        trim_end_ms=int(trimmed_high_ts[-1]),
        high_rate_timestamps_ms=high_targets,
    )


def trim_to_dict(trim: TaskAwareTrim) -> dict[str, Any]:
    def arm_dict(item: ArmMotion) -> dict[str, Any]:
        return {
            "found": item.found,
            "first_active_diff": item.first_active_diff,
            "last_active_diff": item.last_active_diff,
            "threshold": item.threshold,
            "baseline": item.baseline,
        }

    return {
        "algorithm": "PPU-4 convert_arx_to_lerobot21.py task-aware per-arm",
        "original_length": trim.original_length,
        "trim_start": trim.trim_start,
        "trim_end": trim.trim_end,
        "dropped_start": trim.dropped_start,
        "dropped_end": trim.dropped_end,
        "trimmed_length": trim.trimmed_length,
        "reason": trim.reason,
        "arm_mode": trim.arm_mode,
        "left": arm_dict(trim.left),
        "right": arm_dict(trim.right),
        "window": DEFAULT_TRIM_WINDOW,
        "min_active_run": DEFAULT_TRIM_MIN_ACTIVE_RUN,
        "margin": DEFAULT_TRIM_MARGIN,
        "min_length": DEFAULT_TRIM_MIN_LENGTH,
    }
