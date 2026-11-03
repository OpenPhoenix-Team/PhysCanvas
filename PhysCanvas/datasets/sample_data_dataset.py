import copy
import json
import os
import pickle
from typing import Any, Sequence

import cv2
import numpy as np
from giga_datasets import BaseDataset, register_dataset


CAMERA_SPECS = {
    "high": "base_0",
    "left_wrist": "left_wrist_0",
    "right_wrist": "right_wrist_0",
}


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def _nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    if source_timestamps.size == 0:
        return np.zeros((target_timestamps.shape[0],), dtype=np.int64)
    order = np.argsort(source_timestamps)
    sorted_ts = source_timestamps[order]
    pos = np.searchsorted(sorted_ts, target_timestamps)
    left = np.clip(pos - 1, 0, len(sorted_ts) - 1)
    right = np.clip(pos, 0, len(sorted_ts) - 1)
    choose_right = np.abs(sorted_ts[right] - target_timestamps) < np.abs(sorted_ts[left] - target_timestamps)
    nearest = np.where(choose_right, right, left)
    return order[nearest].astype(np.int64)


def _estimate_hz(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 2:
        return 0.0
    delta = np.diff(timestamps)
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if delta.size == 0:
        return 0.0
    return float(1000.0 / np.median(delta))


def _coerce_2d_sequence(value: Any, name: str, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if width is None:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"{name} must be rank-2, got {arr.shape}")
        return arr, np.all(np.isfinite(arr), axis=1)
    out = np.full((len(value), width), np.nan, dtype=np.float32)
    valid = np.zeros((len(value),), dtype=bool)
    for idx, item in enumerate(value):
        if item is None:
            continue
        arr = np.asarray(item, dtype=np.float32).reshape(-1)
        if arr.shape[0] < width or not np.all(np.isfinite(arr[:width])):
            continue
        out[idx] = arr[:width]
        valid[idx] = True
    if not np.any(valid):
        raise RuntimeError(f"No valid rows found in {name}")
    return out, valid


def _extract_arm_state(robot_state: dict, arm_name: str) -> tuple[np.ndarray, np.ndarray]:
    arm = robot_state.get(arm_name, {})
    joint = np.asarray(arm.get("joint", []), dtype=np.float32)
    timestamps = np.asarray(arm.get("timestamps", np.arange(len(joint))), dtype=np.float64)
    if joint.ndim != 2:
        raise RuntimeError(f"robot/{arm_name}/joint must be rank-2, got {joint.shape}")
    if timestamps.ndim != 1 or timestamps.shape[0] != joint.shape[0]:
        timestamps = np.arange(joint.shape[0], dtype=np.float64)
    return joint, timestamps


def _choose_target_timestamps(
    camera_timestamps: dict[str, np.ndarray],
    left_ts: np.ndarray,
    right_ts: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    tactile_timestamps: dict[str, np.ndarray] | None = None,
    target_hz: float = 10.0,
) -> np.ndarray:
    starts = [int(ts[0]) for ts in camera_timestamps.values() if len(ts) > 0]
    ends = [int(ts[-1]) for ts in camera_timestamps.values() if len(ts) > 0]
    if tactile_timestamps:
        starts.extend(int(ts[0]) for ts in tactile_timestamps.values() if len(ts) > 0)
        ends.extend(int(ts[-1]) for ts in tactile_timestamps.values() if len(ts) > 0)
    left_idx = np.flatnonzero(left_valid)
    right_idx = np.flatnonzero(right_valid)
    if not starts or not ends or left_idx.size == 0 or right_idx.size == 0:
        return np.zeros((0,), dtype=np.int64)
    start = max(max(starts), int(left_ts[left_idx[0]]), int(right_ts[right_idx[0]]))
    end = min(min(ends), int(left_ts[left_idx[-1]]), int(right_ts[right_idx[-1]]))
    if end <= start:
        return np.zeros((0,), dtype=np.int64)
    step_ms = max(1, int(round(1000.0 / float(target_hz))))
    return np.arange(start, end + 1, step_ms, dtype=np.int64)


def _build_aligned_episode(
    sample_dir: str,
    cameras: dict[str, str],
    max_frames: int | None = None,
    target_hz: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    robot_state = _load_pickle(os.path.join(sample_dir, "robot", "robot_state_dict.pkl"))
    camera_timestamps = _load_pickle(os.path.join(sample_dir, "camera", "timestamp.pkl"))
    tactile_path = os.path.join(sample_dir, "tactile", "timestamp.pkl")
    tactile_timestamps_raw = _load_pickle(tactile_path) if os.path.exists(tactile_path) else {}

    left_joint, left_joint_valid = _coerce_2d_sequence(robot_state["left"]["joint"], "left.joint", 7)
    right_joint, right_joint_valid = _coerce_2d_sequence(robot_state["right"]["joint"], "right.joint", 7)
    left_eef, left_eef_valid = _coerce_2d_sequence(robot_state["left"].get("eef", []), "left.eef", 7)
    right_eef, right_eef_valid = _coerce_2d_sequence(robot_state["right"].get("eef", []), "right.eef", 7)
    left_ts = np.asarray(robot_state["left"].get("timestamps", np.arange(len(left_joint))), dtype=np.int64)
    right_ts = np.asarray(robot_state["right"].get("timestamps", np.arange(len(right_joint))), dtype=np.int64)
    left_valid = left_joint_valid & left_eef_valid
    right_valid = right_joint_valid & right_eef_valid

    camera_ts = {}
    for view_key, camera_name in cameras.items():
        ts = np.asarray(camera_timestamps.get(camera_name, []), dtype=np.int64)
        camera_ts[view_key] = ts
    tactile_ts = {
        name: np.asarray(ts, dtype=np.int64)
        for name, ts in tactile_timestamps_raw.items()
        if len(ts) > 0
    }
    target_ts = _choose_target_timestamps(
        {camera_name: camera_ts[view_key] for view_key, camera_name in cameras.items()},
        left_ts,
        right_ts,
        left_valid,
        right_valid,
        tactile_timestamps=tactile_ts,
        target_hz=target_hz,
    )
    if max_frames is not None and max_frames > 0:
        target_ts = target_ts[:max_frames]
    if target_ts.size == 0:
        empty = np.zeros((0, 14), dtype=np.float32)
        return (
            empty,
            empty,
            target_ts,
            {k: np.zeros((0,), dtype=np.int64) for k in cameras},
            {k: np.zeros((0,), dtype=np.int64) for k in tactile_ts},
            {},
        )

    left_idx = _nearest_indices(left_ts[np.flatnonzero(left_valid)], target_ts)
    left_idx = np.flatnonzero(left_valid)[left_idx]
    right_idx = _nearest_indices(right_ts[np.flatnonzero(right_valid)], target_ts)
    right_idx = np.flatnonzero(right_valid)[right_idx]
    qpos = np.concatenate([left_joint[left_idx], right_joint[right_idx]], axis=1).astype(np.float32)
    eef_pose = np.concatenate([left_eef[left_idx], right_eef[right_idx]], axis=1).astype(np.float32)
    video_indices = {
        view_key: _nearest_indices(np.asarray(camera_ts[view_key], dtype=np.int64), target_ts)
        for view_key in cameras
    }
    tactile_indices = {
        name: _nearest_indices(np.asarray(ts, dtype=np.int64), target_ts)
        for name, ts in tactile_ts.items()
    }
    alignment_stats = {}
    alignment_stats["target_hz"] = float(target_hz)
    alignment_stats["target_step_ms"] = float(np.median(np.diff(target_ts))) if target_ts.size > 1 else 0.0
    for view_key in cameras:
        ts = camera_ts[view_key][video_indices[view_key]]
        alignment_stats[f"{view_key}_max_image_delta_ms"] = float(np.max(np.abs(ts - target_ts))) if len(ts) else 0.0
        alignment_stats[f"{view_key}_source_hz"] = _estimate_hz(camera_ts[view_key])
    for name, indices in tactile_indices.items():
        ts = tactile_ts[name][indices]
        alignment_stats[f"tactile_{name}_max_delta_ms"] = float(np.max(np.abs(ts - target_ts))) if len(ts) else 0.0
        alignment_stats[f"tactile_{name}_source_hz"] = _estimate_hz(tactile_ts[name])
    alignment_stats["left_max_robot_delta_ms"] = float(np.max(np.abs(left_ts[left_idx] - target_ts)))
    alignment_stats["right_max_robot_delta_ms"] = float(np.max(np.abs(right_ts[right_idx] - target_ts)))
    alignment_stats["left_action_source_hz"] = _estimate_hz(left_ts)
    alignment_stats["right_action_source_hz"] = _estimate_hz(right_ts)
    return qpos, eef_pose, target_ts, video_indices, tactile_indices, alignment_stats


def _normalise_roots(sample_root: str | Sequence[str] | None, sample_roots: Sequence[str] | None = None) -> list[str]:
    roots: list[str] = []
    if sample_roots:
        roots.extend(str(x) for x in sample_roots)
    if sample_root:
        if isinstance(sample_root, (list, tuple)):
            roots.extend(str(x) for x in sample_root)
        else:
            roots.extend(str(x) for x in str(sample_root).split(os.pathsep) if str(x).strip())
    return [os.path.abspath(os.path.expanduser(x)) for x in roots if x]


def _missing_required_files(sample_dir: str, cameras: dict[str, str]) -> list[str]:
    camera_dir = os.path.join(sample_dir, "camera")
    required = [
        os.path.join(sample_dir, "robot", "robot_state_dict.pkl"),
        os.path.join(camera_dir, "timestamp.pkl"),
        os.path.join(sample_dir, "tactile", "timestamp.pkl"),
    ]
    for camera_name in cameras.values():
        required.extend(
            [
                os.path.join(camera_dir, f"{camera_name}_rgb.mp4"),
                os.path.join(camera_dir, f"{camera_name}_ir_left.mkv"),
                os.path.join(camera_dir, f"{camera_name}_ir_right.mkv"),
            ]
        )
    return [path for path in required if not os.path.exists(path)]


@register_dataset
class SampleDataDataset(BaseDataset):
    """Dataset that reads the user's raw sample_data episodes directly.

    Each item is an episode-level dict consumed by WMRNDMixSampleFastFSTransforms.
    Depth is intentionally not materialized here; the transform asks FastFS for
    only the sampled frames and caches the resulting metric depth arrays.
    """

    def __init__(
        self,
        sample_root: str | Sequence[str] | None = None,
        sample_roots: Sequence[str] | None = None,
        params_path: str = "",
        fastfs_dir: str = "",
        fastfs_model_path: str = "",
        depth_cache_dir: str = "",
        motion_cache_dir: str | None = None,
        arx_urdf_path: str | None = None,
        arx_mesh_dir: str | None = None,
        motion_mode: str = "arx_fk",
        motion_arm_sample_count: int = 1200,
        motion_use_cache: bool = True,
        data_size: int | None = None,
        max_episodes: int | None = None,
        max_frames: int | None = None,
        expand_clips: bool = False,
        clip_num_frames: int = 33,
        clip_max_stride: int = 4,
        clip_start_stride: int = 32,
        clip_include_all_strides: bool = False,
        replay_mode: str = "rgb",
        cameras: dict[str, str] | None = None,
        valid_iters: int = 8,
        max_disp: int = 192,
        build_volume: str = "pytorch1",
        zmax_m: float = 5.0,
        target_hz: float = 10.0,
        depth_source: str = "foundation_stereo",
        foundation_stereo_dir: str = "",
        foundation_stereo_ckpt_path: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_roots = _normalise_roots(sample_root, sample_roots)
        if not self.sample_roots:
            raise ValueError("SampleDataDataset requires sample_root or sample_roots")
        self.params_path = params_path
        self.fastfs_dir = fastfs_dir
        self.fastfs_model_path = fastfs_model_path
        self.depth_cache_dir = depth_cache_dir
        self.motion_cache_dir = motion_cache_dir
        self.arx_urdf_path = arx_urdf_path
        self.arx_mesh_dir = arx_mesh_dir
        self.motion_mode = motion_mode
        self.motion_arm_sample_count = int(motion_arm_sample_count)
        self.motion_use_cache = bool(motion_use_cache)
        self.data_size = data_size
        self.max_episodes = max_episodes
        self.max_frames = max_frames
        self.expand_clips = bool(expand_clips)
        self.clip_num_frames = int(clip_num_frames)
        self.clip_max_stride = max(1, int(clip_max_stride))
        self.clip_start_stride = max(1, int(clip_start_stride))
        self.clip_include_all_strides = bool(clip_include_all_strides)
        self.replay_mode = replay_mode
        self.cameras = cameras or CAMERA_SPECS
        self.valid_iters = int(valid_iters)
        self.max_disp = int(max_disp)
        self.build_volume = build_volume
        self.zmax_m = float(zmax_m)
        self.target_hz = float(target_hz)
        self.depth_source = str(depth_source)
        self.foundation_stereo_dir = foundation_stereo_dir
        self.foundation_stereo_ckpt_path = foundation_stereo_ckpt_path
        self.data_list: list[dict[str, Any]] | None = None

    @classmethod
    def load(cls, data_or_config: str | dict) -> "SampleDataDataset":
        from giga_datasets.datasets.dataset import load_config

        config = load_config(data_or_config)
        kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
        kwargs.pop("config_path", None)
        kwargs.pop("data_path", None)
        return cls(config_path=config.get("config_path"), data_path=config.get("data_path"), **kwargs)

    def open(self) -> None:
        if self.data_list is not None:
            return
        with open(self.params_path, "r") as f:
            camera_params = json.load(f)

        episodes: list[tuple[str, str]] = []
        for root in self.sample_roots:
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                if name.isdigit() and os.path.isdir(os.path.join(root, name)):
                    episodes.append((root, name))
        if self.max_episodes is not None and self.max_episodes > 0:
            episodes = episodes[: self.max_episodes]

        data_list = []
        for data_index, (root, episode_name) in enumerate(episodes):
            sample_dir = os.path.join(root, episode_name)
            camera_dir = os.path.join(sample_dir, "camera")
            if not os.path.isdir(camera_dir):
                continue
            missing_files = _missing_required_files(sample_dir, self.cameras)
            if missing_files:
                print(
                    f"[SampleDataDataset] skip incomplete episode {sample_dir}: "
                    f"missing {len(missing_files)} required files, first={missing_files[0]}"
                )
                continue
            try:
                qpos, eef_pose, target_ts, video_indices, tactile_indices, alignment_stats = _build_aligned_episode(
                    sample_dir,
                    self.cameras,
                    max_frames=self.max_frames,
                    target_hz=self.target_hz,
                )
            except Exception as exc:
                print(f"[SampleDataDataset] skip episode {sample_dir}: failed to align timestamps/state: {exc}")
                continue
            root_tag = os.path.basename(os.path.normpath(root))
            episode_id = f"{root_tag}_{episode_name}"
            record = {
                "data_index": data_index,
                "episode_id": episode_id,
                "episode_name": episode_name,
                "sample_root": root,
                "sample_dir": sample_dir,
                "camera_dir": camera_dir,
                "sample_camera_names": dict(self.cameras),
                "camera_params": copy.deepcopy(camera_params),
                "sample_depth_source": self.depth_source,
                "sample_depth_cache_dir": self.depth_cache_dir,
                "sample_fastfs": {
                    "fastfs_dir": self.fastfs_dir,
                    "model_path": self.fastfs_model_path,
                    "valid_iters": self.valid_iters,
                    "max_disp": self.max_disp,
                    "build_volume": self.build_volume,
                    "zmax_m": self.zmax_m,
                },
                "sample_foundation_stereo": {
                    "foundation_stereo_dir": self.foundation_stereo_dir,
                    "ckpt_path": self.foundation_stereo_ckpt_path,
                    "valid_iters": self.valid_iters,
                    "zmax_m": self.zmax_m,
                },
                "sample_motion": {
                    "mode": self.motion_mode,
                    "urdf_path": self.arx_urdf_path,
                    "mesh_dir": self.arx_mesh_dir,
                    "cache_dir": self.motion_cache_dir,
                    "arm_sample_count": self.motion_arm_sample_count,
                    "use_cache": self.motion_use_cache,
                },
                "qpos": qpos,
                "eef_pose": eef_pose,
                "target_timestamps": target_ts,
                "sample_view_indices": video_indices,
                "sample_tactile_indices": tactile_indices,
                "sample_tactile_names": sorted(tactile_indices.keys()),
                "alignment_stats": alignment_stats,
                "bucket_index": 0,
                "video_info": (224, 224),
            }
            min_count = qpos.shape[0]
            for view_key, camera_name in self.cameras.items():
                prefix = f"cam_{view_key}"
                rgb_path = os.path.join(camera_dir, f"{camera_name}_rgb.mp4")
                ir_left_path = os.path.join(camera_dir, f"{camera_name}_ir_left.mkv")
                ir_right_path = os.path.join(camera_dir, f"{camera_name}_ir_right.mkv")
                record[f"{prefix}_video_path"] = rgb_path
                record[f"{prefix}_ir_left_path"] = ir_left_path
                record[f"{prefix}_ir_right_path"] = ir_right_path
                # The raw sample data has no official replay/motion videos. The
                # sample transform generates them online from qpos + ARX URDF.
                record[f"{prefix}_simulator_path"] = None
                try:
                    raw_count = min(
                        _video_frame_count(rgb_path),
                        _video_frame_count(ir_left_path),
                        _video_frame_count(ir_right_path),
                    )
                except Exception as exc:
                    print(f"[SampleDataDataset] skip episode {sample_dir}: failed to read video metadata: {exc}")
                    min_count = 0
                    break
                indices = record["sample_view_indices"][view_key]
                if indices.size:
                    valid_len = int(np.searchsorted(indices, raw_count, side="left"))
                    min_count = min(min_count, valid_len if valid_len > 0 else len(indices))
                else:
                    min_count = 0
            record["sample_num_frames"] = int(min_count if self.max_frames is None or self.max_frames <= 0 else min(min_count, self.max_frames))
            record["qpos"] = record["qpos"][: record["sample_num_frames"]]
            record["eef_pose"] = record["eef_pose"][: record["sample_num_frames"]]
            record["target_timestamps"] = record["target_timestamps"][: record["sample_num_frames"]]
            record["sample_view_indices"] = {
                key: np.asarray(value[: record["sample_num_frames"]], dtype=np.int64)
                for key, value in record["sample_view_indices"].items()
            }
            record["sample_tactile_indices"] = {
                key: np.asarray(value[: record["sample_num_frames"]], dtype=np.int64)
                for key, value in record["sample_tactile_indices"].items()
            }
            if record["sample_num_frames"] > 0:
                data_list.append(record)

        if self.expand_clips:
            data_list = self._expand_episode_records_to_clips(data_list)

        self.data_list = data_list
        self.data_size = len(data_list)

    def _expand_episode_records_to_clips(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded = []
        for record in records:
            num_frames = int(record.get("sample_num_frames", 0))
            max_span = (self.clip_num_frames - 1) * self.clip_max_stride + 1
            max_start = num_frames - max_span
            if max_start < 0:
                continue
            starts = list(range(0, max_start + 1, self.clip_start_stride))
            if starts and starts[-1] != max_start:
                starts.append(max_start)
            strides = range(1, self.clip_max_stride + 1) if self.clip_include_all_strides else (self.clip_max_stride,)
            for start in starts:
                for stride in strides:
                    item = copy.deepcopy(record)
                    item["sample_clip_start"] = int(start)
                    item["sample_clip_stride"] = int(stride)
                    item["sample_clip_num_frames"] = int(self.clip_num_frames)
                    item["sample_clip_id"] = f"{record.get('episode_id')}_clip_s{int(start):06d}_k{int(stride)}"
                    expanded.append(item)
        print(
            f"[SampleDataDataset] expanded {len(records)} episodes to {len(expanded)} clips "
            f"(clip_num_frames={self.clip_num_frames}, clip_max_stride={self.clip_max_stride}, "
            f"clip_start_stride={self.clip_start_stride}, include_all_strides={self.clip_include_all_strides})"
        )
        return expanded

    def close(self) -> None:
        self.data_list = None
        super().close()

    def filter(self, mode: str, **kwargs: Any) -> None:
        self.open()
        if mode in ("none", None):
            return
        if mode == "min_frames":
            min_num_frames = int(kwargs.get("min_num_frames", 1))
            self.data_list = [item for item in self.data_list if item.get("sample_num_frames", 0) >= min_num_frames]
            self.data_size = len(self.data_list)
            return
        raise ValueError(f"Unsupported SampleDataDataset filter mode: {mode}")

    def __len__(self) -> int:
        if self.data_list is None:
            self.open()
        return self.data_size or 0

    def _get_data(self, index: int) -> dict[str, Any]:
        return copy.deepcopy(self.data_list[index])
