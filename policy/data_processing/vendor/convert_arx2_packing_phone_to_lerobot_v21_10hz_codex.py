from __future__ import annotations

import argparse
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tqdm
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = ("base_0", "left_wrist_0", "right_wrist_0")
CAMERA_TO_FEATURE = {
    "base_0": "observation.images.base_0",
    "left_wrist_0": "observation.images.left_wrist_0",
    "right_wrist_0": "observation.images.right_wrist_0",
}
CAMERA_TO_VIDEO = {
    "base_0": "base_0_rgb.mp4",
    "left_wrist_0": "left_wrist_0_rgb.mp4",
    "right_wrist_0": "right_wrist_0_rgb.mp4",
}
JOINT_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
]


def configure_video_codec(vcodec: str) -> None:
    original_encode_video_frames = lerobot_dataset_module.encode_video_frames

    def encode_video_frames_with_codec(imgs_dir, video_path, fps, overwrite=False, **kwargs):
        return original_encode_video_frames(
            imgs_dir,
            video_path,
            fps,
            vcodec=vcodec,
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            overwrite=overwrite,
        )

    lerobot_dataset_module.encode_video_frames = encode_video_frames_with_codec


@dataclass(frozen=True)
class EpisodePlan:
    episode_dir: Path
    timestamps_ms: np.ndarray
    camera_indices: dict[str, np.ndarray]
    states: np.ndarray
    actions: np.ndarray
    static_start_index: int
    dropped_robot_samples: dict[str, int]


class SelectedVideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self.current_index = 0
        self.last_index = -1
        self.last_frame: np.ndarray | None = None

    def read(self, target_index: int) -> np.ndarray:
        if target_index == self.last_index and self.last_frame is not None:
            return self.last_frame.copy()
        if target_index < self.current_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_index))
            self.current_index = target_index

        frame = None
        while self.current_index <= target_index:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame {target_index} from {self.path}")
            self.current_index += 1

        self.last_index = target_index
        # OpenCV returns BGR. LeRobot writes numpy arrays via PIL, which expects RGB.
        self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.last_frame.copy()

    def close(self) -> None:
        self.cap.release()


def nearest_indices(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps for nearest-neighbor alignment.")
    pos = np.searchsorted(timestamps, targets)
    pos = np.clip(pos, 1, len(timestamps) - 1)
    left = pos - 1
    right = pos
    choose_right = np.abs(timestamps[right] - targets) < np.abs(timestamps[left] - targets)
    return np.where(choose_right, right, left)


def clean_joint_stream(robot_side: dict) -> tuple[np.ndarray, np.ndarray, int]:
    joints = robot_side["joint"]
    timestamps = robot_side["timestamps"]
    clean_joints: list[np.ndarray] = []
    clean_timestamps: list[int] = []
    dropped = 0
    for joint, timestamp in zip(joints, timestamps, strict=False):
        arr = np.asarray(joint, dtype=np.float32)
        if arr.shape != (7,) or not np.all(np.isfinite(arr)):
            dropped += 1
            continue
        clean_joints.append(arr)
        clean_timestamps.append(int(timestamp))
    if not clean_joints:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64), dropped
    return np.stack(clean_joints), np.asarray(clean_timestamps, dtype=np.int64), dropped


def first_motion_index(states: np.ndarray, threshold: float) -> int:
    if len(states) == 0:
        return 0
    displacement = np.linalg.norm(states - states[0], axis=1)
    moving = np.flatnonzero(displacement > threshold)
    if len(moving) == 0:
        return 0
    return max(0, int(moving[0]) - 1)


def make_episode_plan(episode_dir: Path, *, target_hz: int, motion_threshold: float) -> EpisodePlan:
    camera_ts_raw = pickle.load(open(episode_dir / "camera" / "timestamp.pkl", "rb"))
    camera_ts = {name: np.asarray(camera_ts_raw[name], dtype=np.int64) for name in CAMERAS}

    robot = pickle.load(open(episode_dir / "robot" / "robot_state_dict.pkl", "rb"))
    left_joint, left_ts, left_dropped = clean_joint_stream(robot["left"])
    right_joint, right_ts, right_dropped = clean_joint_stream(robot["right"])
    n_robot = min(len(left_joint), len(right_joint), len(left_ts), len(right_ts))
    if n_robot < 2:
        raise ValueError(f"{episode_dir.name}: not enough valid robot samples")

    left_joint = left_joint[:n_robot]
    right_joint = right_joint[:n_robot]
    robot_ts = left_ts[:n_robot]
    joints = np.concatenate([left_joint, right_joint], axis=1).astype(np.float32)

    start_ms = max(ts[0] for ts in camera_ts.values())
    end_ms = min(ts[-1] for ts in camera_ts.values())
    start_ms = max(start_ms, int(robot_ts[0]))
    end_ms = min(end_ms, int(robot_ts[-1]))
    step_ms = int(round(1000 / target_hz))
    targets = np.arange(start_ms, end_ms + 1, step_ms, dtype=np.int64)
    if len(targets) < 3:
        raise ValueError(f"{episode_dir.name}: too few aligned 10Hz frames")

    camera_indices = {name: nearest_indices(ts, targets) for name, ts in camera_ts.items()}
    robot_indices = nearest_indices(robot_ts, targets)
    aligned_states = joints[robot_indices]

    start_index = first_motion_index(aligned_states, motion_threshold)
    targets = targets[start_index:]
    aligned_states = aligned_states[start_index:]
    camera_indices = {name: idx[start_index:] for name, idx in camera_indices.items()}
    if len(aligned_states) < 2:
        raise ValueError(f"{episode_dir.name}: too few frames after static filtering")

    return EpisodePlan(
        episode_dir=episode_dir,
        timestamps_ms=targets[:-1],
        camera_indices={name: idx[:-1] for name, idx in camera_indices.items()},
        states=aligned_states[:-1].astype(np.float32),
        actions=aligned_states[1:].astype(np.float32),
        static_start_index=start_index,
        dropped_robot_samples={"left": left_dropped, "right": right_dropped},
    )


def infer_image_shape(first_episode: Path) -> tuple[int, int, int]:
    path = first_episode / "camera" / CAMERA_TO_VIDEO["base_0"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for shape inference: {path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame for shape inference: {path}")
    height, width = frame.shape[:2]
    return int(height), int(width), 3


def create_dataset(repo_id: str, output_root: Path, fps: int, image_shape: tuple[int, int, int]) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": JOINT_NAMES,
        },
    }
    for feature_key in CAMERA_TO_FEATURE.values():
        features[feature_key] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type="arx2",
        features=features,
        use_videos=True,
        image_writer_processes=8,
        image_writer_threads=4,
    )


def add_episode(dataset: LeRobotDataset, plan: EpisodePlan, task: str) -> None:
    readers = {
        camera: SelectedVideoReader(plan.episode_dir / "camera" / CAMERA_TO_VIDEO[camera])
        for camera in CAMERAS
    }
    try:
        for i in range(len(plan.states)):
            frame = {
                "observation.state": plan.states[i],
                "action": plan.actions[i],
                "task": task,
            }
            for camera in CAMERAS:
                frame[CAMERA_TO_FEATURE[camera]] = readers[camera].read(int(plan.camera_indices[camera][i]))
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        for reader in readers.values():
            reader.close()


def link_to_hf_cache(output_root: Path, repo_id: str) -> None:
    cache_path = HF_LEROBOT_HOME / repo_id
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() or cache_path.is_symlink():
        if cache_path.resolve() == output_root.resolve():
            return
        raise FileExistsError(f"HF cache path already exists and points elsewhere: {cache_path}")
    os.symlink(output_root, cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/oss_data/arx2/phone/packing_phone_0626_200"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/oss_data/arx2/phone/packing_phone_0626_200_lerobot_v21_10hz"),
    )
    parser.add_argument("--repo-id", default="zsq/packing_phone_0626_200_lerobot_v21_10hz")
    parser.add_argument("--task", default="Pack the phone.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--motion-threshold", type=float, default=0.02)
    parser.add_argument("--video-codec", default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-cache-link", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root: Path = args.source_root
    output_root: Path = args.output_root
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to remove it.")
        shutil.rmtree(output_root)

    episode_dirs = sorted(p for p in source_root.iterdir() if p.is_dir() and p.name.isdigit())
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {source_root}")

    image_shape = infer_image_shape(episode_dirs[0])
    print(f"source_root={source_root}")
    print(f"output_root={output_root}")
    print(f"repo_id={args.repo_id}")
    print(f"episodes={len(episode_dirs)} fps={args.fps} image_shape={image_shape} video_codec={args.video_codec}")
    configure_video_codec(args.video_codec)

    plans: list[EpisodePlan] = []
    skipped: list[str] = []
    total_static_dropped = 0
    total_frames = 0
    for episode_dir in episode_dirs:
        try:
            plan = make_episode_plan(
                episode_dir,
                target_hz=args.fps,
                motion_threshold=args.motion_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{episode_dir.name}: {exc}")
            continue
        total_static_dropped += plan.static_start_index
        total_frames += len(plan.states)
        plans.append(plan)

    if not plans:
        raise RuntimeError(f"No valid episodes. Skipped: {skipped[:10]}")

    print(
        f"valid_episodes={len(plans)} skipped={len(skipped)} total_frames={total_frames} "
        f"avg_static_dropped={total_static_dropped / len(plans):.2f}"
    )
    if skipped:
        print("Skipped episodes:")
        for item in skipped[:20]:
            print(f"  {item}")

    dataset = create_dataset(args.repo_id, output_root, args.fps, image_shape)
    for plan in tqdm.tqdm(plans, desc="Converting episodes"):
        add_episode(dataset, plan, args.task)

    if not args.no_cache_link:
        link_to_hf_cache(output_root, args.repo_id)
    print(f"Done: {output_root}")
    if not args.no_cache_link:
        print(f"HF cache link: {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    main()
