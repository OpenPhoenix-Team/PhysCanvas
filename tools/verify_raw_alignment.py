"""Compare timestamp-aligned raw episode data with a trusted unified cache."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import av
import cv2
import numpy as np


CAMERAS = ("base_0", "left_wrist_0", "right_wrist_0")
TACTILE_SIDES = ("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")


def nearest_indices(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if source.ndim != 1 or source.size == 0 or np.any(np.diff(source) < 0):
        raise ValueError("source timestamps must be nonempty and sorted")
    right = np.searchsorted(source, targets, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    use_left = np.abs(targets - source[left]) <= np.abs(source[right] - targets)
    return np.where(use_left, left, right)


def decoded_frames(path: Path, wanted: np.ndarray):
    wanted_set = set(map(int, wanted))
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted_set:
                yield index, frame.to_ndarray(format="rgb24")


def compare_video(path: Path, expected: np.ndarray, indices: np.ndarray,
                  width: int, height: int) -> dict:
    found = {}
    for index, frame in decoded_frames(path, indices):
        found[index] = frame
        if len(found) == len(set(map(int, indices))):
            break
    if len(found) != len(set(map(int, indices))):
        raise RuntimeError(f"missing selected frames in {path}")
    results = {}
    variants = (("area", cv2.INTER_AREA, False),
                ("linear", cv2.INTER_LINEAR, False),
                ("area_bgr", cv2.INTER_AREA, True))
    for name, interpolation, reverse_channels in variants:
        diffs = []
        for target, raw_index in zip(expected, indices):
            resized = cv2.resize(found[int(raw_index)], (width, height), interpolation=interpolation)
            if reverse_channels:
                resized = resized[..., ::-1]
            diffs.append(np.abs(resized.astype(np.int16) - target.astype(np.int16)))
        diffs = np.stack(diffs)
        results[name] = {"exact_frames": int(np.sum(np.all(diffs == 0, axis=(1, 2, 3)))),
                         "mae": float(diffs.mean()), "max_error": int(diffs.max())}
    return results


def inspect(raw: Path, cache: Path, frames: int) -> dict:
    targets = np.asarray(np.load(cache / "timestamps.npy", mmap_mode="r")[:frames], dtype=np.int64)
    with (raw / "image_timestamps.pkl").open("rb") as stream:
        image_times = pickle.load(stream)
    with (raw / "tactile_timestamps.pkl").open("rb") as stream:
        tactile_times = pickle.load(stream)
    with (raw / "tactile_pointcloud_dict.pkl").open("rb") as stream:
        pointclouds = pickle.load(stream)
    result = {"frames": len(targets), "cameras": {}, "tactile": {}}
    for camera in CAMERAS:
        source = np.asarray(image_times[f"{camera}_rgb"], dtype=np.int64)
        indices = nearest_indices(source, targets)
        expected = np.asarray(np.load(cache / f"{camera}_rgb.npy", mmap_mode="r")[:frames])
        result["cameras"][camera] = {
            "first_index": int(indices[0]), "last_index": int(indices[-1]),
            "timestamp_max_error_ms": int(np.max(np.abs(source[indices] - targets))),
            "comparison": compare_video(raw / f"{camera}_rgb.mp4", expected, indices,
                                        expected.shape[2], expected.shape[1]),
        }
    for side in TACTILE_SIDES:
        source = np.asarray(tactile_times[side], dtype=np.int64)
        indices = nearest_indices(source, targets)
        expected = np.asarray(np.load(cache / "tactile" / f"{side}.pointcloud.npy", mmap_mode="r")[:frames])
        raw_points = pointclouds[side]["data"]
        selected = np.stack([raw_points[int(index)].reshape(35, 20, 6) for index in indices])
        diff = np.abs(selected.astype(np.float32) - expected.astype(np.float32))
        result["tactile"][side] = {
            "first_index": int(indices[0]), "last_index": int(indices[-1]),
            "pointcloud_exact_frames": int(np.sum(np.all(diff == 0, axis=(1, 2, 3)))),
            "pointcloud_max_error": float(diff.max()),
            "timestamp_max_error_ms": int(np.max(np.abs(source[indices] - targets))),
            "difference_comparison": compare_video(
                raw / f"{side}_difference.mkv",
                np.asarray(np.load(cache / "tactile" / f"{side}.difference.npy", mmap_mode="r")[:frames]),
                indices, 700, 400,
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-episode", type=Path, required=True)
    parser.add_argument("--cache-episode", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.frames <= 256:
        parser.error("--frames must be between 1 and 256")
    print(json.dumps(inspect(args.raw_episode, args.cache_episode, args.frames), indent=2))


if __name__ == "__main__":
    main()
