"""Export and prepare one compact, path-independent 128-frame evaluation fixture.

Export reads trusted source pickles once. Prepare uses only safe NumPy arrays
and the seven original compressed videos; it never reads the source cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


CAMERAS = ("base_0", "left_wrist_0", "right_wrist_0")
SIDES = ("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")
CAMERA_NAMES = {"high": "base_0", "left_wrist": "left_wrist_0", "right_wrist": "right_wrist_0"}
TASK_CACHE_ROOTS = {"bottle": "bottle_cap_rollout", "shiguan": "shiguan_saizi_rollout",
                    "jimu": "jimu_rollout", "mofang": "mofang_v2_rollout"}
FRAMES = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def nearest_indices(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if source.ndim != 1 or not len(source) or np.any(np.diff(source) < 0):
        raise ValueError("source timestamps must be nonempty and sorted")
    right = np.clip(np.searchsorted(source, targets), 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    return np.where(np.abs(targets - source[left]) <= np.abs(source[right] - targets), left, right)


def read_selected_video(path: Path, indices: np.ndarray, size: tuple[int, int],
                        bgr: bool) -> np.ndarray:
    import cv2

    requested = set(map(int, indices))
    frames = {}
    try:
        import av
    except ImportError:
        capture = cv2.VideoCapture(str(path))
        try:
            index = 0
            while capture.isOpened():
                ok, decoded = capture.read()
                if not ok:
                    break
                if index in requested:
                    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                    frames[index] = resized[..., ::-1] if bgr else resized
                    if len(frames) == len(requested):
                        break
                index += 1
        finally:
            capture.release()
    else:
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in requested:
                    rgb = frame.to_ndarray(format="rgb24")
                    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                    frames[index] = resized[..., ::-1] if bgr else resized
                    if len(frames) == len(requested):
                        break
    if len(frames) != len(requested):
        raise RuntimeError(f"{path} has only {len(frames)}/{len(requested)} selected frames")
    return np.stack([frames[int(index)] for index in indices]).astype(np.uint8)


def export_fixture(raw: Path, cache: Path, label: Path, dest: Path, task: str) -> None:
    if dest.exists():
        raise FileExistsError(f"fixture destination must not exist: {dest}")
    dest.mkdir(parents=True)
    raw_dir = dest / "raw"
    raw_dir.mkdir()
    conditioning = dest / "conditioning"
    conditioning.mkdir()
    source_hashes = {}
    for name in [*(f"{camera}_rgb.mp4" for camera in CAMERAS),
                 *(f"{side}_difference.mkv" for side in SIDES)]:
        source = raw / name
        shutil.copyfile(source, raw_dir / name)
        source_hashes[name] = sha256(source)

    with (raw / "image_timestamps.pkl").open("rb") as stream:
        image = pickle.load(stream)
    with (raw / "tactile_timestamps.pkl").open("rb") as stream:
        tactile = pickle.load(stream)
    with (raw / "tactile_pointcloud_dict.pkl").open("rb") as stream:
        pointcloud = pickle.load(stream)
    np.savez_compressed(raw_dir / "timestamps.npz",
                        **{f"camera_{name}": np.asarray(image[f"{name}_rgb"], dtype=np.int64)
                           for name in CAMERAS},
                        **{f"tactile_{name}": np.asarray(tactile[name], dtype=np.int64)
                           for name in SIDES})
    np.savez_compressed(raw_dir / "pointclouds.npz",
                        **{name: np.asarray(pointcloud[name]["data"], dtype=np.float32)
                           for name in SIDES})
    timeline = np.asarray(np.load(cache / "timestamps.npy", mmap_mode="r")[:FRAMES], dtype=np.int64)
    if len(timeline) != FRAMES:
        raise ValueError(f"expected {FRAMES} cached timeline frames")
    np.save(conditioning / "timeline.npy", timeline)
    np.savez_compressed(conditioning / "motion.npz",
                        **{name: np.asarray(np.load(cache / f"{name}_motion.npy", mmap_mode="r")[:FRAMES])
                           for name in CAMERAS})
    with np.load(label, allow_pickle=False) as source:
        np.savez_compressed(conditioning / "labels.npz",
                            labels=np.asarray(source["labels"][:FRAMES]),
                            probabilities=np.asarray(source["probabilities"][:FRAMES]))
    episode_id = cache.name
    manifest = {"schema": "physcanvas_raw_n128_v1", "episode_id": episode_id,
                "task": task, "frames": FRAMES, "output_hw": [224, 224],
                "source_video_sha256": source_hashes,
                "note": "Raw videos and numeric sensors; motion is a required derived condition."}
    manifest["files"] = {
        str(path.relative_to(dest)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(dest.rglob("*")) if path.is_file()
    }
    write_json(dest / "manifest.json", manifest)


def prepare_fixture(fixture: Path, output: Path) -> None:
    manifest = json.loads((fixture / "manifest.json").read_text())
    if manifest.get("schema") not in {"physcanvas_raw_n128_v1", "physcanvas_raw_full_episode_v1"}:
        raise ValueError("unsupported fixture schema")
    frames = int(manifest["frames"])
    if frames < 2 or (manifest["schema"] == "physcanvas_raw_n128_v1" and frames != FRAMES):
        raise ValueError("invalid fixture frame count")
    for name, info in manifest["files"].items():
        path = fixture / name
        if not path.is_file() or path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError(f"fixture file mismatch: {name}")
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    raw = fixture / "raw"
    conditioning = fixture / "conditioning"
    target = np.load(conditioning / "timeline.npy", allow_pickle=False)
    if len(target) != frames or np.any(np.diff(target) <= 0):
        raise ValueError("fixture timeline does not match its frame count")
    cache_root = output / TASK_CACHE_ROOTS[manifest["task"]]
    episode_dir = cache_root / "episodes" / manifest["episode_id"]
    tactile_dir = episode_dir / "tactile"
    tactile_dir.mkdir(parents=True)
    with np.load(raw / "timestamps.npz", allow_pickle=False) as timestamps:
        with np.load(raw / "pointclouds.npz", allow_pickle=False) as pointclouds:
            with np.load(conditioning / "motion.npz", allow_pickle=False) as motion:
                for name in CAMERAS:
                    indices = nearest_indices(timestamps[f"camera_{name}"], target)
                    rgb = read_selected_video(raw / f"{name}_rgb.mp4", indices, (224, 224), False)
                    np.save(episode_dir / f"{name}_rgb.npy", rgb)
                    if motion[name].shape != rgb.shape:
                        raise ValueError(f"motion shape mismatch: {name}")
                    np.save(episode_dir / f"{name}_motion.npy", motion[name])
                for name in SIDES:
                    indices = nearest_indices(timestamps[f"tactile_{name}"], target)
                    diff = read_selected_video(raw / f"{name}_difference.mkv", indices,
                                               (700, 400), True)
                    points = pointclouds[name][indices].reshape(frames, 35, 20, 6)
                    np.save(tactile_dir / f"{name}.difference.npy", diff)
                    np.save(tactile_dir / f"{name}.pointcloud.npy", points)
    np.save(episode_dir / "timestamps.npy", target)
    meta = {"episode_id": manifest["episode_id"], "num_frames": frames,
            "target_fps": 10.0, "output_hw": [224, 224], "tactile_enabled": True}
    write_json(episode_dir / "meta.json", meta)
    with np.load(conditioning / "labels.npz", allow_pickle=False) as labels:
        if any(len(labels[name]) != frames for name in ("labels", "probabilities")):
            raise ValueError("fixture labels do not match its frame count")
    label_path = output / "labels.npz"
    shutil.copyfile(conditioning / "labels.npz", label_path)
    record = {
        "data_index": 0, "bucket_index": 0, "episode_id": manifest["episode_id"],
        "episode_name": manifest["episode_id"], "object_state_task": manifest["task"],
        "object_state_label_path": str(label_path.resolve()),
        "sample_num_frames": frames, "sample_raw_num_frames": frames,
        "sample_frame_step": 1, "sample_source_hz": 10.0, "sample_target_hz": 10.0,
        "sample_cache_source": "unified_cache", "sample_cache_root": str(cache_root.resolve()),
        "sample_cache_episode_dir": str(episode_dir.resolve()), "sample_cache_meta": meta,
        "sample_camera_names": CAMERA_NAMES, "sample_tactile_sides": list(SIDES),
        "target_timestamps": (np.arange(frames) / 10.0).tolist(), "video_info": [224, 224],
    }
    write_json(output / "records.json", {"records": [record]})
    write_json(output / "cohort.json", {"episodes": [{"episode_id": manifest["episode_id"],
                                                      "task": manifest["task"],
                                                      "source_frames": frames}]})
    write_json(output / "prepared_manifest.json", {
        "source_fixture_sha256": sha256(fixture / "manifest.json"),
        "episode_id": manifest["episode_id"], "frames": frames,
        "record": str((output / "records.json").resolve()),
        "cohort": str((output / "cohort.json").resolve()),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Read trusted raw sources and export a portable fixture")
    export.add_argument("--raw-episode", type=Path, required=True)
    export.add_argument("--cache-episode", type=Path, required=True)
    export.add_argument("--label-npz", type=Path, required=True)
    export.add_argument("--task", choices=tuple(TASK_CACHE_ROOTS), required=True)
    export.add_argument("--fixture-dir", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="Decode the fixture without the source cache")
    prepare.add_argument("--fixture-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_fixture(args.raw_episode, args.cache_episode, args.label_npz, args.fixture_dir, args.task)
    else:
        prepare_fixture(args.fixture_dir, args.output_dir)


if __name__ == "__main__":
    main()
