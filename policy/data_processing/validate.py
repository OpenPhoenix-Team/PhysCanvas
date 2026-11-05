"""Validate a converted bundle without unpickling source data or changing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def child(root, relpath):
    relative = Path(relpath)
    path = (root / relative).resolve()
    if relative.is_absolute() or root.resolve() not in path.parents:
        raise ValueError(f"Sidecar path escapes its root: {relpath}")
    return path


def validate_bundle(output, *, require_complete=True):
    import cv2
    import pyarrow.parquet as pq
    output = Path(output).resolve()
    dataset, sidecar = output / "lerobot", output / "tactile"
    info = json.loads((dataset / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1" or info.get("fps") != 10:
        raise ValueError("Expected LeRobot v2.1 at 10 Hz")
    summary = json.loads((sidecar / "summary.json").read_text())
    if require_complete:
        complete = json.loads((output / "COMPLETE.json").read_text())
        if not complete.get("complete") or not summary.get("complete") or (output / "FAILED.json").exists():
            raise ValueError("Bundle is not marked successfully complete")
    episodes = rows(dataset / "meta/episodes.jsonl")
    stats = rows(dataset / "meta/episodes_stats.jsonl")
    tasks = rows(dataset / "meta/tasks.jsonl")
    manifest = rows(dataset / "meta/episodes_multitask_manifest.jsonl")
    mappings = rows(sidecar / "episodes.jsonl")
    sources = rows(sidecar / "sources.jsonl")
    n = int(info["total_episodes"])
    if not n or any(len(r) != n for r in (episodes, stats, manifest, mappings)):
        raise ValueError("Episode/meta/stat/sidecar counts disagree")
    if require_complete and summary.get("mapped_variants") != n:
        raise ValueError("Sidecar mapped_variants mismatch")
    if [x["global_episode_index"] for x in mappings] != list(range(n)):
        raise ValueError("Sidecar episode indices are duplicated, missing or reordered")
    if [x["global_episode_index"] for x in manifest] != list(range(n)):
        raise ValueError("Conversion manifest episode indices mismatch")
    source_by_id = {x["source_id"]: x for x in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("Duplicate physical source records")
    task_map = {int(x["task_index"]): x["task"] for x in tasks}
    total_frames, all_masked_frames, physical = 0, 0, {}
    for index, (episode, item, mapping) in enumerate(zip(episodes, manifest, mappings, strict=True)):
        if int(episode["episode_index"]) != index:
            raise ValueError("LeRobot episode ordering mismatch")
        chunk = index // int(info.get("chunks_size", 1000))
        parquet = dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{index:06d}.parquet"
        table = pq.read_table(parquet)
        state = np.asarray(table["observation.state"].to_pylist(), np.float32)
        action = np.asarray(table["action"].to_pylist(), np.float32)
        count = len(state)
        if state.shape != (count, 14) or action.shape != state.shape or count < 2 or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Invalid finite float32 14-D state/action")
        if not np.allclose(action[:-1], state[1:], atol=1e-6, rtol=0):
            raise ValueError("Action is not next retained measured q[t+1]")
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(count)):
            raise ValueError("frame_index must be contiguous")
        if not np.all(table["episode_index"].to_numpy() == index):
            raise ValueError("Wrong parquet episode_index")
        if not np.array_equal(table["index"].to_numpy(), np.arange(total_frames, total_frames + count)):
            raise ValueError("Global frame index mismatch")
        for task_index in np.unique(table["task_index"].to_numpy()):
            if task_map.get(int(task_index)) != item["prompt"]:
                raise ValueError("Prompt/task index mismatch")
        if not (int(episode["length"]) == item["frames_after_filtering"] == mapping["frame_count"] == count):
            raise ValueError("Episode frame counts disagree")
        xyz = np.load(child(sidecar, mapping["tactile_relpath"]), mmap_mode="r", allow_pickle=False)
        mask = np.load(child(sidecar, mapping["tactile_mask_relpath"]), mmap_mode="r", allow_pickle=False)
        reference = np.load(child(sidecar, mapping["reference_relpath"]), mmap_mode="r", allow_pickle=False)
        if xyz.shape != (count, 4, 700, 3) or xyz.dtype != np.float32 or not np.isfinite(xyz).all():
            raise ValueError("Invalid full-XYZ tactile sidecar")
        if mask.shape != (count, 4) or mask.dtype != np.bool_:
            raise ValueError("Invalid tactile mask")
        if reference.shape != (4, 700, 3) or reference.dtype != np.float32 or not np.isfinite(reference).all():
            raise ValueError("Invalid tactile reference")
        source = source_by_id.get(mapping["source_id"])
        if source is None or source["frame_count"] != count or index not in source["variant_episode_indices"]:
            raise ValueError("Broken physical-source sidecar provenance")
        source_identity = item["physical_episode_id"]
        if source_identity in physical:
            previous = physical[source_identity]
            if mapping["tactile_relpath"] != previous[0] or not np.array_equal(state, previous[1]) or not np.array_equal(action, previous[2]):
                raise ValueError("Visual variants must share actions/state and one tactile array")
        else:
            physical[source_identity] = (mapping["tactile_relpath"], state, action)
        for camera in ("base_0", "left_wrist_0", "right_wrist_0"):
            video = dataset / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{camera}" / f"episode_{index:06d}.mp4"
            cap = cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 10) > .01:
                    raise ValueError(f"Missing/invalid 10 Hz video: {video}")
                decoded = 0
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    if image is None:
                        raise ValueError(f"Corrupt video: {video}")
                    decoded += 1
                if decoded != count:
                    raise ValueError(f"Video length mismatch: {video}: {decoded} != {count}")
            finally:
                cap.release()
        total_frames += count
        all_masked_frames += int(np.count_nonzero(~mask.any(axis=1)))
    if total_frames != info["total_frames"]:
        raise ValueError("Dataset total_frames mismatch")
    return {"valid": True, "episodes": n, "physical_episodes": len(physical),
            "frames": total_frames, "all_sensors_masked_frames": all_masked_frames,
            "video_checks": "all decoded, 10 Hz, frame counts match", "action_semantics": "next measured absolute q"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(validate_bundle(args.output), indent=2))


if __name__ == "__main__":
    main()
