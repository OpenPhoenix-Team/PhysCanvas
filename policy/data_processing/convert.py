"""Manifest-driven, source-read-only conversion; defaults to plan-only.

Planner and tactile algorithms are imported unchanged from checked snapshots.
Only orchestration, safety checks and provenance are new here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .original import HASHES, algorithms, file_sha256

VERSION = "physcanvas-pi05-arx-v21-20261104"
CAMERAS = ("base_0", "left_wrist_0", "right_wrist_0")


def load_manifest(path):
    manifest = Path(path).resolve()
    payload = json.loads(manifest.read_text())
    if payload.get("version") != 1 or not isinstance(payload.get("episodes"), list) or not payload["episodes"]:
        raise ValueError("Expected version=1 and nonempty episodes list")
    if not isinstance(payload.get("repo_id"), str) or not payload["repo_id"].strip():
        raise ValueError("repo_id is required")
    seen_ids, seen_paths = set(), set()
    for row in payload["episodes"]:
        required = ("episode_dir", "physical_episode_id", "task", "prompt", "arm_mode", "swap_robot_sides", "head_variants")
        if any(key not in row for key in required):
            raise ValueError(f"Each episode must explicitly supply {required}")
        path = Path(row["episode_dir"]).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ValueError(f"episode_dir must be an existing absolute directory: {path}")
        row["episode_dir"] = str(path.resolve())
        if row["physical_episode_id"] in seen_ids or row["episode_dir"] in seen_paths:
            raise ValueError("Duplicate physical episode IDs or source directories")
        seen_ids.add(row["physical_episode_id"])
        seen_paths.add(row["episode_dir"])
        if not isinstance(row["physical_episode_id"], str) or not row["physical_episode_id"]:
            raise ValueError("physical_episode_id must be a nonempty string")
        if row["arm_mode"] not in ("left", "right", "bimanual") or type(row["swap_robot_sides"]) is not bool:
            raise ValueError("arm_mode and boolean swap_robot_sides must be explicit; no historical lookup is used")
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError("Explicit nonempty prompt required")
        if not isinstance(row["task"], str) or not row["task"].strip():
            raise ValueError("Explicit nonempty task name required")
        if row["task"] == "shiguan_other":
            raise ValueError("Legacy shiguan_other is not an arm-specific task")
        variants = row["head_variants"]
        if not isinstance(variants, list) or not variants or len(variants) != len(set(variants)) or not set(variants) <= {"raw", "remove_hand"}:
            raise ValueError("head_variants must explicitly list raw and/or remove_hand without duplicates")
    return payload


def ensure_new_output(output, rows):
    raw_output = Path(output).expanduser()
    if not raw_output.is_absolute():
        raise ValueError("output must be absolute")
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(f"Refusing existing output (no overwrite/resume): {raw_output}")
    output = raw_output.resolve()
    for row in rows:
        source = Path(row["episode_dir"]).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Output must not overlap a source or its ancestors: {output} vs {source}")
    return output


def video_paths(row, variant):
    directory = Path(row["episode_dir"]) / "camera"
    head = "base_0_rgb.mp4" if variant == "raw" else "base_0_rgb_remove_hand.mp4"
    return {"base_0": directory / head, "left_wrist_0": directory / "left_wrist_0_rgb.mp4", "right_wrist_0": directory / "right_wrist_0_rgb.mp4"}


def prepare(payload, *, trusted_pickle):
    if not trusted_pickle:
        raise PermissionError("Raw ARX pickles can execute code; pass --trusted-pickle only for data you trust")
    planner, writer, _ = algorithms()
    plans, shape = [], None
    import cv2
    for row in payload["episodes"]:
        # Exact historical alignment/crop and next-retained-q target algorithm.
        plan = planner.make_episode_plan(Path(row["episode_dir"]), target_hz=10,
                                        arm_mode=row["arm_mode"], swap_robot_sides=row["swap_robot_sides"])
        for variant in row["head_variants"]:
            for camera, path in video_paths(row, variant).items():
                cap = cv2.VideoCapture(str(path))
                try:
                    if not cap.isOpened():
                        raise FileNotFoundError(path)
                    actual = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
                    if shape is None:
                        shape = actual
                    if actual != shape or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= int(plan.camera_indices[camera].max()):
                        raise ValueError(f"Video shape/frame coverage mismatch: {path}")
                finally:
                    cap.release()
        plans.append(plan)
    return plans, shape


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _inputs(row):
    episode = Path(row["episode_dir"])
    paths = {episode / "camera/timestamp.pkl", episode / "robot/robot_state_dict.pkl"}
    for variant in row["head_variants"]:
        paths.update(video_paths(row, variant).values())
    for filename in ("timestamp.pkl", "tactile_pointcloud_dict.pkl"):
        p = episode / "tactile" / filename
        if p.exists():
            paths.add(p)
    return {str(path): {"sha256": file_sha256(path), "bytes": path.stat().st_size} for path in sorted(paths)}


def execute(payload, output, *, trusted_pickle=False, manifest_sha256=None):
    output = ensure_new_output(output, payload["episodes"])
    plans, shape = prepare(payload, trusted_pickle=trusted_pickle)
    planner, writer, tactile = algorithms()
    # Verify paths again immediately before the sole output-root creation.
    ensure_new_output(output, payload["episodes"])
    output.mkdir(parents=True, exist_ok=False)
    lerobot_root, sidecar_root = output / "lerobot", output / "tactile"
    sidecar_root.mkdir()
    dataset = None
    original_video_encoder = writer.lerobot_dataset_module.encode_video_frames
    try:
        _json(output / "source_manifest.json", payload)
        source_hashes = [_inputs(row) for row in payload["episodes"]]
        _json(output / "source_hashes.json", source_hashes)
        _json(sidecar_root / "summary.json", {"complete": False, "mapped_variants": 0})
        features = {
            key: {"dtype": "float32", "shape": (14,), "names": writer.JOINT_NAMES}
            for key in ("observation.state", "action")
        }
        features.update({f"observation.images.{camera}": {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]} for camera in CAMERAS})
        # Same LeRobot writer/API/features as the original converter. Synchronous
        # image writing bounds process count; it does not change data semantics.
        writer.configure_video_codec("h264")
        dataset = writer.LeRobotDataset.create(repo_id=payload["repo_id"], root=lerobot_root,
                                              fps=10, robot_type="arx2", features=features,
                                              use_videos=True, image_writer_processes=0,
                                              image_writer_threads=0)
        manifest_rows, source_rows, mapping_rows = [], [], []
        for source_index, (row, plan) in enumerate(zip(payload["episodes"], plans, strict=True)):
            variants = []
            for variant in row["head_variants"]:
                index = len(manifest_rows)
                readers = {camera: writer.SelectedVideoReader(path) for camera, path in video_paths(row, variant).items()}
                try:
                    for frame_index in range(len(plan.states)):
                        frame = {"observation.state": plan.states[frame_index], "action": plan.actions[frame_index], "task": row["prompt"]}
                        frame.update({f"observation.images.{camera}": reader.read(int(plan.camera_indices[camera][frame_index])) for camera, reader in readers.items()})
                        dataset.add_frame(frame)
                    dataset.save_episode()
                finally:
                    for reader in readers.values():
                        reader.close()
                item = dict(global_episode_index=index, physical_episode_id=row["physical_episode_id"],
                            variant_id=f"source{source_index:06d}__{variant}", head_variant=variant,
                            source_episode_path=row["episode_dir"], task=row["task"], prompt=row["prompt"],
                            arm_mode=row["arm_mode"], swap_robot_sides=row["swap_robot_sides"],
                            camera_swap_applied=False, tactile_swap_applied=False,
                            frames_after_filtering=len(plan.states), trim=planner.trim_to_dict(plan.trim),
                            pre_crop_start_ms=int(plan.pre_crop_start_ms), target_fps=10)
                variants.append(item)
                manifest_rows.append(item)
            source_id = tactile.make_source_id((row["physical_episode_id"],))
            # Original function recomputes plan, verifies cached q/action, uses
            # PRE-CROP tactile reference and all XYZ; no new approximation.
            source = tactile.process_source({"source_id": source_id, "source_index": source_index,
                "episode_dir": row["episode_dir"], "lerobot_root": str(lerobot_root),
                "output_root": str(sidecar_root), "variant_rows": variants, "target_hz": 10,
                "max_tactile_skew_ms": 60.0, "state_atol": 1e-6})
            source_rows.append(source)
            for item in variants:
                mapping_rows.append({"global_episode_index": item["global_episode_index"],
                    "variant_id": item["variant_id"], "head_variant": item["head_variant"],
                    "source_id": source_id, "frame_count": source["frame_count"],
                    "tactile_relpath": source["tactile_relpath"],
                    "tactile_mask_relpath": source["tactile_mask_relpath"],
                    "reference_relpath": source["reference_relpath"]})
        dataset.stop_image_writer()
        dataset = None
        _jsonl(lerobot_root / "meta/episodes_multitask_manifest.jsonl", manifest_rows)
        _jsonl(sidecar_root / "episodes.jsonl", mapping_rows)
        _jsonl(sidecar_root / "sources.jsonl", source_rows)
        metadata = {"conversion_version": VERSION, "source_manifest_sha256": manifest_sha256,
                    "algorithm_sha256": HASHES, "target_hz": 10,
                    "action_semantics": "q[t+1] absolute measured joints at next retained aligned sample; NOT action_traces",
                    "alignment_slop_ms": 50, "tactile_max_skew_ms": 60,
                    "state_order": writer.JOINT_NAMES, "tactile_sides": list(tactile.SENSORS),
                    "tactile_channels": ["dx", "dy", "dz"], "source_data_read_only": True,
                    "physical_episodes": len(source_rows), "visual_variants": len(manifest_rows),
                    "frames": sum(x["frames_after_filtering"] for x in manifest_rows)}
        _json(lerobot_root / "meta/conversion_summary.json", metadata)
        from .validate import validate_bundle
        report = validate_bundle(output, require_complete=False)
        _json(output / "validation.json", report)
        # Compare all input bytes again, then publish completion markers last.
        if [_inputs(row) for row in payload["episodes"]] != source_hashes:
            raise RuntimeError("Source files changed during conversion; refusing completion")
        _json(sidecar_root / "summary.json", {"complete": True, "mapped_variants": len(mapping_rows),
              "sensor_order": list(tactile.SENSORS), "channels": ["dx", "dy", "dz"],
              "feature": "episode-relative displacement D(t)-D(t_ref)",
              "reference_semantics": "common physical reference before leading-static-frame cropping",
              "mask_semantics": "nearest tactile sample skew <= 60 ms", "target_hz": 10,
              "expected_sources": len(source_rows), "completed_sources": len(source_rows)})
        _json(output / "COMPLETE.json", {**metadata, "complete": True, "validation": "validation.json"})
        return report
    except BaseException as error:
        # Even an I/O failure while writing COMPLETE must not leave a sidecar
        # advertised as ready. The training wrapper requires both markers.
        _json(sidecar_root / "summary.json", {"complete": False, "error": f"{type(error).__name__}: {error}"})
        _json(output / "FAILED.json", {"complete": False, "error": f"{type(error).__name__}: {error}"})
        # Failure artifacts deliberately remain inspectable; no deletion/resume.
        raise
    finally:
        writer.lerobot_dataset_module.encode_video_frames = original_video_encoder
        if dataset is not None:
            dataset.stop_image_writer()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="default: inspect only, create no output")
    mode.add_argument("--execute", action="store_true", help="create the new dataset bundle")
    parser.add_argument("--trusted-pickle", action="store_true", help="acknowledge raw pickle inputs are trusted executable data")
    args = parser.parse_args(argv)
    payload = load_manifest(args.manifest)
    output = ensure_new_output(args.output, payload["episodes"])
    if args.execute:
        report = execute(payload, output, trusted_pickle=args.trusted_pickle, manifest_sha256=file_sha256(args.manifest))
    else:
        plans, shape = prepare(payload, trusted_pickle=args.trusted_pickle)
        report = {"mode": "plan_only", "output_created": False, "physical_episodes": len(plans),
                  "visual_variants": sum(len(x["head_variants"]) for x in payload["episodes"]),
                  "frames": sum(len(p.states) * len(row["head_variants"]) for p, row in zip(plans, payload["episodes"], strict=True)),
                  "image_shape": shape, "lerobot_root": str(output / "lerobot"), "sidecar_root": str(output / "tactile")}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
