from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def load_state_action(root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(
        parquet_path(root, episode_index),
        columns=["observation.state", "action", "episode_index", "frame_index"],
    )
    states = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
    episode_indices = np.asarray(table["episode_index"].combine_chunks().to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64)
    return states, actions, episode_indices, frame_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--allow-main-incomplete", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip loading tactile arrays; still validate all main/sidecar mapping metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_summary = read_json(args.lerobot_root / "meta" / "conversion_summary.json")
    if not main_summary.get("complete", False) and not args.allow_main_incomplete:
        raise ValueError("main conversion summary is not complete")
    manifest = read_jsonl(args.lerobot_root / "meta" / "episodes_multitask_manifest.jsonl")
    info = read_json(args.lerobot_root / "meta" / "info.json")
    if len(manifest) != int(info["total_episodes"]):
        raise ValueError(f"manifest episodes {len(manifest)} != info total_episodes {info['total_episodes']}")

    by_physical: dict[
        str,
        list[tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]],
    ] = defaultdict(list)
    total_frames = 0
    swap_variant_count = 0
    for expected_index, row in enumerate(manifest):
        episode_index = int(row["global_episode_index"])
        if episode_index != expected_index:
            raise ValueError(f"non-contiguous manifest episode index {episode_index} at {expected_index}")
        expected_frames = int(row["frames_after_filtering"])
        states: np.ndarray | None = None
        actions: np.ndarray | None = None
        if not args.metadata_only:
            states, actions, episode_indices, frame_indices = load_state_action(
                args.lerobot_root, episode_index
            )
            if states.shape != (expected_frames, 14) or actions.shape != (expected_frames, 14):
                raise ValueError(
                    f"episode {episode_index}: invalid state/action shapes {states.shape} {actions.shape}"
                )
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
                raise ValueError(f"episode {episode_index}: state/action contains NaN/Inf")
            if not np.all(episode_indices == episode_index):
                raise ValueError(f"episode {episode_index}: parquet episode_index mismatch")
            if not np.array_equal(frame_indices, np.arange(expected_frames, dtype=np.int64)):
                raise ValueError(f"episode {episode_index}: frame_index is not contiguous")
        if ("camera_swap_applied" in row and row["camera_swap_applied"] is not False) or (
            "tactile_swap_applied" in row and row["tactile_swap_applied"] is not False
        ):
            raise ValueError(f"episode {episode_index}: camera/tactile swap invariant violated")
        paths = row.get("camera_source_paths", {})
        if paths:
            if "left_wrist_0" not in paths or "right_wrist_0" not in paths:
                raise ValueError(f"episode {episode_index}: missing wrist camera paths")
            if "left_wrist_0_rgb.mp4" not in paths["left_wrist_0"]:
                raise ValueError(f"episode {episode_index}: left wrist camera identity mismatch")
            if "right_wrist_0_rgb.mp4" not in paths["right_wrist_0"]:
                raise ValueError(f"episode {episode_index}: right wrist camera identity mismatch")
        swap_variant_count += int(bool(row["swap_robot_sides"]))
        total_frames += expected_frames
        by_physical[str(row["physical_episode_id"])].append((row, states, actions))

    for physical_id, variants in by_physical.items():
        if args.metadata_only:
            continue
        reference_states = variants[0][1]
        reference_actions = variants[0][2]
        assert reference_states is not None and reference_actions is not None
        for row, states, actions in variants[1:]:
            assert states is not None and actions is not None
            if not np.array_equal(states, reference_states) or not np.array_equal(actions, reference_actions):
                raise ValueError(f"{physical_id}: visual variants do not share identical state/action")

    sidecar_summary = read_json(args.sidecar_root / "summary.json")
    if not sidecar_summary.get("complete", False):
        raise ValueError("tactile sidecar summary is not complete")
    episode_map = read_jsonl(args.sidecar_root / "episodes.jsonl")
    source_rows = read_jsonl(args.sidecar_root / "sources.jsonl")
    if len(episode_map) != len(manifest):
        raise ValueError(f"sidecar maps {len(episode_map)} variants, expected {len(manifest)}")
    if int(sidecar_summary["mapped_variants"]) != len(manifest):
        raise ValueError("sidecar summary mapped_variants mismatch")

    source_by_id = {row["source_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("duplicate tactile source_id entries")
    for row in source_rows:
        frame_count = int(row["frame_count"])
        if args.metadata_only:
            continue
        tactile = np.load(args.sidecar_root / row["tactile_relpath"], mmap_mode="r", allow_pickle=False)
        tactile_mask = np.load(args.sidecar_root / row["tactile_mask_relpath"], mmap_mode="r", allow_pickle=False)
        reference = np.load(args.sidecar_root / row["reference_relpath"], mmap_mode="r", allow_pickle=False)
        if tactile.shape != (frame_count, 4, 700, 3) or tactile.dtype != np.float32:
            raise ValueError(f"{row['source_id']}: invalid tactile {tactile.shape} {tactile.dtype}")
        if tactile_mask.shape != (frame_count, 4) or tactile_mask.dtype != np.bool_:
            raise ValueError(f"{row['source_id']}: invalid mask {tactile_mask.shape} {tactile_mask.dtype}")
        if reference.shape != (4, 700, 3) or reference.dtype != np.float32:
            raise ValueError(f"{row['source_id']}: invalid reference {reference.shape} {reference.dtype}")
        if not np.all(np.isfinite(tactile)) or not np.all(np.isfinite(reference)):
            raise ValueError(f"{row['source_id']}: tactile/reference contains NaN/Inf")
        if row["reference_time_ms"] is not None and int(row["reference_time_ms"]) > int(row["first_output_time_ms"]):
            raise ValueError(f"{row['source_id']}: tactile reference is after first retained frame")

    mapped_indices: list[int] = []
    mapped_by_source: dict[str, list[int]] = defaultdict(list)
    for row in episode_map:
        episode_index = int(row["global_episode_index"])
        mapped_indices.append(episode_index)
        source_id = str(row["source_id"])
        if source_id not in source_by_id:
            raise ValueError(f"episode {episode_index}: unknown tactile source_id {source_id}")
        source = source_by_id[source_id]
        mapped_by_source[source_id].append(episode_index)
        manifest_row = manifest[episode_index]
        if str(row["variant_id"]) != str(manifest_row["variant_id"]):
            raise ValueError(f"episode {episode_index}: sidecar/main variant_id mismatch")
        if str(row["head_variant"]) != str(manifest_row["head_variant"]):
            raise ValueError(f"episode {episode_index}: sidecar/main head_variant mismatch")
        if int(row["frame_count"]) != int(manifest_row["frames_after_filtering"]):
            raise ValueError(f"episode {episode_index}: sidecar/main frame count mismatch")
        if int(source["frame_count"]) != int(manifest_row["frames_after_filtering"]):
            raise ValueError(f"episode {episode_index}: source sidecar/main frame count mismatch")
        for relpath_key in ("tactile_relpath", "tactile_mask_relpath", "reference_relpath"):
            if str(row[relpath_key]) != str(source[relpath_key]):
                raise ValueError(
                    f"episode {episode_index}: {relpath_key} differs from tactile source"
                )
    if sorted(mapped_indices) != list(range(len(manifest))):
        raise ValueError("sidecar episode mapping is not contiguous and complete")
    for source_id, source in source_by_id.items():
        declared = [int(index) for index in source.get("variant_episode_indices", [])]
        mapped = sorted(mapped_by_source.get(source_id, []))
        if declared != mapped:
            raise ValueError(
                f"{source_id}: source variant_episode_indices {declared} != mapped {mapped}"
            )

    result = {
        "valid": True,
        "main_complete": bool(main_summary.get("complete", False)),
        "lerobot_episodes": len(manifest),
        "physical_episodes": len(by_physical),
        "lerobot_frames": total_frames,
        "sidecar_sources": len(source_rows),
        "sidecar_mapped_variants": len(episode_map),
        "swap_visual_variants": swap_variant_count,
        "fully_masked_tactile_sources": int(sidecar_summary["fully_masked_sources"]),
        "stale_sensor_frames": int(sidecar_summary["stale_sensor_frames"]),
        "all_sensors_stale_frames": int(sidecar_summary["all_sensors_stale_frames"]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
