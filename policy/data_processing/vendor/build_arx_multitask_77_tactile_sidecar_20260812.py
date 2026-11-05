from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

from arx_episode_plan_ppu4_20260812 import make_episode_plan
from arx_episode_plan_ppu4_20260812 import nearest_indices
import numpy as np
import pyarrow.parquet as pq
import tqdm

SENSORS = ("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")
POINTS_PER_SENSOR = 700
DISPLACEMENT_CHANNELS = ("dx", "dy", "dz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated tactile sidecar for the existing ARX multitask LeRobot cache."
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path(
            "/mnt/oss_data_raw/xzb/data/hf_lerobot/xzb/"
            "arx_multitask_pi05_raw_remove_nofail_77src_lerobot_v21_10hz_20260811"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/oss_data_raw/xzb/data/hf_lerobot/xzb/"
            "arx_multitask_pi05_raw_remove_nofail_77src_tactile_sidecar_v1_10hz_20260811"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument(
        "--episode-index",
        type=int,
        action="append",
        default=None,
        help="Only build source episodes referenced by these processed global episode indices.",
    )
    parser.add_argument("--target-hz", type=int, default=10)
    parser.add_argument("--max-tactile-skew-ms", type=float, default=60.0)
    parser.add_argument("--state-atol", type=float, default=1e-6)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temp, path)


def resolve_source_path(manifest_path: str) -> Path:
    path = Path(manifest_path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def source_key(row: dict[str, Any]) -> tuple[str]:
    return (str(row["physical_episode_id"]),)


def make_source_id(key: tuple[str, ...]) -> str:
    readable = "__".join("".join(ch if ch.isalnum() else "_" for ch in part) for part in key)
    digest = hashlib.sha1("/".join(key).encode()).hexdigest()[:10]
    return f"{readable}__{digest}"


def parquet_path(lerobot_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return lerobot_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def verify_plan_against_cache(
    lerobot_root: Path,
    episode_index: int,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    atol: float,
) -> None:
    path = parquet_path(lerobot_root, episode_index)
    table = pq.read_table(path, columns=["observation.state", "action", "frame_index"])
    cached_states = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
    cached_actions = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64)
    if cached_states.shape != states.shape or cached_actions.shape != actions.shape:
        raise ValueError(
            f"cache shape mismatch for episode {episode_index}: "
            f"states {cached_states.shape} != {states.shape}, actions {cached_actions.shape} != {actions.shape}"
        )
    if not np.array_equal(frame_indices, np.arange(len(states), dtype=np.int64)):
        raise ValueError(f"non-contiguous frame_index in cache episode {episode_index}")
    state_error = float(np.max(np.abs(cached_states - states), initial=0.0))
    action_error = float(np.max(np.abs(cached_actions - actions), initial=0.0))
    if state_error > atol or action_error > atol:
        raise ValueError(
            f"cache content mismatch for episode {episode_index}: "
            f"state max error={state_error}, action max error={action_error}, atol={atol}"
        )


def load_tactile(episode_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tactile_dir = episode_dir / "tactile"
    with (tactile_dir / "tactile_pointcloud_dict.pkl").open("rb") as stream:
        pointcloud_raw = pickle.load(stream)
    with (tactile_dir / "timestamp.pkl").open("rb") as stream:
        timestamp_raw = pickle.load(stream)

    pointclouds: dict[str, np.ndarray] = {}
    timestamps: dict[str, np.ndarray] = {}
    for sensor in SENSORS:
        if sensor not in pointcloud_raw or sensor not in timestamp_raw:
            raise KeyError(f"{episode_dir}: missing tactile sensor {sensor}")
        frames = np.asarray(pointcloud_raw[sensor]["data"], dtype=np.float32)
        sensor_timestamps = np.asarray(timestamp_raw[sensor], dtype=np.int64)
        if frames.ndim != 3 or frames.shape[1:] != (POINTS_PER_SENSOR, 6):
            raise ValueError(f"{episode_dir}: {sensor} has invalid pointcloud shape {frames.shape}")
        if len(frames) != len(sensor_timestamps):
            raise ValueError(
                f"{episode_dir}: {sensor} pointcloud/timestamp length mismatch "
                f"{len(frames)} != {len(sensor_timestamps)}"
            )
        if len(frames) < 2 or np.any(np.diff(sensor_timestamps) < 0):
            raise ValueError(f"{episode_dir}: {sensor} timestamps move backwards")
        if not np.all(np.isfinite(frames)):
            raise ValueError(f"{episode_dir}: {sensor} pointcloud contains NaN/Inf")
        pointclouds[sensor] = frames
        timestamps[sensor] = sensor_timestamps
    return pointclouds, timestamps


def write_fully_masked_source(
    *,
    output_root: Path,
    source_id: str,
    source_index: int,
    episode_dir: Path,
    variant_rows: list[dict[str, Any]],
    plan: Any,
    pre_crop_start_ms: int,
    reason: str,
) -> dict[str, Any]:
    """Represent an episode with unavailable/non-causal tactile as masked tokens."""
    frame_count = len(plan.states)
    tactile = np.zeros((frame_count, len(SENSORS), POINTS_PER_SENSOR, len(DISPLACEMENT_CHANNELS)), np.float32)
    tactile_mask = np.zeros((frame_count, len(SENSORS)), np.bool_)
    reference = np.zeros((len(SENSORS), POINTS_PER_SENSOR, len(DISPLACEMENT_CHANNELS)), np.float32)
    chunk = source_index // 1000
    tactile_relpath = Path("episodes") / f"chunk-{chunk:03d}" / f"{source_id}.npy"
    tactile_mask_relpath = Path("masks") / f"chunk-{chunk:03d}" / f"{source_id}.npy"
    reference_relpath = Path("references") / f"chunk-{chunk:03d}" / f"{source_id}.npy"
    atomic_save_npy(output_root / tactile_relpath, tactile)
    atomic_save_npy(output_root / tactile_mask_relpath, tactile_mask)
    atomic_save_npy(output_root / reference_relpath, reference)
    return {
        "source_id": source_id,
        "source_index": source_index,
        "source_episode_path": str(episode_dir),
        "variant_episode_indices": [int(row["global_episode_index"]) for row in variant_rows],
        "frame_count": frame_count,
        "trim_dropped_start": int(plan.trim.dropped_start),
        "trim_dropped_end": int(plan.trim.dropped_end),
        "pre_crop_start_ms": pre_crop_start_ms,
        "reference_time_ms": None,
        "first_output_time_ms": int(plan.timestamps_ms[0]),
        "last_output_time_ms": int(plan.timestamps_ms[-1]),
        "reference_sample_times_ms": [None] * len(SENSORS),
        "reference_indices": [None] * len(SENSORS),
        "raw_tactile_frame_counts": [0] * len(SENSORS),
        "max_tactile_skew_ms": [None] * len(SENSORS),
        "mean_tactile_skew_ms": [None] * len(SENSORS),
        "tactile_relpath": tactile_relpath.as_posix(),
        "tactile_mask_relpath": tactile_mask_relpath.as_posix(),
        "reference_relpath": reference_relpath.as_posix(),
        "tactile_available": False,
        "fallback_reason": reason,
        "stale_sensor_frames": int(tactile_mask.size),
        "all_sensors_stale_frames": frame_count,
        "dtype": str(tactile.dtype),
        "shape": list(tactile.shape),
        "bytes": int(tactile.nbytes),
    }


def process_source(job: dict[str, Any]) -> dict[str, Any]:
    source_id = str(job["source_id"])
    episode_dir = Path(job["episode_dir"])
    lerobot_root = Path(job["lerobot_root"])
    output_root = Path(job["output_root"])
    variant_rows = job["variant_rows"]
    target_hz = int(job["target_hz"])
    max_tactile_skew_ms = float(job["max_tactile_skew_ms"])
    state_atol = float(job["state_atol"])
    source_index = int(job["source_index"])

    first_row = variant_rows[0]
    plan = make_episode_plan(
        episode_dir,
        target_hz=target_hz,
        swap_robot_sides=bool(first_row["swap_robot_sides"]),
        arm_mode=str(first_row["arm_mode"]),
    )
    expected_lengths = {int(row["frames_after_filtering"]) for row in variant_rows}
    expected_trim = {
        (int(row["trim"]["trim_start"]), int(row["trim"]["trim_end"])) for row in variant_rows
    }
    if expected_lengths != {len(plan.states)}:
        raise ValueError(f"{episode_dir}: manifest/cache length {expected_lengths} != plan length {len(plan.states)}")
    if expected_trim != {(plan.trim.trim_start, plan.trim.trim_end)}:
        raise ValueError(
            f"{episode_dir}: manifest trim {expected_trim} != plan trim "
            f"{(plan.trim.trim_start, plan.trim.trim_end)}"
        )

    verification_episode = int(variant_rows[0]["global_episode_index"])
    verify_plan_against_cache(
        lerobot_root,
        verification_episode,
        plan.states,
        plan.actions,
        atol=state_atol,
    )

    pre_crop_start_ms = int(plan.pre_crop_start_ms)
    try:
        pointclouds, tactile_timestamps = load_tactile(episode_dir)
    except FileNotFoundError as exc:
        return write_fully_masked_source(
            output_root=output_root,
            source_id=source_id,
            source_index=source_index,
            episode_dir=episode_dir,
            variant_rows=variant_rows,
            plan=plan,
            pre_crop_start_ms=pre_crop_start_ms,
            reason=f"missing tactile file: {exc.filename}",
        )
    tactile_common_start_ms = max(int(tactile_timestamps[sensor][0]) for sensor in SENSORS)
    reference_time_ms = max(pre_crop_start_ms, tactile_common_start_ms)
    first_output_time_ms = int(plan.timestamps_ms[0])
    if reference_time_ms > first_output_time_ms:
        return write_fully_masked_source(
            output_root=output_root,
            source_id=source_id,
            source_index=source_index,
            episode_dir=episode_dir,
            variant_rows=variant_rows,
            plan=plan,
            pre_crop_start_ms=pre_crop_start_ms,
            reason=(
                f"common tactile reference {reference_time_ms} is after first retained frame {first_output_time_ms}"
            ),
        )

    aligned_sensors: list[np.ndarray] = []
    valid_sensors: list[np.ndarray] = []
    references: list[np.ndarray] = []
    reference_indices: list[int] = []
    reference_sample_times: list[int] = []
    max_skews: list[float] = []
    mean_skews: list[float] = []
    raw_frame_counts: list[int] = []
    for sensor in SENSORS:
        timestamps = tactile_timestamps[sensor]
        frames = pointclouds[sensor]
        ref_index = int(nearest_indices(timestamps, np.asarray([reference_time_ms], dtype=np.int64))[0])
        target_indices = nearest_indices(timestamps, plan.timestamps_ms)
        skews = np.abs(timestamps[target_indices] - plan.timestamps_ms).astype(np.float64)
        max_skew = float(np.max(skews, initial=0.0))
        reference = frames[ref_index, :, 3:6].astype(np.float32, copy=True)
        aligned = frames[target_indices, :, 3:6] - reference[None, :, :]
        aligned_sensors.append(aligned.astype(np.float32, copy=False))
        # Keep the episode, but mask this sensor token at frames where a
        # dropped/stalled tactile stream has no sufficiently close sample.
        valid_sensors.append(skews <= max_tactile_skew_ms)
        references.append(reference)
        reference_indices.append(ref_index)
        reference_sample_times.append(int(timestamps[ref_index]))
        max_skews.append(max_skew)
        mean_skews.append(float(np.mean(skews)))
        raw_frame_counts.append(len(frames))

    tactile = np.stack(aligned_sensors, axis=1)
    tactile_mask = np.stack(valid_sensors, axis=1).astype(np.bool_, copy=False)
    reference = np.stack(references, axis=0)
    if tactile.shape != (len(plan.states), len(SENSORS), POINTS_PER_SENSOR, len(DISPLACEMENT_CHANNELS)):
        raise AssertionError(f"unexpected aligned tactile shape {tactile.shape}")
    if reference.shape != (len(SENSORS), POINTS_PER_SENSOR, len(DISPLACEMENT_CHANNELS)):
        raise AssertionError(f"unexpected tactile reference shape {reference.shape}")
    if tactile_mask.shape != (len(plan.states), len(SENSORS)):
        raise AssertionError(f"unexpected tactile mask shape {tactile_mask.shape}")
    if not np.all(np.isfinite(tactile)):
        raise ValueError(f"{episode_dir}: aligned tactile contains NaN/Inf")

    chunk = source_index // 1000
    tactile_relpath = Path("episodes") / f"chunk-{chunk:03d}" / f"{source_id}.npy"
    tactile_mask_relpath = Path("masks") / f"chunk-{chunk:03d}" / f"{source_id}.npy"
    reference_relpath = Path("references") / f"chunk-{chunk:03d}" / f"{source_id}.npy"
    atomic_save_npy(output_root / tactile_relpath, tactile)
    atomic_save_npy(output_root / tactile_mask_relpath, tactile_mask)
    # Use NPY instead of NPZ: the OSS mount does not support the random seeks required
    # by ZipFile while writing compressed archives. Timing/index metadata is stored in
    # sources.jsonl alongside this array.
    atomic_save_npy(output_root / reference_relpath, reference)

    return {
        "source_id": source_id,
        "source_index": source_index,
        "source_episode_path": str(episode_dir),
        "variant_episode_indices": [int(row["global_episode_index"]) for row in variant_rows],
        "frame_count": len(plan.states),
        "trim_dropped_start": int(plan.trim.dropped_start),
        "trim_dropped_end": int(plan.trim.dropped_end),
        "pre_crop_start_ms": pre_crop_start_ms,
        "reference_time_ms": reference_time_ms,
        "first_output_time_ms": first_output_time_ms,
        "last_output_time_ms": int(plan.timestamps_ms[-1]),
        "reference_sample_times_ms": reference_sample_times,
        "reference_indices": reference_indices,
        "raw_tactile_frame_counts": raw_frame_counts,
        "max_tactile_skew_ms": max_skews,
        "mean_tactile_skew_ms": mean_skews,
        "tactile_relpath": tactile_relpath.as_posix(),
        "tactile_mask_relpath": tactile_mask_relpath.as_posix(),
        "reference_relpath": reference_relpath.as_posix(),
        "tactile_available": True,
        "validity_threshold_ms": max_tactile_skew_ms,
        "stale_sensor_frames": int(np.count_nonzero(~tactile_mask)),
        "all_sensors_stale_frames": int(np.count_nonzero(~np.any(tactile_mask, axis=1))),
        "dtype": str(tactile.dtype),
        "shape": list(tactile.shape),
        "bytes": int(tactile.nbytes),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.lerobot_root / "meta" / "episodes_multitask_manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    grouped: dict[tuple[str], list[dict[str, Any]]] = {}
    for row in manifest_rows:
        grouped.setdefault(source_key(row), []).append(row)

    sorted_groups = sorted(grouped.items())
    if args.episode_index:
        selected_episode_indices = set(args.episode_index)
        sorted_groups = [
            item
            for item in sorted_groups
            if any(int(row["global_episode_index"]) in selected_episode_indices for row in item[1])
        ]
        found_episode_indices = {
            int(row["global_episode_index"])
            for _, rows in sorted_groups
            for row in rows
            if int(row["global_episode_index"]) in selected_episode_indices
        }
        missing_episode_indices = selected_episode_indices - found_episode_indices
        if missing_episode_indices:
            raise ValueError(f"episode indices not found in manifest: {sorted(missing_episode_indices)}")
    if args.max_sources is not None:
        sorted_groups = sorted_groups[: args.max_sources]
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "sources.progress.jsonl"
    failures_path = args.output_root / "failures.jsonl"
    completed = {row["source_id"]: row for row in read_jsonl(progress_path)} if progress_path.exists() else {}

    jobs: list[dict[str, Any]] = []
    group_by_id: dict[str, list[dict[str, Any]]] = {}
    for source_index, (key, variant_rows) in enumerate(sorted_groups):
        source_id = make_source_id(key)
        group_by_id[source_id] = variant_rows
        if source_id in completed:
            continue
        episode_dir = resolve_source_path(variant_rows[0]["source_episode_path"])
        jobs.append(
            {
                "source_id": source_id,
                "source_index": source_index,
                "episode_dir": str(episode_dir),
                "lerobot_root": str(args.lerobot_root),
                "output_root": str(args.output_root),
                "variant_rows": variant_rows,
                "target_hz": args.target_hz,
                "max_tactile_skew_ms": args.max_tactile_skew_ms,
                "state_atol": args.state_atol,
            }
        )

    failures: list[dict[str, Any]] = []
    if jobs:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(process_source, job): job for job in jobs}
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(future_to_job), total=len(future_to_job), desc="Tactile sources"
            ):
                job = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    failure = {
                        "source_id": job["source_id"],
                        "source_episode_path": job["episode_dir"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    append_jsonl(failures_path, failure)
                else:
                    completed[row["source_id"]] = row
                    append_jsonl(progress_path, row)

    source_rows = sorted(completed.values(), key=lambda row: int(row["source_index"]))
    selected_ids = {make_source_id(key) for key, _ in sorted_groups}
    source_rows = [row for row in source_rows if row["source_id"] in selected_ids]
    # A completed tactile array is intentionally reused when a new visual variant of the
    # same physical episode is appended.  Refresh this denormalized field from the current
    # manifest so sources.jsonl cannot retain the older phase-1 variant list.
    source_rows = [
        {
            **row,
            "variant_episode_indices": [
                int(variant["global_episode_index"])
                for variant in group_by_id[row["source_id"]]
            ],
        }
        for row in source_rows
    ]
    episode_rows: list[dict[str, Any]] = []
    for row in source_rows:
        episode_rows.extend(
            [
                {
                    "global_episode_index": int(variant["global_episode_index"]),
                    "variant_id": variant["variant_id"],
                    "head_variant": variant["head_variant"],
                    "source_id": row["source_id"],
                    "frame_count": row["frame_count"],
                    "tactile_relpath": row["tactile_relpath"],
                    "tactile_mask_relpath": row.get("tactile_mask_relpath"),
                    "reference_relpath": row["reference_relpath"],
                }
                for variant in group_by_id[row["source_id"]]
            ]
        )
    episode_rows.sort(key=lambda row: row["global_episode_index"])

    expected_sources = len(sorted_groups)
    complete = len(source_rows) == expected_sources and not failures
    summary = {
        "complete": complete,
        "lerobot_root": str(args.lerobot_root),
        "output_root": str(args.output_root),
        "sensor_order": list(SENSORS),
        "feature": "episode-relative displacement D(t)-D(t_ref)",
        "channels": list(DISPLACEMENT_CHANNELS),
        "reference_semantics": "common physical reference before leading-static-frame cropping",
        "mask_semantics": "sensor token is valid when nearest tactile sample skew <= max_tactile_skew_ms",
        "max_tactile_skew_ms": args.max_tactile_skew_ms,
        "target_hz": args.target_hz,
        "expected_sources": expected_sources,
        "completed_sources": len(source_rows),
        "failed_sources_this_run": len(failures),
        "mapped_variants": len(episode_rows),
        "total_unique_frames": int(sum(int(row["frame_count"]) for row in source_rows)),
        "total_tactile_bytes": int(sum(int(row["bytes"]) for row in source_rows)),
        "stale_sensor_frames": int(sum(int(row.get("stale_sensor_frames", 0)) for row in source_rows)),
        "all_sensors_stale_frames": int(sum(int(row.get("all_sensors_stale_frames", 0)) for row in source_rows)),
        "fully_masked_sources": int(sum(row.get("tactile_available") is False for row in source_rows)),
    }
    write_jsonl(args.output_root / "sources.jsonl", source_rows)
    write_jsonl(args.output_root / "episodes.jsonl", episode_rows)
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
