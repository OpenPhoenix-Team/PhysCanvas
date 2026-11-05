"""Verify videos, aggregate N=128 metrics, then discard temporary features."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import linalg

from .cli import load_records


def frechet_distance(real: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(real) < 2 or len(predicted) < 2:
        return None
    real = np.asarray(real, np.float64)
    predicted = np.asarray(predicted, np.float64)
    real_mean, predicted_mean = real.mean(0), predicted.mean(0)
    real_cov = np.cov(real, rowvar=False)
    predicted_cov = np.cov(predicted, rowvar=False)
    cov_sqrt = linalg.sqrtm(real_cov @ predicted_cov)
    if not np.isfinite(cov_sqrt).all():
        offset = np.eye(real_cov.shape[0]) * 1e-6
        cov_sqrt = linalg.sqrtm((real_cov + offset) @ (predicted_cov + offset))
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    result = float(np.square(real_mean - predicted_mean).sum() + np.trace(real_cov + predicted_cov - 2 * cov_sqrt))
    return result if np.isfinite(result) else None


def verify_video(path: Path, expected_frames: int) -> None:
    try:
        import av
    except ImportError:
        av = None
    if not path.is_file():
        raise FileNotFoundError(path)
    if av is not None:
        with av.open(str(path)) as container:
            decoded = sum(1 for _ in container.decode(video=0))
    else:
        process = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path), "-f", "null", "-"],
            capture_output=True, text=True, check=True,
        )
        counts = re.findall(r"frame=\s*(\d+)", process.stderr)
        if not counts:
            raise RuntimeError(f"ffmpeg did not report decoded frames for {path}")
        decoded = int(counts[-1])
    if decoded != expected_frames:
        raise ValueError(f"{path}: decoded {decoded} frames, expected {expected_frames}")


def load_episode_results(run_dir: Path, cohort: Path, split: str, horizon: int | None) -> list[dict]:
    selected_path = run_dir / "selected_records.json"
    selected = load_records(selected_path)
    fixed = json.loads(cohort.read_text())["episodes"]
    expected = [item for item in fixed if horizon is None or int(item["source_frames"]) >= horizon]
    if [row["episode_id"] for row in selected] != [item["episode_id"] for item in expected]:
        raise ValueError("selected records differ from the fixed cohort or horizon eligibility")
    records = []
    import re
    for source in selected:
        frames = min(int(source["sample_num_frames"]), horizon) if horizon else int(source["sample_num_frames"])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source["episode_id"]))
        path = run_dir / split / "videos" / f"{safe}_{frames}f.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text())
        if record.get("episode_id") != source["episode_id"] or record.get("frames") != frames:
            raise ValueError(f"metadata mismatch: {path}")
        if record.get("smoke_only"):
            raise ValueError(f"smoke result cannot be finalized: {path}")
        video = Path(record["video"])
        if video.resolve() != path.with_suffix(".mp4").resolve():
            raise ValueError(f"video path mismatch: {path}")
        verify_video(video, frames)
        if (run_dir / "run_spec.json").is_file() and json.loads((run_dir / "run_spec.json").read_text()).get("export_clean_rgb"):
            clean = Path(record.get("clean_rgb_video") or "")
            expected_clean = run_dir / split / "clean_rgb" / f"{safe}_{frames}f.mp4"
            if clean.resolve() != expected_clean.resolve():
                raise ValueError(f"clean RGB video path mismatch: {path}")
            verify_video(clean, frames)
            clean_gt = Path(record.get("clean_rgb_gt_video") or "")
            expected_gt = run_dir / split / "clean_rgb_gt" / f"{safe}_{frames}f.mp4"
            if clean_gt.resolve() != expected_gt.resolve():
                raise ValueError(f"clean GT RGB video path mismatch: {path}")
            verify_video(clean_gt, frames)
        records.append(record)
    if len(records) != len(selected):
        raise ValueError("episode count mismatch")
    return records


def weighted_image_summary(records: list[dict], modality: str) -> dict:
    values = [row["metrics"]["images"][modality] for row in records]
    output = {"future_frames": sum(item["frames"] for item in values)}
    for key in ("mse_01", "psnr", "ssim", "lpips"):
        valid = [(item[key], item["frames"]) for item in values
                 if item.get(key) is not None and np.isfinite(item[key])]
        output[key] = float(sum(value * frames for value, frames in valid) / sum(frames for _, frames in valid)) if valid else None
        output[key + "_episode_macro"] = float(np.mean([value for value, _ in valid])) if valid else None
    return output


def flow_summary(records: list[dict]) -> dict:
    output = {}
    for name in ("overall", "active", "idle_fp"):
        items = [row["metrics"]["flow"][name] for row in records]
        counts = np.sum([item["point_counts_per_side"] for item in items], axis=0)
        sse = np.sum([item["sse_per_side_axis"] for item in items], axis=0)
        axis = np.divide(sse, counts[:, None], out=np.full_like(sse, np.nan), where=counts[:, None] > 0)
        episode_values = [item["mse"] for item in items if item["mse"] is not None]
        output[name] = {
            "micro_mse": float(sse.sum() / (3 * counts.sum())) if counts.sum() else None,
            "episode_macro_mse": float(np.mean(episode_values)) if episode_values else None,
            "point_counts_per_side": counts.tolist(),
            "sse_per_side_axis": sse.tolist(),
            "axis_mse": (sse.sum(0) / counts.sum()).tolist() if counts.sum() else None,
            "side_axis_mse": [[float(value) if np.isfinite(value) else None for value in side] for side in axis],
        }
    total = sum(output["overall"]["point_counts_per_side"])
    active = sum(output["active"]["point_counts_per_side"])
    output["active_coverage"] = active / total if total else None
    coverage = [row["metrics"]["flow"]["active_coverage"] for row in records
                if row["metrics"]["flow"]["active_coverage"] is not None]
    output["active_coverage_episode_macro"] = float(np.mean(coverage)) if coverage else None
    output["tau_per_side"] = records[0]["metrics"]["flow"]["tau_per_side"]
    return output


def marker_displacement_summary(records: list[dict]) -> dict:
    items = [row["metrics"]["marker_displacement"] for row in records]
    active_count = np.sum([row["active_point_counts_per_side"] for row in items], axis=0)
    idle_count = np.sum([row["idle_point_counts_per_side"] for row in items], axis=0)
    active_abs = np.sum([row["active_abs_sum_per_side_axis"] for row in items], axis=0)
    idle_abs = np.sum([row["idle_pred_abs_sum_per_side_axis"] for row in items], axis=0)
    amplitude_abs = np.sum([row["active_amplitude_abs_sum_per_side"] for row in items], axis=0)
    denominators = {"active_mae": 3 * active_count.sum(), "idle_fp_mae": 3 * idle_count.sum(),
                    "active_amplitude_mae": active_count.sum()}
    numerators = {"active_mae": active_abs.sum(), "idle_fp_mae": idle_abs.sum(),
                  "active_amplitude_mae": amplitude_abs.sum()}
    summary = {"display_scale": 1000, "active_points": int(active_count.sum()),
               "idle_points": int(idle_count.sum())}
    for key in denominators:
        summary[key + "_micro"] = float(numerators[key] / denominators[key]) if denominators[key] else None
        values = [row[key] for row in items if row[key] is not None]
        summary[key + "_episode_macro"] = float(np.mean(values)) if values else None
    summary["active_mae_per_side"] = [
        float(active_abs[i].sum() / (3 * count)) if count else None
        for i, count in enumerate(active_count)
    ]
    summary["idle_fp_mae_per_side"] = [
        float(idle_abs[i].sum() / (3 * count)) if count else None
        for i, count in enumerate(idle_count)
    ]
    summary["active_amplitude_mae_per_side"] = [
        float(amplitude_abs[i] / count) if count else None
        for i, count in enumerate(active_count)
    ]
    return summary


def tactile_event_summary(records: list[dict]) -> dict | None:
    present = ["tactile_events" in row["metrics"] for row in records]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("contact/event metrics are incomplete across episodes")
    sides = [side for row in records for side in row["metrics"]["tactile_events"]["sides"].values()]
    output = {"episode_side_count": len(sides), "aggregation": "episode-side macro; undefined events omitted"}
    for name, field in (("contact_mae_01", None),
                        ("contact_event_f1", "event_f1"),
                        ("contact_onset_sec", "onset_error_sec"),
                        ("contact_offset_sec", "offset_error_sec"),
                        ("flow_event_f1", "event_f1"),
                        ("flow_onset_sec", "onset_error_sec"),
                        ("flow_offset_sec", "offset_error_sec")):
        values = []
        for side in sides:
            value = side[name] if field is None else side["contact_events" if name.startswith("contact_") else "flow_events"][field]
            if value is not None and np.isfinite(value):
                values.append(value)
        output[name] = float(np.mean(values)) if values else None
        output[name + "_defined_sides"] = len(values)
    return output


def distribution_summary(records: list[dict], run_dir: Path, modality: str) -> dict:
    output = {}
    root = run_dir.resolve()
    for metric in ("fid", "fvd"):
        groups = {}
        for role in ("gt", "pred"):
            features = []
            for row in records:
                feature_file = Path(row["metrics"]["temporary_features"]).resolve()
                if not feature_file.is_relative_to(root):
                    raise ValueError(f"feature file outside run directory: {feature_file}")
                with np.load(feature_file) as data:
                    features.append(data[f"{modality}_{metric}_{role}"])
            groups[role] = np.concatenate(features)
        if len(groups["gt"]) != len(groups["pred"]):
            raise ValueError(f"{metric} feature count mismatch")
        output[metric + "_count"] = len(groups["gt"])
        output[metric] = frechet_distance(groups["gt"], groups["pred"])
    return output


def summarize(records: list[dict], run_dir: Path, include_distributions: bool, profile: str = "all") -> dict:
    output = {
        "episodes": len(records),
        "frames": sum(row["frames"] for row in records),
        "generated_future_frames": sum(row["frames"] - 1 for row in records),
        "tasks": dict(Counter(row["task"] for row in records)),
        "images": {},
    }
    if profile == "all":
        output["flow"] = flow_summary(records)
        output["marker_displacement"] = marker_displacement_summary(records)
    for modality in ("rgb", "tactile_diff"):
        image = weighted_image_summary(records, modality)
        if include_distributions and profile != "reconstruction":
            image.update(distribution_summary(records, run_dir, modality))
        output["images"][modality] = image
    events = tactile_event_summary(records)
    if events is not None:
        output["tactile_events"] = events
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "generalization", "evaluation"), required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--horizon", default="128")
    parser.add_argument("--threshold-json", type=Path)
    parser.add_argument("--infer-only", action="store_true")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    horizon = None if args.horizon == "full" else int(args.horizon)
    records = load_episode_results(run_dir, args.cohort, args.split, horizon)
    spec = json.loads((run_dir / "run_spec.json").read_text())
    profile = spec.get("metric_profile", "all")
    if spec["episode_ids"] != [row["episode_id"] for row in records]:
        raise ValueError("run specification and finished episodes differ")
    if any(row["num_inference_steps"] != spec["denoising_steps"] for row in records):
        raise ValueError("denoising-step count differs from run specification")
    if spec.get("contact_threshold_sha256"):
        if any(row["metrics"].get("tactile_events", {}).get("threshold_sha256")
               != spec["contact_threshold_sha256"] for row in records):
            raise ValueError("contact/event threshold hash differs from run specification")
    if args.infer_only:
        if any(row.get("metrics") is not None for row in records):
            raise ValueError("infer-only run unexpectedly contains metrics")
        summary = {"episodes": len(records), "decoded_frames": sum(row["frames"] for row in records),
                   "metrics": False, "run_spec": spec}
    else:
        if horizon != 128 or any(row.get("metrics") is None for row in records):
            raise ValueError("metrics mode requires N=128 and per-episode metrics")
        if any(row["metrics"].get("metric_profile", "all") != profile for row in records):
            raise ValueError("episode metric profile differs from run specification")
        if profile == "all":
            if args.threshold_json is None:
                raise ValueError("all metrics require threshold evidence")
            threshold = json.loads(args.threshold_json.read_text())
            if threshold["tau_per_side"] != records[0]["metrics"]["flow"]["tau_per_side"]:
                raise ValueError("flow threshold differs from frozen evidence")
        summary = {
            "run_spec": spec,
            "protocol": {
                "horizon_frames": 128,
                "denoising_steps": records[0]["num_inference_steps"],
                "frame_zero_excluded": True,
                "fvd_clip_length": 16,
                "fvd_stride": 8,
                "flow_reference": "raw physical XYZ; GT valid-marker mask; no prediction filtering",
                "prefix_success_warning": "N=128 state is not compared to full-episode outcome",
            },
            "split": args.split,
            "overall": summarize(records, run_dir, True, profile),
            "by_task": {task: summarize([row for row in records if row["task"] == task], run_dir, False, profile)
                        for task in sorted({row["task"] for row in records})},
        }
        if profile != "all":
            summary["protocol"].pop("flow_reference")
        if profile == "reconstruction":
            summary["protocol"].pop("fvd_clip_length")
            summary["protocol"].pop("fvd_stride")
        path = run_dir / "metrics_aggregate.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if json.loads(path.read_text())["overall"]["episodes"] != len(records):
            raise ValueError("aggregate readback failed")
        feature_dir = run_dir / "temporary_features"
        if feature_dir.is_dir():
            shutil.rmtree(feature_dir)
    (run_dir / "COMPLETED.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"verified {len(records)} videos; {sum(row['frames'] for row in records)} decoded frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
