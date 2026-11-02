"""Two public entry points sharing a fixed, auditable episode cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tools.cohort_fixture import prepare_cohort, verify_archives, verify_prepared


REPO = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = Path(__file__).resolve().parent / "data" / "fixed_20_ids.json"
TASKS = ("bottle", "shiguan", "jimu", "mofang")


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a records list")
    ids = [str(row["episode_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate episode IDs in {path}")
    return records


def select_cohort(cohort_path: Path, records_path: Path | list[dict],
                  horizon: int | None) -> list[dict]:
    cohort = json.loads(cohort_path.read_text())["episodes"]
    if not cohort or any(row["task"] not in TASKS for row in cohort):
        raise ValueError("cohort needs one or more episodes from the four supported tasks")
    records = load_records(records_path) if isinstance(records_path, Path) else records_path
    all_records = {str(row["episode_id"]): row for row in records}
    selected = []
    for item in cohort:
        episode_id = str(item["episode_id"])
        if episode_id not in all_records:
            raise ValueError(f"missing episode in supplied data manifest: {episode_id}")
        row = all_records[episode_id]
        if row.get("object_state_task") != item["task"]:
            raise ValueError(f"task mismatch for {episode_id}")
        if int(row["sample_num_frames"]) != int(item["source_frames"]):
            raise ValueError(f"source frame count mismatch for {episode_id}")
        if horizon is None or int(row["sample_num_frames"]) >= horizon:
            selected.append(row)
    if len({row["episode_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate selected episode")
    return selected


def parse_horizons(text: str) -> list[int | None]:
    result = []
    for token in text.split(","):
        token = token.strip().lower()
        value = None if token == "full" else int(token)
        if value is not None and value < 9:
            raise ValueError("N must be at least 9 frames")
        if value in result:
            raise ValueError("duplicate horizon")
        result.append(value)
    if not result:
        raise ValueError("at least one horizon is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("infer-all-n", "metrics-128"), required=True)
    parser.add_argument("--evaluation-suite", choices=("custom", "reconstruction", "all"),
                        default="custom", help="Strict metric selection for the named evaluation entry points")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--records-json", type=Path,
                        help="Private runtime cache manifest containing full records for the selected IDs")
    parser.add_argument("--allow-external-records", action="store_true",
                        help="Internal compatibility only; public evaluation uses --archives-dir")
    parser.add_argument("--archives-dir", type=Path,
                        help="Verified 20-episode archive set; no external data manifest is read")
    parser.add_argument("--prepared-dir", type=Path,
                        help="Temporary input directory derived only from --archives-dir")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, default=REPO,
                        help="Compatible PhysCanvas inference/data-loader source tree (default: this package)")
    parser.add_argument("--prompt-embedding", type=Path,
                        help="Path to the default prompt embedding required by the model transforms")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Dedicated visual_eval_* folder, a sibling of the checkpoint's models folder")
    parser.add_argument("--pretrained-vae", type=Path,
                        help="Override the archived config's pretrained VAE/scheduler directory")
    parser.add_argument("--threshold-json", type=Path,
                        help="Frozen training-side tactile-flow threshold, required for metrics-128")
    parser.add_argument("--contact-threshold-json", type=Path,
                        help="Optional frozen four-side model-space tactile contact/event thresholds")
    parser.add_argument("--i3d-weights", type=Path,
                        help="TorchScript I3D detector, required for FVD in metrics-128")
    parser.add_argument("--export-clean-rgb", action="store_true",
                        help="Save separate annotation-free RGB videos for official VBench scoring")
    parser.add_argument("--vbench-repo", type=Path,
                        help="Official VBench repository and its installed metric assets")
    parser.add_argument("--cotracker-repo", type=Path,
                        help="Official CoTracker3 repository")
    parser.add_argument("--cotracker-checkpoint", type=Path,
                        help="CoTracker3 offline checkpoint")
    parser.add_argument("--queries-json", type=Path,
                        help="Reviewed EEF/object frame-0 query points for the fixed cohort")
    parser.add_argument("--horizons", default="128,192,256,full",
                        help="Comma-separated frame horizons for infer-all-n")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection without loading a model")
    return parser


def run_extra_metrics(args: argparse.Namespace, run_dir: Path, spec: dict) -> None:
    if args.mode != "metrics-128":
        return
    if not (run_dir / "COMPLETED.json").is_file():
        raise FileNotFoundError(run_dir / "COMPLETED.json")
    output = {"run_spec": spec, "base": json.loads((run_dir / "metrics_aggregate.json").read_text())}
    if args.evaluation_suite == "all" and "tactile_events" not in output["base"]["overall"]:
        raise ValueError("all-metrics evaluation is missing tactile contact/event scores")
    if args.vbench_repo:
        directory = run_dir / "vbench"
        summary_path = directory / "physcanvas_summary.json"
        if not summary_path.is_file():
            subprocess.run([
                sys.executable, "-m", "evaluation.vbench_eval",
                "--run-dir", str(run_dir), "--vbench-repo", str(args.vbench_repo),
                "--output-dir", str(directory),
            ], cwd=REPO, check=True)
        summary = json.loads(summary_path.read_text())
        if summary.get("source_run_spec") != spec or summary.get("episodes") != len(spec["episode_ids"]):
            raise ValueError("existing VBench scores do not match this evaluation")
        output["vbench"] = summary
    if args.cotracker_repo:
        path = run_dir / "tracking_summary.json"
        if not path.is_file():
            subprocess.run([
                sys.executable, "-m", "evaluation.tracking_eval",
                "--run-dir", str(run_dir), "--queries-json", str(args.queries_json),
                "--cotracker-repo", str(args.cotracker_repo),
                "--checkpoint", str(args.cotracker_checkpoint),
                "--output-json", str(path), "--device", args.device,
            ], cwd=REPO, check=True)
        summary = json.loads(path.read_text())
        from .vbench_eval import sha256
        if (summary.get("source_run_spec") != spec
                or summary.get("episodes") != len(spec["episode_ids"])
                or summary.get("queries_sha256") != sha256(args.queries_json)
                or summary.get("checkpoint_sha256") != sha256(args.cotracker_checkpoint)):
            raise ValueError("existing tracking scores do not match this evaluation")
        output["tracking"] = summary
    path = run_dir / "evaluation_suite.json"
    if args.evaluation_suite == "all" and not {"vbench", "tracking"} <= output.keys():
        raise ValueError("all-metrics evaluation is incomplete")
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    optional = (args.contact_threshold_json, args.vbench_repo, args.cotracker_repo,
                args.cotracker_checkpoint, args.queries_json)
    if args.evaluation_suite != "custom" and args.mode != "metrics-128":
        raise ValueError("named evaluation suites require metrics-128 mode")
    if args.evaluation_suite == "reconstruction" and any(optional):
        raise ValueError("reconstruction-only evaluation does not accept optional metric assets")
    if args.evaluation_suite == "all" and not all(optional):
        raise ValueError("all-metrics evaluation requires contact thresholds, VBench, CoTracker3, and reviewed queries")
    if args.num_inference_steps <= 0:
        raise ValueError("num-inference-steps must be positive")
    if args.vbench_repo:
        from .vbench_eval import evaluation_script
        evaluation_script(args.vbench_repo)
    tracking_inputs = (args.cotracker_repo, args.cotracker_checkpoint, args.queries_json)
    if any(tracking_inputs) and not all(tracking_inputs):
        raise ValueError("tracking requires --cotracker-repo, --cotracker-checkpoint, and --queries-json")
    if args.cotracker_repo:
        for path in (args.cotracker_repo / "hubconf.py", args.cotracker_checkpoint, args.queries_json):
            if not path.is_file():
                raise FileNotFoundError(path)
    if (args.vbench_repo or args.cotracker_repo) and args.mode != "metrics-128":
        raise ValueError("VBench and tracking require metrics-128 mode")
    if args.vbench_repo or args.cotracker_repo:
        args.export_clean_rgb = True
    if args.archives_dir:
        if args.records_json or args.cohort != DEFAULT_COHORT or args.allow_external_records:
            raise ValueError("--archives-dir cannot be combined with external records or cohort options")
        archive_index = verify_archives(args.archives_dir)
        short_archive = archive_index["schema"] == "physcanvas_cohort_n128_v1"
        if short_archive and args.mode == "infer-all-n" and args.horizons not in ("128", "128,full"):
            raise ValueError("these archives contain 128 prepared frames; use --horizons 128")
        args.prepared_dir = args.prepared_dir or args.output_dir / (
            "prepared_inputs_n128" if short_archive else "prepared_inputs_full_episode")
        if args.prepared_dir.exists():
            args.records_json, args.cohort = verify_prepared(args.archives_dir, args.prepared_dir)
        else:
            prepare_cohort(args.archives_dir, args.prepared_dir)
            args.records_json, args.cohort = verify_prepared(args.archives_dir, args.prepared_dir)
        if short_archive and args.mode == "infer-all-n":
            args.horizons = "128"
    elif not args.records_json:
        raise ValueError("public evaluation requires --archives-dir")
    elif not args.allow_external_records:
        raise ValueError("external records require --allow-external-records; public evaluation uses --archives-dir")
    if args.mode == "metrics-128" and (args.threshold_json is None or args.i3d_weights is None):
        raise ValueError("metrics-128 requires --threshold-json and --i3d-weights")
    if args.contact_threshold_json and args.mode != "metrics-128":
        raise ValueError("contact/event metrics require metrics-128 mode")
    horizons = parse_horizons(args.horizons) if args.mode == "infer-all-n" else [128]
    records = load_records(args.records_json)
    selections = [(horizon, select_cohort(args.cohort, records, horizon)) for horizon in horizons]
    if args.mode == "metrics-128" and len(selections[0][1]) != len(json.loads(args.cohort.read_text())["episodes"]):
        raise ValueError("metrics-128 requires every selected episode to contain 128 frames")
    for horizon, selected in selections:
        print(f"N={horizon or 'full'}: {len(selected)} episodes, "
              f"tasks={dict(Counter(row['object_state_task'] for row in selected))}")
    if args.dry_run:
        return 0
    if not (args.runtime_source / "PhysCanvas" / "models").is_dir():
        raise FileNotFoundError("runtime-source lacks PhysCanvas/models")
    if args.prompt_embedding is not None and not args.prompt_embedding.is_file():
        raise FileNotFoundError(args.prompt_embedding)
    if args.prompt_embedding is None and not os.environ.get("CVPR2026_WM_PROMPT_EMBEDDING"):
        raise ValueError("set --prompt-embedding or CVPR2026_WM_PROMPT_EMBEDDING")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.mode == "metrics-128":
        for path in (args.threshold_json, args.i3d_weights):
            if not path.is_file():
                raise FileNotFoundError(path)
        if args.contact_threshold_json and not args.contact_threshold_json.is_file():
            raise FileNotFoundError(args.contact_threshold_json)
    config = json.loads(args.config_json.read_text())
    if "training_config" not in config:
        raise ValueError("config-json must contain training_config")
    if config["training_config"].get("models", {}).get("pretrained") == "SET_WITH_--pretrained-vae" and not args.pretrained_vae:
        raise ValueError("public_eval_config.json requires --pretrained-vae")
    if args.pretrained_vae:
        if not args.pretrained_vae.is_dir():
            raise FileNotFoundError(args.pretrained_vae)
        config["training_config"]["models"]["pretrained"] = str(args.pretrained_vae)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "runtime_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    threshold_hash = hashlib.sha256(args.threshold_json.read_bytes()).hexdigest() if args.threshold_json else None
    env = dict(
        os.environ,
        PHYSCANVAS_EVAL_RUNTIME_SOURCE=str(args.runtime_source.resolve()),
        PHYSCANVAS_UNIFIED_WAN_INCLUDE_DEPTH="0",
        PHYSCANVAS_PERF_COMPACT_SEPARATE_VAE_BATCH="1",
        PHYSCANVAS_TACTILE_DIFF_ROTATE_CW="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False",
    )
    if args.prompt_embedding is not None:
        env["CVPR2026_WM_PROMPT_EMBEDDING"] = str(args.prompt_embedding.resolve())
    for horizon, selected in selections:
        name = "full" if horizon is None else f"n{horizon}"
        run_dir = args.output_dir / args.mode / name
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "mode": args.mode,
            "split": "evaluation",
            "horizon": name,
            "episode_ids": [row["episode_id"] for row in selected],
            "checkpoint": str(args.checkpoint.resolve()),
            "config_sha256": config_hash,
            "threshold_sha256": threshold_hash,
            "i3d_weights": str(args.i3d_weights.resolve()) if args.i3d_weights else None,
            "denoising_steps": args.num_inference_steps,
        }
        if args.export_clean_rgb:
            spec["export_clean_rgb"] = True
        if args.contact_threshold_json:
            spec["contact_threshold_sha256"] = hashlib.sha256(args.contact_threshold_json.read_bytes()).hexdigest()
        completion = run_dir / "COMPLETED.json"
        if completion.is_file():
            prior = json.loads(completion.read_text())
            count = prior["overall"]["episodes"] if args.mode == "metrics-128" else prior["episodes"]
            if count != len(selected) or prior.get("run_spec") != spec:
                raise ValueError(f"existing completion is for another run: {completion}")
            print(f"already verified: {run_dir}")
            run_extra_metrics(args, run_dir, spec)
            continue
        spec_path = run_dir / "run_spec.json"
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError(f"existing partial run is for another specification: {spec_path}")
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        selection_path = run_dir / "selected_records.json"
        selection_path.write_text(json.dumps({"records": selected}, indent=2) + "\n")
        command = [
            sys.executable, "-m", "evaluation.model_eval",
            "--selection-manifest", str(selection_path),
            "--config-json", str(config_path),
            "--checkpoint", str(args.checkpoint),
            "--output-dir", str(run_dir),
            "--split", "evaluation",
            "--device", args.device,
            "--dtype", args.dtype,
            "--num-inference-steps", str(args.num_inference_steps),
            "--max-frames", str(horizon or 0),
        ]
        if args.resume:
            command.append("--resume")
        if args.export_clean_rgb:
            command.append("--export-clean-rgb")
        if args.mode == "metrics-128":
            command.extend(["--metrics", "--threshold-json", str(args.threshold_json),
                            "--i3d-weights", str(args.i3d_weights)])
            if args.contact_threshold_json:
                command.extend(["--contact-threshold-json", str(args.contact_threshold_json)])
        subprocess.run(command, cwd=REPO, env=env, check=True)
        aggregate = [
            sys.executable, "-m", "evaluation.aggregate",
            "--run-dir", str(run_dir), "--split", "evaluation",
            "--cohort", str(args.cohort), "--horizon", name[1:] if horizon is not None else "full",
        ]
        if args.mode == "metrics-128":
            aggregate.extend(["--threshold-json", str(args.threshold_json)])
        else:
            aggregate.append("--infer-only")
        subprocess.run(aggregate, cwd=REPO, check=True)
        run_extra_metrics(args, run_dir, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


def infer_all_n() -> int:
    return run(["--mode", "infer-all-n", *sys.argv[1:]])


def evaluate_n128() -> int:
    return run(["--mode", "metrics-128", *sys.argv[1:]])


def evaluate_reconstruction() -> int:
    return run(["--mode", "metrics-128", "--evaluation-suite", "reconstruction", *sys.argv[1:]])


def evaluate_all() -> int:
    return run(["--mode", "metrics-128", "--evaluation-suite", "all", *sys.argv[1:]])
