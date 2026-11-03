"""Run the official VBench custom-input metrics on verified clean RGB videos."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DIMENSIONS = ("subject_consistency", "background_consistency", "motion_smoothness")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(run_dir):
    run_dir = Path(run_dir).resolve()
    spec = json.loads((run_dir / "run_spec.json").read_text())
    if not spec.get("export_clean_rgb"):
        raise ValueError("the inference run must use --export-clean-rgb")
    if not (run_dir / "COMPLETED.json").is_file():
        raise ValueError("the inference run must be verified and complete")
    records = json.loads((run_dir / "selected_records.json").read_text())["records"]
    horizon = None if spec["horizon"] == "full" else int(spec["horizon"].removeprefix("n"))
    expected = []
    for record in records:
        episode_id = record["episode_id"]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id))
        frames = min(int(record["sample_num_frames"]), horizon) if horizon else int(record["sample_num_frames"])
        path = run_dir / "evaluation" / "clean_rgb" / f"{safe}_{frames}f.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        expected.append(path.name)
    actual = {path.name for path in (run_dir / "evaluation" / "clean_rgb").glob("*.mp4")}
    if set(expected) != actual or len(set(expected)) != len(records):
        raise ValueError("clean RGB videos differ from the frozen episode selection")
    return spec, records, run_dir / "evaluation" / "clean_rgb"


def evaluation_script(vbench_repo):
    root = Path(vbench_repo)
    for script in (root / "evaluate.py", root / "vbench" / "launch" / "evaluate.py"):
        if script.is_file():
            return script
    raise FileNotFoundError(f"VBench evaluate.py not found under {root}")


def command(vbench_repo, videos_dir, output_dir):
    script = evaluation_script(vbench_repo)
    invocation = [sys.executable, str(script), "--mode", "custom_input",
            "--videos_path", str(videos_dir), "--output_path", str(output_dir),
            "--dimension", *DIMENSIONS]
    if os.environ.get("VBENCH_CACHE_DIR"):
        invocation.extend(["--load_ckpt_from_local", "True"])
    return invocation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vbench-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    spec, records, videos_dir = validate_inputs(args.run_dir)
    invocation = command(args.vbench_repo, videos_dir, args.output_dir)
    print(f"verified {len(records)} clean RGB videos; dimensions={','.join(DIMENSIONS)}")
    if args.dry_run:
        print(" ".join(invocation))
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("VBench output directory must be empty to prevent stale scores")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    paths = [str(args.vbench_repo.resolve())]
    extra_deps = args.vbench_repo.resolve().parent / "extra_deps"
    if extra_deps.is_dir():
        paths.append(str(extra_deps))
    clip_source = args.vbench_repo.resolve().parent / "CLIP-main"
    if clip_source.is_dir():
        paths.append(str(clip_source))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    subprocess.run(invocation, cwd=args.vbench_repo, env=env, check=True)
    candidates = list(args.output_dir.glob("*_eval_results.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one official VBench result, found {len(candidates)}")
    raw = json.loads(candidates[0].read_text())
    scores = {}
    for metric in DIMENSIONS:
        if metric not in raw or not isinstance(raw[metric], list) or len(raw[metric]) != 2:
            raise ValueError(f"official VBench result missing {metric}")
        score, rows = raw[metric]
        if len(rows) != len(records):
            raise ValueError(f"{metric}: scored {len(rows)} videos, expected {len(records)}")
        scores[metric + "_x100"] = 100 * float(score)
    summary = {
        "source_run_spec": spec, "episodes": len(records), "dimensions": scores,
        "official_result_sha256": sha256(candidates[0]),
        "vbench_source": str(args.vbench_repo.resolve()),
        "note": "Official VBench custom-input scores on clean predicted RGB; not paired GT reconstruction or task success",
    }
    (args.output_dir / "physcanvas_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
