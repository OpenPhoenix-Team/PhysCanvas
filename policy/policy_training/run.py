"""Plan by default; heavy operations require two explicit write confirmations."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .plan import Plan, absolute_path, asset_preflight, check_norm_stats, identifier, overlaps, require_write_authority, sha256
from .validate_torch import validate_checkpoint

HERE = Path(__file__).resolve().parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"isolated_openpi_{name}", HERE / "openpi" / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def imports_check(plan: Plan) -> dict:
    from .build_config import build_train_config
    config = build_train_config(plan)  # No .create(), model, tokenizer, or checkpoint load.
    import importlib.metadata as metadata
    import jax
    import openpi
    from lerobot.common.constants import HF_LEROBOT_HOME
    from rlinf.utils.ckpt_convertor.openpi import jax_to_openpi_rlinf
    root = Path(HF_LEROBOT_HOME) / config.data.repo_id
    if root.resolve() != Path(plan.lerobot_root):
        raise ValueError(f"LeRobot root was captured before env was set: {root}")
    return {"ok": True, "openpi_source": str(Path(openpi.__file__).resolve()),
            "tactile_converter_source": str(Path(jax_to_openpi_rlinf.__file__).resolve()),
            "lerobot_resolved_root": str(root), "jax_cpu_devices": [str(d) for d in jax.devices("cpu")],
            "versions": {name: metadata.version(name) for name in ("jax", "flax", "optax", "orbax-checkpoint", "torch", "lerobot")},
            "model": {"use_tactile": config.model.use_tactile, "action_dim": config.model.action_dim,
                      "action_horizon": config.model.action_horizon},
            "checkpoint_dir": str(config.checkpoint_dir), "norm_stats": str(plan.norm_stats)}


def gpu_check(plan: Plan) -> dict:
    code = "import json,jax; d=jax.devices('gpu'); print(json.dumps({'count':len(d),'devices':[str(x) for x in d]}))"
    env = {**os.environ, "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    try:
        result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "JAX CUDA device enumeration timed out after 90 seconds"}
    if result.returncode:
        return {"ok": False, "error": "JAX CUDA backend unavailable", "stderr": result.stderr[-6000:]}
    devices = json.loads(result.stdout.strip().splitlines()[-1])
    ok = devices["count"] >= plan.fsdp_devices and devices["count"] % plan.fsdp_devices == 0 and plan.batch_size % devices["count"] == 0
    return {"ok": ok, **devices, "required_fsdp_devices": plan.fsdp_devices,
            "note": "Device enumeration only, not a multi-GPU train/collective validation"}


def norm_stats(plan: Plan) -> dict:
    from .build_config import build_train_config, stats_data_config
    import numpy as np
    from openpi.shared import normalize
    cfg = build_train_config(plan)
    data_cfg = stats_data_config(plan)
    batches, count = load_script("compute_norm_stats").create_tactile_acwm_array_dataloader(
        data_cfg, cfg.model.action_horizon, plan.batch_size, cfg.model)
    if not count:
        raise ValueError("No complete normalization batches")
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in batches:
        for key, accumulator in stats.items():
            accumulator.update(np.asarray(batch[key]))
    # Reserve output atomically after computing; never overwrite another run.
    plan.norm_stats.parent.mkdir(parents=True, exist_ok=False)
    normalize.save(plan.norm_stats.parent, {key: value.get_statistics() for key, value in stats.items()})
    provenance = {"operation": "norm-stats", "execution": "completed", "plan": plan.describe(), "batches": count,
                  "dataset_info_sha256": sha256(Path(plan.lerobot_root) / "meta/info.json"),
                  "sidecar_manifest_sha256": sha256(Path(plan.tactile_sidecar_root) / "episodes.jsonl"),
                  "norm_stats": check_norm_stats(plan.norm_stats),
                  "semantics": "state/action raw quantiles; action_horizon=16, episode-tail clamp, drop-last; no tactile normalization"}
    (plan.norm_stats.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def conversion(args) -> dict:
    identifier(args.asset_id)
    params, norm, output = map(absolute_path, (args.input_params, args.norm_stats, args.output))
    if params.name != "params" or not params.is_dir():
        raise ValueError("--input-params must be an existing JAX checkpoint's params directory")
    if overlaps(output, params.parent) or overlaps(output, norm) or len(output.parts) < 3:
        raise ValueError("Conversion output must be a dedicated new directory disjoint from all inputs")
    if output.exists():
        raise ValueError(f"Refusing existing conversion output: {output}")
    # An explicit override would make provenance ambiguous: insist on checkpoint-carried stats.
    if params.parent / "assets" not in norm.parents:
        raise ValueError("Use norm_stats.json carried in the SAME JAX checkpoint/assets, not a historical default")
    stats = check_norm_stats(norm)
    result = {"mode": "conversion_plan", "input_params": str(params), "input_norm_stats": stats,
              "output": str(output), "action_horizon": 16, "use_tactile": True,
              "PI05_CHECKPOINT": str(output), "PI05_NORM_STATS": str(output / "assets" / args.asset_id / "norm_stats.json")}
    if not args.execute:
        return result
    require_write_authority(args.execute)
    from rlinf.utils.ckpt_convertor.openpi.jax_to_openpi_rlinf import convert
    output.mkdir(parents=True, exist_ok=False)
    # Existing tactile-aware RLinf converter; preserves all 23 tactile tensors.
    convert(params.parent, norm, output, output / "assets" / args.asset_id / "norm_stats.json",
            pi05=True, action_dim=32, action_horizon=16, max_token_len=200, use_tactile=True)
    result.update(mode="converted", validation=validate_checkpoint(output, args.asset_id))
    (output / "conversion_provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("plan", "preflight", "norm-stats", "train-jax", "train-torch"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--execute", action="store_true")
        if name == "preflight":
            cmd.add_argument("--stage", choices=("norm-stats", "train-jax"), default="train-jax")
            cmd.add_argument("--check-imports", action="store_true")
            cmd.add_argument("--check-gpu", action="store_true")
    cmd = sub.add_parser("convert")
    for name in ("input-params", "norm-stats", "asset-id", "output"):
        cmd.add_argument("--" + name, required=True)
    cmd.add_argument("--execute", action="store_true")
    cmd = sub.add_parser("validate-torch")
    cmd.add_argument("--checkpoint", required=True)
    cmd.add_argument("--asset-id", required=True)
    args = parser.parse_args(argv)
    if args.operation == "convert":
        report = conversion(args)
    elif args.operation == "validate-torch":
        report = validate_checkpoint(absolute_path(args.checkpoint), args.asset_id)
    else:
        plan = Plan.load(args.config)
        if args.operation == "train-torch":
            raise ValueError("Tactile PyTorch supervised training is NOT supported by the archived trainer. Use train-jax; do not use vision-only PI0Pytorch.")
        if args.operation == "plan" or (args.operation != "preflight" and not args.execute):
            report = plan.describe()
            report["requested_operation"] = args.operation
        elif args.operation == "preflight":
            report = asset_preflight(plan, args.stage)
            if args.check_imports:
                try:
                    report["imports"] = imports_check(plan)
                except Exception as exc:
                    report["ok"] = False
                    report["errors"].append(f"CPU imports: {type(exc).__name__}: {exc}")
            if args.check_gpu:
                report["gpu"] = gpu_check(plan)
                report["ok"] &= report["gpu"]["ok"]
        else:
            require_write_authority(args.execute)
            report = asset_preflight(plan, args.operation)
            if not report["ok"]:
                print(json.dumps(report, indent=2))
                return 2
            if args.operation == "norm-stats":
                report = norm_stats(plan)
            else:
                check = gpu_check(plan)
                if not check["ok"]:
                    print(json.dumps(check, indent=2))
                    return 2
                from .build_config import build_train_config
                cfg = build_train_config(plan)
                # run.sh chooses CUDA before this Python process imports JAX.
                if os.environ.get("JAX_PLATFORMS") != "cuda":
                    raise ValueError("JAX_PLATFORMS must be cuda for actual SFT, never inherit the GRPO CPU setting")
                script = load_script("train")
                import logging
                logging.basicConfig(level=logging.INFO)
                # New output-local working directory also contains incidental run logs.
                work_dir = Path(plan.output_root) / "run_metadata" / plan.name / plan.exp_name
                work_dir.mkdir(parents=True, exist_ok=False)
                (work_dir / "plan.json").write_text(json.dumps(plan.describe(), indent=2) + "\n")
                os.chdir(work_dir)
                script.main(cfg)
                report = {"completed": True, "checkpoint_dir": str(cfg.checkpoint_dir)}
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
