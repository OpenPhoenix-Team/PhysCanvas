"""Dependency-free configuration, path safety, and read-only asset checks."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from pathlib import Path


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Expected a single safe path component, got {value!r}")
    return value


def absolute_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute() or "$" in str(path):
        raise ValueError(f"Use an explicit absolute local path, got {value!r}")
    return path.resolve()


def overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


@dataclasses.dataclass(frozen=True)
class Plan:
    name: str
    exp_name: str
    lerobot_root: str
    tactile_sidecar_root: str
    initial_params: str
    tokenizer_file: str
    output_root: str
    asset_id: str
    batch_size: int = 64
    num_train_steps: int = 60000
    fsdp_devices: int = 8
    num_workers: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    save_interval: int = 1000
    keep_period: int = 5000
    log_interval: int = 100
    seed: int = 42
    wandb_enabled: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        plan = cls(**json.loads(Path(path).read_text()))
        for key in ("name", "exp_name", "asset_id"):
            identifier(getattr(plan, key))
        for key in ("lerobot_root", "tactile_sidecar_root", "initial_params", "tokenizer_file", "output_root"):
            object.__setattr__(plan, key, str(absolute_path(getattr(plan, key))))
        identifier(plan.repo_id)
        out = Path(plan.output_root)
        if out in (Path("/"), Path.home()) or len(out.parts) < 3:
            raise ValueError("output_root must be a dedicated directory, not a broad root")
        for key in ("lerobot_root", "tactile_sidecar_root", "initial_params", "tokenizer_file"):
            if overlaps(out, Path(getattr(plan, key))):
                raise ValueError(f"output_root must be disjoint from input {key}")
        for key in ("batch_size", "num_train_steps", "fsdp_devices", "save_interval", "keep_period", "log_interval"):
            if type(getattr(plan, key)) is not int or getattr(plan, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if plan.num_workers < 0 or plan.warmup_steps < 0 or plan.batch_size % plan.fsdp_devices:
            raise ValueError("Invalid workers/warmup or batch_size is not divisible by fsdp_devices")
        if not math.isfinite(plan.learning_rate) or plan.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        return plan

    @property
    def repo_id(self) -> str:
        # LeRobot resolves HF_LEROBOT_HOME / repo_id, independently of meta.repo_id.
        return Path(self.lerobot_root).name

    @property
    def norm_stats(self) -> Path:
        return Path(self.output_root) / "assets" / self.name / self.asset_id / "norm_stats.json"

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_root) / "checkpoints" / self.name / self.exp_name

    def describe(self) -> dict:
        return {
            **dataclasses.asdict(self), "mode": "plan_only", "repo_id": self.repo_id,
            "HF_LEROBOT_HOME": str(Path(self.lerobot_root).parent),
            "norm_stats": str(self.norm_stats), "checkpoint_dir": str(self.checkpoint_dir),
            "model": {"pi05": True, "action_dim": 32, "robot_action_dim": 14, "action_horizon": 16,
                      "tactile_shape": [4, 700, 3], "tactile_semantics": "dx,dy,dz", "gate_init": .01},
            "optimizer": {"type": "AdamW", "beta1": .9, "beta2": .95, "eps": 1e-8,
                          "weight_decay": 1e-10, "clip_gradient_norm": 1., "ema_decay": .999},
            "schedule": "1000-step warmup by default, then constant peak_lr (decay_lr equals peak_lr)",
            "torch_supervised_training": "unsupported for tactile; use train-jax",
            "note": "This plan description itself performs no model load, dataset decode, download, directory creation, or training.",
        }


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_norm_stats(path: Path) -> dict:
    stats = json.loads(path.read_text())["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = stats[key][field]
            if not isinstance(values, list) or len(values) != 14 or not all(
                isinstance(x, (int, float)) and math.isfinite(x) for x in values
            ):
                raise ValueError(f"{key}.{field} must contain 14 finite numbers")
        if any(x < 0 for x in stats[key]["std"]):
            raise ValueError(f"{key}.std contains negatives")
        if any(a > b for a, b in zip(stats[key]["q01"], stats[key]["q99"])):
            raise ValueError(f"{key} quantile ordering invalid")
    return {"path": str(path), "sha256": sha256(path), "dimensions": 14}


def asset_preflight(plan: Plan, stage: str = "train-jax") -> dict:
    errors: list[str] = []
    details: dict = {}
    root, tactile = Path(plan.lerobot_root), Path(plan.tactile_sidecar_root)
    required = [root / "meta/info.json", root / "meta/tasks.jsonl", root / "meta/episodes.jsonl",
                root / "meta/episodes_stats.jsonl", tactile / "summary.json", tactile / "episodes.jsonl"]
    for path in required:
        if not path.is_file():
            errors.append(f"Missing required file: {path}")
    conversion_summary = root / "meta/conversion_summary.json"
    is_new_bundle = (root.parent / "source_manifest.json").exists()
    if conversion_summary.exists():
        try:
            version = json.loads(conversion_summary.read_text()).get("conversion_version", "")
            is_new_bundle |= str(version).startswith("physcanvas-pi05")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"Invalid conversion summary: {exc}")
    if is_new_bundle:
        try:
            complete = json.loads((root.parent / "COMPLETE.json").read_text())
            if complete.get("complete") is not True or (root.parent / "FAILED.json").exists():
                errors.append("New data_processing bundle is not successfully complete")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"New bundle COMPLETE gate: {exc}")
    try:
        info = json.loads((root / "meta/info.json").read_text())
        if info["codebase_version"] != "v2.1" or info["fps"] != 10:
            errors.append("Expected LeRobot v2.1 at 10 Hz")
        for key in ("observation.state", "action"):
            if info["features"][key]["shape"] != [14]:
                errors.append(f"Expected {key} shape [14]")
        for view in ("base_0", "left_wrist_0", "right_wrist_0"):
            feature = info["features"][f"observation.images.{view}"]
            if feature["dtype"] != "video":
                errors.append(f"Expected video feature {view}")
        total = int(info["total_episodes"])
        details.update(episodes=total, frames=int(info["total_frames"]))
        if int(info["total_frames"]) < plan.batch_size:
            errors.append("Dataset has fewer frames than one full batch; historical normalizer drops last")
        summary = json.loads((tactile / "summary.json").read_text())
        if summary.get("complete") is not True or summary.get("mapped_variants") != total:
            errors.append("Sidecar summary must be complete and mapped_variants equal total_episodes")
        rows = [json.loads(line) for line in (tactile / "episodes.jsonl").read_text().splitlines() if line.strip()]
        if sorted(row["global_episode_index"] for row in rows) != list(range(total)):
            errors.append("Sidecar episode indices must be unique contiguous [0,total_episodes)")
        episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        by_ep = {row["episode_index"]: row for row in episodes}
        for row in rows:
            ep = row["global_episode_index"]
            if row["frame_count"] != by_ep[ep]["length"]:
                errors.append(f"Sidecar / RGB frame count mismatch at episode {ep}")
            for key in ("tactile_relpath", "tactile_mask_relpath"):
                path = (tactile / row[key]).resolve()
                if tactile not in path.parents or not path.is_file():
                    errors.append(f"Missing/escaping sidecar file: {path}")
            chunk = ep // int(info.get("chunks_size", 1000))
            paths = [root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)]
            for key, feature in info["features"].items():
                if feature["dtype"] == "video":
                    paths.append(root / info["video_path"].format(episode_chunk=chunk, episode_index=ep, video_key=key))
            for path in paths:
                if root not in path.resolve().parents or not path.is_file():
                    errors.append(f"Missing/escaping LeRobot media/parquet: {path}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Dataset metadata validation: {exc}")
    if stage == "train-jax":
        if not Path(plan.initial_params).is_dir():
            errors.append(f"Missing local JAX initial params: {plan.initial_params}")
        if not Path(plan.tokenizer_file).is_file():
            errors.append(f"Missing local tokenizer: {plan.tokenizer_file}")
        try:
            details["norm_stats"] = check_norm_stats(plan.norm_stats)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Norm stats not ready: {exc}")
        if plan.checkpoint_dir.exists():
            errors.append(f"Checkpoint directory already exists; overwrite/resume forbidden: {plan.checkpoint_dir}")
    elif plan.norm_stats.parent.exists():
        errors.append(f"Norm-stat output already exists; will not overwrite: {plan.norm_stats.parent}")
    return {"ok": not errors, "stage": stage, "errors": errors, "details": details,
            "lerobot_resolved_root": str(Path(plan.lerobot_root).parent / plan.repo_id)}


def require_write_authority(execute: bool) -> None:
    if not execute or os.environ.get("POLICY_TRAINING_ALLOW_WRITE") != "1":
        raise ValueError("Execution requires both --execute and POLICY_TRAINING_ALLOW_WRITE=1")
