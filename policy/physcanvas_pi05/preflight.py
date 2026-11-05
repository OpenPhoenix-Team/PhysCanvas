"""Read-only configuration, asset and import checks; never starts Ray or models."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from physcanvas_pi05.config import compose_config, compose_rollout_config, TASK_NAMES


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def validate_experiment(
    cfg: DictConfig, *, check_assets: bool = True, check_imports: bool = False
) -> dict:
    """Validate without RLinf validate_cfg, which would initialize a cluster."""
    env = cfg.env.train
    wm = env.world_model
    _require(wm.backend == "physcanvas", "Only the public PhysCanvas backend is allowed")
    _require(env.action_horizon == 16 and env.wm_sub_frames == 8, "Expected policy16 / WM8x2")
    _require(env.action_dim == 14, "Expected 14 absolute joint targets")
    _require(env.max_episode_steps == env.max_policy_chunks * 16, "Inconsistent episode horizon")
    _require(env.max_steps_per_rollout_epoch == env.max_episode_steps, "Rollout horizon mismatch")
    _require(cfg.actor.model.num_action_chunks == 16, "Actor horizon must be 16")
    _require(cfg.actor.model.action_dim == 14, "Actor action dimension must be 14")
    _require(cfg.algorithm.group_size == env.group_size == 8, "GRPO groups must contain 8 candidates")
    _require(env.total_num_envs % env.group_size == 0, "Environments must form complete groups")
    _require(env.total_num_envs * env.rollout_epoch == 512, "Expected 512 trajectories per update")
    _require(wm.tactile_flow_encoding == "xyz_linear", "WM flow must preserve signed XYZ")
    _require(wm.tactile_third_channel == "xyz", "Policy must receive generated dz, not magnitude/zero")
    _require(abs(wm.tactile_flow_linear_max_disp - 0.06) < 1e-9, "Expected public WM scale .06")
    _require(int(wm.num_inference_steps) > 0, "WM steps must be positive")
    _require(env.policy_tactile_mode == "enabled", "Policy tactile must be enabled")
    _require(env.reward_model.threshold == 0.7, "Task RM threshold must be .7")
    _require(env.terminate_on_rm_success and env.skip_terminated_wm_rollouts, "Success must stop future WM calls")
    _require(cfg.runner.save_interval == 4, "Save every 4 updates")
    _require(not cfg.actor.fsdp_config.enable_gradient_accumulation, "Gradient accumulation is disabled")
    _require(cfg.algorithm.adv_type == "grpo" and cfg.algorithm.loss_type == "actor", "Expected GRPO actor-only objective")
    _require(not cfg.critic.use_critic_model, "No trained critic in GRPO")
    _require(cfg.algorithm.filter_rewards, "Invalid groups must be filtered")
    _require(cfg.algorithm.rewards_lower_bound == 0.125 and cfg.algorithm.rewards_upper_bound == 0.875,
             "Keep groups with 1..7 successes out of 8")
    _require(not cfg.reward.use_reward_model, "Task RM runs inside EnvWorker, not a separate RLinf reward worker")
    expected_mode = "relative_sum" if cfg.task == "building_blocks" else "binary_any_success"
    _require(env.trajectory_reward_aggregation == expected_mode, "Task reward mode mismatch")
    _require(env.use_rel_reward == (cfg.task == "building_blocks"), "Task relative reward switch mismatch")
    if check_assets:
        model_root = _path(cfg.actor.model.model_path, "Pi0.5 checkpoint")
        _require((model_root / "model.safetensors").is_file(), "Initial Pi0.5 must be the converted 59999 safetensors checkpoint")
        _path(cfg.actor.model.openpi_data.norm_stats_path, "Pi0.5 normalization")
        _path(env.reward_model.checkpoint, "Private task RM checkpoint")
        _path(wm.reset_cache, "Reset cache")
        _path(wm.selection_manifest, "Reset selection manifest")
        from physcanvas_pi05.reset_data import ResetDataset
        from physcanvas_pi05.canvas import SIDE_ORDER, VIEW_ORDER
        reset_dataset = ResetDataset(wm.reset_cache, wm.selection_manifest)
        _require(len(reset_dataset) == 128, "These experiments require exactly 128 reset observations")
        required_names = ["qpos.npy"] + [f"{view}_rgb.npy" for view in VIEW_ORDER]
        required_names += [f"tactile/{side}.{kind}.npy"
                           for side in SIDE_ORDER for kind in ("pointcloud", "difference")]
        for episode_id in reset_dataset.episode_ids:
            if cfg.task == "test_tube":
                _require(episode_id.startswith("use_both_arms_collaboratively_"),
                         f"Shiguan reset is not from the double-arm pool: {episode_id}")
            for relative in required_names:
                _path(str(reset_dataset.root / "episodes" / episode_id / relative), "Reset modality")
        _path(wm.renderer.settings_file, "Motion renderer settings")
        _path(wm.pretrained_vae, "Wan VAE/scheduler")
        _path(wm.prompt_embedding, "WM prompt embedding")
        source = _path(wm.source_root, "Public PhysCanvas source")
        _require((source / "evaluation/model_eval.py").is_file(), "Public evaluation inference source missing")
        public = json.loads(_path(wm.public_config, "Public WM config").read_text())
        dl = public["training_config"]["dataloaders"]["test"]
        transform = dl["transform"]
        _require(transform["unified_wan_tactile_flow_encoding"] == "xyz_linear", "Public config flow contract differs")
        _require(transform["unified_wan_tactile_flow_linear_max_disp"] == 0.06, "Public config flow scale differs")
        _require(transform["unified_wan_include_depth"] is False, "This release is the no-depth WM")
        _require(transform["dual_reference_mode"] == "dual_memory", "Expected fixed anchor plus generated local reference")
        ckpt = _path(wm.checkpoint, "Public WM checkpoint")
        manifest_path = _path(str(ckpt / "manifest.json"), "Public checkpoint manifest")
        manifest_bytes = manifest_path.read_bytes()
        _require(hashlib.sha256(manifest_bytes).hexdigest() ==
                 "72cd121e50d40c874df7716e6086d06c98af6c956754e1d3193dd8b00aef70ea",
                 "Manifest is not the released epoch30_step300 checkpoint")
        manifest = json.loads(manifest_bytes)
        weights = _path(str(ckpt / manifest["source_filename"]), "Assembled public WM weights")
        _require(weights.stat().st_size == manifest["source_size"], "Assembled WM weight size mismatch")
        # Full multi-GB SHA verification is performed by the download/WM loader;
        # this preflight does not claim a content verification from a size check.
    imports = {}
    if check_imports:
        for name in [
            "torch", "diffusers", "hydra", "ray", "openpi",
            "rlinf.models.embodiment.openpi_rlinf",
            "rlinf.envs.world_model.jimu_wm_grpo_env",
            "physcanvas_pi05.backend",
        ]:
            module = importlib.import_module(name)
            imports[name] = getattr(module, "__file__", None)
    return {
        "task": cfg.task,
        "nodes": cfg.cluster.num_nodes,
        "trajectories_per_update": env.total_num_envs * env.rollout_epoch,
        "groups_per_update": env.total_num_envs * env.rollout_epoch // env.group_size,
        "max_frames": env.max_episode_steps,
        "wm_steps": wm.num_inference_steps,
        "wm_checkpoint": wm.checkpoint,
        "flow": "xyz_linear, .06; signed dx/dy/dz passed to policy",
        "noise_level": cfg.actor.model.openpi.noise_level,
        "reward_mode": env.trajectory_reward_aggregation,
        "threshold": env.reward_model.threshold,
        "save_interval": cfg.runner.save_interval,
        "imports": imports,
        "status": "static/import checks passed; not a GPU rollout or training test",
    }


def validate_rollout(cfg, *, check_assets=True, check_imports=False):
    """Validate standalone inference without examining any reward-model assets."""
    env = cfg.env.train
    _require(env.reward_model is None and cfg.actor.model.openpi.noise_method == "flow_ode",
             "Standalone rollout must be ODE without an RM")
    if check_assets:
        from .wm_runtime import validate_public_contract
        from .reset_data import ResetDataset
        validate_public_contract(env.world_model, hash_weights=False)
        _path(Path(cfg.actor.model.model_path) / "model.safetensors", "Pi0.5 base")
        _path(cfg.actor.model.openpi_data.norm_stats_path, "Normalization")
        _path(env.world_model.renderer.settings_file, "Motion settings")
        ResetDataset(env.world_model.reset_cache, env.world_model.selection_manifest)
    if check_imports:
        for name in ("physcanvas_pi05.backend", "rlinf.models.embodiment.openpi_rlinf", "diffusers"):
            importlib.import_module(name)
    return {"task": cfg.task, "sampler": "flow_ode", "rm_scoring": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--rollout", action="store_true", help="Check RM-free ODE inference instead of GRPO")
    parser.add_argument("--config-only", action="store_true", help="Skip assets/imports")
    parser.add_argument("--imports", action="store_true", help="Also import runtime modules, no model loads")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = (compose_rollout_config if args.rollout else compose_config)(args.task, args.overrides)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
    summary = (validate_rollout if args.rollout else validate_experiment)(
        cfg, check_assets=not args.config_only,
        check_imports=args.imports and not args.config_only,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
