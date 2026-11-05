"""Build the historical arm-fix config without mutating its registry or assets."""
import dataclasses
import os
from pathlib import Path

from .plan import Plan

TEMPLATE = "pi05_arx_multitask_nofail_77src_armfix_tactile_60k_finetune"


def set_dataset_environment(plan: Plan) -> None:
    # Set before importing LeRobot; it captures these constants at import time.
    os.environ["HF_LEROBOT_HOME"] = str(Path(plan.lerobot_root).parent)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_CACHE"] = str(Path(plan.output_root) / "cache" / "datasets")
    os.environ["OPENPI_TOKENIZER_PATH"] = plan.tokenizer_file
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(Path(plan.output_root) / "cache" / "jax")
    os.environ["WANDB_DIR"] = str(Path(plan.output_root) / "wandb")
    os.environ["WANDB_MODE"] = "offline"


def build_train_config(plan: Plan):
    set_dataset_environment(plan)
    from openpi.training import config as cfg
    from openpi.training import optimizer, weight_loaders
    template = cfg.get_config(TEMPLATE)
    data = dataclasses.replace(
        template.data, repo_id=plan.repo_id,
        assets=cfg.AssetsConfig(asset_id=plan.asset_id),
        tactile_sidecar_root=plan.tactile_sidecar_root,
    )
    return dataclasses.replace(
        template, name=plan.name, exp_name=plan.exp_name, data=data,
        weight_loader=weight_loaders.TactileCheckpointWeightLoader(plan.initial_params),
        assets_base_dir=str(Path(plan.output_root) / "assets"),
        checkpoint_base_dir=str(Path(plan.output_root) / "checkpoints"),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=plan.warmup_steps,
            peak_lr=plan.learning_rate, decay_steps=1000000, decay_lr=plan.learning_rate),
        batch_size=plan.batch_size, num_train_steps=plan.num_train_steps,
        fsdp_devices=plan.fsdp_devices, num_workers=plan.num_workers,
        save_interval=plan.save_interval, keep_period=plan.keep_period,
        log_interval=plan.log_interval, seed=plan.seed,
        wandb_enabled=plan.wandb_enabled, overwrite=False, resume=False,
    )


def stats_data_config(plan: Plan):
    """No tokenizer, network, or model-transform construction for norm stats."""
    set_dataset_environment(plan)
    from openpi.training.config import DataConfig
    return DataConfig(repo_id=plan.repo_id, asset_id=plan.asset_id,
                      tactile_sidecar_root=plan.tactile_sidecar_root,
                      action_sequence_keys=("action",), prompt_from_task=False)
