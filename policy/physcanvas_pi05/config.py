"""Shared, side-effect-free configuration for rollout and GRPO."""

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


TASK_ALIASES = {"jimu": "building_blocks", "shiguan": "test_tube"}
TASK_NAMES = ("building_blocks", "test_tube", *TASK_ALIASES)


def compose_rollout_config(task: str, overrides: Sequence[str] = ()) -> DictConfig:
    # Remove the RM node *before* interpolation, so RM_CHECKPOINT is neither
    # read nor required. Do not apply any of these overrides to GRPO.
    return compose_config(task, [*overrides, "env.train.reward_model=null",
        "env.train.terminate_on_rm_success=false", "env.eval.terminate_on_rm_success=false",
        "actor.model.openpi.noise_method=flow_ode"])


def compose_config(task: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose an experiment without initializing Ray, CUDA, or models."""
    task = TASK_ALIASES.get(task, task)
    if task not in ("building_blocks", "test_tube"):
        raise ValueError("task must be building_blocks or test_tube (both-arm test tube)")
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=task, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg
