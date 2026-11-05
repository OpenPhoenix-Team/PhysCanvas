"""Shared, side-effect-free configuration for rollout and GRPO."""

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def compose_config(task: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose an experiment without initializing Ray, CUDA, or models."""
    if task not in {"jimu", "shiguan"}:
        raise ValueError("task must be jimu or shiguan (both arms)")
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=task, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg
