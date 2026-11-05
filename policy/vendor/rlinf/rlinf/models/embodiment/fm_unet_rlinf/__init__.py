"""RLinf adapters for the task-specific tactile FM-UNet policy."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from .rl_action_model import FMUnetRLActionModel, FMUnetRLConfig
from .sampling import flow_step_mean_std, gaussian_logprob
from .timing import FmWmTiming, DEFAULT_FM_WM_TIMING


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def get_model(cfg: Any, torch_dtype: torch.dtype | None = None) -> FMUnetRLActionModel:
    """Load a task FM-UNet checkpoint and wrap it for exact GRPO replay."""

    fm_cfg = _config_value(cfg, "fm", {})
    policy_root = Path(_config_value(fm_cfg, "policy_root")).expanduser().resolve()
    run_dir = Path(_config_value(fm_cfg, "run_dir")).expanduser().resolve()
    checkpoint = Path(
        _config_value(cfg, "model_path", _config_value(fm_cfg, "checkpoint"))
    ).expanduser().resolve()
    for path, label in (
        (policy_root, "policy_root"),
        (run_dir, "run_dir"),
        (checkpoint, "checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"FM {label} does not exist: {path}")

    policy_root_text = str(policy_root)
    if policy_root_text not in sys.path:
        sys.path.insert(0, policy_root_text)

    from infer.config import (  # type: ignore[import-not-found]
        build_policy_from_cfg,
        load_run_config,
        policy_config_from_checkpoint_state,
    )
    from models.policy_factory import resolve_policy_type  # type: ignore[import-not-found]
    from tools.normalizer import DatasetNormalizer  # type: ignore[import-not-found]

    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fallback_config = load_run_config(run_dir)
    policy_config = policy_config_from_checkpoint_state(
        checkpoint_state, fallback_config
    )
    if resolve_policy_type(policy_config) != "flow_matching":
        raise ValueError("fm_unet_rlinf requires a flow_matching checkpoint")
    velocity_model = str(policy_config["models"]["fm"].get("velocity_model", "unet"))
    if velocity_model != "unet":
        raise ValueError(
            f"fm_unet_rlinf requires velocity_model='unet', got {velocity_model!r}"
        )
    # The full checkpoint contains the DINO trunk.  Avoid any hub access while
    # constructing the model before loading its state dict.
    policy_config["models"]["fm"]["image_pretrained"] = False
    policy = build_policy_from_cfg(
        policy_config,
        match_training=True,
        policy_state_dict=checkpoint_state["policy_state_dict"],
    )
    policy.load_state_dict(checkpoint_state["policy_state_dict"], strict=True)
    if torch_dtype is not None:
        policy.to(dtype=torch_dtype)
    normalizer = DatasetNormalizer.load_state_dict(
        checkpoint_state["normalizer_state_dict"]
    )
    resolved = (
        OmegaConf.to_container(fm_cfg, resolve=True)
        if OmegaConf.is_config(fm_cfg)
        else dict(fm_cfg)
    )
    rl_config = FMUnetRLConfig.from_mapping(resolved)
    checkpoint_action_arm = str(policy_config["data"].get("action_arm", "both"))
    configured_action_arm = resolved.get("action_arm")
    if (
        configured_action_arm is not None
        and str(configured_action_arm) != checkpoint_action_arm
    ):
        raise ValueError(
            "configured FM action_arm conflicts with checkpoint metadata: "
            f"configured={configured_action_arm!r}, "
            f"checkpoint={checkpoint_action_arm!r}"
        )
    rl_config = dataclasses.replace(rl_config, action_arm=checkpoint_action_arm)
    return FMUnetRLActionModel(policy, normalizer=normalizer, config=rl_config)


__all__ = [
    "DEFAULT_FM_WM_TIMING",
    "FMUnetRLActionModel",
    "FMUnetRLConfig",
    "FmWmTiming",
    "flow_step_mean_std",
    "gaussian_logprob",
    "get_model",
]
