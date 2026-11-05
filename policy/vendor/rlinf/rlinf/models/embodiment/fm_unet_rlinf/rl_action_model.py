"""GRPO actor adapter for the task-specific tactile FM-UNet policy."""

from __future__ import annotations

import dataclasses
import random
from contextlib import nullcontext
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .sampling import flow_step_mean_std, flow_timesteps, gaussian_logprob
from .timing import DEFAULT_FM_WM_TIMING, FmWmTiming


@dataclasses.dataclass(frozen=True)
class FMUnetRLConfig:
    """Static rollout/replay settings for FM-UNet GRPO."""

    num_steps: int = 32
    noise_method: str = "flow_sde"
    noise_level: float = 0.3
    policy_horizon: int = 64
    executed_policy_steps: int = 48
    policy_hz: int = 30
    wm_hz: int = 10
    wm_steps_per_call: int = 8
    action_arm: str = "both"
    compute_precision: str = "bf16"

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        if self.noise_method != "flow_sde":
            raise ValueError("FM GRPO currently requires noise_method='flow_sde'")
        if self.noise_level <= 0:
            raise ValueError("noise_level must be positive for GRPO exploration")
        if self.action_arm not in {"both", "left", "right"}:
            raise ValueError(
                "action_arm must be one of {'both', 'left', 'right'}, "
                f"got {self.action_arm!r}"
            )
        if self.compute_precision not in {"bf16", "fp32"}:
            raise ValueError("FM compute_precision must be 'bf16' or 'fp32'")
        self.timing

    @property
    def action_dim(self) -> int:
        return 14 if self.action_arm == "both" else 7

    @property
    def action_slice(self) -> slice:
        if self.action_arm == "right":
            return slice(7, 14)
        if self.action_arm == "left":
            return slice(0, 7)
        return slice(0, 14)

    @property
    def timing(self) -> FmWmTiming:
        return FmWmTiming(
            policy_horizon=self.policy_horizon,
            executed_policy_steps=self.executed_policy_steps,
            policy_hz=self.policy_hz,
            wm_hz=self.wm_hz,
            wm_steps_per_call=self.wm_steps_per_call,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FMUnetRLConfig":
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in fields})


class FMUnetRLActionModel(nn.Module, BasePolicy):
    """Wrap the existing FM policy with rollout-chain collection and replay.

    The env receives all 48 executed 30 Hz absolute targets.  PPO/GRPO stores
    one log-probability token per 10 Hz WM frame by summing each corresponding
    group of three transition-coordinate log-probabilities.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        normalizer: Any,
        config: FMUnetRLConfig,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.observation_steps = int(getattr(policy, "cond_steps", 8))
        if self.observation_steps not in (1, 8):
            raise ValueError("FM GRPO supports checkpoint cond_steps=1 or 8")
        self.policy.eval()
        self.normalizer = normalizer
        self.rl_config = config
        self.timing = config.timing
        action_horizon = int(getattr(policy, "action_horizon", -1))
        action_dim = int(getattr(policy, "action_dim", -1))
        if action_horizon != self.timing.policy_horizon:
            raise ValueError(
                "FM checkpoint horizon does not match the rollout contract: "
                f"checkpoint={action_horizon}, required={self.timing.policy_horizon}"
            )
        if action_dim != self.rl_config.action_dim:
            raise ValueError(
                "FM checkpoint action dimension does not match action_arm: "
                f"checkpoint={action_dim}, action_arm={self.rl_config.action_arm!r}, "
                f"expected={self.rl_config.action_dim}"
            )

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def train(self, mode: bool = True) -> "FMUnetRLActionModel":
        """Train weights while retaining the rollout network's deterministic mode.

        The FM condition encoder has dropout. PPO must replay the sampled
        transition with the same network computation, so enabling dropout or
        updating BatchNorm statistics only in the actor would corrupt ratios.
        Evaluation module mode does not disable parameter autograd.
        """
        super().train(mode)
        self.policy.eval()
        return self

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.policy.parameters()).dtype

    def set_global_step(self, global_step: int) -> None:
        """Keep a constant exploration level; retained for RLinf compatibility."""

        del global_step

    def _autocast_context(self):
        if self.device.type != "cuda" or self.rl_config.compute_precision == "fp32":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _required_tensor(env_obs: Mapping[str, Any], key: str) -> torch.Tensor:
        if key not in env_obs:
            raise KeyError(
                f"FM policy observation is missing {key!r}; use the FM 30 Hz "
                "history environment adapter, not the Pi0.5 single-frame payload"
            )
        return torch.as_tensor(env_obs[key])

    def _normalize_observation(
        self, env_obs: Mapping[str, Any]
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        image_keys = ("main_images", "wrist_images", "extra_view_images")
        images = [self._required_tensor(env_obs, key) for key in image_keys]
        if any(image.ndim != 4 or image.shape[1] != 3 for image in images):
            raise ValueError("FM RGB views must each have shape [B,3,H,W]")
        image = torch.stack(images, dim=1).unsqueeze(1).contiguous()

        raw_state = self._required_tensor(env_obs, "state_history_30hz").float()
        raw_tactile = self._required_tensor(env_obs, "tactile_history_30hz").float()
        batch_size = image.shape[0]
        if tuple(raw_state.shape) != (batch_size, 8, 14):
            raise ValueError(
                f"state_history_30hz must be [B,8,14], got {tuple(raw_state.shape)}"
            )
        if tuple(raw_tactile.shape) != (batch_size, 8, 35, 20, 12):
            raise ValueError(
                "tactile_history_30hz must be [B,8,35,20,12], "
                f"got {tuple(raw_tactile.shape)}"
            )

        # The environment retains its aligned history; the checkpoint decides
        # whether the policy consumes its current endpoint or all eight frames.
        raw_state = raw_state[:, -self.observation_steps:]
        raw_tactile = raw_tactile[:, -self.observation_steps:]
        full_state_np = (
            raw_state.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        model_state_np = full_state_np[..., self.rl_config.action_slice]
        tactile_np = raw_tactile.detach().cpu().numpy().astype(np.float32, copy=False)
        state_norm = self.normalizer.normalize_state_np(model_state_np)
        tactile_norm = self.normalizer.normalize_tactile_np(tactile_np)
        observation = {
            "image": image.to(self.device, non_blocking=True),
            "state": torch.from_numpy(np.asarray(state_norm, dtype=np.float32)).to(
                self.device, non_blocking=True
            ),
            "tactile": torch.from_numpy(
                np.asarray(tactile_norm, dtype=np.float32)
            ).to(self.device, non_blocking=True),
        }
        return observation, model_state_np, full_state_np

    def _unnormalize_actions(
        self,
        normalized: torch.Tensor,
        model_state_history: np.ndarray,
        full_state_history: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_np = normalized.detach().float().cpu().numpy()
        model_absolute = [
            self.normalizer.unnormalize_action_np(actions, state_history)
            for actions, state_history in zip(normalized_np, model_state_history)
        ]
        model_absolute_np = np.ascontiguousarray(
            np.stack(model_absolute), dtype=np.float32
        )
        if self.rl_config.action_arm == "both":
            full_absolute_np = model_absolute_np.copy()
        else:
            anchors = full_state_history[:, -1, :]
            full_absolute_np = np.broadcast_to(
                anchors[:, None, :], normalized_np.shape[:2] + (14,)
            ).copy()
            full_absolute_np[..., self.rl_config.action_slice] = model_absolute_np
        model_absolute_tensor = torch.from_numpy(model_absolute_np).to(self.device)
        full_absolute_tensor = torch.from_numpy(
            np.ascontiguousarray(full_absolute_np, dtype=np.float32)
        ).to(self.device)
        return model_absolute_tensor, full_absolute_tensor

    def _condition(
        self, observation: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with self._autocast_context():
            return self.policy._build_condition(observation)

    def _velocity(
        self,
        trajectory: torch.Tensor,
        timestep: torch.Tensor,
        global_condition: torch.Tensor,
        condition_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        with self._autocast_context():
            result = self.policy._model_forward(
                trajectory,
                timestep,
                global_cond=global_condition,
                condition_tokens=condition_tokens,
            )
        return result.float()

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate 64 actions, execute 48, and retain exact replay inputs."""

        del compute_values, kwargs
        if mode not in {"train", "eval"}:
            raise ValueError(f"unsupported prediction mode {mode!r}")
        observation, model_state_history, full_state_history = (
            self._normalize_observation(env_obs)
        )
        global_condition, condition_tokens = self._condition(observation)
        batch_size = int(global_condition.shape[0])
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.timing.policy_horizon,
                self.rl_config.action_dim,
                device=self.device,
                dtype=torch.float32,
                generator=rng,
            )
        expected_noise_shape = (
            batch_size,
            self.timing.policy_horizon,
            self.rl_config.action_dim,
        )
        if tuple(noise.shape) != expected_noise_shape:
            raise ValueError(f"noise has incompatible shape {tuple(noise.shape)}")

        chosen = random.randint(0, self.rl_config.num_steps - 1)
        denoise_indices = torch.full(
            (batch_size,), chosen, device=self.device, dtype=torch.long
        )
        times = flow_timesteps(
            self.rl_config.num_steps, device=self.device, dtype=torch.float32
        )
        trajectory = noise.float()
        chains = [trajectory]
        chosen_logprobs = torch.zeros_like(trajectory)
        for step_index in range(self.rl_config.num_steps):
            step_indices = torch.full_like(denoise_indices, step_index)
            timestep = times[step_index].expand(batch_size)
            velocity = self._velocity(
                trajectory, timestep, global_condition, condition_tokens
            )
            stochastic = mode == "train" and step_index == chosen
            mean, std = flow_step_mean_std(
                trajectory,
                velocity,
                step_indices,
                num_steps=self.rl_config.num_steps,
                noise_method=self.rl_config.noise_method if stochastic else "flow_ode",
                noise_level=self.rl_config.noise_level,
            )
            if stochastic:
                step_noise = torch.randn(
                    trajectory.shape,
                    device=self.device,
                    dtype=torch.float32,
                    generator=rng,
                )
                trajectory = mean + step_noise * std
                chosen_logprobs = gaussian_logprob(trajectory, mean, std)
            else:
                trajectory = mean
            chains.append(trajectory)

        model_absolute, full_absolute = self._unnormalize_actions(
            trajectory, model_state_history, full_state_history
        )
        executed = self.timing.executed_actions(full_absolute)
        wm_actions = self.timing.wm_actions(full_absolute)
        policy_actions = self.timing.wm_actions(model_absolute)
        compressed_logprobs = self.timing.compress_logprobs(chosen_logprobs)
        forward_inputs = {
            "chains": torch.stack(chains, dim=1).contiguous(),
            "denoise_indices": denoise_indices.contiguous(),
            "fm_obs_image": observation["image"].contiguous(),
            "fm_obs_state": observation["state"].contiguous(),
            "fm_obs_tactile": observation["tactile"].contiguous(),
            # GRPO-facing actions stay in the checkpoint's trained action space.
            "action": policy_actions.reshape(batch_size, -1).contiguous(),
            # Environment-facing actions are always full dual-arm 14-D targets.
            "wm_action_14d": wm_actions.reshape(batch_size, -1).contiguous(),
            "executed_action_30hz": executed.reshape(batch_size, -1).contiguous(),
            "model_action": trajectory.reshape(batch_size, -1).contiguous(),
        }
        return executed, {
            "prev_logprobs": compressed_logprobs.contiguous(),
            "prev_values": torch.zeros(
                batch_size, 1, device=self.device, dtype=torch.float32
            ),
            "forward_inputs": forward_inputs,
        }

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs: Any):
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError("FM-UNet GRPO only implements default_forward")
        return self.default_forward(**kwargs)

    def default_forward(
        self, forward_inputs: dict[str, torch.Tensor], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Recompute log-probability for the exact stored SDE transition."""

        del kwargs
        chains = forward_inputs["chains"]
        denoise_indices = forward_inputs["denoise_indices"].long()
        batch_size = int(chains.shape[0])
        rows = torch.arange(batch_size, device=chains.device)
        before = chains[rows, denoise_indices]
        after = chains[rows, denoise_indices + 1]
        observation = {
            "image": forward_inputs["fm_obs_image"],
            "state": forward_inputs["fm_obs_state"],
            "tactile": forward_inputs["fm_obs_tactile"],
        }
        global_condition, condition_tokens = self._condition(observation)
        times = flow_timesteps(
            self.rl_config.num_steps, device=chains.device, dtype=torch.float32
        )
        velocity = self._velocity(
            before,
            times[denoise_indices],
            global_condition,
            condition_tokens,
        )
        mean, std = flow_step_mean_std(
            before,
            velocity,
            denoise_indices,
            num_steps=self.rl_config.num_steps,
            noise_method=self.rl_config.noise_method,
            noise_level=self.rl_config.noise_level,
        )
        logprobs = self.timing.compress_logprobs(
            gaussian_logprob(after.float(), mean, std)
        )
        return {
            "logprobs": logprobs.float().contiguous(),
            "values": torch.zeros(batch_size, device=chains.device),
            "entropy": torch.zeros(batch_size, 1, device=chains.device),
        }
