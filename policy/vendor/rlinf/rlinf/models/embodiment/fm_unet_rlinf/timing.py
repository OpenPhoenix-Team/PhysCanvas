"""Timebase conversion between the 30 Hz FM policy and 10 Hz Jimu WM."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FmWmTiming:
    """Validated timing contract for one closed-loop planning iteration."""

    policy_horizon: int = 64
    executed_policy_steps: int = 48
    policy_hz: int = 30
    wm_hz: int = 10
    wm_steps_per_call: int = 8

    def __post_init__(self) -> None:
        values = (
            self.policy_horizon,
            self.executed_policy_steps,
            self.policy_hz,
            self.wm_hz,
            self.wm_steps_per_call,
        )
        if any(value <= 0 for value in values):
            raise ValueError(f"all timing values must be positive, got {values}")
        if self.executed_policy_steps > self.policy_horizon:
            raise ValueError("executed_policy_steps cannot exceed policy_horizon")
        if self.policy_hz % self.wm_hz:
            raise ValueError("policy_hz must be an integer multiple of wm_hz")
        if self.executed_policy_steps % self.downsample_ratio:
            raise ValueError(
                "executed_policy_steps must be divisible by the downsample ratio"
            )
        if self.wm_output_steps % self.wm_steps_per_call:
            raise ValueError("WM output must split into complete WM calls")
        if self.policy_duration_seconds != self.wm_duration_seconds:
            raise ValueError("FM execution duration and WM rollout duration differ")

    @property
    def downsample_ratio(self) -> int:
        return self.policy_hz // self.wm_hz

    @property
    def wm_output_steps(self) -> int:
        return self.executed_policy_steps // self.downsample_ratio

    @property
    def wm_calls_per_plan(self) -> int:
        return self.wm_output_steps // self.wm_steps_per_call

    @property
    def policy_duration_seconds(self) -> float:
        return self.executed_policy_steps / self.policy_hz

    @property
    def wm_duration_seconds(self) -> float:
        return self.wm_output_steps / self.wm_hz

    def endpoint_indices(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        """Return ``[2,5,...,47]`` for the default 30 Hz → 10 Hz mapping."""

        return torch.arange(
            self.downsample_ratio - 1,
            self.executed_policy_steps,
            self.downsample_ratio,
            device=device,
            dtype=torch.long,
        )

    def executed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Discard the unused FM tail and return all 48 executed 30 Hz targets."""

        self._validate_action_tensor(actions)
        return actions[:, : self.executed_policy_steps].contiguous()

    def wm_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Select the endpoint of every three executed FM actions for the WM."""

        executed = self.executed_actions(actions)
        return executed.index_select(1, self.endpoint_indices(device=actions.device)).contiguous()

    def compress_logprobs(self, logprobs: torch.Tensor) -> torch.Tensor:
        """Map 48 per-action transition log-probs to 16 WM reward tokens.

        Each 10 Hz token is the sum of the three independent Gaussian
        transition-coordinate log-probabilities executed during the same
        100 ms interval.  This preserves log-probability addition while
        aligning PPO/GRPO's token axis with the WM reward axis.
        """

        self._validate_action_tensor(logprobs)
        executed = logprobs[:, : self.executed_policy_steps]
        batch, _, action_dim = executed.shape
        return executed.reshape(
            batch, self.wm_output_steps, self.downsample_ratio, action_dim
        ).sum(dim=2).contiguous()

    def _validate_action_tensor(self, value: torch.Tensor) -> None:
        if value.ndim != 3 or value.shape[1] != self.policy_horizon:
            raise ValueError(
                f"expected [B,{self.policy_horizon},A], got {tuple(value.shape)}"
            )


DEFAULT_FM_WM_TIMING = FmWmTiming()
