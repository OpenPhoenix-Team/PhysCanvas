"""Stochastic flow-matching transitions used by FM policy GRPO.

The task FM integrates from noise at ``t=0`` to actions at ``t=1``.  This is
the same convention as RLinf's GR00T adapter and the opposite of OpenPI.  A
rollout stores the sampled transition itself; actor replay evaluates that exact
transition instead of drawing a new action.
"""

from __future__ import annotations

import math

import torch


def flow_timesteps(
    num_steps: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the inclusive ``0 → 1`` FM time grid."""

    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    return torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)


def flow_step_mean_std(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    step_indices: torch.Tensor,
    *,
    num_steps: int,
    noise_method: str,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one ``0 → 1`` ODE/SDE transition distribution.

    The Flow-SDE formula matches RLinf's GR00T implementation.  ``step_indices``
    is per batch row so the exact stored rollout transition can be replayed by
    the updated policy.
    """

    if x_t.shape != velocity.shape or x_t.ndim != 3:
        raise ValueError(
            "x_t and velocity must share shape [B,H,A], "
            f"got x_t={tuple(x_t.shape)} velocity={tuple(velocity.shape)}"
        )
    if step_indices.shape != (x_t.shape[0],):
        raise ValueError(
            f"step_indices must have shape ({x_t.shape[0]},), "
            f"got {tuple(step_indices.shape)}"
        )
    if noise_level < 0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")

    work_dtype = torch.float32
    x_t = x_t.to(work_dtype)
    velocity = velocity.to(work_dtype)
    indices = step_indices.to(device=x_t.device, dtype=torch.long)
    if bool(((indices < 0) | (indices >= num_steps)).any()):
        raise ValueError("step_indices must be in [0, num_steps)")

    times = flow_timesteps(num_steps, device=x_t.device, dtype=work_dtype)
    t = times[indices][:, None, None].expand_as(x_t)
    delta = (times[indices + 1] - times[indices])[:, None, None].expand_as(x_t)
    x0_pred = x_t - velocity * t
    x1_pred = x_t + velocity * (1.0 - t)

    if noise_method == "flow_ode":
        mean = x_t + velocity * delta
        std = torch.zeros_like(x_t)
        return mean, std
    if noise_method != "flow_sde":
        raise NotImplementedError(
            f"noise_method={noise_method!r} is unsupported; use flow_ode or flow_sde"
        )

    safe_denominator = torch.where(times == 0, times[1], times)
    sigmas = float(noise_level) * torch.sqrt(
        (1.0 - times) / safe_denominator
    )
    sigma_i = sigmas[:-1][indices][:, None, None].expand_as(x_t)
    x0_weight = 1.0 - (t + delta) - sigma_i.square() * delta / (2.0 * (1.0 - t))
    x1_weight = t + delta
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    std = torch.sqrt(delta) * sigma_i
    return mean, std


def gaussian_logprob(
    sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Return elementwise Gaussian log-probability.

    Deterministic ODE coordinates receive zero log-probability, matching the
    existing RLinf flow-policy adapters.
    """

    if sample.shape != mean.shape or sample.shape != std.shape:
        raise ValueError(
            "sample, mean and std must have identical shapes, "
            f"got {tuple(sample.shape)}, {tuple(mean.shape)}, {tuple(std.shape)}"
        )
    deterministic = std == 0
    safe_std = torch.where(deterministic, torch.ones_like(std), std)
    result = (
        -torch.log(safe_std)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((sample - mean) / safe_std).square()
    )
    return torch.where(deterministic, torch.zeros_like(result), result)
