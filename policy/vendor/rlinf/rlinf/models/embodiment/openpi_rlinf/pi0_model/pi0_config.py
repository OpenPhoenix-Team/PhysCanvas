# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi0 config for PyTorch, aligned with JAX models/pi0_config.py."""

from __future__ import annotations

import dataclasses

import torch

from . import gemma, model, pointnet


@dataclasses.dataclass
class Pi0Config(model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    pointnet_variant: pointnet.Variant = "pcd"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    pi05: bool = False
    discrete_state_input: bool | None = None
    pcd: bool = False

    use_tactile: bool = False
    tactile_num_sensors: int = 4
    tactile_points_per_sensor: int = 700
    tactile_input_channels: int = 3
    tactile_grid_height: int = 35
    tactile_grid_width: int = 20
    tactile_hidden_dim: int = 1024
    tactile_gate_init: float = 0.01

    def __post_init__(self):
        if self.pi05 and self.max_token_len == 48:
            object.__setattr__(self, "max_token_len", 200)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.use_tactile:
            if not self.pi05:
                raise ValueError("The tactile prefix is supported only for pi0.5")
            if (
                self.tactile_grid_height * self.tactile_grid_width
                != self.tactile_points_per_sensor
            ):
                raise ValueError(
                    "The tactile grid size must equal tactile_points_per_sensor"
                )

    def create(self, **kwargs) -> model.BaseModel:
        from .pi0 import Pi0

        return Pi0(self)

    def fake_obs(self, batch_size: int = 1) -> model.Observation:
        image = torch.ones(batch_size, *model.IMAGE_RESOLUTION, 3)
        image_mask = torch.ones(batch_size, dtype=torch.bool)
        return model.Observation(
            images={
                "base_0_rgb": image,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": image,
            },
            image_masks={
                "base_0_rgb": image_mask,
                "left_wrist_0_rgb": image_mask,
                "right_wrist_0_rgb": image_mask,
            },
            state=torch.ones(batch_size, self.action_dim),
            tokenized_prompt=torch.ones(
                batch_size, self.max_token_len, dtype=torch.long
            ),
            tokenized_prompt_mask=torch.ones(
                batch_size, self.max_token_len, dtype=torch.bool
            ),
            pcd_xyz=torch.ones(batch_size, 16, 2025, 3) if self.pcd else None,
            tactile=(
                torch.ones(
                    batch_size,
                    self.tactile_num_sensors,
                    self.tactile_points_per_sensor,
                    self.tactile_input_channels,
                )
                if self.use_tactile
                else None
            ),
            tactile_mask=(
                torch.ones(batch_size, self.tactile_num_sensors, dtype=torch.bool)
                if self.use_tactile
                else None
            ),
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        return torch.ones(batch_size, self.action_horizon, self.action_dim)
