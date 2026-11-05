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

"""PyTorch port of the arm-fixed Pi0.5 four-sensor tactile encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TactileEncoder(nn.Module):
    """Encode four episode-relative Xense displacement grids as prefix tokens.

    The module mirrors ``openpi.models.tactile.TactileEncoder``.  Each sensor's
    700 points are reshaped to a 35x20 grid, processed by a shared CNN and MLP,
    then augmented with sensor/modality embeddings and a learned scalar gate.
    """

    def __init__(
        self,
        *,
        num_sensors: int = 4,
        points_per_sensor: int = 700,
        input_channels: int = 3,
        grid_height: int = 35,
        grid_width: int = 20,
        hidden_dim: int = 1024,
        output_dim: int = 2048,
        gate_init: float = 0.01,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if grid_height * grid_width != points_per_sensor:
            raise ValueError(
                f"tactile grid {grid_height}x{grid_width} does not match "
                f"{points_per_sensor} points"
            )
        self.num_sensors = int(num_sensors)
        self.points_per_sensor = int(points_per_sensor)
        self.input_channels = int(input_channels)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.output_dim = int(output_dim)
        self.output_dtype = output_dtype

        # padding=1 exactly matches JAX padding="SAME" for 3x3, stride-2 here:
        # 35x20 -> 18x10 -> 9x5 -> 5x3.
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.LayerNorm(64, eps=1e-6)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.LayerNorm(128, eps=1e-6)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.LayerNorm(256, eps=1e-6)

        self.spatial_projection = nn.Linear(5 * 3 * 256, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim, eps=1e-6)

        self.sensor_embedding = nn.Embedding(num_sensors, output_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, output_dim))
        # FSDP rejects zero-dimensional parameters.  Keep the gate as a
        # one-element vector; broadcasting is numerically identical to the
        # original scalar parameter.
        self.gate = nn.Parameter(torch.tensor([float(gate_init)]))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Accept legacy checkpoints that stored ``gate`` as a scalar."""

        gate_key = f"{prefix}gate"
        gate = state_dict.get(gate_key)
        if gate is not None and gate.ndim == 0:
            state_dict[gate_key] = gate.reshape(1)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _conv_norm_gelu(
        x: torch.Tensor,
        conv: nn.Conv2d,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        """Apply NCHW convolution and channel-last LayerNorm/GELU."""

        x = conv(x).permute(0, 2, 3, 1)
        x = F.gelu(norm(x), approximate="tanh")
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        """Return tactile prefix tokens with shape ``[B,S,output_dim]``."""

        if tactile.ndim != 4:
            raise ValueError(
                f"expected tactile [B,S,P,C], got shape {tuple(tactile.shape)}"
            )
        expected = (
            self.num_sensors,
            self.points_per_sensor,
            self.input_channels,
        )
        if tuple(tactile.shape[1:]) != expected:
            raise ValueError(
                f"expected tactile trailing shape {expected}, "
                f"got {tuple(tactile.shape[1:])}"
            )

        batch_size = tactile.shape[0]
        compute_dtype = self.conv1.weight.dtype
        x = tactile.to(dtype=compute_dtype).reshape(
            batch_size * self.num_sensors,
            self.grid_height,
            self.grid_width,
            self.input_channels,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self._conv_norm_gelu(x, self.conv1, self.norm1)
        x = self._conv_norm_gelu(x, self.conv2, self.norm2)
        x = self._conv_norm_gelu(x, self.conv3, self.norm3)
        x = x.reshape(batch_size * self.num_sensors, -1)
        x = F.gelu(self.hidden_norm(self.spatial_projection(x)), approximate="tanh")
        x = self.output_norm(self.output_projection(x))
        x = x.reshape(batch_size, self.num_sensors, self.output_dim)

        sensor_ids = torch.arange(self.num_sensors, device=x.device)
        sensor_embedding = self.sensor_embedding(sensor_ids).unsqueeze(0)
        x = self.gate * (x + sensor_embedding + self.modality_embedding)
        return x.to(dtype=self.output_dtype)
