# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen three-view ResNet-18 reward model used by the Jimu WM environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from rlinf.envs.world_model.jimu_wm_protocol import (
    ACTION_HORIZON,
    WM_SUB_FRAMES,
    panel_video_to_views,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ThreeViewResNet18(nn.Module):
    """Exact inference architecture from ``train_three_view_reward.py``."""

    def __init__(
        self,
        pretrained: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fusion_head = nn.Sequential(
            nn.Linear(3 * feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or tuple(images.shape[1:3]) != (3, 3):
            raise ValueError(f"Expected [B,3,3,H,W], got {tuple(images.shape)}")
        batch_size, num_views = images.shape[:2]
        images = images.reshape(batch_size * num_views, *images.shape[2:])
        images = images.float().div(255.0)
        images = (images - self.image_mean) / self.image_std
        features = self.backbone(images)
        features = features.reshape(batch_size, num_views * features.shape[-1])
        return self.fusion_head(features).squeeze(-1)


class ThreeViewRewardScorer:
    """Emit continuous frame probabilities plus thresholded diagnostics."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: torch.device | str,
        threshold: float = 0.5,
        batch_size: int = 256,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0,1), got {threshold}")
        if batch_size <= 0:
            raise ValueError("reward batch_size must be positive")
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)

        checkpoint_path = Path(checkpoint).expanduser()
        payload: dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if "model_state_dict" not in payload:
            raise KeyError(
                f"Reward checkpoint lacks model_state_dict: {checkpoint_path}"
            )
        saved_order = tuple(
            payload.get("view_order", ("base", "left_wrist", "right_wrist"))
        )
        if saved_order != ("base", "left_wrist", "right_wrist"):
            raise ValueError(f"Unexpected reward checkpoint view order: {saved_order}")

        self.model = ThreeViewResNet18(pretrained=False)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.checkpoint_epoch = int(payload.get("epoch", -1)) + 1

    @torch.inference_mode()
    def score_panel_video(self, panel_video) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps = int(panel_video.shape[1])
        if time_steps not in {WM_SUB_FRAMES, ACTION_HORIZON}:
            raise ValueError(
                "Reward scorer expects one or two WM segments "
                f"({WM_SUB_FRAMES} or {ACTION_HORIZON} frames), got {time_steps}"
            )
        views = panel_video_to_views(
            panel_video, expected_frames=time_steps
        )  # [B,T,V,C,H,W]
        batch_size, time_steps = views.shape[:2]
        flattened = views.reshape(-1, *views.shape[2:])
        if not torch.is_tensor(flattened):
            flattened = torch.from_numpy(flattened)
        probabilities = []
        for start in range(0, flattened.shape[0], self.batch_size):
            images = flattened[start : start + self.batch_size].to(
                self.device, non_blocking=True
            )
            probabilities.append(torch.sigmoid(self.model(images)).float())
        probs = torch.cat(probabilities, dim=0).reshape(batch_size, time_steps)
        labels = (probs >= self.threshold).to(torch.float32)
        return probs, labels

    def to(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.model.to(self.device)
