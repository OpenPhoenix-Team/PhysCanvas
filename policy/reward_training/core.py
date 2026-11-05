"""Shared three-view reward model, frame dataset and evaluation metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ThreeViewFrameDataset(Dataset):
    def __init__(self, rows: list[dict], cache_dir: Path, indices: np.ndarray | None = None):
        self.rows = rows
        self.cache_dir = cache_dir
        self.indices = indices
        self._episode_arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows) if self.indices is None else len(self.indices)

    def global_index(self, idx: int) -> int:
        return idx if self.indices is None else int(self.indices[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[self.global_index(idx)]
        sample_id = row["sample_id"]
        if sample_id not in self._episode_arrays:
            path = self.cache_dir / f"{sample_id}.npy"
            self._episode_arrays[sample_id] = np.load(path, mmap_mode="r")
        frame_idx = int(row["aligned_frame_index"])
        frame = np.array(self._episode_arrays[sample_id][frame_idx], copy=True)
        return torch.from_numpy(frame), torch.tensor(float(row["label"]), dtype=torch.float32)


class ThreeViewResNet18(nn.Module):
    def __init__(self, pretrained: bool = True, hidden_dim: int = 256, dropout: float = 0.1):
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
        for module in self.fusion_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError(f"Expected [B,3,3,H,W], got {tuple(images.shape)}")
        batch_size, num_views, channels, height, width = images.shape
        images = images.reshape(batch_size * num_views, channels, height, width)
        images = images.float().div_(255.0)
        images = (images - self.image_mean) / self.image_std
        features = self.backbone(images)
        features = features.reshape(batch_size, num_views * features.shape[-1])
        return self.fusion_head(features).squeeze(-1)

    @torch.no_grad()
    def predict_video(self, frames: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        if frames.ndim != 6:
            raise ValueError(f"Expected [B,T,3,3,H,W], got {tuple(frames.shape)}")
        batch_size, time_steps = frames.shape[:2]
        logits = self(frames.reshape(batch_size * time_steps, *frames.shape[2:]))
        return (torch.sigmoid(logits) >= threshold).long().reshape(batch_size, time_steps)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    tp = tn = fp = fn = count = 0
    probability_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.binary_cross_entropy_with_logits(
            logits, labels, reduction="sum"
        ).item()
        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= threshold
        targets = labels >= 0.5
        tp += int((predictions & targets).sum().item())
        tn += int((~predictions & ~targets).sum().item())
        fp += int((predictions & ~targets).sum().item())
        fn += int((~predictions & targets).sum().item())
        probability_sum += probabilities.sum().item()
        count += labels.numel()
    accuracy = (tp + tn) / max(1, count)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "loss": loss_sum / max(1, count),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(1, fp + tn),
        "probability_mean": probability_sum / max(1, count),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "count": count,
    }
