"""Image scoring and tactile display helpers used by the evaluation runner."""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resize_panel(panel: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(panel).convert("RGB")
    src_w, src_h = image.size
    scale = min(width / src_w, height / src_h)
    size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))
    image = image.resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), (238, 238, 235))
    canvas.paste(image, ((width - size[0]) // 2, (height - size[1]) // 2))
    return np.asarray(canvas)


def resize_tactile_diff_panel(panel: np.ndarray, width: int, height: int, side_count: int) -> np.ndarray:
    """Restore the four tactile strips' clockwise orientation for display only."""
    side_count = max(1, int(side_count))
    original_h = int(os.environ.get("PHYSCANVAS_TACTILE_DIFF_ORIGINAL_H", "400"))
    original_w = int(os.environ.get("PHYSCANVAS_TACTILE_DIFF_ORIGINAL_W", "700"))
    if side_count <= 1 or panel.shape[1] % side_count:
        image = Image.fromarray(panel).convert("RGB").resize(
            (original_w, original_h), Image.Resampling.BILINEAR
        )
        return resize_panel(np.asarray(image.transpose(Image.Transpose.ROTATE_270)), width, height)
    side_width = panel.shape[1] // side_count
    slot_width = max(1, width // side_count)
    parts = []
    for index in range(side_count):
        side = Image.fromarray(panel[:, index * side_width : (index + 1) * side_width]).convert("RGB")
        side = side.resize((original_w, original_h), Image.Resampling.BILINEAR)
        side = side.transpose(Image.Transpose.ROTATE_270)
        parts.append(resize_panel(np.asarray(side), slot_width, height))
    merged = np.concatenate(parts, axis=1)
    return merged if merged.shape[1] == width else resize_panel(merged, width, height)


class ImageMetricAccumulator:
    """Future-frame PSNR, SSIM and LPIPS with the archived run's conventions."""

    def __init__(self, layers: list[str], device: torch.device, frame_dir=None) -> None:
        if frame_dir is not None:
            raise ValueError("per-frame files are not part of this release")
        self.layers = set(layers)
        self.device = device
        self.records: list[dict] = []
        self.lpips_model = None
        self.lpips_error = ""
        self.lpips_batch = max(1, int(os.environ.get("PHYSCANVAS_EVAL_LPIPS_BATCH", "2")))
        self.lpips_max_side = int(os.environ.get("PHYSCANVAS_EVAL_LPIPS_MAX_SIDE", "0"))

    def _ensure_lpips(self):
        if self.lpips_error:
            return None
        if self.lpips_model is None:
            try:
                import lpips

                self.lpips_model = lpips.LPIPS(net=os.environ.get("PHYSCANVAS_EVAL_LPIPS_NET", "alex"), verbose=False)
                self.lpips_model.to(self.device).eval()
            except Exception as exc:
                self.lpips_error = repr(exc)
        return self.lpips_model

    def _lpips_values(self, gt: torch.Tensor, pred: torch.Tensor) -> list[float]:
        model = self._ensure_lpips()
        if model is None:
            return [float("nan")] * len(gt)
        gt = gt.detach().float().clamp(-1, 1).to(self.device)
        pred = pred.detach().float().clamp(-1, 1).to(self.device)
        if self.lpips_max_side > 0:
            height, width = gt.shape[-2:]
            scale = min(1.0, self.lpips_max_side / max(height, width))
            if scale < 1:
                size = (max(1, round(height * scale)), max(1, round(width * scale)))
                gt = F.interpolate(gt, size=size, mode="bilinear", align_corners=False)
                pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
        values = []
        with torch.no_grad():
            for start in range(0, len(gt), self.lpips_batch):
                score = model(gt[start : start + self.lpips_batch], pred[start : start + self.lpips_batch])
                values.extend(score.detach().float().reshape(-1).cpu().tolist())
        return values

    def add(self, layer, gt_uint8, pred_uint8, gt_tensor, pred_tensor, *_):
        if layer not in self.layers:
            return
        from skimage.metrics import structural_similarity
        gt_uint8 = np.asarray(gt_uint8)
        pred_uint8 = np.asarray(pred_uint8)
        if len(gt_uint8) != len(pred_uint8) or len(gt_uint8) != len(gt_tensor):
            raise ValueError("image metric sequence length mismatch")
        difference = (gt_uint8.astype(np.float32) - pred_uint8.astype(np.float32)) / 255.0
        mse = np.mean(difference * difference, axis=(1, 2, 3))
        perceptual = self._lpips_values(gt_tensor, pred_tensor)
        for index, value in enumerate(mse):
            ssim = structural_similarity(gt_uint8[index], pred_uint8[index], channel_axis=-1, data_range=255)
            self.records.append({
                "layer": layer,
                "mse_01": float(value),
                "psnr": float("inf") if value <= 0 else -10.0 * math.log10(float(value)),
                "ssim": float(ssim),
                "lpips": float(perceptual[index]),
            })

    def summary(self) -> dict:
        overall = {}
        for layer in sorted(self.layers):
            rows = [row for row in self.records if row["layer"] == layer]
            if not rows:
                continue
            overall[layer] = {"frames": len(rows)}
            for key in ("mse_01", "psnr", "ssim", "lpips"):
                values = [row[key] for row in rows if np.isfinite(row[key])]
                overall[layer][key] = float(np.mean(values)) if values else None
        return {"overall": overall}
