"""CPU canvas conversions matching PhysCanvas release c49ab996.

The public transforms are the authority: RGB uses resize-with-pad, sensor
images are packed horizontally, XYZ is scaled by 0.06, and there is no depth
row. No policy normalization belongs here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SIDE_ORDER = ("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")
VIEW_ORDER = ("base_0", "left_wrist_0", "right_wrist_0")
FLOW_SCALE = 0.06


def normalize_rgb(images: np.ndarray) -> torch.Tensor:
    """Official uint8 RGB preprocessing, returning [T,3,224,224]."""
    from PhysCanvas.utils import resize_with_pad

    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3] RGB, got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError("RGB must be uint8; refusing an ambiguous input range")
    images = resize_with_pad(images, 224, 224)
    return torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2).float().div(127.5).sub(1)


def _side_panel(sides: torch.Tensor) -> torch.Tensor:
    """[T,4,3,H,W] -> [T,3,224,672], official packing/interpolation."""
    if sides.ndim != 5 or sides.shape[1:3] != (4, 3):
        raise ValueError(f"Expected [T,4,3,H,W], got {tuple(sides.shape)}")
    t, s, c, h, w = sides.shape
    panel = sides.permute(0, 2, 3, 1, 4).reshape(t, c, h, s * w)
    return F.interpolate(panel, size=(224, 672), mode="bilinear", align_corners=False)


def build_visual(reset: dict) -> torch.Tensor:
    """Pack one real reset as normalized [1,1,3,672,672] (RGB/diff/XYZ)."""
    rgb = np.asarray(reset["rgb"])
    if rgb.ndim != 4 or rgb.shape[0] != 3:
        raise ValueError(f"Expected three RGB views, got {rgb.shape}")
    views = normalize_rgb(rgb)
    rgb_row = torch.cat(tuple(views.unbind(0)), dim=-1).unsqueeze(0)

    diff = np.asarray(reset["tactile_diff"])
    if diff.ndim != 4 or diff.shape[0] != 4 or diff.shape[-1] != 3 or diff.dtype != np.uint8:
        raise ValueError("tactile_diff must be uint8 [4,H,W,3]")
    # Match _read_unified_tactile_frames before packing; INTER_AREA is not
    # equivalent to one direct resize of the already concatenated canvas.
    if diff.shape[1:3] != (350, 200):
        import cv2
        diff = np.stack([cv2.resize(frame, (200, 350), interpolation=cv2.INTER_AREA) for frame in diff])
    diff_t = torch.from_numpy(np.array(diff, copy=True)).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    diff_row = _side_panel(diff_t.unsqueeze(0))

    xyz = np.asarray(reset["tactile_xyz"], dtype=np.float32)
    if xyz.shape != (4, 700, 3) or not np.isfinite(xyz).all():
        raise ValueError("tactile_xyz must be finite [4,700,3] raw XYZ displacement")
    xyz_t = torch.from_numpy(np.array(xyz, copy=True)).reshape(1, 4, 35, 20, 3)
    encoded = xyz_t.div(FLOW_SCALE).clamp(-1, 1).permute(0, 1, 4, 2, 3)
    flow_row = _side_panel(encoded)
    return torch.cat([rgb_row, diff_row, flow_row], dim=-2).unsqueeze(0).contiguous()


def build_state(label: int = 0, task: str = "jimu") -> torch.Tensor:
    """Use the public state-canvas implementation, not a copied color table."""
    from PhysCanvas.object_state_canvas import CanvasSpec, render_state_sequence

    return render_state_sequence([label], task, CanvasSpec()).unsqueeze(0)


def build_motion_replay(motion: np.ndarray) -> torch.Tensor:
    """Pack [9,224,672,3] motion as [1,9,3,672,672].

    Official full/no-depth transforms repeat the motion row for RGB, tactile
    difference and tactile flow; zero rows would change the conditioning.
    """
    motion = np.asarray(motion)
    if motion.shape != (9, 224, 672, 3) or motion.dtype != np.uint8:
        raise ValueError(f"Expected uint8 [9,224,672,3], got {motion.shape}/{motion.dtype}")
    row = torch.from_numpy(np.array(motion, copy=True)).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    return torch.cat([row, row, row], dim=-2).unsqueeze(0).contiguous()


def build_state_replay(motion: np.ndarray) -> torch.Tensor:
    """Official bilinear motion strip [1,9,3,32,672], masked by the model."""
    row = build_motion_replay(motion)[0, :, :, :224]
    return F.interpolate(row, size=(32, 672), mode="bilinear", align_corners=False).unsqueeze(0)


def decode_policy_visual(visual: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB row [T,224,672,3] and physical XYZ [T,4,700,3].

    Follows model_eval.decode_xyz_flow_panel exactly, including no clipping of
    predicted XYZ. RGB is quantized for the existing Pi0.5/RM image interface.
    This function does not subtract a new reference or normalize tactile again.
    """
    if visual.ndim != 5 or visual.shape[0] != 1 or tuple(visual.shape[2:]) != (3, 672, 672):
        raise ValueError(f"Expected [1,T,3,672,672], got {tuple(visual.shape)}")
    frames = visual[0].detach().float().cpu()
    if not torch.isfinite(frames).all():
        raise ValueError("WM produced non-finite visual values")
    rgb = frames[:, :, :224].permute(0, 2, 3, 1)
    rgb = ((rgb + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    packed = F.interpolate(frames[:, :, 448:672], size=(35, 80), mode="bilinear", align_corners=False)
    t = packed.shape[0]
    sides = packed.reshape(t, 3, 35, 4, 20).permute(0, 3, 2, 4, 1)
    xyz = (sides * FLOW_SCALE).reshape(t, 4, 700, 3).contiguous().numpy()
    return np.ascontiguousarray(rgb), xyz
