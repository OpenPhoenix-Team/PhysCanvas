import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


_MODEL_CACHE: dict[tuple[str, str, int, int], Any] = {}
_FASTFS_IMPORTS: dict[str, Any] = {}


def _read_frame(path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok and frame_idx > 0:
        cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"failed to reopen video: {path}")
        frame = None
        for _ in range(int(frame_idx) + 1):
            ok, frame = cap.read()
            if not ok:
                break
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx} from {path}")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _import_fastfs(fastfs_dir: str) -> dict[str, Any]:
    fastfs_dir = str(Path(fastfs_dir))
    if fastfs_dir not in sys.path:
        sys.path.insert(0, fastfs_dir)
    if fastfs_dir not in _FASTFS_IMPORTS:
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        _FASTFS_IMPORTS[fastfs_dir] = {
            "InputPadder": InputPadder,
            "AMP_DTYPE": AMP_DTYPE,
        }
    return _FASTFS_IMPORTS[fastfs_dir]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _load_model(fastfs_dir: str, model_path: str, valid_iters: int, max_disp: int):
    device = _get_device()
    key = (str(Path(model_path)), str(device), int(valid_iters), int(max_disp))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    _import_fastfs(fastfs_dir)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if hasattr(model, "args"):
        model.args.valid_iters = int(valid_iters)
        model.args.max_disp = int(max_disp)
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def _run_model(model, fastfs_dir: str, left_bgr: np.ndarray, right_bgr: np.ndarray, valid_iters: int, build_volume: str) -> np.ndarray:
    imports = _import_fastfs(fastfs_dir)
    InputPadder = imports["InputPadder"]
    amp_dtype = imports["AMP_DTYPE"]
    device = _get_device()

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    img0 = torch.as_tensor(left_rgb, device=device).float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb, device=device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=amp_dtype):
            disp = model.forward(img0, img1, iters=int(valid_iters), test_mode=True, optimize_build_volume=build_volume)
    disp = padder.unpad(disp.float())
    return disp.detach().cpu().numpy().reshape(left_bgr.shape[:2]).clip(0, None)


class FastFoundationStereoDepth:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.fastfs_dir = settings["fastfs_dir"]
        self.model_path = settings["model_path"]
        self.valid_iters = int(settings.get("valid_iters", 8))
        self.max_disp = int(settings.get("max_disp", 192))
        self.build_volume = settings.get("build_volume", "pytorch1")
        self.zmax_m = float(settings.get("zmax_m", 5.0))

    @staticmethod
    def _cache_path(sample: dict[str, Any], camera_name: str, frame_idx: int) -> str:
        cache_dir = sample["sample_depth_cache_dir"]
        episode_id = sample["episode_id"]
        return os.path.join(cache_dir, episode_id, camera_name, f"frame_{int(frame_idx):06d}.npy")

    def _depth_for_frame(self, sample: dict[str, Any], view_key: str, frame_idx: int) -> np.ndarray:
        camera_name = sample["sample_camera_names"][view_key]
        raw_frame_idx = int(frame_idx)
        view_indices = sample.get("sample_view_indices", {})
        if view_key in view_indices:
            aligned = np.asarray(view_indices[view_key], dtype=np.int64)
            if aligned.size:
                safe_pos = int(np.clip(raw_frame_idx, 0, aligned.size - 1))
                raw_frame_idx = int(aligned[safe_pos])
        cache_path = self._cache_path(sample, camera_name, raw_frame_idx)
        if os.path.exists(cache_path):
            return np.load(cache_path).astype(np.float32)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cam = sample["camera_params"][camera_name]
        intr = cam.get("intrinsics_640x480_30", {})
        fx = float(intr.get("fx", cam.get("fx")))
        baseline_m = float(cam.get("baseline_m", cam.get("baseline_mm") / 1000.0))

        left = _read_frame(sample[f"cam_{view_key}_ir_left_path"], raw_frame_idx)
        right = _read_frame(sample[f"cam_{view_key}_ir_right_path"], raw_frame_idx)
        model = _load_model(self.fastfs_dir, self.model_path, self.valid_iters, self.max_disp)
        disp = _run_model(model, self.fastfs_dir, left, right, self.valid_iters, self.build_volume)

        valid = np.isfinite(disp) & (disp > 0.1)
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid] = fx * baseline_m / disp[valid]
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        if self.zmax_m > 0:
            depth = np.clip(depth, 0.0, self.zmax_m)

        tmp_path = f"{cache_path}.tmp.{os.getpid()}.npy"
        np.save(tmp_path, depth)
        os.replace(tmp_path, cache_path)
        return depth

    def depth_frames(self, sample: dict[str, Any], view_key: str, frame_indices: np.ndarray) -> np.ndarray:
        frames = [self._depth_for_frame(sample, view_key, int(idx)) for idx in frame_indices]
        return np.stack(frames, axis=0)
