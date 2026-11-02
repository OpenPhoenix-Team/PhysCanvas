import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


_MODEL_CACHE: dict[tuple[str, str, int], Any] = {}
_IMPORT_CACHE: dict[str, dict[str, Any]] = {}


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
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {path}")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _read_frames(path: str, frame_indices: list[int]) -> list[np.ndarray]:
    if not frame_indices:
        return []
    if len(frame_indices) == 1:
        return [_read_frame(path, frame_indices[0])]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    unique_sorted = sorted(set(int(idx) for idx in frame_indices))
    frame_cache: dict[int, np.ndarray] = {}
    current_pos = -1
    try:
        for target_idx in unique_sorted:
            # For small forward gaps, read sequentially to avoid repeated codec seeks.
            # For large gaps or backward jumps, seek once on the already-open capture.
            if current_pos < 0 or target_idx < current_pos or target_idx - current_pos > 32:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
                current_pos = target_idx
                ok, frame = cap.read()
                current_pos += 1
            else:
                ok, frame = True, None
                while current_pos <= target_idx:
                    ok, frame = cap.read()
                    current_pos += 1
                    if not ok:
                        break
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame {target_idx} from {path}")
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame_cache[target_idx] = frame
    finally:
        cap.release()

    return [frame_cache[int(idx)] for idx in frame_indices]


def _import_foundation_stereo(foundation_stereo_dir: str) -> dict[str, Any]:
    foundation_stereo_dir = str(Path(foundation_stereo_dir))
    if foundation_stereo_dir not in sys.path:
        sys.path.insert(0, foundation_stereo_dir)
    if foundation_stereo_dir not in _IMPORT_CACHE:
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        _IMPORT_CACHE[foundation_stereo_dir] = {
            "FoundationStereo": FoundationStereo,
            "InputPadder": InputPadder,
        }
    return _IMPORT_CACHE[foundation_stereo_dir]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _load_model(foundation_stereo_dir: str, ckpt_path: str, valid_iters: int):
    device = _get_device()
    ckpt_path = str(Path(ckpt_path))
    key = (ckpt_path, str(device), int(valid_iters))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"FoundationStereo checkpoint not found: {ckpt_path}. "
            "Set ckpt_path to the official model_best_bp2.pth checkpoint."
        )
    imports = _import_foundation_stereo(foundation_stereo_dir)
    cfg = OmegaConf.load(str(Path(ckpt_path).parent / "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = int(valid_iters)
    model = imports["FoundationStereo"](cfg)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def _run_model_batch(
    model,
    foundation_stereo_dir: str,
    left_bgr_frames: list[np.ndarray],
    right_bgr_frames: list[np.ndarray],
    valid_iters: int,
) -> np.ndarray:
    t0 = time.perf_counter()
    imports = _import_foundation_stereo(foundation_stereo_dir)
    InputPadder = imports["InputPadder"]
    device = _get_device()
    if len(left_bgr_frames) != len(right_bgr_frames):
        raise ValueError("left/right frame count mismatch")
    left_rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in left_bgr_frames], axis=0)
    right_rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in right_bgr_frames], axis=0)
    img0 = torch.as_tensor(left_rgb, device=device).float().permute(0, 3, 1, 2).contiguous()
    img1 = torch.as_tensor(right_rgb, device=device).float().permute(0, 3, 1, 2).contiguous()
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            disp = model.forward(img0, img1, iters=int(valid_iters), test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    disp = padder.unpad(disp.float())
    h, w = left_bgr_frames[0].shape[:2]
    if os.environ.get("PHYSCANVAS_FOUNDATION_STEREO_PROFILE", "0") in ("1", "true", "True"):
        allocated = reserved = 0.0
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(
            "[foundation-stereo] "
            f"batch={len(left_bgr_frames)} prep={t1 - t0:.3f}s "
            f"forward={t2 - t1:.3f}s cuda_alloc={allocated:.1f}MiB cuda_reserved={reserved:.1f}MiB",
            flush=True,
        )
    return disp.detach().cpu().numpy().reshape(len(left_bgr_frames), h, w).clip(0, None)


def _run_model(model, foundation_stereo_dir: str, left_bgr: np.ndarray, right_bgr: np.ndarray, valid_iters: int) -> np.ndarray:
    return _run_model_batch(model, foundation_stereo_dir, [left_bgr], [right_bgr], valid_iters)[0]


class FoundationStereoDepth:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.foundation_stereo_dir = settings["foundation_stereo_dir"]
        self.ckpt_path = settings["ckpt_path"]
        self.valid_iters = int(settings.get("valid_iters", 8))
        self.zmax_m = float(settings.get("zmax_m", 5.0))

    @staticmethod
    def _cache_path(sample: dict[str, Any], camera_name: str, frame_idx: int) -> str:
        cache_dir = sample["sample_depth_cache_dir"]
        episode_id = sample["episode_id"]
        return os.path.join(cache_dir, "foundation_stereo", episode_id, camera_name, f"frame_{int(frame_idx):06d}.npy")

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
        model = _load_model(self.foundation_stereo_dir, self.ckpt_path, self.valid_iters)
        disp = _run_model(model, self.foundation_stereo_dir, left, right, self.valid_iters)

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
        camera_name = sample["sample_camera_names"][view_key]
        view_indices = sample.get("sample_view_indices", {})
        raw_frame_indices: list[int] = []
        if view_key in view_indices:
            aligned = np.asarray(view_indices[view_key], dtype=np.int64)
        else:
            aligned = None
        for frame_idx in np.asarray(frame_indices, dtype=np.int64).tolist():
            raw_frame_idx = int(frame_idx)
            if aligned is not None and aligned.size:
                safe_pos = int(np.clip(raw_frame_idx, 0, aligned.size - 1))
                raw_frame_idx = int(aligned[safe_pos])
            raw_frame_indices.append(raw_frame_idx)

        frames: list[np.ndarray | None] = []
        missing_positions: list[int] = []
        missing_raw_indices: list[int] = []
        missing_cache_paths: list[str] = []
        for pos, raw_frame_idx in enumerate(raw_frame_indices):
            cache_path = self._cache_path(sample, camera_name, raw_frame_idx)
            if os.path.exists(cache_path):
                frames.append(np.load(cache_path).astype(np.float32))
            else:
                frames.append(None)
                missing_positions.append(pos)
                missing_raw_indices.append(raw_frame_idx)
                missing_cache_paths.append(cache_path)

        if missing_positions:
            t_read0 = time.perf_counter()
            for cache_path in missing_cache_paths:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cam = sample["camera_params"][camera_name]
            intr = cam.get("intrinsics_640x480_30", {})
            fx = float(intr.get("fx", cam.get("fx")))
            baseline_m = float(cam.get("baseline_m", cam.get("baseline_mm") / 1000.0))

            left_frames = _read_frames(sample[f"cam_{view_key}_ir_left_path"], missing_raw_indices)
            right_frames = _read_frames(sample[f"cam_{view_key}_ir_right_path"], missing_raw_indices)
            t_read1 = time.perf_counter()
            model = _load_model(self.foundation_stereo_dir, self.ckpt_path, self.valid_iters)
            disp_batch = _run_model_batch(model, self.foundation_stereo_dir, left_frames, right_frames, self.valid_iters)
            t_model1 = time.perf_counter()

            for pos, cache_path, disp in zip(missing_positions, missing_cache_paths, disp_batch):
                valid = np.isfinite(disp) & (disp > 0.1)
                depth = np.zeros_like(disp, dtype=np.float32)
                depth[valid] = fx * baseline_m / disp[valid]
                depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
                if self.zmax_m > 0:
                    depth = np.clip(depth, 0.0, self.zmax_m)
                tmp_path = f"{cache_path}.tmp.{os.getpid()}.npy"
                np.save(tmp_path, depth)
                os.replace(tmp_path, cache_path)
                frames[pos] = depth
            if os.environ.get("PHYSCANVAS_FOUNDATION_STEREO_PROFILE", "0") in ("1", "true", "True"):
                t_save1 = time.perf_counter()
                print(
                    "[foundation-stereo-depth] "
                    f"view={view_key} missing={len(missing_positions)} "
                    f"read={t_read1 - t_read0:.3f}s "
                    f"model={t_model1 - t_read1:.3f}s "
                    f"save={t_save1 - t_model1:.3f}s",
                    flush=True,
                )

        return np.stack(frames, axis=0)
