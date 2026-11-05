"""Read-only first-observation loader for existing unified reset caches.

No raw_path in the manifest is followed, and no future RGB/action/label is read.
Order is the manifest order, not directory enumeration order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .canvas import SIDE_ORDER, VIEW_ORDER


class ResetDataset:
    def __init__(self, reset_cache: str | Path, selection_manifest: str | Path | None = None):
        root = Path(reset_cache).expanduser()
        if not (root / "episodes").is_dir() and (root / "cache" / "episodes").is_dir():
            root = root / "cache"
        if not (root / "episodes").is_dir():
            raise FileNotFoundError(f"No episodes directory in reset cache {root}")
        self.root = root.resolve()
        self.manifest = Path(selection_manifest) if selection_manifest else self.root / "selection_manifest.json"
        with self.manifest.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            rows = payload
        else:
            rows = next((payload[k] for k in ("episodes", "records", "clips") if k in payload), None)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Reset selection manifest needs a nonempty episodes/records/clips list")
        ids = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each reset record must be an object")
            episode_id = row.get("cache_episode_id", row.get("episode_id"))
            if not isinstance(episode_id, str) or not episode_id or Path(episode_id).name != episode_id or episode_id in (".", ".."):
                raise ValueError(f"Invalid cache episode_id: {episode_id!r}")
            if int(row.get("start", 0)) != 0:
                raise ValueError("Only real frame-0 resets are allowed; nonzero clip start is not a reset")
            ids.append(episode_id)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate episode IDs in reset selection manifest")
        self.episode_ids = tuple(ids)

    def __len__(self) -> int:
        return len(self.episode_ids)

    @staticmethod
    def _first(path: Path) -> np.ndarray:
        # mmap_mode='r' plus explicit [0] is intentional: there is no full-array
        # materialization, write access or fallback to a different raw source.
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim == 0 or array.shape[0] < 1:
            raise ValueError(f"Empty time series: {path}")
        return np.array(array[0], copy=True)

    def read(self, index: int) -> dict:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        episode_id = self.episode_ids[int(index)]
        directory = self.root / "episodes" / episode_id
        views = [self._first(directory / f"{view}_rgb.npy") for view in VIEW_ORDER]
        if any(v.ndim != 3 or v.shape[-1] != 3 or v.dtype != np.uint8 for v in views):
            raise ValueError(f"Invalid uint8 RGB cache for {episode_id}")
        # Use PhysCanvas's own aspect-ratio preserving image convention.
        from PhysCanvas.utils import resize_with_pad
        rgb = np.stack([resize_with_pad(v[None], 224, 224)[0] for v in views])
        differences = []
        xyz = []
        for side in SIDE_ORDER:
            diff = self._first(directory / "tactile" / f"{side}.difference.npy")
            if diff.ndim != 3 or diff.shape[-1] != 3 or diff.dtype != np.uint8:
                raise ValueError(f"Invalid difference image for {episode_id}/{side}")
            differences.append(diff)
            points = self._first(directory / "tactile" / f"{side}.pointcloud.npy")
            if points.shape != (35, 20, 6) or not np.isfinite(points).all():
                raise ValueError(f"Expected finite [35,20,6] pointcloud for {episode_id}/{side}")
            displacement0 = points[..., 3:6].astype(np.float32)
            xyz.append((displacement0 - displacement0).reshape(700, 3))
        qpos = self._first(directory / "qpos.npy").astype(np.float32)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError(f"Expected finite [14] qpos for {episode_id}")
        return {
            "episode_id": episode_id,
            "rgb": np.ascontiguousarray(rgb),
            "tactile_diff": np.stack(differences),
            "tactile_xyz": np.stack(xyz),
            "qpos": qpos,
            "initial_label": 0,
        }
