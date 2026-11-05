"""Explicit-asset wrapper around the public PhysCanvas motion renderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MotionRenderer:
    def __init__(self, settings: dict):
        settings = dict(settings)
        settings_file = settings.pop("settings_file", None)
        if settings_file is not None:
            path = Path(settings_file).expanduser()
            with path.open() as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict) or "settings_file" in loaded:
                raise ValueError("motion.settings_file must contain a non-recursive JSON settings object")
            # Explicit launch-site overrides take precedence over the file;
            # paths remain absolute and source assets are never rewritten.
            settings = {**loaded, **settings}
        # Do not inherit the public renderer's historical Jimu/calibration
        # defaults. Both tasks must specify their previously audited assets.
        for key in ("debug_dir", "x5a_root", "urdf_path", "fk_offset_path", "left_calib", "right_calib"):
            value = settings.get(key)
            if not value or not Path(value).is_absolute() or not Path(value).exists():
                raise ValueError(f"motion.{key} must name an existing absolute asset path")
        if "render_size_hw" not in settings:
            raise ValueError("motion.render_size_hw must explicitly match the audited motion reference")
        self.render_size_hw = tuple(int(x) for x in settings.pop("render_size_hw"))
        if len(self.render_size_hw) != 2 or min(self.render_size_hw) <= 0:
            raise ValueError("motion.render_size_hw must be [height,width]")
        settings["use_cache"] = False
        settings["cache_dir"] = None
        self.settings = settings
        from PhysCanvas.motion.arxx5_link678_gray_renderer import ARXLink678GrayMotionRenderer
        self.renderer = ARXLink678GrayMotionRenderer(settings)

    def frames(self, qpos: np.ndarray, episode_id: str) -> np.ndarray:
        """Render current qpos + 8 actions, return [9,224,672,3] uint8."""
        from PhysCanvas.utils import resize_with_pad

        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (9, 14) or not np.isfinite(qpos).all():
            raise ValueError("Motion needs finite qpos [9,14]: current state plus eight absolute targets")
        frames = self.renderer.frames(
            {"qpos": qpos, "episode_id": str(episode_id)},
            np.arange(9, dtype=np.int64), output_size=self.render_size_hw,
        )
        views = []
        for name in ("high", "left_wrist", "right_wrist"):
            value = np.asarray(frames[name])
            if value.shape != (9, *self.render_size_hw, 3) or value.dtype != np.uint8:
                raise ValueError(f"Invalid motion output for {name}: {value.shape}/{value.dtype}")
            views.append(resize_with_pad(value, 224, 224))
        return np.ascontiguousarray(np.concatenate(views, axis=2))
