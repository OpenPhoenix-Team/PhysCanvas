# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pinned motion rendering for Jimu, Shiguan-Saizi, and red bottle-cap rollouts.

All supported source groups, including the single-right-arm 0823/0824 data,
intentionally use the 0822 standard head-camera calibration.  The base view uses the filled
link6/7/8 renderer from iMac commit ``a14a3df``.  The wrist views deliberately
use that renderer's zero tool-local offset: replay against the preserved
``wirefast_v5`` wrist reference is pixel exact, while the previously inferred
``(+0.015, 0, +0.005)`` compatibility offset moved both wrist projections about
22--24 pixels above the real RGB grippers.  Every active source artifact is
hash checked before use.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import subprocess
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RENDERER_VERSION = "arxx5_gripper_v10_head0822_wrist_zero_offset_v3"
RENDERER_IMPLEMENTATION_VERSION = RENDERER_VERSION
PINNED_SOURCE_RENDERER_VERSION = "arxx5_gripper_allfacecontour_partialfill_v10"
# The compact reset cache predates the standard-head repair and retains this
# old metadata tag.  It is provenance for the copied cache, not the identity of
# the online renderer selected by this module.
LEGACY_RESET_CACHE_MOTION_ALGORITHM = "arxx5_gripper_wirefast_v5"
RENDERER_GIT_COMMIT = "a14a3df347a26950a3f8a5388e458aca23920f4f"
RENDERER_GIT_PATH = "iMac/motion/arxx5_link678_gray_renderer.py"
RENDERER_SHA256 = (
    "a219bc1abeec671b77585882b0e97d6dec2d0d73dbe31ce5577fed14a361010e"
)
STANDARD_0822_CALIBRATION_SHA256 = (
    "79953e55b9d0bcd7066628fc987031519ea3700b02f4b22e86899a72830558c6"
)
LEFT_WRIST_CALIBRATION_SHA256 = (
    "3b2a71efc0c74fb9695290818e883bb4dce0fd4044e0cb762412cdbf4f1595dc"
)
RIGHT_WRIST_CALIBRATION_SHA256 = (
    "f5502c34f7f77dc8ab15d38d8607c857d8f0d6c724e5a37624c53a8d8a9917f0"
)
FK_OFFSET_SHA256 = (
    "e14104c8a6c2cefd44c44acbecaebae5e05de58b871adab7e86d9fc985cdc1ef"
)
FIXED_RENDER_PARAMETERS = {
    "gray_background": 190,
    "gripper_gray": 55,
    "mesh_max_faces": 1000,
    "outline_max_faces": 0,
    "line_thickness": 1,
    "output_height": 224,
    "output_width": 224,
    "wrist_mesh_tool_offset_x": 0.0,
    "wrist_mesh_tool_offset_y": 0.0,
    "wrist_mesh_tool_offset_z": 0.0,
}
SUPPORTED_SOURCE_GROUPS = (
    "separate_the_building_blocks_and_place_them_into_the_bags_with_matching_colors_",
    "saizi_0822_110",
    "saizi_0822_139",
    "saizi_0824_100",
    "dan_saizi_0823_52",
    "dan_saizi_0823_100",
    "dan_saizi_0823_100_1",
    "dan_saizi_0824_50",
    "unscrew_the_bottle_cap_red_zhf_",
)


@dataclass(frozen=True)
class CalibrationSpec:
    calibration_id: str
    calibration_sha256: str
    episode_markers: tuple[str, ...]
    debug_dir: Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}, "
            f"path={path}"
        )


def _load_calibration_module(specification: CalibrationSpec):
    path = specification.debug_dir / "export_jimu1_0625_157_temp_head_calib.py"
    _require_file_hash(
        path,
        specification.calibration_sha256,
        f"{specification.calibration_id} head calibration",
    )
    module_name = f"_shiguan_calibration_{specification.calibration_id}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot import calibration module: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def _load_historical_renderer_class(
    git_repository: str,
    git_commit: str,
    git_path: str,
    expected_sha256: str,
):
    repository = Path(git_repository).resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"Missing iMac Git repository: {repository}")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "show",
            f"{git_commit}:{git_path}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    actual_sha256 = _sha256_bytes(result.stdout)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Pinned motion renderer SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}, "
            f"source={repository}@{git_commit}:{git_path}"
        )
    module = types.ModuleType("_shiguan_pinned_motion_renderer")
    module.__file__ = f"{repository}@{git_commit}:{git_path}"
    exec(compile(result.stdout, module.__file__, "exec"), module.__dict__)
    renderer_class = getattr(module, "ARXLink678GrayMotionRenderer", None)
    if renderer_class is None:
        raise ImportError("Pinned renderer does not define the expected class")
    return renderer_class


class SourceRoutedShiguanMotionRenderer:
    """Route every supported rollout session to the standard head calibration."""

    renderer_version = RENDERER_VERSION
    renderer_implementation_version = RENDERER_IMPLEMENTATION_VERSION

    def __init__(self, settings: dict[str, Any]) -> None:
        project_root = Path(__file__).resolve().parents[3]
        calibration_root = Path(
            settings.get(
                "calibration_root",
                project_root / "assets/motion_calibration",
            )
        ).resolve()
        self.git_repository = Path(
            settings.get(
                "renderer_git_repository",
                "/mnt/workspace/xzb/workspace/iMac_sample_fastfs",
            )
        ).resolve()
        self.git_commit = str(
            settings.get("renderer_git_commit", RENDERER_GIT_COMMIT)
        )
        self.git_path = str(settings.get("renderer_git_path", RENDERER_GIT_PATH))
        self.renderer_sha256 = str(
            settings.get("renderer_sha256", RENDERER_SHA256)
        )
        fixed_source = {
            "renderer_git_commit": (self.git_commit, RENDERER_GIT_COMMIT),
            "renderer_git_path": (self.git_path, RENDERER_GIT_PATH),
            "renderer_sha256": (self.renderer_sha256, RENDERER_SHA256),
            "renderer_version": (
                str(settings.get("renderer_version", RENDERER_VERSION)),
                RENDERER_VERSION,
            ),
            "renderer_implementation_version": (
                str(
                    settings.get(
                        "renderer_implementation_version",
                        RENDERER_IMPLEMENTATION_VERSION,
                    )
                ),
                RENDERER_IMPLEMENTATION_VERSION,
            ),
        }
        mismatched_source = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in fixed_source.items()
            if actual != expected
        }
        if mismatched_source:
            raise ValueError(
                f"Motion renderer source is not the pinned handoff version: "
                f"{mismatched_source}"
            )

        configured_parameters = {
            "gray_background": int(settings.get("gray_background", 190)),
            "gripper_gray": int(settings.get("gripper_gray", 55)),
            "mesh_max_faces": int(settings.get("mesh_max_faces", 1000)),
            "outline_max_faces": int(settings.get("outline_max_faces", 0)),
            "line_thickness": int(settings.get("line_thickness", 1)),
            "output_height": int(settings.get("output_height", 224)),
            "output_width": int(settings.get("output_width", 224)),
            "wrist_mesh_tool_offset_x": float(
                settings.get("wrist_mesh_tool_offset_x", 0.0)
            ),
            "wrist_mesh_tool_offset_y": float(
                settings.get("wrist_mesh_tool_offset_y", 0.0)
            ),
            "wrist_mesh_tool_offset_z": float(
                settings.get("wrist_mesh_tool_offset_z", 0.0)
            ),
        }
        if configured_parameters != FIXED_RENDER_PARAMETERS:
            raise ValueError(
                "Motion renderer parameters do not match the pinned runtime contract: "
                f"expected={FIXED_RENDER_PARAMETERS}, "
                f"actual={configured_parameters}"
            )
        self.render_parameters = configured_parameters
        self.wrist_mesh_tool_offset = np.eye(4, dtype=np.float64)
        self.wrist_mesh_tool_offset[:3, 3] = np.asarray(
            [
                configured_parameters["wrist_mesh_tool_offset_x"],
                configured_parameters["wrist_mesh_tool_offset_y"],
                configured_parameters["wrist_mesh_tool_offset_z"],
            ],
            dtype=np.float64,
        )

        self.specs = (
            CalibrationSpec(
                calibration_id="shiguan_0822_standard_20260822_1450",
                calibration_sha256=STANDARD_0822_CALIBRATION_SHA256,
                episode_markers=SUPPORTED_SOURCE_GROUPS,
                debug_dir=calibration_root / "0822_shiguan_standard",
            ),
        )
        renderer_class = _load_historical_renderer_class(
            str(self.git_repository),
            self.git_commit,
            self.git_path,
            self.renderer_sha256,
        )
        renderer_settings = dict(settings)
        renderer_settings.update(
            {
                "use_cache": False,
                "cache_dir": "",
                "mesh_max_faces": FIXED_RENDER_PARAMETERS["mesh_max_faces"],
                "outline_max_faces": FIXED_RENDER_PARAMETERS[
                    "outline_max_faces"
                ],
                "line_thickness": FIXED_RENDER_PARAMETERS["line_thickness"],
                "wrist_mesh_tool_offset_x": FIXED_RENDER_PARAMETERS[
                    "wrist_mesh_tool_offset_x"
                ],
                "wrist_mesh_tool_offset_y": FIXED_RENDER_PARAMETERS[
                    "wrist_mesh_tool_offset_y"
                ],
                "wrist_mesh_tool_offset_z": FIXED_RENDER_PARAMETERS[
                    "wrist_mesh_tool_offset_z"
                ],
            }
        )
        self.renderers: dict[str, Any] = {}
        for specification in self.specs:
            debug_dir = specification.debug_dir
            if not debug_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing motion calibration directory: {debug_dir}"
                )
            _require_file_hash(
                debug_dir
                / "assets/calib_eye_in_hand_left_M/result_eye_in_hand.json",
                LEFT_WRIST_CALIBRATION_SHA256,
                "left wrist calibration",
            )
            _require_file_hash(
                debug_dir
                / "assets/calib_eye_in_hand_right_M/result_eye_in_hand.json",
                RIGHT_WRIST_CALIBRATION_SHA256,
                "right wrist calibration",
            )
            _require_file_hash(
                debug_dir / "assets/arxx5_fk_offsets_jimu1_0625_157.json",
                FK_OFFSET_SHA256,
                "ARX FK offset",
            )
            local_settings = dict(renderer_settings)
            local_settings["debug_dir"] = str(debug_dir)
            renderer = renderer_class(local_settings)
            calibration = _load_calibration_module(specification)
            # Bare imports in the historical renderer are process-global.  Set
            # the route-specific extrinsics explicitly so the second instance
            # cannot inherit the first route's head matrix from sys.modules.
            renderer.extr = calibration.make_updated_extrinsics(
                renderer.args.left_calib,
                renderer.args.right_calib,
            )
            self.renderers[specification.calibration_id] = renderer

        self.active_episode_id: str | None = None
        self.active_spec: CalibrationSpec | None = None

    def _spec_for_episode(self, episode_id: str) -> CalibrationSpec:
        matches = [
            specification
            for specification in self.specs
            if any(
                marker in episode_id
                for marker in specification.episode_markers
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "Episode must match exactly one standard motion calibration route: "
                f"episode_id={episode_id!r}, "
                f"matches={[item.calibration_id for item in matches]}"
            )
        return matches[0]

    def set_episode(self, episode_id: str) -> dict[str, Any]:
        episode_id = str(episode_id)
        self.active_episode_id = episode_id
        self.active_spec = self._spec_for_episode(episode_id)
        return self.active_metadata()

    def implementation_metadata(self) -> dict[str, Any]:
        return {
            "motion_renderer_version": self.renderer_version,
            "motion_renderer_implementation_version": (
                self.renderer_implementation_version
            ),
            "motion_renderer_git_repository": str(self.git_repository),
            "motion_renderer_git_commit": self.git_commit,
            "motion_renderer_git_path": self.git_path,
            "motion_renderer_sha256": self.renderer_sha256,
            "motion_renderer_pinned_source_version": (
                PINNED_SOURCE_RENDERER_VERSION
            ),
            "legacy_reset_cache_motion_algorithm": (
                LEGACY_RESET_CACHE_MOTION_ALGORITHM
            ),
            "motion_renderer_parameters": dict(self.render_parameters),
            "motion_renderer_view_algorithms": {
                "base": PINNED_SOURCE_RENDERER_VERSION,
                "left_wrist": "pinned_v10_zero_offset_wirefast_v5_exact_v1",
                "right_wrist": "pinned_v10_zero_offset_wirefast_v5_exact_v1",
            },
            "motion_renderer_validation_scope": (
                "128 unique reset qpos, frame 0, three-view runtime replay; "
                "base uses the unified 0822 reference, wrists are pixel-exact "
                "against the preserved zero-offset reference and independently "
                "gated against RGB"
            ),
            "motion_calibration_routes": {
                marker: {
                    "calibration_id": specification.calibration_id,
                    "calibration_sha256": specification.calibration_sha256,
                }
                for specification in self.specs
                for marker in specification.episode_markers
            },
        }

    def active_metadata(self) -> dict[str, Any]:
        metadata = self.implementation_metadata()
        metadata.update(
            {
                "motion_calibration_id": (
                    None
                    if self.active_spec is None
                    else self.active_spec.calibration_id
                ),
                "motion_calibration_sha256": (
                    None
                    if self.active_spec is None
                    else self.active_spec.calibration_sha256
                ),
                "motion_episode_id": self.active_episode_id,
            }
        )
        return metadata

    def frames(
        self,
        sample: dict[str, Any],
        frame_indices,
        output_size: tuple[int, int] = (224, 224),
    ):
        expected_size = (
            FIXED_RENDER_PARAMETERS["output_height"],
            FIXED_RENDER_PARAMETERS["output_width"],
        )
        output_size = (int(output_size[0]), int(output_size[1]))
        if output_size != expected_size:
            raise ValueError(
                "Standard motion renderer requires 224x224 output: "
                f"expected={expected_size}, actual={output_size}"
            )
        if self.active_spec is None or self.active_episode_id is None:
            episode_id = str(sample.get("episode_id", ""))
            if not episode_id:
                raise RuntimeError("Motion renderer has no active episode route")
            self.set_episode(episode_id)
        renderer = self.renderers[self.active_spec.calibration_id]
        routed_sample = dict(sample)
        routed_sample["episode_id"] = self.active_episode_id
        return renderer.frames(
            routed_sample,
            frame_indices,
            output_size=output_size,
        )


def renderer_settings_from_args(args: Any) -> dict[str, Any]:
    """Build explicit, hash-pinned renderer settings from the WM config."""

    return {
        "calibration_root": str(args.motion_calibration_root),
        "renderer_version": str(args.motion_renderer_version),
        "renderer_implementation_version": str(
            args.motion_renderer_implementation_version
        ),
        "renderer_git_repository": str(args.motion_renderer_git_repository),
        "renderer_git_commit": str(args.motion_renderer_git_commit),
        "renderer_git_path": str(args.motion_renderer_git_path),
        "renderer_sha256": str(args.motion_renderer_sha256),
        "gray_background": int(args.motion_gray_background),
        "gripper_gray": int(args.motion_gripper_gray),
        "mesh_max_faces": int(args.motion_mesh_max_faces),
        "outline_max_faces": int(args.motion_outline_max_faces),
        "line_thickness": int(args.motion_line_thickness),
        "output_height": int(args.motion_output_height),
        "output_width": int(args.motion_output_width),
        "wrist_mesh_tool_offset_x": float(args.motion_wrist_tool_offset_x),
        "wrist_mesh_tool_offset_y": float(args.motion_wrist_tool_offset_y),
        "wrist_mesh_tool_offset_z": float(args.motion_wrist_tool_offset_z),
    }
