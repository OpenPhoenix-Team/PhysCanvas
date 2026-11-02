import hashlib
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np


_CAM_TO_COND = {
    "high": "front",
    "left_wrist": "left",
    "right_wrist": "right",
}


def _as_matrix(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"missing calibration matrix: {name}")
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {mat.shape}")
    return mat


def _calibration_payload(sample: dict[str, Any]) -> dict[str, Any]:
    camera_names = sample["sample_camera_names"]
    params = sample["camera_params"]
    base = params[camera_names["high"]]
    left = params[camera_names["left_wrist"]]
    right = params[camera_names["right_wrist"]]
    return {
        "base_T_cam2left_base": base.get("extrinsics", {}).get("T_cam2left_base"),
        "base_T_cam2right_base": base.get("extrinsics", {}).get("T_cam2right_base"),
        "left_T_cam2gripper": left.get("extrinsics", {}).get("T_cam2gripper"),
        "right_T_cam2gripper": right.get("extrinsics", {}).get("T_cam2gripper"),
    }


def _calibration_tag(sample: dict[str, Any]) -> str:
    payload = _calibration_payload(sample)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class ARXMotionImageRenderer:
    """Render online replay/motion images from ARX qpos using URDF FK.

    The output is a black-background RGB robot mask for each requested view. It
    replaces the temporary raw-RGB replay fallback used by the sample-data path.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self.urdf_path = settings["urdf_path"]
        self.mesh_dir = settings["mesh_dir"]
        self.cache_dir = settings.get("cache_dir")
        self.device = settings.get("device", "cpu")
        self.arm_sample_count = int(settings.get("arm_sample_count", 1200))
        self.use_cache = bool(settings.get("use_cache", True))
        self.robot_color = tuple(int(x) for x in settings.get("robot_color", (230, 230, 230)))
        self.gripper_color = tuple(int(x) for x in settings.get("gripper_color", (255, 120, 60)))
        self.mask_dilate_iterations = int(settings.get("mask_dilate_iterations", 1))
        self._generator = None
        self._params_signature = None

    @staticmethod
    def _intrinsic_matrix(camera_params: dict[str, Any], camera_name: str) -> np.ndarray:
        cam = camera_params[camera_name]
        intr = cam.get("intrinsics_640x480_30", cam)
        return np.array(
            [
                [float(intr["fx"]), 0.0, float(intr["cx"])],
                [0.0, float(intr["fy"]), float(intr["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _get_generator(self, sample: dict[str, Any]):
        calibration = _calibration_payload(sample)
        calibration_by_view = {
            "high": calibration.get("base_T_cam2left_base"),
            "left_wrist": calibration.get("left_T_cam2gripper"),
            "right_wrist": calibration.get("right_T_cam2gripper"),
        }
        signature = tuple(
            (
                view_key,
                sample["sample_camera_names"][view_key],
                tuple(np.ravel(self._intrinsic_matrix(sample["camera_params"], sample["sample_camera_names"][view_key])).tolist()),
                tuple(np.ravel(np.asarray(calibration_by_view[view_key], dtype=np.float64)).tolist()),
            )
            for view_key in ("high", "left_wrist", "right_wrist")
        ) + (tuple(np.ravel(np.asarray(calibration.get("base_T_cam2right_base"), dtype=np.float64)).tolist()),)
        if self._generator is not None and self._params_signature == signature:
            return self._generator

        from third_party.func_flow_control_urdf import CondGenerator

        generator = CondGenerator(
            urdf_path=self.urdf_path,
            gripper_mesh_dir=self.mesh_dir,
            device=self.device,
            load_da3_model=False,
            arm_sample_count=self.arm_sample_count,
            rendermask_dilate_iterations=1,
        )
        generator.intrinsics = {
            _CAM_TO_COND[view_key]: self._intrinsic_matrix(sample["camera_params"], sample["sample_camera_names"][view_key])
            for view_key in ("high", "left_wrist", "right_wrist")
        }
        generator.camera_names = ["front", "left", "right"]
        self._apply_handeye_calibration(generator, sample)
        self._generator = generator
        self._params_signature = signature
        return generator


    def _apply_handeye_calibration(self, generator: Any, sample: dict[str, Any]) -> None:
        camera_names = sample["sample_camera_names"]
        params = sample["camera_params"]
        base_params = params[camera_names["high"]]
        left_params = params[camera_names["left_wrist"]]
        right_params = params[camera_names["right_wrist"]]

        base_ext = base_params.get("extrinsics", {})
        left_ext = left_params.get("extrinsics", {})
        right_ext = right_params.get("extrinsics", {})
        T_front2base_left = _as_matrix(base_ext.get("T_cam2left_base"), "base_0.extrinsics.T_cam2left_base")
        T_front2base_right = _as_matrix(base_ext.get("T_cam2right_base"), "base_0.extrinsics.T_cam2right_base")
        T_left_cam2gripper = _as_matrix(left_ext.get("T_cam2gripper"), "left_wrist_0.extrinsics.T_cam2gripper")
        T_right_cam2gripper = _as_matrix(right_ext.get("T_cam2gripper"), "right_wrist_0.extrinsics.T_cam2gripper")

        generator.T_front2base_left = T_front2base_left
        generator.T_front2base_right = T_front2base_right
        generator.T_base2front_left = np.linalg.inv(T_front2base_left)
        generator.T_base2front_right = np.linalg.inv(T_front2base_right)
        generator.T_left_cam2gripper = T_left_cam2gripper
        generator.T_right_cam2gripper = T_right_cam2gripper

        def forward_kinematics_with_handeye(gen, current_action: np.ndarray) -> dict[str, np.ndarray]:
            qpos = gen._convert_gripper_qpos(current_action)
            q_left = gen._arm_q7_to_q8(qpos[:7])
            q_right = gen._arm_q7_to_q8(qpos[7:])

            extrinsics = {"front": np.eye(4)}

            # T_front2base_* maps the base/front camera frame to each arm base.
            # T_*_cam2gripper maps the wrist camera frame to link6/end-effector.
            # The renderer projects points already expressed in the front camera frame.
            T_base_gripper_L = gen.ets_arm_body.get("link6", gen.ets_link7).fkine(q_left).A
            T_front_cam_L = gen.T_base2front_left @ T_base_gripper_L @ gen.T_left_cam2gripper
            extrinsics["left"] = np.linalg.inv(T_front_cam_L)

            T_base_gripper_R = gen.ets_arm_body.get("link6", gen.ets_link7).fkine(q_right).A
            T_front_cam_R = gen.T_base2front_right @ T_base_gripper_R @ gen.T_right_cam2gripper
            extrinsics["right"] = np.linalg.inv(T_front_cam_R)
            return extrinsics

        generator.forward_kinematics = MethodType(forward_kinematics_with_handeye, generator)

    def _cache_path(self, sample: dict[str, Any], view_key: str, frame_idx: int, output_size: tuple[int, int]) -> str | None:
        if not self.cache_dir:
            return None
        camera_name = sample["sample_camera_names"][view_key]
        h, w = output_size
        return os.path.join(
            self.cache_dir,
            f"calib_{_calibration_tag(sample)}",
            str(sample.get("episode_id", "episode")),
            camera_name,
            f"h{h}_w{w}_frame_{int(frame_idx):06d}.png",
        )

    @staticmethod
    def _read_png(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read cached motion image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _write_png_atomic(path: str, image_rgb: np.ndarray) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}.png"
        ok = cv2.imwrite(tmp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"failed to write motion image cache: {tmp_path}")
        os.replace(tmp_path, path)

    def _mask_to_rgb(self, mask: np.ndarray, gripper_mask: np.ndarray | None = None) -> np.ndarray:
        mask = np.asarray(mask).astype(bool)
        if self.mask_dilate_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=self.mask_dilate_iterations).astype(bool)
        image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        image[mask] = self.robot_color
        if gripper_mask is not None:
            gmask = np.asarray(gripper_mask).astype(bool)
            if gmask.shape == mask.shape:
                image[gmask] = self.gripper_color
        return image

    def _render_missing(
        self,
        sample: dict[str, Any],
        qpos_seq: np.ndarray,
        frame_indices: np.ndarray,
        output_size: tuple[int, int],
    ) -> dict[tuple[str, int], np.ndarray]:
        generator = self._get_generator(sample)
        qpos_seq = np.asarray(qpos_seq, dtype=np.float32)
        points_list, point_masks, render_masks, render_poses, output_intrinsics = generator._prepare_robot_data_from_fk(qpos_seq, output_size)

        rendered: dict[tuple[str, int], np.ndarray] = {}
        cond_names = ["front", "left", "right"]
        for pos, frame_idx in enumerate(frame_indices.tolist()):
            pts = points_list[pos]
            is_gripper = point_masks[pos]
            extr = {cam: render_poses[pos][cond_names.index(cam)] for cam in cond_names}
            intrinsics = output_intrinsics
            for view_key in ("high", "left_wrist", "right_wrist"):
                cam = _CAM_TO_COND[view_key]
                view_idx = cond_names.index(cam)
                mask = render_masks[pos][view_idx]

                # A second gripper-only projection makes the future-contact area easier to read.
                gripper_rgb = None
                if is_gripper.any():
                    gripper_mask = generator._render_robot_mask_from_points(
                        pts[is_gripper], extr[cam], intrinsics[cam], output_size
                    )
                    gripper_rgb = gripper_mask
                image = self._mask_to_rgb(mask, gripper_rgb)
                rendered[(view_key, int(frame_idx))] = image
        return rendered

    def frames(
        self,
        sample: dict[str, Any],
        frame_indices: np.ndarray,
        output_size: tuple[int, int] = (224, 224),
    ) -> dict[str, np.ndarray]:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        qpos = np.asarray(sample["qpos"], dtype=np.float32)
        safe_indices = np.clip(frame_indices, 0, len(qpos) - 1)
        output_size = (int(output_size[0]), int(output_size[1]))

        result: dict[str, list[np.ndarray | None]] = {"high": [], "left_wrist": [], "right_wrist": []}
        missing_positions = []
        missing_indices = []

        for pos, frame_idx in enumerate(safe_indices.tolist()):
            all_cached = True
            cached_for_pos = {}
            for view_key in result:
                cache_path = self._cache_path(sample, view_key, int(frame_idx), output_size)
                if self.use_cache and cache_path is not None and os.path.exists(cache_path):
                    cached_for_pos[view_key] = self._read_png(cache_path)
                else:
                    all_cached = False
                    break
            if all_cached:
                for view_key in result:
                    result[view_key].append(cached_for_pos[view_key])
            else:
                missing_positions.append(pos)
                missing_indices.append(int(frame_idx))
                for view_key in result:
                    result[view_key].append(None)

        if missing_positions:
            qpos_missing = qpos[np.asarray(missing_indices, dtype=np.int64)]
            rendered = self._render_missing(sample, qpos_missing, np.asarray(missing_indices, dtype=np.int64), output_size)
            for pos, frame_idx in zip(missing_positions, missing_indices):
                for view_key in result:
                    image = rendered[(view_key, int(frame_idx))]
                    cache_path = self._cache_path(sample, view_key, int(frame_idx), output_size)
                    if self.use_cache and cache_path is not None:
                        self._write_png_atomic(cache_path, image)
                    result[view_key][pos] = image

        return {view_key: np.stack(images, axis=0) for view_key, images in result.items()}
