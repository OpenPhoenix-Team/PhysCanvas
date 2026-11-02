import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


_VIEW_TO_CAM = {
    "high": "head",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}
_CAM_TO_VIEW = {value: key for key, value in _VIEW_TO_CAM.items()}


class ARXLink678GrayMotionRenderer:
    """Render the whole gripper mesh projection on a flat gray background.

    This follows scripts/debug/check_traj_map_for_wujie_data/export_arxx5_link78_mesh_overlay.py:
    joint state -> ARX X5 FK -> link6/7/8 mesh transforms -> per-camera w2c -> pinhole projection.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        debug_dir = settings.get("debug_dir") or os.environ.get("PHYSCANVAS_ARXX5_DEBUG_DIR")
        if not debug_dir:
            raise ValueError("ARX mesh rendering requires debug_dir or PHYSCANVAS_ARXX5_DEBUG_DIR")
        self.debug_dir = Path(debug_dir)
        self.cache_dir = settings.get("cache_dir")
        self.use_cache = bool(settings.get("use_cache", True))
        self.background = int(settings.get("gray_background", 190))
        self.gripper_gray = int(settings.get("gripper_gray", 55))
        # 0 means draw every face. Filled rendering should not decimate faces by default,
        # otherwise the gripper surface can look patchy rather than solid.
        self.mesh_max_faces = int(settings.get("mesh_max_faces", 0))
        self.outline_max_faces = int(settings.get("outline_max_faces", 0))
        self.line_thickness = int(settings.get("line_thickness", 1))
        self.links = ("link6", "link7", "link8")
        gripper_color = (self.gripper_gray, self.gripper_gray, self.gripper_gray)
        self.colors = {(arm, link): gripper_color for arm in ("left", "right") for link in self.links}
        if str(self.debug_dir) not in sys.path:
            sys.path.insert(0, str(self.debug_dir))

        from arxx5_urdf_fk import configure as configure_fk
        from arxx5_dataset4_wancontrolmultiview import scale_intrinsic
        from export_arxx5_link78_mesh_overlay import (
            GEOMETRY_LINK_FOR_LINK,
            fk_pose_and_link_mesh_transforms,
            load_stl_triangles,
            load_urdf_model,
            make_transform,
            rpy_to_rot,
            selected_links,
            transform_triangles,
            transforms_for_camera,
            draw_mesh_wireframe,
        )
        from export_jimu1_0625_157_temp_head_calib import get_color_intrinsics, make_updated_extrinsics

        self._scale_intrinsic = scale_intrinsic
        self._fk_pose_and_link_mesh_transforms = fk_pose_and_link_mesh_transforms
        self._transform_triangles = transform_triangles
        self._transforms_for_camera = transforms_for_camera
        self._draw_mesh_wireframe = draw_mesh_wireframe
        self._get_color_intrinsics = get_color_intrinsics

        self.args = self._make_args(settings)
        configure_fk(
            urdf_path=self.args.urdf_path,
            offset_path=self.args.fk_offset_path,
            offset_mode=self.args.fk_offset_mode,
            state_unit="rad",
        )
        self.extr = make_updated_extrinsics(self.args.left_calib, self.args.right_calib)
        child_to_joint, visuals = load_urdf_model(self.args.urdf_path, self.args.x5a_root)

        def chain_to_link(link: str, base_link: str = "base_link"):
            chain = []
            cur = link
            while cur != base_link:
                joint = child_to_joint[cur]
                chain.append(joint)
                cur = joint.parent
            return list(reversed(chain))

        links = selected_links(self.args)
        chain_keys = tuple(dict.fromkeys((*links, "link6")))
        self.chain_by_link = {link: chain_to_link(link) for link in chain_keys}
        if self.args.ignore_visual_origin:
            self.visual_T = {link: np.eye(4, dtype=np.float64) for link in links}
        else:
            self.visual_T = {
                link: make_transform(
                    rpy_to_rot(visuals[GEOMETRY_LINK_FOR_LINK[link]].origin_rpy),
                    visuals[GEOMETRY_LINK_FOR_LINK[link]].origin_xyz,
                )
                for link in links
            }
        self.tris_by_link = {
            link: load_stl_triangles(visuals[GEOMETRY_LINK_FOR_LINK[link]].mesh_path, self.args.mesh_scale)
            for link in links
        }

    def _make_args(self, settings: dict[str, Any]) -> SimpleNamespace:
        def resolve(path: str) -> Path:
            p = Path(path).expanduser()
            return p if p.is_absolute() else self.debug_dir / p

        x5a_root = settings.get("x5a_root") or os.environ.get("PHYSCANVAS_X5A_ROOT")
        if not x5a_root:
            raise ValueError("ARX mesh rendering requires x5a_root or PHYSCANVAS_X5A_ROOT")
        x5a_root = Path(x5a_root).expanduser()

        return SimpleNamespace(
            mesh_selection="both",
            mesh_max_faces=int(settings.get("mesh_max_faces", 0)),
            mesh_scale=float(settings.get("mesh_scale", 0.0)),
            ignore_visual_origin=bool(settings.get("ignore_visual_origin", False)),
            line_thickness=int(settings.get("line_thickness", 1)),
            alpha=1.0,
            gripper_mode=settings.get("gripper_mode", "distance_m"),
            gripper_distance_is_total_gap=bool(settings.get("gripper_distance_is_total_gap", True)),
            gripper_state_scale=float(settings.get("gripper_state_scale", 1.0)),
            auto_gripper_zero_offset=bool(settings.get("auto_gripper_zero_offset", False)),
            gripper_zero_offset_percentile=float(settings.get("gripper_zero_offset_percentile", 5.0)),
            gripper_left_state_offset=settings.get("gripper_left_state_offset"),
            gripper_right_state_offset=settings.get("gripper_right_state_offset"),
            gripper_max_state=float(settings.get("gripper_max_state", 3.0)),
            gripper_max_meter=float(settings.get("gripper_max_meter", 0.044)),
            head_mesh_tool_offset_x=float(settings.get("head_mesh_tool_offset_x", 0.0)),
            head_mesh_tool_offset_y=float(settings.get("head_mesh_tool_offset_y", 0.0)),
            head_mesh_tool_offset_z=float(settings.get("head_mesh_tool_offset_z", 0.0)),
            wrist_mesh_tool_offset_x=float(settings.get("wrist_mesh_tool_offset_x", 0.0)),
            wrist_mesh_tool_offset_y=float(settings.get("wrist_mesh_tool_offset_y", 0.0)),
            wrist_mesh_tool_offset_z=float(settings.get("wrist_mesh_tool_offset_z", 0.0)),
            real_eef_uses_fk_offset=True,
            x5a_root=x5a_root,
            urdf_path=Path(settings.get("urdf_path") or x5a_root / "urdf" / "X5A.urdf"),
            fk_offset_path=resolve(settings.get("fk_offset_path", "assets/arxx5_fk_offsets_jimu1_0625_157.json")),
            fk_offset_mode=settings.get("fk_offset_mode", "tool"),
            left_calib=resolve(settings.get("left_calib", "assets/calib_eye_in_hand_left_M/result_eye_in_hand.json")),
            right_calib=resolve(settings.get("right_calib", "assets/calib_eye_in_hand_right_M/result_eye_in_hand.json")),
        )

    def _cache_path(self, sample: dict[str, Any], view_key: str, frame_idx: int, output_size: tuple[int, int]) -> str | None:
        if not self.cache_dir:
            return None
        h, w = output_size
        return os.path.join(
            self.cache_dir,
            f"arxx5_gripper_allfacecontour_partialfill_v10_gray_{self.gripper_gray:03d}_faces{self.mesh_max_faces}_outline{self.outline_max_faces}",
            str(sample.get("episode_id", "episode")),
            view_key,
            f"h{h}_w{w}_frame_{int(frame_idx):06d}.png",
        )

    @staticmethod
    def _project_mesh_faces(
        tris_world: np.ndarray,
        w2c: np.ndarray,
        K: np.ndarray,
        output_size: tuple[int, int],
        max_faces: int,
    ) -> tuple[np.ndarray, dict]:
        h, w = output_size
        if tris_world.shape[0] > max_faces > 0:
            step = int(np.ceil(tris_world.shape[0] / max_faces))
            tris_world = tris_world[::step]

        if tris_world.size == 0:
            return np.empty((0, 3, 2), dtype=np.int32), {"faces_considered": 0, "faces_drawn": 0}

        flat = tris_world.reshape(-1, 3).astype(np.float64, copy=False)
        homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
        cam = (homog @ w2c.T)[:, :3].reshape(tris_world.shape)

        z = cam[:, :, 2]
        valid_z = np.all(z > 1e-5, axis=1)
        if not np.any(valid_z):
            return np.empty((0, 3, 2), dtype=np.int32), {
                "faces_considered": int(tris_world.shape[0]),
                "faces_drawn": 0,
            }

        cam = cam[valid_z]
        z = z[valid_z]
        uv = np.empty((cam.shape[0], 3, 2), dtype=np.float64)
        uv[:, :, 0] = K[0, 0] * cam[:, :, 0] / (z + 1e-8) + K[0, 2]
        uv[:, :, 1] = K[1, 1] * cam[:, :, 1] / (z + 1e-8) + K[1, 2]

        offscreen = np.all(
            (uv[:, :, 0] < -w) | (uv[:, :, 0] > 2 * w) | (uv[:, :, 1] < -h) | (uv[:, :, 1] > 2 * h),
            axis=1,
        )
        pts = np.rint(uv[~offscreen]).astype(np.int32)
        return pts, {
            "faces_considered": int(tris_world.shape[0]),
            "faces_drawn": int(len(pts)),
        }

    @staticmethod
    def _paint_projected_faces(
        canvas: np.ndarray,
        face_pts: np.ndarray,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        if len(face_pts) == 0:
            return canvas
        h, w = canvas.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, face_pts.astype(np.int32), 255, lineType=cv2.LINE_AA)
        canvas[mask > 0] = color
        return canvas

    @staticmethod
    def _draw_mesh_filled(
        rgb: np.ndarray,
        tris_world: np.ndarray,
        w2c: np.ndarray,
        K: np.ndarray,
        color: tuple[int, int, int],
        max_faces: int,
    ) -> tuple[np.ndarray, dict]:
        canvas = rgb.copy()
        pts, stats = ARXLink678GrayMotionRenderer._project_mesh_faces(
            tris_world,
            w2c,
            K,
            rgb.shape[:2],
            max_faces,
        )
        canvas = ARXLink678GrayMotionRenderer._paint_projected_faces(canvas, pts, color)
        return canvas, stats

    @staticmethod
    def _draw_mesh_face_contours(
        rgb: np.ndarray,
        tris_world: np.ndarray,
        w2c: np.ndarray,
        K: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
        max_faces: int,
    ) -> tuple[np.ndarray, dict]:
        canvas = rgb.copy()
        pts, stats = ARXLink678GrayMotionRenderer._project_mesh_faces(
            tris_world,
            w2c,
            K,
            rgb.shape[:2],
            max_faces,
        )
        if len(pts) == 0:
            return canvas, stats
        cv2.polylines(
            canvas,
            pts.astype(np.int32),
            isClosed=True,
            color=color,
            thickness=max(1, thickness),
            lineType=cv2.LINE_AA,
        )
        return canvas, stats

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

    def _intrinsics(self, output_size: tuple[int, int]) -> dict[str, np.ndarray]:
        out_h, out_w = output_size
        base = self._get_color_intrinsics()
        return {cam: self._scale_intrinsic(K, 480, 640, out_h, out_w) for cam, K in base.items()}

    def _render_missing(
        self,
        sample: dict[str, Any],
        frame_indices: np.ndarray,
        output_size: tuple[int, int],
    ) -> dict[tuple[str, int], np.ndarray]:
        from arxx5_dataset4_wancontrolmultiview import make_per_camera_sequences

        qpos = np.asarray(sample["qpos"], dtype=np.float32)
        safe = np.clip(np.asarray(frame_indices, dtype=np.int64), 0, len(qpos) - 1)
        state = qpos[safe]
        fk_ee, mesh_by_arm_base = self._fk_pose_and_link_mesh_transforms(
            state,
            self.chain_by_link,
            self.visual_T,
            self.args,
        )
        _, w2c_by_cam = make_per_camera_sequences(fk_ee, state, self.extr, "identity")
        Ks = self._intrinsics(output_size)
        rendered: dict[tuple[str, int], np.ndarray] = {}
        for local_i, frame_idx in enumerate(safe.tolist()):
            for cam in ("head", "left_wrist", "right_wrist"):
                canvas = np.full((output_size[0], output_size[1], 3), self.background, dtype=np.uint8)
                for arm, link, T_mesh_world in self._transforms_for_camera(mesh_by_arm_base, self.extr, cam, local_i):
                    tris_world = self._transform_triangles(self.tris_by_link[link], T_mesh_world)
                    canvas, _ = self._draw_mesh_filled(
                        canvas,
                        tris_world,
                        w2c_by_cam[cam][local_i],
                        Ks[cam],
                        self.colors[(arm, link)],
                        self.mesh_max_faces,
                    )
                    canvas, _ = self._draw_mesh_face_contours(
                        canvas,
                        tris_world,
                        w2c_by_cam[cam][local_i],
                        Ks[cam],
                        self.colors[(arm, link)],
                        self.line_thickness,
                        self.outline_max_faces,
                    )
                rendered[(_CAM_TO_VIEW[cam], int(frame_idx))] = canvas
        return rendered

    def frames(
        self,
        sample: dict[str, Any],
        frame_indices: np.ndarray,
        output_size: tuple[int, int] = (224, 224),
    ) -> dict[str, np.ndarray]:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        output_size = (int(output_size[0]), int(output_size[1]))
        result: dict[str, list[np.ndarray | None]] = {"high": [], "left_wrist": [], "right_wrist": []}
        missing_positions = []
        missing_indices = []
        for pos, frame_idx in enumerate(frame_indices.tolist()):
            cached = {}
            all_cached = True
            for view_key in result:
                cache_path = self._cache_path(sample, view_key, frame_idx, output_size)
                if self.use_cache and cache_path and os.path.exists(cache_path):
                    cached[view_key] = self._read_png(cache_path)
                else:
                    all_cached = False
                    break
            if all_cached:
                for view_key in result:
                    result[view_key].append(cached[view_key])
            else:
                missing_positions.append(pos)
                missing_indices.append(frame_idx)
                for view_key in result:
                    result[view_key].append(None)
        if missing_positions:
            rendered = self._render_missing(sample, np.asarray(missing_indices, dtype=np.int64), output_size)
            for pos, frame_idx in zip(missing_positions, missing_indices):
                for view_key in result:
                    image = rendered[(view_key, int(frame_idx))]
                    cache_path = self._cache_path(sample, view_key, frame_idx, output_size)
                    if self.use_cache and cache_path:
                        self._write_png_atomic(cache_path, image)
                    result[view_key][pos] = image
        return {view_key: np.stack(images, axis=0) for view_key, images in result.items()}
