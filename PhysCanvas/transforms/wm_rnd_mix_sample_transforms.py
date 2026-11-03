import copy
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader as DecordVideoReader
from giga_datasets import video_utils
from giga_train import TRANSFORMS

from ..depth import FastFoundationStereoDepth, FoundationStereoDepth
from ..motion import ARXLink678GrayMotionRenderer, ARXMotionImageRenderer
from .wm_rnd_mix_transforms import WMRNDMixTransforms


REMOTE_CACHE_PREFIXES = ("/mnt/oss_models", "/mnt/oss_data")
PI05_TACTILE_SIDES = (
    "left_wrist_0",
    "left_wrist_1",
    "right_wrist_0",
    "right_wrist_1",
)
REWARD_TASK_IDS = {
    "bottle": 0,
    "shiguan_saizi": 1,
    "jimu": 2,
}


def _reward_task_id_from_cache_root(cache_root):
    root_name = Path(str(cache_root)).name.lower()
    if "bottle" in root_name:
        return REWARD_TASK_IDS["bottle"]
    if "shiguan_saizi" in root_name or "saizi" in root_name:
        return REWARD_TASK_IDS["shiguan_saizi"]
    if "jimu" in root_name:
        return REWARD_TASK_IDS["jimu"]
    return -1


def _cache_mmap_mode(path):
    mode = os.environ.get("PHYSCANVAS_CACHE_MMAP", "auto").strip().lower()
    if mode in ("1", "true", "yes", "on", "r"):
        return "r"
    if mode in ("0", "false", "no", "off", "none"):
        return None
    path_str = os.path.abspath(os.path.expanduser(str(path)))
    if path_str.startswith(REMOTE_CACHE_PREFIXES):
        return None
    return "r"


def _load_cache_npy(path):
    return np.load(path, mmap_mode=_cache_mmap_mode(path), allow_pickle=False)


@TRANSFORMS.register
class WMRNDMixSampleFastFSTransforms(WMRNDMixTransforms):
    """Stage-one transform for raw sample_data episodes.

    RGB frames are read from the sample videos. Metric depth frames are generated
    lazily from IR-left/IR-right via Fast-FoundationStereo. Replay/motion frames
    are generated online from qpos + ARX URDF FK and cached as PNGs.
    """

    def __init__(self, *args, **kwargs):
        include_depth = kwargs.pop("unified_wan_include_depth", None)
        if include_depth is None:
            include_depth = os.environ.get("PHYSCANVAS_UNIFIED_WAN_INCLUDE_DEPTH", "1")
        self.unified_wan_include_depth = str(include_depth).lower() not in ("0", "false", "no", "none", "")
        append_tactile = kwargs.pop("unified_wan_append_tactile_diff", None)
        if append_tactile is None:
            append_tactile = os.environ.get("PHYSCANVAS_UNIFIED_WAN_APPEND_TACTILE_DIFF", "0")
        append_flow = kwargs.pop("unified_wan_append_tactile_flow", None)
        if append_flow is None:
            append_flow = os.environ.get("PHYSCANVAS_UNIFIED_WAN_APPEND_TACTILE_FLOW", "0")
        self.unified_wan_append_tactile_diff = str(append_tactile).lower() not in (
            "0",
            "false",
            "no",
            "none",
            "",
        )
        self.unified_wan_append_tactile_flow = str(append_flow).lower() not in (
            "0",
            "false",
            "no",
            "none",
            "",
        )
        self.unified_wan_tactile_flow_encoding = str(
            kwargs.pop(
                "unified_wan_tactile_flow_encoding",
                os.environ.get("PHYSCANVAS_UNIFIED_WAN_TACTILE_FLOW_ENCODING", "xyz_linear"),
            )
        ).lower()
        self.unified_wan_tactile_flow_sincos_scale = float(
            kwargs.pop(
                "unified_wan_tactile_flow_sincos_scale",
                os.environ.get("PHYSCANVAS_UNIFIED_WAN_TACTILE_FLOW_SINCOS_SCALE", "60.0"),
            )
        )
        self.unified_wan_tactile_flow_linear_max_disp = float(
            kwargs.pop(
                "unified_wan_tactile_flow_linear_max_disp",
                os.environ.get("PHYSCANVAS_UNIFIED_WAN_TACTILE_FLOW_LINEAR_MAX_DISP", "0.08"),
            )
        )
        self.unified_wan_tactile_flow_vqvae_path = str(
            kwargs.pop(
                "unified_wan_tactile_flow_vqvae_path",
                os.environ.get(
                    "PHYSCANVAS_UNIFIED_WAN_TACTILE_FLOW_VQVAE",
                    os.environ.get("TACTILE_FLOW_VQVAE", ""),
                ),
            )
        )
        self.unified_wan_tactile_flow_vqvae_latent_scale = float(
            kwargs.pop(
                "unified_wan_tactile_flow_vqvae_latent_scale",
                os.environ.get("PHYSCANVAS_UNIFIED_WAN_TACTILE_FLOW_VQVAE_LATENT_SCALE", "20.0"),
            )
        )
        self.dual_reference_mode = str(
            kwargs.pop(
                "dual_reference_mode",
                os.environ.get("PHYSCANVAS_DUAL_REFERENCE_MODE", "dual_memory"),
            )
        ).lower()
        if self.dual_reference_mode not in ("clip_only", "naive_concat", "dual_memory"):
            raise ValueError(f"Unsupported dual_reference_mode={self.dual_reference_mode}")
        super().__init__(*args, **kwargs)
        self._depth_generator = None
        self._motion_renderer = None
        self._flow_vqvae = None
        self._flow_vqvae_cfg = None
        self._cache_frame_files: dict[tuple[str, str, str], list[Path]] = {}

    def _get_depth_generator(self, data_dict):
        if self._depth_generator is None:
            source = data_dict.get("sample_depth_source", "fast_foundation_stereo")
            if source == "foundation_stereo":
                self._depth_generator = FoundationStereoDepth(data_dict["sample_foundation_stereo"])
            elif source == "fast_foundation_stereo":
                self._depth_generator = FastFoundationStereoDepth(data_dict["sample_fastfs"])
            else:
                raise ValueError(f"Unsupported sample depth source: {source}")
        return self._depth_generator

    def _get_motion_renderer(self, data_dict):
        settings = data_dict.get("sample_motion", {})
        mode = settings.get("mode", "arx_fk")
        if mode == "arxx5_link678_gray":
            if self._motion_renderer is None or not isinstance(self._motion_renderer, ARXLink678GrayMotionRenderer):
                self._motion_renderer = ARXLink678GrayMotionRenderer(settings)
            return self._motion_renderer
        if mode != "arx_fk":
            raise ValueError(f"Unsupported sample motion mode: {mode}")
        if not settings.get("urdf_path") or not settings.get("mesh_dir"):
            raise ValueError("sample motion generation requires arx_urdf_path and arx_mesh_dir")
        if self._motion_renderer is None:
            self._motion_renderer = ARXMotionImageRenderer(settings)
        return self._motion_renderer

    def __call__(self, data_dict):
        if self.num_views != 3:
            raise ValueError("WMRNDMixSampleFastFSTransforms currently supports num_views=3 only.")
        if data_dict.get("sample_cache_source") == "unified_cache":
            return self._call_unified_cache(data_dict)
        if data_dict.get("sample_cache_source") == "tactile_cache":
            return self._call_tactile_cache(data_dict)

        front_view_images = DecordVideoReader(data_dict["cam_high_video_path"])
        left_view_images = DecordVideoReader(data_dict["cam_left_wrist_video_path"])
        right_view_images = DecordVideoReader(data_dict["cam_right_wrist_video_path"])
        qpos_len = len(data_dict["qpos"]) if data_dict.get("qpos") is not None else data_dict.get("sample_num_frames", 0)
        video_length = min(
            len(front_view_images),
            len(left_view_images),
            len(right_view_images),
            int(data_dict.get("sample_num_frames", qpos_len)),
            qpos_len,
        )
        if video_length <= 0:
            raise RuntimeError(f"empty sample episode: {data_dict.get('episode_id')}")

        if "sample_clip_start" in data_dict:
            stride = int(data_dict.get("sample_clip_stride", self.debug_sample_stride))
            start_frame = max(0, min(int(data_dict["sample_clip_start"]), video_length - 1))
            last_frame = start_frame + stride * (self.num_frames - 1)
            if last_frame > video_length - 1:
                start_frame = max(0, (video_length - 1) - stride * (self.num_frames - 1))
                last_frame = min(video_length - 1, start_frame + stride * (self.num_frames - 1))
            sample_indexes = np.linspace(start_frame, last_frame, num=self.num_frames, dtype=int)
        elif self.debug_fixed_sampling:
            start_frame = max(0, min(self.debug_sample_start, video_length - 1))
            stride = self.debug_sample_stride
            last_frame = start_frame + stride * (self.num_frames - 1)
            if last_frame > video_length - 1:
                start_frame = max(0, (video_length - 1) - stride * (self.num_frames - 1))
                last_frame = min(video_length - 1, start_frame + stride * (self.num_frames - 1))
            sample_indexes = np.linspace(start_frame, last_frame, num=self.num_frames, dtype=int)
        elif self.max_stride is not None:
            stride = random.randint(1, self.max_stride)
            start_frame = random.randint(0, max(0, video_length - stride * (self.num_frames - 1) - 1))
            end_frame = min(video_length - 1, start_frame + stride * (self.num_frames - 1))
            sample_indexes = np.linspace(start_frame, end_frame, num=self.num_frames, dtype=int)
        else:
            sample_indexes = np.linspace(0, video_length - 1, self.num_frames, dtype=int)

        def view_raw_indexes(view_key, video_len):
            view_indices = data_dict.get("sample_view_indices", {}).get(view_key)
            if view_indices is None:
                return np.clip(sample_indexes, 0, video_len - 1)
            view_indices = np.asarray(view_indices, dtype=np.int64)
            aligned = np.clip(sample_indexes, 0, len(view_indices) - 1)
            return np.clip(view_indices[aligned], 0, video_len - 1)

        def get_input_images(view_key, video):
            safe_indexes = view_raw_indexes(view_key, len(video))
            input_images = video_utils.sample_video(video, safe_indexes, method=2)
            data_dict["input_fps"] = self.fps
            return self._normalize_rgb_like_frames(input_images)

        def get_fastfs_depth(view_key):
            generator = self._get_depth_generator(data_dict)
            safe_indexes = np.clip(sample_indexes, 0, video_length - 1)
            depth = generator.depth_frames(data_dict, view_key, safe_indexes)
            return self._normalize_metric_depth_frames(depth)

        def get_motion_images(view_key):
            renderer = self._get_motion_renderer(data_dict)
            safe_indexes = np.clip(sample_indexes, 0, video_length - 1)
            if self.dst_size is None:
                output_size = (224, 224)
            else:
                output_size = (int(self.dst_size[1]), int(self.dst_size[0]))
            motion = renderer.frames(data_dict, safe_indexes, output_size=output_size)[view_key]
            data_dict["input_fps"] = self.fps
            return self._normalize_rgb_like_frames(motion)

        front_images = get_input_images("high", front_view_images)
        left_images = get_input_images("left_wrist", left_view_images)
        right_images = get_input_images("right_wrist", right_view_images)
        data_dict["input_front_images"] = torch.cat([front_images, left_images, right_images], dim=-1)

        front_depth = get_fastfs_depth("high")
        left_depth = get_fastfs_depth("left_wrist")
        right_depth = get_fastfs_depth("right_wrist")
        data_dict["input_depth_images"] = torch.cat([front_depth, left_depth, right_depth], dim=-1)

        front_replay = get_motion_images("high")
        left_replay = get_motion_images("left_wrist")
        right_replay = get_motion_images("right_wrist")
        data_dict["input_replay_images"] = torch.cat([front_replay, left_replay, right_replay], dim=-1)

        ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_front_images"].shape[0])
        ref_masks = ref_masks[:, None, None, None]
        ref_latent_masks = ref_latent_masks[None, :, None, None]
        data_dict["input_front_ref_images"] = copy.deepcopy(data_dict["input_front_images"]) * ref_masks
        data_dict["input_front_ref_depth_images"] = copy.deepcopy(data_dict["input_depth_images"]) * ref_masks
        data_dict["input_front_ref_masks"] = ref_latent_masks

        if not self.is_train:
            raise AssertionError("WMRNDMixSampleFastFSTransforms is train-only.")

        new_data_dict = {
            "fps": data_dict.get("input_fps", self.fps),
            "front_images": torch.cat([data_dict["input_front_images"], data_dict["input_depth_images"]], dim=-2),
            "front_depth_images": data_dict["input_depth_images"],
            "front_ref_images": torch.cat([data_dict["input_front_ref_images"], data_dict["input_front_ref_depth_images"]], dim=-2),
            "front_ref_depth_images": data_dict["input_front_ref_depth_images"],
            "front_ref_masks": data_dict["input_front_ref_masks"],
            "replay_condition": torch.cat([data_dict["input_replay_images"], data_dict["input_replay_images"]], dim=-2),
            "prompt_embeds": self.default_prompt_embeds,
        }
        if "qpos" in data_dict and data_dict["qpos"] is not None:
            qpos_seq = data_dict["qpos"]
            if isinstance(qpos_seq, torch.Tensor):
                new_data_dict["qpos"] = qpos_seq[sample_indexes].float()
            else:
                new_data_dict["qpos"] = torch.from_numpy(np.asarray(qpos_seq)[sample_indexes]).float()
        return new_data_dict

    def _sample_indexes(self, data_dict, video_length):
        if "sample_clip_start" in data_dict:
            stride = int(data_dict.get("sample_clip_stride", self.debug_sample_stride))
            start_frame = max(0, min(int(data_dict["sample_clip_start"]), video_length - 1))
            last_frame = start_frame + stride * (self.num_frames - 1)
            if last_frame > video_length - 1:
                start_frame = max(0, (video_length - 1) - stride * (self.num_frames - 1))
                last_frame = min(video_length - 1, start_frame + stride * (self.num_frames - 1))
            return np.linspace(start_frame, last_frame, num=self.num_frames, dtype=int)
        if self.debug_fixed_sampling:
            start_frame = max(0, min(self.debug_sample_start, video_length - 1))
            stride = self.debug_sample_stride
            last_frame = start_frame + stride * (self.num_frames - 1)
            if last_frame > video_length - 1:
                start_frame = max(0, (video_length - 1) - stride * (self.num_frames - 1))
                last_frame = min(video_length - 1, start_frame + stride * (self.num_frames - 1))
            return np.linspace(start_frame, last_frame, num=self.num_frames, dtype=int)
        if self.max_stride is not None:
            stride = random.randint(1, self.max_stride)
            start_frame = random.randint(0, max(0, video_length - stride * (self.num_frames - 1) - 1))
            end_frame = min(video_length - 1, start_frame + stride * (self.num_frames - 1))
            return np.linspace(start_frame, end_frame, num=self.num_frames, dtype=int)
        return np.linspace(0, video_length - 1, self.num_frames, dtype=int)

    @staticmethod
    def _dual_reference_indices(sample_indexes, frame_step, raw_len):
        sample_indexes = np.asarray(sample_indexes, dtype=np.int64)
        if sample_indexes.size == 0:
            raise ValueError("sample_indexes must not be empty")
        clip_index = int(sample_indexes[0])
        clip_stride = int(sample_indexes[1] - sample_indexes[0]) if sample_indexes.size > 1 else 1
        frame_step = max(1, int(frame_step))
        raw_len = max(1, int(raw_len))
        return {
            "episode_index": 0,
            "episode_raw_index": 0,
            "clip_index": clip_index,
            "clip_stride": max(1, clip_stride),
            "clip_raw_index": int(np.clip(clip_index * frame_step, 0, raw_len - 1)),
            "is_duplicate": clip_index == 0,
        }

    @staticmethod
    def _read_cache_npy_frames(path, indexes):
        arr = _load_cache_npy(path)
        safe = np.clip(np.asarray(indexes, dtype=np.int64), 0, arr.shape[0] - 1)
        return np.asarray(arr[safe])

    @staticmethod
    def _normalize_tactile_diff_frames(frames):
        return torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _normalize_tactile_diff_side_frames(frames):
        # Input: T, S, H, W, C. Output: T, S, C, H, W.
        return torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0).permute(0, 1, 4, 2, 3).contiguous()

    @staticmethod
    def _tactile_sides_to_wan_panel(tactile_diff_images, target_hw):
        if tactile_diff_images.ndim != 5:
            raise ValueError(
                "Expected tactile_diff_images with shape (T, S, C, H, W), "
                f"got {tuple(tactile_diff_images.shape)}"
            )
        num_frames, num_sides, channels, height, width = tactile_diff_images.shape
        panel = (
            tactile_diff_images.permute(0, 2, 3, 1, 4)
            .reshape(num_frames, channels, height, num_sides * width)
            .contiguous()
        )
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if panel.shape[-2:] != (target_h, target_w):
            panel = F.interpolate(panel, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return panel.contiguous()

    @staticmethod
    def _resize_flow_grid(flow_grid, target_grid_hw):
        target_h, target_w = int(target_grid_hw[0]), int(target_grid_hw[1])
        if flow_grid.shape[-3:-1] == (target_h, target_w):
            return flow_grid
        if flow_grid.shape[-3:-1] == (35, 20) and (target_h, target_w) == (18, 10):
            return flow_grid[..., ::2, ::2, :]
        flat = flow_grid.reshape(-1, flow_grid.shape[-3], flow_grid.shape[-2], 2).permute(0, 3, 1, 2)
        resized = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1).reshape(*flow_grid.shape[:-3], target_h, target_w, 2).contiguous()

    @staticmethod
    def _infer_flow_vqvae_grid(flow_vqvae, cfg):
        grid = getattr(flow_vqvae, "flow_grid_shape", None)
        if grid is not None:
            return int(grid[0]), int(grid[1])
        num_markers = int(
            getattr(flow_vqvae, "num_markers", 0)
            or cfg.get("model", {}).get("vqvae", {}).get("num_markers", 0)
            or cfg.get("num_markers", 0)
            or 0
        )
        if num_markers == 180:
            return 18, 10
        if num_markers == 700:
            return 35, 20
        if num_markers <= 0:
            raise ValueError("Cannot infer flow VQ-VAE marker grid")
        h = int(np.sqrt(num_markers))
        while h > 1 and num_markers % h != 0:
            h -= 1
        return h, num_markers // h

    @staticmethod
    def _infer_latent_hw_from_tokens(num_tokens):
        h = int(np.sqrt(int(num_tokens)))
        while h > 1 and int(num_tokens) % h != 0:
            h -= 1
        return h, int(num_tokens) // h

    def _get_flow_vqvae(self):
        if self._flow_vqvae is not None:
            return self._flow_vqvae, self._flow_vqvae_cfg
        path = self.unified_wan_tactile_flow_vqvae_path
        if not path:
            raise ValueError(
                "unified_wan_tactile_flow_encoding=vqvae requires "
                "PHYSCANVAS_UNIFIED_WAN_TACTILE_FLOW_VQVAE or TACTILE_FLOW_VQVAE"
            )
        root = os.environ.get("TACTILE_MMDIT_ROOT")
        if not root:
            raise ValueError("vqvae flow encoding requires TACTILE_MMDIT_ROOT")
        if Path(root).exists() and root not in sys.path:
            sys.path.insert(0, root)
        from models.vqvae import TactileFlowCNNVQVAE, TactileFlowTactileVAE2DVQVAE, TactileVQVAE

        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("config", {})
        state = ckpt["model_state_dict"]
        keys = set(state.keys())
        if "enc_queries" in keys:
            model = TactileVQVAE.from_config(cfg)
        elif any(key.startswith("to_latent.") for key in keys):
            model = TactileFlowCNNVQVAE.from_config(cfg)
        else:
            model = TactileFlowTactileVAE2DVQVAE.from_config(cfg)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)
        self._flow_vqvae = model
        self._flow_vqvae_cfg = cfg
        return self._flow_vqvae, self._flow_vqvae_cfg

    def _tactile_flow_grid_to_sincos_panel(self, flow_grids, episode0_flow_grids, target_hw):
        displacement = flow_grids - episode0_flow_grids[None]
        dx = displacement[..., 0] * self.unified_wan_tactile_flow_sincos_scale
        dy = displacement[..., 1] * self.unified_wan_tactile_flow_sincos_scale
        norm = torch.sqrt(dx.square() + dy.square() + 1e-8)
        encoded = torch.stack([torch.sin(dx), torch.sin(dy), torch.cos(norm)], dim=2)
        return self._tactile_sides_to_wan_panel(encoded, target_hw)

    def _tactile_flow_grid_to_linear_panel(self, flow_grids, episode0_flow_grids, target_hw):
        displacement = flow_grids - episode0_flow_grids[None]
        max_disp = max(float(self.unified_wan_tactile_flow_linear_max_disp), 1e-6)
        dx = (displacement[..., 0] / max_disp).clamp(-1.0, 1.0)
        dy = (displacement[..., 1] / max_disp).clamp(-1.0, 1.0)
        norm = (torch.sqrt(displacement[..., 0].square() + displacement[..., 1].square() + 1e-8) / max_disp).clamp(
            0.0, 1.0
        )
        encoded = torch.stack([dx, dy, norm], dim=2)
        return self._tactile_sides_to_wan_panel(encoded, target_hw)

    def _tactile_flow_xyz_grid_to_linear_panel(self, xyz_displacement_grids, target_hw):
        if xyz_displacement_grids.ndim != 5 or xyz_displacement_grids.shape[-1] != 3:
            raise ValueError(
                "Expected XYZ tactile displacement grids with shape (T,S,H,W,3), "
                f"got {tuple(xyz_displacement_grids.shape)}"
            )
        max_disp = max(float(self.unified_wan_tactile_flow_linear_max_disp), 1e-6)
        encoded = (xyz_displacement_grids / max_disp).clamp(-1.0, 1.0)
        encoded = encoded.permute(0, 1, 4, 2, 3).contiguous()
        return self._tactile_sides_to_wan_panel(encoded, target_hw)

    def _tactile_flow_grid_to_vqvae_panel(self, flow_grids, episode0_flow_grids, target_hw):
        flow_vqvae, cfg = self._get_flow_vqvae()
        target_grid = self._infer_flow_vqvae_grid(flow_vqvae, cfg)
        flow = self._resize_flow_grid(flow_grids, target_grid)
        ref = self._resize_flow_grid(episode0_flow_grids, target_grid)
        displacement = (flow - ref[None]).float()
        num_frames, num_sides, grid_h, grid_w, _ = displacement.shape
        flow_input = displacement.reshape(num_frames * num_sides, grid_h * grid_w, 2)
        with torch.no_grad():
            if hasattr(flow_vqvae, "encode_vq"):
                latent = flow_vqvae.encode_vq(flow_input)[0]
            else:
                latent = flow_vqvae.encode(flow_input)
        latent = latent.detach().float()
        scale = max(float(self.unified_wan_tactile_flow_vqvae_latent_scale), 1e-6)
        latent = (latent / scale).clamp(-1.0, 1.0)
        latent_hw = getattr(flow_vqvae, "_last_latent_hw", None)
        if latent_hw is None:
            latent_hw = self._infer_latent_hw_from_tokens(latent.shape[1])
        latent_h, latent_w = int(latent_hw[0]), int(latent_hw[1])
        chunks = list(torch.chunk(latent, 3, dim=-1))
        while len(chunks) < 3:
            chunks.append(torch.zeros_like(chunks[0]))
        channels = torch.stack([chunk.mean(dim=-1) for chunk in chunks[:3]], dim=-1)
        encoded = channels.reshape(num_frames, num_sides, latent_h, latent_w, 3).permute(0, 1, 4, 2, 3)
        return self._tactile_sides_to_wan_panel(encoded.contiguous(), target_hw)

    def _tactile_flow_grid_to_wan_panel(self, tactile, target_hw):
        if self.unified_wan_tactile_flow_encoding in (
            "xyz",
            "xyz_linear",
            "linear_xyz",
            "physical_xyz",
        ):
            if "tactile_flow_xyz_grids" not in tactile:
                raise RuntimeError("XYZ tactile flow panel requires tactile_flow_xyz_grids")
            return self._tactile_flow_xyz_grid_to_linear_panel(
                tactile["tactile_flow_xyz_grids"].float(),
                target_hw,
            )
        if "tactile_flow_grids" not in tactile or "tactile_episode0_flow_grids" not in tactile:
            raise RuntimeError("Tactile flow panel requires tactile_flow_grids and tactile_episode0_flow_grids")
        flow_grids = tactile["tactile_flow_grids"].float()
        episode0_flow_grids = tactile["tactile_episode0_flow_grids"].float()
        if self.unified_wan_tactile_flow_encoding in ("sincos", "sin_cos", "grid", "direct"):
            return self._tactile_flow_grid_to_sincos_panel(flow_grids, episode0_flow_grids, target_hw)
        if self.unified_wan_tactile_flow_encoding in ("linear", "lin", "direct_linear"):
            return self._tactile_flow_grid_to_linear_panel(flow_grids, episode0_flow_grids, target_hw)
        if self.unified_wan_tactile_flow_encoding in ("vqvae", "flow_vqvae", "vae", "vae_latent"):
            return self._tactile_flow_grid_to_vqvae_panel(flow_grids, episode0_flow_grids, target_hw)
        raise ValueError(f"Unsupported unified_wan_tactile_flow_encoding={self.unified_wan_tactile_flow_encoding}")

    def _read_tactile_cache_frames(self, episode_dir, data_dict, sample_indexes):
        meta = data_dict.get("sample_cache_meta", {})
        tactile_side = meta.get("primary_tactile_side") or meta.get("tactile_side") or "left_wrist_0"
        point_path = episode_dir / f"point_{tactile_side}_xy.npy"
        if not point_path.exists():
            point_path = episode_dir / "point_xy.npy"
        diff_path = episode_dir / f"tactile_{tactile_side}_diff.npy"
        if not point_path.exists() or not diff_path.exists():
            return {}
        points = self._read_cache_npy_frames(point_path, sample_indexes).astype(np.float32)
        episode0_points = self._read_cache_npy_frames(point_path, [0]).astype(np.float32)[0]
        diff = self._read_cache_npy_frames(diff_path, sample_indexes)
        return {
            "tactile_points": torch.from_numpy(points),
            "tactile_episode0_points": torch.from_numpy(episode0_points),
            "tactile_diff_images": self._normalize_tactile_diff_frames(diff),
        }

    def _read_unified_tactile_frames(self, episode_dir, data_dict, raw_indexes):
        tactile_dir = episode_dir / "tactile"
        sides = list(data_dict.get("sample_tactile_sides", []))
        if not sides:
            return {}
        if tuple(sides) != PI05_TACTILE_SIDES:
            raise ValueError(
                "Pi0.5 tactile sensor order must be "
                f"{PI05_TACTILE_SIDES}, got {tuple(sides)}"
            )
        diff_h = int(os.environ.get("PHYSCANVAS_TACTILE_DIFF_IMAGE_H", "350"))
        diff_w = int(os.environ.get("PHYSCANVAS_TACTILE_DIFF_IMAGE_W", "200"))
        point_frames = []
        episode0_point_frames = []
        flow_grid_frames = []
        episode0_flow_grid_frames = []
        xyz_displacement_grid_frames = []
        diff_frames = []
        pi05_displacement_frames = []
        for side in sides:
            point_path = tactile_dir / f"{side}.pointcloud.npy"
            diff_path = tactile_dir / f"{side}.difference.npy"
            if not point_path.exists() or not diff_path.exists():
                return {}
            point = self._read_cache_npy_frames(point_path, raw_indexes).astype(np.float32)
            episode0_point = self._read_cache_npy_frames(point_path, [0]).astype(np.float32)[0]
            # pointcloud is T,H,W,6. The first two channels are used as 2D flow markers.
            point_xy = point[..., :2].reshape(point.shape[0], -1, 2)
            episode0_point_xy = episode0_point[..., :2].reshape(-1, 2)
            point_xyz = point[..., 3:6].reshape(point.shape[0], -1, 3)
            episode0_point_xyz = episode0_point[..., 3:6].reshape(-1, 3)
            if point_xyz.shape[1:] != (700, 3):
                raise ValueError(
                    f"Expected 700 Xense marker displacements for {side}, "
                    f"got {point_xyz.shape[1:]}"
                )
            pi05_displacement_frames.append(
                np.ascontiguousarray(point_xyz - episode0_point_xyz[None])
            )
            xyz_displacement_grid_frames.append(
                np.ascontiguousarray(point[..., 3:6] - episode0_point[None, ..., 3:6])
            )
            flow_grid_frames.append(point[..., :2])
            episode0_flow_grid_frames.append(episode0_point[..., :2])
            diff = self._read_cache_npy_frames(diff_path, raw_indexes)
            if diff.shape[1] != diff_h or diff.shape[2] != diff_w:
                diff = np.stack(
                    [cv2.resize(frame, (diff_w, diff_h), interpolation=cv2.INTER_AREA) for frame in diff],
                    axis=0,
                )
            point_frames.append(point_xy)
            episode0_point_frames.append(episode0_point_xy)
            diff_frames.append(diff)
        points = np.stack(point_frames, axis=1)
        episode0_points = np.stack(episode0_point_frames, axis=0)
        flow_grids = np.stack(flow_grid_frames, axis=1).astype(np.float32)
        episode0_flow_grids = np.stack(episode0_flow_grid_frames, axis=0).astype(np.float32)
        xyz_displacement_grids = np.stack(xyz_displacement_grid_frames, axis=1).astype(np.float32)
        diffs = np.stack(diff_frames, axis=1)
        pi05_tactile = np.stack(pi05_displacement_frames, axis=1).astype(np.float32)
        return {
            "tactile_points": torch.from_numpy(points),
            "tactile_episode0_points": torch.from_numpy(episode0_points),
            "tactile_flow_grids": torch.from_numpy(flow_grids),
            "tactile_episode0_flow_grids": torch.from_numpy(episode0_flow_grids),
            "tactile_flow_xyz_grids": torch.from_numpy(xyz_displacement_grids),
            "tactile_diff_images": self._normalize_tactile_diff_side_frames(diffs),
            "tactile_sides": sides,
            "pi05_tactile": torch.from_numpy(pi05_tactile),
        }

    def _cache_file_list(self, root, episode_id, view_or_camera, suffix):
        key = (str(root), str(episode_id), str(view_or_camera), str(suffix))
        if key not in self._cache_frame_files:
            frame_dir = Path(root) / episode_id / view_or_camera
            files = sorted(frame_dir.glob(suffix))
            if not files:
                raise FileNotFoundError(f"missing cache frames under {frame_dir} matching {suffix}")
            self._cache_frame_files[key] = files
        return self._cache_frame_files[key]

    @staticmethod
    def _nearest_existing_file(files, index):
        if not files:
            return None
        index = int(np.clip(index, 0, len(files) - 1))
        if files[index].exists():
            return files[index]
        for radius in range(1, len(files)):
            left = index - radius
            if left >= 0 and files[left].exists():
                return files[left]
            right = index + radius
            if right < len(files) and files[right].exists():
                return files[right]
        return None

    def _read_depth_cache_frames(self, data_dict, view_key, indexes):
        camera_name = data_dict["sample_camera_names"][view_key]
        root = Path(data_dict["sample_depth_cache_dir"]) / "foundation_stereo"
        files = self._cache_file_list(root, data_dict["episode_id"], camera_name, "frame_*.npy")
        frames = []
        for idx in np.clip(np.asarray(indexes, dtype=np.int64), 0, len(files) - 1):
            path = files[int(idx)]
            try:
                frames.append(np.load(path).astype(np.float32))
                continue
            except FileNotFoundError:
                fallback = self._nearest_existing_file(files, int(idx))
                if fallback is None:
                    raise
                path = fallback
            try:
                frames.append(np.load(path).astype(np.float32))
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"missing depth cache frame episode={data_dict['episode_id']} "
                    f"view={view_key} camera={camera_name} positional_index={int(idx)} root={root}"
                ) from exc
        return np.stack(frames, axis=0)

    def _read_motion_cache_frames(self, data_dict, view_key, indexes):
        settings = data_dict.get("sample_motion", {})
        variant = settings.get("cache_variant", "arxx5_gripper_filled_gray_055")
        root = Path(settings["cache_dir"]) / variant
        h = int(self.dst_size[1]) if self.dst_size is not None else 224
        w = int(self.dst_size[0]) if self.dst_size is not None else 224
        try:
            files = self._cache_file_list(root, data_dict["episode_id"], view_key, f"h{h}_w{w}_frame_*.png")
        except FileNotFoundError:
            files = self._cache_file_list(root, data_dict["episode_id"], view_key, "h*_w*_frame_*.png")
        frames = []
        for idx in np.clip(np.asarray(indexes, dtype=np.int64), 0, len(files) - 1):
            path = files[int(idx)]
            if not path.exists():
                fallback = self._nearest_existing_file(files, int(idx))
                if fallback is None:
                    raise FileNotFoundError(
                        f"missing motion cache frame episode={data_dict['episode_id']} "
                        f"view={view_key} positional_index={int(idx)} root={root}"
                    )
                path = fallback
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                fallback = self._nearest_existing_file(files, int(idx))
                if fallback is None or fallback == path:
                    raise RuntimeError(f"failed to read cached motion image: {path}")
                image = cv2.imread(str(fallback), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(
                    f"missing motion cache frame episode={data_dict['episode_id']} "
                    f"view={view_key} positional_index={int(idx)} root={root}"
                )
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.stack(frames, axis=0)

    def _call_tactile_cache(self, data_dict):
        video_length = int(data_dict.get("sample_num_frames", 0))
        if video_length <= 0:
            raise RuntimeError(f"empty tactile cache episode: {data_dict.get('episode_id')}")
        sample_indexes = self._sample_indexes(data_dict, video_length)
        episode_dir = Path(data_dict["sample_cache_episode_dir"])
        rgb_by_view = {}
        depth_by_view = {}
        motion_by_view = {}

        def get_rgb(view_key):
            camera_name = data_dict["sample_camera_names"][view_key]
            frames = self._read_cache_npy_frames(episode_dir / f"camera_{camera_name}_rgb.npy", sample_indexes)
            data_dict["input_fps"] = self.fps
            rgb_by_view[view_key] = self._normalize_rgb_like_frames(frames)
            return rgb_by_view[view_key]

        def get_depth(view_key):
            depth = self._read_depth_cache_frames(data_dict, view_key, sample_indexes)
            depth_by_view[view_key] = self._normalize_metric_depth_frames(depth)
            return depth_by_view[view_key]

        def get_motion(view_key):
            motion = self._read_motion_cache_frames(data_dict, view_key, sample_indexes)
            data_dict["input_fps"] = self.fps
            motion_by_view[view_key] = self._normalize_rgb_like_frames(motion)
            return motion_by_view[view_key]

        front_images = get_rgb("high")
        left_images = get_rgb("left_wrist")
        right_images = get_rgb("right_wrist")
        data_dict["input_front_images"] = torch.cat([front_images, left_images, right_images], dim=-1)

        if self.unified_wan_include_depth:
            front_depth = get_depth("high")
            left_depth = get_depth("left_wrist")
            right_depth = get_depth("right_wrist")
            data_dict["input_depth_images"] = torch.cat([front_depth, left_depth, right_depth], dim=-1)

        front_replay = get_motion("high")
        left_replay = get_motion("left_wrist")
        right_replay = get_motion("right_wrist")
        data_dict["input_replay_images"] = torch.cat([front_replay, left_replay, right_replay], dim=-1)

        ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_front_images"].shape[0])
        ref_masks = ref_masks[:, None, None, None]
        ref_latent_masks = ref_latent_masks[None, :, None, None]
        data_dict["input_front_ref_images"] = copy.deepcopy(data_dict["input_front_images"]) * ref_masks
        if self.unified_wan_include_depth:
            data_dict["input_front_ref_depth_images"] = copy.deepcopy(data_dict["input_depth_images"]) * ref_masks
        data_dict["input_front_ref_masks"] = ref_latent_masks

        if not self.is_train:
            raise AssertionError("WMRNDMixSampleFastFSTransforms is train-only.")

        out = {
            "fps": data_dict.get("input_fps", self.fps),
            "front_images": torch.cat([data_dict["input_front_images"], data_dict["input_depth_images"]], dim=-2),
            "front_depth_images": data_dict["input_depth_images"],
            "front_ref_images": torch.cat([data_dict["input_front_ref_images"], data_dict["input_front_ref_depth_images"]], dim=-2),
            "front_ref_depth_images": data_dict["input_front_ref_depth_images"],
            "front_ref_masks": data_dict["input_front_ref_masks"],
            "replay_condition": torch.cat([data_dict["input_replay_images"], data_dict["input_replay_images"]], dim=-2),
            "prompt_embeds": self.default_prompt_embeds,
        }
        tactile = self._read_tactile_cache_frames(episode_dir, data_dict, sample_indexes)
        if tactile:
            view_order = ("high", "left_wrist", "right_wrist")
            out.update(tactile)
            out["tactile_rgb_condition"] = torch.stack([rgb_by_view[key] for key in view_order], dim=1)
            out["tactile_depth_condition"] = torch.stack([depth_by_view[key] for key in view_order], dim=1)
            out["tactile_motion_condition"] = torch.stack([motion_by_view[key] for key in view_order], dim=1)
        return out

    def _call_unified_cache(self, data_dict):
        video_length = int(data_dict.get("sample_num_frames", 0))
        if video_length <= 0:
            raise RuntimeError(f"empty unified cache episode: {data_dict.get('episode_id')}")
        forced_sample_indexes = data_dict.pop(
            "_object_state_forced_sample_indexes", None
        )
        if forced_sample_indexes is None:
            sample_indexes = self._sample_indexes(data_dict, video_length)
        else:
            sample_indexes = np.asarray(forced_sample_indexes, dtype=np.int64)
            if sample_indexes.shape != (self.num_frames,):
                raise ValueError(
                    "forced object-state sample indexes must have shape "
                    f"({self.num_frames},), got {sample_indexes.shape}"
                )
            if np.any(sample_indexes < 0) or np.any(sample_indexes >= video_length):
                raise IndexError(
                    "forced object-state sample indexes are outside the episode: "
                    f"min={int(sample_indexes.min())}, max={int(sample_indexes.max())}, "
                    f"video_length={video_length}"
                )
        frame_step = max(1, int(data_dict.get("sample_frame_step", 1)))
        raw_len = int(data_dict.get("sample_raw_num_frames", video_length * frame_step))
        raw_indexes = np.clip(np.asarray(sample_indexes, dtype=np.int64) * frame_step, 0, raw_len - 1)
        reference_indices = self._dual_reference_indices(sample_indexes, frame_step, raw_len)
        episode_raw_indexes = np.asarray([reference_indices["episode_raw_index"]], dtype=np.int64)
        episode_dir = Path(data_dict["sample_cache_episode_dir"])
        rgb_by_view = {}
        depth_by_view = {}
        motion_by_view = {}

        def get_rgb(view_key, indexes=raw_indexes, store=True):
            camera_name = data_dict["sample_camera_names"][view_key]
            frames = self._read_cache_npy_frames(episode_dir / f"{camera_name}_rgb.npy", indexes)
            data_dict["input_fps"] = float(data_dict.get("sample_target_hz", self.fps))
            normalized = self._normalize_rgb_like_frames(frames)
            if store:
                rgb_by_view[view_key] = normalized
            return normalized

        def get_depth(view_key, indexes=raw_indexes, store=True):
            camera_name = data_dict["sample_camera_names"][view_key]
            depth = self._read_cache_npy_frames(episode_dir / f"{camera_name}_depth.npy", indexes).astype(np.float32)
            normalized = self._normalize_metric_depth_frames(depth)
            if store:
                depth_by_view[view_key] = normalized
            return normalized

        def get_motion(view_key, indexes=raw_indexes, store=True):
            camera_name = data_dict["sample_camera_names"][view_key]
            motion = self._read_cache_npy_frames(episode_dir / f"{camera_name}_motion.npy", indexes)
            data_dict["input_fps"] = float(data_dict.get("sample_target_hz", self.fps))
            normalized = self._normalize_rgb_like_frames(motion)
            if store:
                motion_by_view[view_key] = normalized
            return normalized

        front_images = get_rgb("high")
        left_images = get_rgb("left_wrist")
        right_images = get_rgb("right_wrist")
        data_dict["input_front_images"] = torch.cat([front_images, left_images, right_images], dim=-1)

        if self.unified_wan_include_depth:
            front_depth = get_depth("high")
            left_depth = get_depth("left_wrist")
            right_depth = get_depth("right_wrist")
            data_dict["input_depth_images"] = torch.cat([front_depth, left_depth, right_depth], dim=-1)

        front_replay = get_motion("high")
        left_replay = get_motion("left_wrist")
        right_replay = get_motion("right_wrist")
        data_dict["input_replay_images"] = torch.cat([front_replay, left_replay, right_replay], dim=-1)

        episode_rgb = torch.cat(
            [get_rgb(view, episode_raw_indexes, store=False) for view in ("high", "left_wrist", "right_wrist")],
            dim=-1,
        )
        ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_front_images"].shape[0])
        ref_masks = ref_masks[:, None, None, None]
        ref_latent_masks = ref_latent_masks[None, :, None, None]
        data_dict["input_front_ref_images"] = copy.deepcopy(data_dict["input_front_images"]) * ref_masks
        if self.unified_wan_include_depth:
            data_dict["input_front_ref_depth_images"] = copy.deepcopy(data_dict["input_depth_images"]) * ref_masks
        data_dict["input_front_ref_masks"] = ref_latent_masks

        if not self.is_train:
            raise AssertionError("WMRNDMixSampleFastFSTransforms is train-only.")

        tactile = self._read_unified_tactile_frames(episode_dir, data_dict, raw_indexes)
        front_blocks = [data_dict["input_front_images"]]
        ref_blocks = [data_dict["input_front_ref_images"]]
        # The global anchor describes scene appearance only. Depth is derived
        # from RGB, while tactile diff/flow has no useful signal at episode t=0.
        episode_ref_blocks = [episode_rgb]
        episode_ref_modality_mask = [1.0]
        episode_ref_modality_scores = [1.0]
        replay_blocks = [data_dict["input_replay_images"]]
        if self.unified_wan_include_depth:
            front_blocks.append(data_dict["input_depth_images"])
            ref_blocks.append(data_dict["input_front_ref_depth_images"])
            replay_blocks.append(data_dict["input_replay_images"])
        if self.unified_wan_append_tactile_diff:
            if not tactile or "tactile_diff_images" not in tactile:
                raise RuntimeError(
                    "unified_wan_append_tactile_diff=True requires tactile diff cache. "
                    f"episode={data_dict.get('episode_id')}"
                )
            tactile_panel = self._tactile_sides_to_wan_panel(
                tactile["tactile_diff_images"],
                data_dict["input_front_images"].shape[-2:],
            )
            data_dict["input_tactile_diff_images"] = tactile_panel
            data_dict["input_front_ref_tactile_diff_images"] = copy.deepcopy(tactile_panel) * ref_masks
            front_blocks.append(data_dict["input_tactile_diff_images"])
            ref_blocks.append(data_dict["input_front_ref_tactile_diff_images"])
            replay_blocks.append(data_dict["input_replay_images"])
        if self.unified_wan_append_tactile_flow:
            if not tactile:
                raise RuntimeError(
                    "unified_wan_append_tactile_flow=True requires tactile cache. "
                    f"episode={data_dict.get('episode_id')}"
                )
            tactile_flow_panel = self._tactile_flow_grid_to_wan_panel(
                tactile,
                data_dict["input_front_images"].shape[-2:],
            )
            data_dict["input_tactile_flow_images"] = tactile_flow_panel
            data_dict["input_front_ref_tactile_flow_images"] = copy.deepcopy(tactile_flow_panel) * ref_masks
            front_blocks.append(data_dict["input_tactile_flow_images"])
            ref_blocks.append(data_dict["input_front_ref_tactile_flow_images"])
            replay_blocks.append(data_dict["input_replay_images"])

        clip_ref_images = torch.cat(ref_blocks, dim=-2)
        episode_ref_images = torch.cat(episode_ref_blocks, dim=-2)
        episode_ref_enabled = self.dual_reference_mode != "clip_only"
        episode_ref_mask = float(episode_ref_enabled)
        out = {
            "fps": data_dict.get("input_fps", self.fps),
            "front_images": torch.cat(front_blocks, dim=-2),
            "front_ref_images": clip_ref_images,
            "front_ref_masks": data_dict["input_front_ref_masks"],
            "front_ref_pixel_masks": ref_masks,
            "clip_ref_images": clip_ref_images,
            "clip_ref_masks": data_dict["input_front_ref_masks"],
            "episode_ref_images": episode_ref_images,
            "episode_ref_mask": torch.tensor([episode_ref_mask], dtype=torch.float32),
            "episode_ref_available_mask": torch.ones(1, dtype=torch.float32),
            "episode_ref_role_id": torch.tensor(0, dtype=torch.long),
            "clip_ref_role_id": torch.tensor(1, dtype=torch.long),
            "episode_ref_frame_index": torch.tensor(reference_indices["episode_index"], dtype=torch.long),
            "episode_ref_raw_index": torch.tensor(reference_indices["episode_raw_index"], dtype=torch.long),
            "clip_ref_frame_index": torch.tensor(reference_indices["clip_index"], dtype=torch.long),
            "clip_ref_raw_index": torch.tensor(reference_indices["clip_raw_index"], dtype=torch.long),
            "episode_ref_time_offset_frames": torch.tensor(
                reference_indices["clip_index"] - reference_indices["episode_index"], dtype=torch.float32
            ),
            "clip_ref_stride_frames": torch.tensor(reference_indices["clip_stride"], dtype=torch.float32),
            "episode_ref_is_duplicate": torch.tensor(reference_indices["is_duplicate"], dtype=torch.bool),
            "reward_task_id": torch.tensor(
                _reward_task_id_from_cache_root(data_dict.get("sample_cache_root", "")),
                dtype=torch.long,
            ),
            "episode_ref_modality_mask": torch.tensor(episode_ref_modality_mask, dtype=torch.float32),
            "episode_ref_modality_scores": torch.tensor(episode_ref_modality_scores, dtype=torch.float32),
            "replay_condition": torch.cat(replay_blocks, dim=-2),
            "prompt_embeds": self.default_prompt_embeds,
            "sample_frame_indices": torch.from_numpy(raw_indexes.copy()).long(),
        }
        if self.unified_wan_include_depth:
            out["front_depth_images"] = data_dict["input_depth_images"]
            out["front_ref_depth_images"] = data_dict["input_front_ref_depth_images"]
        if self.unified_wan_append_tactile_diff:
            out["front_tactile_diff_images"] = data_dict["input_tactile_diff_images"]
            out["front_ref_tactile_diff_images"] = data_dict["input_front_ref_tactile_diff_images"]
        if self.unified_wan_append_tactile_flow:
            out["front_tactile_flow_images"] = data_dict["input_tactile_flow_images"]
            out["front_ref_tactile_flow_images"] = data_dict["input_front_ref_tactile_flow_images"]
        qpos_path = episode_dir / "qpos.npy"
        if qpos_path.exists():
            out["qpos"] = torch.from_numpy(self._read_cache_npy_frames(qpos_path, raw_indexes).astype(np.float32))
        if tactile:
            view_order = ("high", "left_wrist", "right_wrist")
            out.update(tactile)
            out["tactile_rgb_condition"] = torch.stack([rgb_by_view[key] for key in view_order], dim=1)
            if self.unified_wan_include_depth:
                out["tactile_depth_condition"] = torch.stack([depth_by_view[key] for key in view_order], dim=1)
            out["tactile_motion_condition"] = torch.stack([motion_by_view[key] for key in view_order], dim=1)
        return out
