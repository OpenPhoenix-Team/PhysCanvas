"""RGB + object-state-canvas transform for unified cache episodes."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from giga_train import TRANSFORMS

from ..object_state_canvas import CanvasSpec, TASK_TO_ID, normalize_task_name, render_state_sequence
from .wm_rnd_mix_sample_transforms import WMRNDMixSampleFastFSTransforms


SEPARATE_VAE_UNUSED_VIDEO_FIELDS = (
    "front_images", "front_ref_images", "clip_ref_images",
    "replay_condition", "front_visual_images",
    "front_visual_ref_images", "replay_visual_images",
    "front_tactile_diff_images", "front_ref_tactile_diff_images",
    "front_tactile_flow_images", "front_ref_tactile_flow_images",
    "tactile_rgb_condition", "tactile_motion_condition",
)


def compact_separate_vae_batch(batch):
    # Only compatibility duplicates are removed. Retained tensors are untouched.
    for key in SEPARATE_VAE_UNUSED_VIDEO_FIELDS:
        batch.pop(key, None)
    return batch


@TRANSFORMS.register
class WMRNDMixObjectStateCanvasTransforms(WMRNDMixSampleFastFSTransforms):
    """Treat object state as a separate VAE modality beside a visual canvas."""

    def __init__(self, *args, **kwargs):
        self.object_state_canvas_encoding = kwargs.pop("object_state_canvas_encoding", "lowfreq_bands_v1")
        self.object_state_canvas_height = int(kwargs.pop("object_state_canvas_height", 32))
        self.object_state_canvas_header_height = int(kwargs.pop("object_state_canvas_header_height", 8))
        self.object_state_visual_canvas_mode = str(
            kwargs.pop("object_state_visual_canvas_mode", "rgb_only")
        ).strip().lower()
        if self.object_state_visual_canvas_mode not in ("rgb_only", "full"):
            raise ValueError(
                "object_state_visual_canvas_mode must be 'rgb_only' or 'full', got "
                f"{self.object_state_visual_canvas_mode!r}"
            )
        self.object_state_canvas_label_map = self._parse_canvas_label_map(
            kwargs.pop("object_state_canvas_label_map", "")
        )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _parse_canvas_label_map(value):
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            if text.startswith("{"):
                value = json.loads(text)
            else:
                value = dict(
                    item.strip().split(":", 1)
                    for item in text.split(",")
                    if item.strip()
                )
        if not isinstance(value, dict):
            raise TypeError(f"object_state_canvas_label_map must be a mapping, got {type(value)}")
        return {int(source): int(target) for source, target in value.items()}

    def _canvas_labels(self, labels, data_dict):
        mapping = self._parse_canvas_label_map(
            data_dict.get("object_state_canvas_label_map", self.object_state_canvas_label_map)
        )
        canvas_labels = np.asarray(labels, dtype=np.int64).copy()
        for source, target in mapping.items():
            canvas_labels[labels == source] = target
        return canvas_labels, mapping

    def _sample_balanced_object_state_indexes(
        self,
        data_dict,
        video_length,
        all_labels,
        frame_step,
    ):
        category = str(data_dict.get("object_state_sampling_category", "uniform"))
        category_ids = {"uniform": -1, "all_zero": 0, "transition": 1, "all_one": 2}
        if category not in category_ids:
            raise ValueError(f"unsupported object-state sampling category: {category!r}")
        if category == "uniform":
            indexes = np.asarray(self._sample_indexes(data_dict, video_length), dtype=np.int64)
            return indexes, category_ids[category], 0

        logical_raw_indexes = np.arange(video_length, dtype=np.int64) * frame_step
        logical_raw_indexes = np.clip(logical_raw_indexes, 0, len(all_labels) - 1)
        logical_labels = np.asarray(all_labels[logical_raw_indexes], dtype=np.int64)
        if category == "all_zero":
            if np.any(logical_labels != 0):
                raise AssertionError(
                    f"all_zero pool contains state 1: {data_dict.get('episode_id')}"
                )
            indexes = np.asarray(self._sample_indexes(data_dict, video_length), dtype=np.int64)
            return indexes, category_ids[category], 0

        max_stride = max(1, int(self.max_stride or 1))
        num_frames = int(self.num_frames)
        if category == "transition":
            positive = np.flatnonzero(logical_labels == 1)
            negative = np.flatnonzero(logical_labels == 0)
            if positive.size == 0 or negative.size == 0:
                raise AssertionError(
                    f"transition pool lacks both states: {data_dict.get('episode_id')}"
                )
            first_one = int(positive[0])
            stride = random.randint(1, max_stride)
            max_offset = min(
                int(data_dict.get("object_state_transition_max_offset", 32)),
                num_frames - 1,
            )
            offset = random.randint(1, max_offset)
            start = max(0, first_one - offset * stride)
            unbounded = start + stride * np.arange(num_frames, dtype=np.int64)
            indexes = np.clip(unbounded, 0, video_length - 1)
            sampled = logical_labels[indexes]
            if not (np.any(sampled == 0) and np.any(sampled == 1)):
                stride = 1
                offset = min(max_offset, max(1, first_one))
                start = max(0, first_one - offset)
                unbounded = start + np.arange(num_frames, dtype=np.int64)
                indexes = np.clip(unbounded, 0, video_length - 1)
                sampled = logical_labels[indexes]
            if not (np.any(sampled == 0) and np.any(sampled == 1)):
                raise AssertionError(
                    f"failed to sample a 0->1 clip: {data_dict.get('episode_id')}"
                )
        else:
            min_real = min(
                num_frames,
                int(data_dict.get("object_state_all_one_min_real_frames", 65)),
            )
            if logical_labels[-1] != 1:
                raise AssertionError(
                    f"all_one pool has no terminal state-1 suffix: {data_dict.get('episode_id')}"
                )
            zeros = np.flatnonzero(logical_labels == 0)
            suffix_start = int(zeros[-1] + 1) if zeros.size else 0
            suffix_length = video_length - suffix_start
            feasible_max_stride = min(
                max_stride,
                (suffix_length - 1) // max(1, min_real - 1),
            )
            if feasible_max_stride < 1:
                raise AssertionError(
                    f"all_one pool has fewer than {min_real} real state-1 frames: "
                    f"{data_dict.get('episode_id')}"
                )
            stride = random.randint(1, feasible_max_stride)
            latest_start = video_length - 1 - stride * (min_real - 1)
            start = random.randint(suffix_start, latest_start)
            unbounded = start + stride * np.arange(num_frames, dtype=np.int64)
            indexes = np.clip(unbounded, 0, video_length - 1)
            if np.any(logical_labels[indexes] != 1):
                raise AssertionError(
                    f"failed to sample an all-state-1 clip: {data_dict.get('episode_id')}"
                )

        padded_frames = int(np.count_nonzero(unbounded >= video_length))
        return indexes, category_ids[category], padded_frames

    def _call_unified_cache(self, data_dict):
        video_length = int(data_dict.get("sample_num_frames", 0))
        if video_length <= 0:
            raise RuntimeError(f"empty unified cache episode: {data_dict.get('episode_id')}")
        if os.environ.get("PHYSCANVAS_OBJECT_STATE_ONLINE_REWARDS") == "1":
            from PhysCanvas.object_state_online_labels import ensure_online_labels
            ensure_online_labels(data_dict)
        label_path = Path(str(data_dict.get("object_state_label_path", "")))
        if not label_path.is_file():
            raise FileNotFoundError(f"missing object-state labels for {data_dict.get('episode_id')}: {label_path}")

        frame_step = max(1, int(data_dict.get("sample_frame_step", 1)))
        raw_len = int(data_dict.get("sample_raw_num_frames", video_length * frame_step))
        with np.load(label_path, allow_pickle=False) as state_file:
            all_labels = np.asarray(state_file["labels"], dtype=np.int64)
            all_probabilities = np.asarray(state_file["probabilities"], dtype=np.float32)
        if os.environ.get("PHYSCANVAS_OBJECT_STATE_ONLINE_REWARDS") == "1":
            positive = np.flatnonzero(all_labels == 1)
            if positive.size == 0:
                data_dict["object_state_sampling_category"] = "all_zero"
            elif positive[0] == 0:
                data_dict["object_state_sampling_category"] = "uniform"
            else:
                data_dict["object_state_sampling_category"] = "transition"
        sample_indexes, sampling_category_id, padded_frames = (
            self._sample_balanced_object_state_indexes(
                data_dict,
                video_length,
                all_labels,
                frame_step,
            )
        )
        raw_indexes = np.clip(np.asarray(sample_indexes, dtype=np.int64) * frame_step, 0, raw_len - 1)
        reference_indices = self._dual_reference_indices(sample_indexes, frame_step, raw_len)
        episode_raw_indexes = np.asarray(
            [reference_indices["episode_raw_index"]], dtype=np.int64
        )
        episode_dir = Path(data_dict["sample_cache_episode_dir"])

        parent_out = None
        if self.object_state_visual_canvas_mode == "full":
            parent_data = dict(data_dict)
            parent_data["_object_state_forced_sample_indexes"] = sample_indexes
            parent_out = super()._call_unified_cache(parent_data)
            visual = parent_out["front_images"]
            visual_ref = parent_out["front_ref_images"]
            replay_visual = parent_out["replay_condition"]
            episode_rgb = parent_out["episode_ref_images"]
            motion_for_state = parent_data["input_replay_images"]
            ref_masks = parent_out.pop("front_ref_pixel_masks")
            ref_latent_masks = parent_out["front_ref_masks"]
            data_dict["input_fps"] = float(parent_out["fps"])
        else:
            def get_rgb(view_key, indexes=raw_indexes):
                camera_name = data_dict["sample_camera_names"][view_key]
                frames = self._read_cache_npy_frames(
                    episode_dir / f"{camera_name}_rgb.npy", indexes
                )
                return self._normalize_rgb_like_frames(frames)

            def get_motion(view_key, indexes=raw_indexes):
                camera_name = data_dict["sample_camera_names"][view_key]
                frames = self._read_cache_npy_frames(
                    episode_dir / f"{camera_name}_motion.npy", indexes
                )
                return self._normalize_rgb_like_frames(frames)

            visual = torch.cat(
                [get_rgb(view) for view in ("high", "left_wrist", "right_wrist")],
                dim=-1,
            )
            motion_for_state = torch.cat(
                [get_motion(view) for view in ("high", "left_wrist", "right_wrist")],
                dim=-1,
            )
            replay_visual = motion_for_state
            episode_rgb = torch.cat(
                [
                    get_rgb(view, episode_raw_indexes)
                    for view in ("high", "left_wrist", "right_wrist")
                ],
                dim=-1,
            )
            data_dict["input_fps"] = float(
                data_dict.get("sample_target_hz", self.fps)
            )
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(visual.shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            visual_ref = copy.deepcopy(visual) * ref_masks

        if int(raw_indexes.max()) >= len(all_labels):
            raise IndexError(
                f"object-state label length={len(all_labels)} cannot cover raw index={int(raw_indexes.max())} "
                f"for {data_dict.get('episode_id')}"
            )
        labels = all_labels[raw_indexes]
        probabilities = all_probabilities[raw_indexes]
        canvas_labels, canvas_label_map = self._canvas_labels(labels, data_dict)
        task = normalize_task_name(data_dict.get("object_state_task", data_dict.get("sample_cache_root", "")))
        spec = CanvasSpec(
            height=self.object_state_canvas_height,
            width=visual.shape[-1],
            header_height=self.object_state_canvas_header_height,
            encoding=self.object_state_canvas_encoding,
        )
        state_canvas = render_state_sequence(canvas_labels, task, spec)

        front_state_ref_images = copy.deepcopy(state_canvas) * ref_masks

        # Keep the legacy joint tensors for checkpoint/evaluation compatibility.
        # The separate-VAE trainer consumes the modality-specific tensors below.
        front_images = torch.cat([visual, state_canvas], dim=-2)
        front_ref_images = torch.cat([visual_ref, front_state_ref_images], dim=-2)
        motion_state_strip = F.interpolate(
            motion_for_state,
            size=(self.object_state_canvas_height, motion_for_state.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        replay_condition = torch.cat([replay_visual, motion_state_strip], dim=-2)
        episode_ref_enabled = self.dual_reference_mode != "clip_only"

        # Only GT current state is visible through the first-frame mask. Future
        # labels live exclusively in front_images, the noised prediction target.
        if torch.count_nonzero(front_ref_images[1:]).item() != 0:
            raise AssertionError("future RGB/state leaked through front_ref_images")
        if not self.is_train:
            raise AssertionError("WMRNDMixObjectStateCanvasTransforms is train-only")

        out = {} if parent_out is None else parent_out
        out.update({
            "fps": data_dict["input_fps"],
            "front_images": front_images,
            "front_ref_images": front_ref_images,
            # These legacy names are consumed by the separate-VAE trainer. In
            # full mode they carry RGB+depth+tactile as one visual canvas.
            "front_rgb_images": visual,
            "front_visual_images": visual,
            "front_state_images": state_canvas,
            "front_rgb_ref_images": visual_ref,
            "front_visual_ref_images": visual_ref,
            "front_state_ref_images": front_state_ref_images,
            "front_ref_masks": ref_latent_masks,
            "clip_ref_images": front_ref_images,
            "clip_ref_masks": ref_latent_masks,
            "replay_condition": replay_condition,
            "replay_rgb_images": replay_visual,
            "replay_visual_images": replay_visual,
            "replay_state_images": motion_state_strip,
            # Match the proven dual-reference path: episode frame 0 contributes
            # scene RGB only. The local clip reference carries RGB + state.
            "episode_ref_images": episode_rgb,
            "episode_ref_mask": torch.tensor(
                [float(episode_ref_enabled)], dtype=torch.float32
            ),
            "episode_ref_available_mask": torch.ones(1, dtype=torch.float32),
            "episode_ref_role_id": torch.tensor(0, dtype=torch.long),
            "clip_ref_role_id": torch.tensor(1, dtype=torch.long),
            "episode_ref_frame_index": torch.tensor(
                reference_indices["episode_index"], dtype=torch.long
            ),
            "episode_ref_raw_index": torch.tensor(
                reference_indices["episode_raw_index"], dtype=torch.long
            ),
            "clip_ref_frame_index": torch.tensor(
                reference_indices["clip_index"], dtype=torch.long
            ),
            "clip_ref_raw_index": torch.tensor(
                reference_indices["clip_raw_index"], dtype=torch.long
            ),
            "episode_ref_time_offset_frames": torch.tensor(
                reference_indices["clip_index"] - reference_indices["episode_index"],
                dtype=torch.float32,
            ),
            "clip_ref_stride_frames": torch.tensor(
                reference_indices["clip_stride"], dtype=torch.float32
            ),
            "episode_ref_is_duplicate": torch.tensor(
                reference_indices["is_duplicate"], dtype=torch.bool
            ),
            "episode_ref_modality_mask": torch.tensor([1.0], dtype=torch.float32),
            "episode_ref_modality_scores": torch.tensor([1.0], dtype=torch.float32),
            "prompt_embeds": self.default_prompt_embeds,
            "object_state_labels": torch.from_numpy(labels.copy()).long(),
            "object_state_canvas_labels": torch.from_numpy(canvas_labels.copy()).long(),
            "object_state_probabilities": torch.from_numpy(probabilities.copy()).float(),
            "object_state_task_id": torch.tensor(TASK_TO_ID[task], dtype=torch.long),
            "object_state_rgb_height": torch.tensor(visual.shape[-2], dtype=torch.long),
            "object_state_visual_height": torch.tensor(visual.shape[-2], dtype=torch.long),
            "object_state_visual_canvas_mode_id": torch.tensor(
                int(self.object_state_visual_canvas_mode == "full"), dtype=torch.long
            ),
            "object_state_canvas_height": torch.tensor(self.object_state_canvas_height, dtype=torch.long),
            "object_state_current_is_gt": torch.tensor(True),
            "object_state_future_is_condition": torch.tensor(False),
            "object_state_sampling_category_id": torch.tensor(
                sampling_category_id, dtype=torch.long
            ),
            "object_state_clip_state1_fraction": torch.tensor(
                float(np.mean(labels == 1)), dtype=torch.float32
            ),
            "object_state_clip_has_state1": torch.tensor(
                bool(np.any(labels == 1)), dtype=torch.bool
            ),
            "object_state_clip_all_state1": torch.tensor(
                bool(np.all(labels == 1)), dtype=torch.bool
            ),
            "object_state_clip_padded_frames": torch.tensor(
                padded_frames, dtype=torch.long
            ),
            "object_state_canvas_label_remapped_fraction": torch.tensor(
                float(np.mean(canvas_labels != labels)), dtype=torch.float32
            ),
            "object_state_canvas_label_map_enabled": torch.tensor(
                bool(canvas_label_map), dtype=torch.bool
            ),
            "sample_frame_indices": torch.from_numpy(raw_indexes.copy()).long(),
        })
        qpos_path = episode_dir / "qpos.npy"
        if parent_out is None and qpos_path.exists():
            out["qpos"] = torch.from_numpy(
                self._read_cache_npy_frames(qpos_path, raw_indexes).astype(np.float32)
            )
        if os.environ.get("PHYSCANVAS_PERF_COMPACT_SEPARATE_VAE_BATCH", "0") == "1":
            # These are compatibility duplicates, not inputs to the separate-VAE
            # training or its periodic visual evaluation. Drop before collation,
            # pinning, prefetch and transfer replicate their large video buffers.
            out = compact_separate_vae_batch(out)
        return out
