#!/usr/bin/env python3
"""Full-episode visual-canvas/object-state evaluation with persistent loading."""

from __future__ import annotations

import argparse
import copy
import errno
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

PROJECT_DIR = Path(os.environ['PHYSCANVAS_EVAL_RUNTIME_SOURCE'])
sys.path.insert(0, str(PROJECT_DIR / "third_party" / "giga-datasets"))
sys.path.insert(0, str(PROJECT_DIR / "third_party" / "giga-train"))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from giga_datasets import DefaultCollator, load_dataset
from giga_train import build_transform
from PIL import Image, ImageDraw

import PhysCanvas.datasets  # noqa: F401,E402  Register UnifiedCacheDataset for standalone evaluation.
from PhysCanvas.transforms import object_state_canvas_transforms  # Explicit registry side effect.
from PhysCanvas.models import WanConditionRNDMixStageOneModel  # noqa: E402
from PhysCanvas.object_state_canvas import CanvasSpec, render_state_sequence  # noqa: E402
from .image_metrics import resize_tactile_diff_panel
from .metrics import EpisodeMetrics


def local_staging_dir():
    return os.environ.get("PHYS_CANVAS_TMPDIR") or (
        "/dev/shm" if Path("/dev/shm").is_dir() else tempfile.gettempdir()
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("bottle", "shiguan", "jimu", "mofang"), default="bottle")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--threshold-json")
    parser.add_argument("--contact-threshold-json")
    parser.add_argument("--smoke-frames", type=int, default=0)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("fit", "generalization", "evaluation"), required=True)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--metric-profile", choices=("reconstruction", "visual", "all"), default="all")
    parser.add_argument("--export-clean-rgb", action="store_true")
    parser.add_argument("--i3d-weights")
    parser.add_argument(
        "--visual-canvas-mode", choices=("rgb_only", "full"), default="full"
    )
    parser.add_argument("--flow-encoding", default="xyz_linear")
    parser.add_argument("--flow-max-disp", type=float, default=0.06)
    parser.add_argument("--flow-vis-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reward-checkpoint")
    parser.add_argument("--reward-model-source")
    parser.add_argument("--reward-device", default="cuda:1")
    parser.add_argument("--reward-model-name", default="base_plus_reward_state")
    parser.add_argument("--reward-tactile-max-disp", type=float, default=0.08)
    parser.add_argument("--reward-only", action="store_true")
    return parser.parse_args()


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=local_staging_dir())
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
        try:
            shutil.copyfile(tmp_name, path)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise
            with path.open("w") as handle:
                json.dump(payload, handle, indent=2)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def checkpoint_transformer_dir(path):
    path = Path(path)
    if (path / "transformer" / "config.json").is_file():
        return path / "transformer"
    if (path / "config.json").is_file():
        return path
    raise FileNotFoundError(f"invalid checkpoint: {path}")


def load_transformer(path, device, dtype):
    transformer_dir = checkpoint_transformer_dir(path)
    transformer = WanConditionRNDMixStageOneModel.from_config(
        str(transformer_dir), torch_dtype=dtype
    ).to(device)
    weight = transformer_dir / "diffusion_pytorch_model.bin"
    if not weight.is_file():
        weight = transformer_dir / "diffusion_pytorch_model.safetensors"
    if weight.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weight), device=str(device))
    else:
        state = torch.load(weight, map_location=device, weights_only=True)
    transformer.load_state_dict(state, strict=True)
    del state
    print('[checkpoint] strict=True missing=0 unexpected=0', flush=True)
    transformer.to(device=device, dtype=dtype).eval()
    return transformer


def forward_vae(vae, images, mean, scale):
    images = images.to(vae.dtype)
    images = rearrange(images, "b t c h w -> b c t h w")
    return (vae.encode(images).latent_dist.mode() - mean) * scale


def decode_vae(vae, latents, mean, scale):
    latents = latents.to(vae.dtype) / scale + mean
    return vae.decode(latents, return_dict=False)[0].detach().transpose(1, 2)


def num_ref_images_from_masks(front_ref_masks):
    per_sample = front_ref_masks[0]
    return (torch.sum(per_sample).int() - 1).item() * 4 + 1


def pipeline_timestep(transformer, first_frame_mask, timestep, batch_size):
    patch_size = tuple(transformer.patch_embedding.kernel_size)
    if len(patch_size) == 3 and patch_size[0] == 1:
        patch_mask = first_frame_mask[0][0][:, :: patch_size[1], :: patch_size[2]]
    else:
        patch_mask = F.max_pool3d(
            first_frame_mask.float(), kernel_size=patch_size, stride=patch_size
        )[0][0]
    return (patch_mask * timestep).flatten().unsqueeze(0).expand(batch_size, -1)


def dual_reference_kwargs(transformer, condition, first_frame_mask):
    if condition.get("reference_fusion_mode") != "dual_memory":
        return {"reference_fusion_mode": "clip_only"}
    patch_size = tuple(transformer.patch_embedding.kernel_size)
    patch_mask = F.max_pool3d(
        first_frame_mask.float(), kernel_size=patch_size, stride=patch_size
    )
    return {
        "reference_fusion_mode": "dual_memory",
        "global_reference_latents": condition["global_reference_latents"],
        "global_reference_mask": condition["global_reference_mask"],
        "global_reference_role_ids": condition["global_reference_role_ids"],
        "global_reference_time_offsets": condition["global_reference_time_offsets"],
        "clip_reference_patch_mask": 1.0 - patch_mask,
        "clip_reference_role_id": condition["clip_reference_role_id"],
    }


def denoise_pipeline(transformer, shape_like, condition, scheduler, steps, generator):
    sample_dtype = torch.float32
    ref_latents = condition["ref_latents"].to(sample_dtype)
    first_frame_mask = condition["first_frame_mask"].to(sample_dtype)
    latents = torch.randn(
        shape_like.shape,
        device=shape_like.device,
        dtype=sample_dtype,
        generator=generator,
    )
    scheduler.set_timesteps(steps, device=shape_like.device)
    params = inspect.signature(transformer.forward).parameters
    for timestep in scheduler.timesteps:
        model_input = (1.0 - first_frame_mask) * ref_latents + first_frame_mask * latents
        kwargs = {
            "hidden_states": model_input.to(transformer.dtype),
            "timestep": pipeline_timestep(
                transformer, first_frame_mask, timestep, latents.shape[0]
            ),
            "encoder_hidden_states": condition["prompt_embeds"],
            "return_dict": False,
        }
        if "replay_latents" in params:
            kwargs["replay_latents"] = condition["replay_latents"]
        else:
            kwargs["replay_condition_latents"] = condition["replay_latents"]
        kwargs.update(dual_reference_kwargs(transformer, condition, first_frame_mask))
        kwargs['mask_state_replay'] = True
        kwargs['state_latent_height'] = condition['state_latent_height']
        context = transformer.cache_context("cond") if hasattr(transformer, "cache_context") else nullcontext()
        with context:
            prediction = transformer(**kwargs)[0]
        latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
    return ((1.0 - first_frame_mask) * ref_latents + first_frame_mask * latents).detach()


def tensor_video_to_uint8(video):
    video = video.detach().float().cpu().permute(0, 2, 3, 1).contiguous()
    if video.shape[-1] == 1:
        video = video.repeat(1, 1, 1, 3)
    return ((video + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()


def resize_panel(panel, width, height, nearest=False):
    image = Image.fromarray(panel).convert("RGB")
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = image.resize(size, resample)
    canvas = Image.new("RGB", (width, height), (238, 238, 235))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return np.asarray(canvas)


def title_panel(panel, title):
    image = Image.fromarray(panel).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(255, 255, 255))
    draw.text((6, 6), title, fill=(20, 20, 20))
    return np.asarray(image)


def timeline_panel(gt_labels, pred_labels, frame_index, width, height, colors):
    image = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    count = max(1, len(gt_labels))
    left, right = 70, width - 8
    tracks = [("GT", gt_labels, 31, 55), ("Pred", pred_labels, 64, 88)]
    for name, labels, y0, y1 in tracks:
        draw.text((8, y0 + 4), name, fill=(20, 20, 20))
        for index, label in enumerate(labels):
            x0 = left + round(index * (right - left) / count)
            x1 = left + round((index + 1) * (right - left) / count)
            draw.rectangle((x0, y0, max(x0, x1), y1), fill=colors[int(label)])
    cursor = left + round(frame_index * (right - left) / max(count - 1, 1))
    draw.line((cursor, 25, cursor, height - 2), fill=(220, 45, 45), width=2)
    draw.rectangle((0, 0, width, 23), fill=(255, 255, 255))
    draw.text((6, 6), f"State timeline  frame={frame_index:04d}/{count - 1:04d}", fill=(20, 20, 20))
    return np.asarray(image)


def write_mp4(path, frames, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="object_state_eval_", dir=local_staging_dir()))
    try:
        for index, frame in enumerate(frames):
            h, w = frame.shape[:2]
            padded = np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
            Image.fromarray(padded).save(tmp_dir / f"frame_{index:05d}.png")
        tmp_video = tmp_dir / "video.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(fps), "-i", str(tmp_dir / "frame_%05d.png"),
                "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp_video),
            ],
            check=True,
        )
        shutil.copyfile(tmp_video, path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_episode_segment(
    record,
    selection_manifest,
    output_dir,
    config,
    start,
    num_frames,
    tag,
    visual_canvas_mode,
    flow_encoding,
    flow_max_disp,
):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["episode_id"]))
    manifest_dir = Path(output_dir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    episode_manifest = manifest_dir / f"{safe}.{tag}.episodes.json"
    fixed_manifest = manifest_dir / f"{safe}.{tag}.fixed.json"
    true_frames = int(record["sample_num_frames"])
    padded_record = copy.deepcopy(record)
    padded_record["data_index"] = 0
    write_json(
        episode_manifest,
        {"manifest_type": "unified_cache_episodes_v1", "records": [padded_record]},
    )
    write_json(
        fixed_manifest,
        {
            "manifest_type": "unified_cache_fixed_clips_v1",
            "source": "unified_cache",
            "fixed_clip_start_mode": "episode_start",
            "clips": [{
                "episode_id": record["episode_id"],
                "sample_cache_root": record["sample_cache_root"],
                "start": int(start),
                "stride": 1,
                "num_frames": int(num_frames),
                "true_num_frames": true_frames,
                "mode": f"full_episode_{tag}",
            }],
        },
    )
    data_cfg = copy.deepcopy(dict(config["dataloaders"]["test"]))
    dataset_cfg = data_cfg["data_or_config"]
    full_visual = visual_canvas_mode == "full"
    dataset_cfg.update(
        cache_roots=[record["sample_cache_root"]],
        cameras=record["sample_camera_names"],
        max_episodes=1,
        clip_num_frames=int(num_frames),
        clip_max_stride=1,
        fixed_clip_count=1,
        fixed_clip_manifest_path=str(fixed_manifest),
        fixed_clip_override_num_frames=True,
        fixed_clip_start_mode="episode_start",
        cache_manifest_path=str(episode_manifest),
        use_cache_manifest=True,
        write_cache_manifest=False,
        object_state_manifest_path=str(selection_manifest),
        require_object_state_labels=True,
        require_tactile=full_visual,
        require_depth=False,
        require_motion=True,
        object_state_online_rewards=False,
    )
    data_cfg["filter"] = {"mode": "min_frames", "min_num_frames": int(num_frames)}
    data_cfg["transform"].update(
        num_frames=int(num_frames),
        max_stride=1,
        dual_reference_mode="dual_memory",
        object_state_canvas_encoding="lowfreq_bands_v1",
        object_state_canvas_height=32,
        object_state_canvas_header_height=8,
        object_state_visual_canvas_mode=visual_canvas_mode,
        unified_wan_append_tactile_diff=full_visual,
        unified_wan_append_tactile_flow=full_visual,
        unified_wan_tactile_flow_encoding=flow_encoding,
        unified_wan_tactile_flow_linear_max_disp=flow_max_disp,
        unified_wan_include_depth=False,
    )
    dataset = load_dataset(dataset_cfg)
    dataset.filter(**data_cfg["filter"])
    if len(dataset) != 1:
        raise RuntimeError(f"expected one dataset item for {record['episode_id']}, got {len(dataset)}")
    dataset.set_transform(build_transform(data_cfg["transform"]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=DefaultCollator(**data_cfg.get("collator", {})),
        num_workers=0,
    )
    return next(iter(loader))


def make_episode_dataset(record, selection_manifest, output_dir, config, args):
    true_frames = int(record["sample_num_frames"])
    if args.max_frames:
        true_frames = min(true_frames, args.max_frames)
    if true_frames < 9:
        raise RuntimeError(f"episode {record['episode_id']} is too short: {true_frames} frames")
    prefix_frames = ((true_frames - 1) // 8) * 8 + 1
    batch = load_episode_segment(
        record, selection_manifest, output_dir, config,
        start=0,
        num_frames=prefix_frames,
        tag=f"prefix_{prefix_frames}",
        visual_canvas_mode=args.visual_canvas_mode,
        flow_encoding=args.flow_encoding,
        flow_max_disp=args.flow_max_disp,
    )
    if prefix_frames == true_frames:
        return batch

    tail_start = true_frames - 9
    tail = load_episode_segment(
        record, selection_manifest, output_dir, config,
        start=tail_start,
        num_frames=9,
        tag=f"tail_{tail_start}_9",
        visual_canvas_mode=args.visual_canvas_mode,
        flow_encoding=args.flow_encoding,
        flow_max_disp=args.flow_max_disp,
    )
    tail_offset = (prefix_frames - 1) - tail_start
    temporal_keys = (
        "front_rgb_images",
        "front_state_images",
        "replay_rgb_images",
        "replay_state_images",
        "object_state_labels",
        "tactile_flow_xyz_grids",
    )
    for key in temporal_keys:
        if key not in batch or key not in tail:
            continue
        batch[key] = torch.cat([batch[key], tail[key][:, tail_offset + 1 :]], dim=1)
        if batch[key].shape[1] != true_frames:
            raise RuntimeError(
                f"failed to stitch {key} for {record['episode_id']}: "
                f"expected {true_frames}, got {batch[key].shape[1]}"
            )
    return batch


def pad_time_by_repeating_last(tensor, target_frames):
    if tensor.shape[1] >= target_frames:
        return tensor[:, :target_frames]
    repeat_shape = [1] * tensor.ndim
    repeat_shape[1] = target_frames - tensor.shape[1]
    return torch.cat([tensor, tensor[:, -1:].repeat(*repeat_shape)], dim=1)


def first_transition(labels):
    indices = np.flatnonzero(np.asarray(labels, dtype=np.int64) == 1)
    return int(indices[0]) if len(indices) else None


def split_visual_canvas(video, mode):
    names = ("RGB",) if mode == "rgb_only" else (
        "RGB",
        "Tactile diff",
        "Tactile flow (xyz_linear encoded)",
    )
    if len(names) == 1:
        return [(names[0], video)]
    if video.shape[1] % len(names) != 0:
        raise ValueError(
            f"full visual canvas height {video.shape[1]} is not divisible by {len(names)}"
        )
    return list(zip(names, np.split(video, len(names), axis=1)))


def blank_panel(width, height, title):
    panel = np.full((height, width, 3), 238, dtype=np.uint8)
    return title_panel(panel, title)


def _flow_grid_shape(num_markers):
    if num_markers == 180:
        return 18, 10
    if num_markers == 700:
        return 35, 20
    height = int(np.sqrt(max(num_markers, 1)))
    while height > 1 and num_markers % height != 0:
        height -= 1
    return height, max(1, num_markers // max(height, 1))


def _downsample_flow_grid(values):
    values = np.asarray(values)
    if values.shape[0] == 700:
        return values.reshape(35, 20, *values.shape[1:])[::2, ::2].reshape(
            180, *values.shape[1:]
        )
    return values


def _sparse_flow_indices(num_markers):
    stride_text = os.environ.get("PHYSCANVAS_TACTILE_FLOW_MARKER_STRIDE", "1,1").replace(":", ",")
    parts = [int(value.strip()) for value in stride_text.split(",") if value.strip()]
    stride_y, stride_x = (parts + [3, 3])[:2]
    if num_markers in (180, 700):
        height, width = _flow_grid_shape(num_markers)
        mask = np.zeros((height, width), dtype=bool)
        mask[::max(1, stride_y), ::max(1, stride_x)] = True
        return np.flatnonzero(mask.reshape(-1))
    step = max(1, int(round(num_markers / 60.0)))
    return np.arange(0, num_markers, step, dtype=np.int64)


def _draw_sparse_arrow(draw, start, end, color, width=1, head_len=4.0, head_w=2.4):
    start_x, start_y = start
    end_x, end_y = end
    draw.line((start_x, start_y, end_x, end_y), fill=color, width=width)
    delta_x, delta_y = end_x - start_x, end_y - start_y
    length = max(float((delta_x * delta_x + delta_y * delta_y) ** 0.5), 1e-6)
    unit_x, unit_y = delta_x / length, delta_y / length
    perp_x, perp_y = -unit_y, unit_x
    draw.polygon(
        [
            (end_x, end_y),
            (end_x - unit_x * head_len + perp_x * head_w, end_y - unit_y * head_len + perp_y * head_w),
            (end_x - unit_x * head_len - perp_x * head_w, end_y - unit_y * head_len - perp_y * head_w),
        ],
        fill=color,
    )


def _flow_panel_for_side(points, flow, width, height, arrow_color, flow_vis_scale):
    points = points.detach().float().cpu().numpy() if torch.is_tensor(points) else np.asarray(points, dtype=np.float32)
    flow = flow.detach().float().cpu().numpy() if torch.is_tensor(flow) else np.asarray(flow, dtype=np.float32)
    points = _downsample_flow_grid(points.reshape(-1, 2))
    flow = _downsample_flow_grid(flow.reshape(-1, 2))
    count = min(len(points), len(flow))
    indices = _sparse_flow_indices(count)
    points = points[:count][indices]
    flow = flow[:count][indices]

    pad = int(os.environ.get("PHYSCANVAS_TACTILE_FLOW_PANEL_PAD", "4"))
    plot_width = max(1.0, float(width - 2 * pad))
    plot_height = max(1.0, float(height - 2 * pad))
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span_x = max(float(maximum[0] - minimum[0]), 1e-6)
    span_y = max(float(maximum[1] - minimum[1]), 1e-6)
    origins = np.empty_like(points, dtype=np.float32)
    origins[:, 0] = pad + (points[:, 0] - minimum[0]) / span_x * plot_width
    origins[:, 1] = pad + (1.0 - (points[:, 1] - minimum[1]) / span_y) * plot_height
    delta = np.empty_like(flow, dtype=np.float32)
    delta[:, 0] = flow[:, 0] / span_x * plot_width * float(flow_vis_scale)
    delta[:, 1] = -flow[:, 1] / span_y * plot_height * float(flow_vis_scale)
    if int(os.environ.get("PHYSCANVAS_TACTILE_FLOW_FLIP_X", "0")):
        delta[:, 0] *= -1.0
    if int(os.environ.get("PHYSCANVAS_TACTILE_FLOW_FLIP_Y", "0")):
        delta[:, 1] *= -1.0
    endpoints = origins + delta
    endpoints[:, 0] = np.clip(endpoints[:, 0], 0, width - 1)
    endpoints[:, 1] = np.clip(endpoints[:, 1], 0, height - 1)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(214, 214, 208))
    arrow_width = int(os.environ.get("PHYSCANVAS_TACTILE_FLOW_ARROW_WIDTH", "1"))
    for start, end in zip(origins, endpoints):
        if float(np.linalg.norm(end - start)) >= 0.5:
            _draw_sparse_arrow(draw, tuple(start), tuple(end), arrow_color, width=arrow_width)
    radius = int(os.environ.get("PHYSCANVAS_TACTILE_FLOW_MARKER_RADIUS", "1"))
    for point_x, point_y in origins:
        draw.ellipse(
            (point_x - radius - 1, point_y - radius - 1, point_x + radius + 1, point_y + radius + 1),
            fill=(150, 150, 150),
        )
        draw.ellipse(
            (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
            fill=(70, 70, 70),
        )
    return np.asarray(image)


def flow_panel_for_sides(points, flows, width, height, arrow_color, flow_vis_scale, stat_label=""):
    del stat_label
    side_count = int(min(points.shape[0], flows.shape[0]))
    side_width = max(1, width // side_count)
    panels = [
        _flow_panel_for_side(points[index], flows[index], side_width, height, arrow_color, flow_vis_scale)
        for index in range(side_count)
    ]
    panel = np.concatenate(panels, axis=1)
    if panel.shape[1] != width:
        panel = np.asarray(Image.fromarray(panel).resize((width, height), Image.Resampling.BILINEAR))
    return panel


def _unpack_wan_side_panel(panel, side_count, side_hw):
    side_height, side_width = int(side_hw[0]), int(side_hw[1])
    packed = F.interpolate(
        panel.detach().float(), size=(side_height, side_count * side_width),
        mode="bilinear", align_corners=False,
    )
    frames, channels, _, _ = packed.shape
    return packed.reshape(frames, channels, side_height, side_count, side_width).permute(
        0, 3, 1, 2, 4
    ).contiguous()


def decode_xyz_flow_panel(panel, side_count, grid_hw, max_disp):
    """Invert the exact four-side xyz_linear packing used by the training transform."""
    sides = _unpack_wan_side_panel(panel, side_count, grid_hw)
    xyz = sides[:, :, :3] * max(float(max_disp), 1e-6)
    return xyz.permute(0, 1, 3, 4, 2).reshape(panel.shape[0], side_count, -1, 3)


@torch.inference_mode()
def evaluate_episode(record, args, config, transformer, vae, scheduler, mean, scale, device, dtype, reward_scorer=None):
    source_frames = int(record["sample_num_frames"])
    true_frames = min(source_frames, args.max_frames) if args.max_frames else source_frames
    if args.smoke_frames:
        true_frames = min(true_frames, args.smoke_frames)
    sub_frames = 8
    rollout_steps = int(math.ceil(max(true_frames - 1, 1) / sub_frames))
    clip_frames = rollout_steps * sub_frames + 1
    batch = make_episode_dataset(
        record, args.selection_manifest, args.output_dir, config, args
    )
    # Keep full episodes on CPU; only the current rollout belongs on the GPU.
    for key in (
        "front_rgb_images",
        "front_state_images",
        "replay_rgb_images",
        "replay_state_images",
        "object_state_labels",
        "tactile_flow_xyz_grids",
    ):
        if key not in batch:
            continue
        batch[key] = pad_time_by_repeating_last(batch[key], clip_frames)
    rgb = batch["front_rgb_images"]
    state = batch["front_state_images"]
    replay_rgb = batch["replay_rgb_images"]
    replay_state = batch["replay_state_images"]
    if rgb.shape[-2:] != (672, 672) or state.shape[-2:] != (32, 672):
        raise RuntimeError(f'Unexpected no-depth shapes: {rgb.shape}, {state.shape}')
    rgb_refs = batch["front_rgb_ref_images"][:, : sub_frames + 1].to(device).clone()
    state_refs = batch["front_state_ref_images"][:, : sub_frames + 1].to(device).clone()
    num_ref_images = num_ref_images_from_masks(batch["front_ref_masks"])
    sub_latents = sub_frames // int(vae.config.scale_factor_temporal) + 1
    front_ref_masks = batch["front_ref_masks"][:, :, :sub_latents].to(device)
    generator = torch.Generator(device=device).manual_seed(
        args.seed + int(record.get("data_index", 0))
    )

    gt_rgb_parts, pred_rgb_parts, motion_parts = [], [], []
    gt_state_parts, pred_state_parts = [], []
    gt_label_parts = []
    anchor_latents = forward_vae(vae, batch["episode_ref_images"].to(device), mean, scale)
    for rollout_idx in range(rollout_steps):
        start = rollout_idx * sub_frames
        end = (rollout_idx + 1) * sub_frames + 1
        roll_rgb = rgb[:, start:end].to(device)
        roll_state = state[:, start:end].to(device)
        roll_replay_rgb = replay_rgb[:, start:end].to(device)
        roll_replay_state = replay_state[:, start:end].to(device)

        rgb_latents = forward_vae(vae, roll_rgb, mean, scale)
        state_latents = forward_vae(vae, roll_state, mean, scale)
        joint_shape = torch.cat([rgb_latents, state_latents], dim=-2)
        rgb_height = rgb_latents.shape[-2]
        replay_latents = torch.cat(
            [
                forward_vae(vae, roll_replay_rgb, mean, scale),
                forward_vae(vae, roll_replay_state, mean, scale),
            ],
            dim=-2,
        )
        ref_latents = torch.cat(
            [
                forward_vae(vae, rgb_refs[:, :num_ref_images], mean, scale),
                forward_vae(vae, state_refs[:, :num_ref_images], mean, scale),
            ],
            dim=-2,
        )
        num_ref_latents = (num_ref_images - 1) // int(vae.config.scale_factor_temporal) + 1
        first_frame_mask = torch.ones(
            1, 1, joint_shape.shape[2], joint_shape.shape[-2], joint_shape.shape[-1],
            device=device, dtype=joint_shape.dtype,
        )
        first_frame_mask[:, :, :num_ref_latents] = 0
        episode_ref_latents = anchor_latents
        episode_mask = batch["episode_ref_mask"].to(joint_shape).reshape(1, -1)
        duplicate = batch["episode_ref_is_duplicate"].to(device).reshape(1, -1).bool() & (rollout_idx == 0)
        episode_mask = episode_mask * (~duplicate).to(episode_mask.dtype)
        stride = batch.get("clip_ref_stride_frames", torch.ones_like(batch["episode_ref_time_offset_frames"]))
        time_offset = batch["episode_ref_time_offset_frames"] + rollout_idx * sub_frames * stride
        condition = {
            "ref_latents": ref_latents,
            "replay_latents": replay_latents,
            "first_frame_mask": first_frame_mask,
            "prompt_embeds": batch["prompt_embeds"].to(device=device, dtype=joint_shape.dtype),
            "reference_fusion_mode": "dual_memory",
            "state_latent_height": state_latents.shape[-2],
            "global_reference_latents": episode_ref_latents,
            "global_reference_mask": episode_mask,
            "global_reference_role_ids": batch["episode_ref_role_id"].to(device=device, dtype=torch.long),
            "global_reference_time_offsets": time_offset.to(device=device, dtype=joint_shape.dtype).reshape(1, 1),
            "clip_reference_role_id": batch["clip_ref_role_id"].to(device=device, dtype=torch.long),
        }
        prediction = denoise_pipeline(
            transformer, joint_shape, condition, scheduler,
            args.num_inference_steps, generator,
        )
        pred_rgb = decode_vae(vae, prediction[..., :rgb_height, :], mean, scale)
        pred_state = decode_vae(vae, prediction[..., rgb_height:, :], mean, scale)
        rgb_refs[:, :num_ref_images] = pred_rgb[:, -num_ref_images:]
        state_refs[:, :num_ref_images] = pred_state[:, -num_ref_images:]

        offset = 0 if rollout_idx == 0 else 1
        gt_rgb_parts.append(roll_rgb[0, offset:].detach().cpu())
        pred_rgb_part = pred_rgb[0, offset:].detach().cpu()
        gt_state_part = roll_state[0, offset:].detach().cpu()
        pred_state_part = pred_state[0, offset:].detach().cpu()
        if rollout_idx == 0:
            pred_rgb_part[0] = gt_rgb_parts[-1][0]
            pred_state_part[0] = gt_state_part[0]
        pred_rgb_parts.append(pred_rgb_part)
        motion_parts.append(roll_replay_rgb[0, offset:].detach().cpu())
        gt_state_parts.append(gt_state_part)
        pred_state_parts.append(pred_state_part)
        gt_label_parts.append(batch["object_state_labels"][0, start + offset:end].detach().cpu())
        print(f'[rollout] {record["episode_id"]} {rollout_idx+1}/{rollout_steps}', flush=True)

    gt_rgb = torch.cat(gt_rgb_parts)[:true_frames]
    pred_rgb = torch.cat(pred_rgb_parts)[:true_frames]
    motion = torch.cat(motion_parts)[:true_frames]
    gt_state = torch.cat(gt_state_parts)[:true_frames]
    pred_state = torch.cat(pred_state_parts)[:true_frames]
    gt_labels = torch.cat(gt_label_parts)[:true_frames].numpy().astype(np.int64)

    spec = CanvasSpec(height=32, width=gt_state.shape[-1], header_height=8, encoding="lowfreq_bands_v1")
    record_task = str(record.get("object_state_task", args.task)).lower()
    if record_task != args.task:
        raise ValueError(
            f"selection record task mismatch: expected {args.task!r}, got {record_task!r}"
        )
    prototypes = render_state_sequence(np.asarray([0, 1]), args.task, spec).float()
    distances = torch.stack(
        [((pred_state.float() - prototype.unsqueeze(0)) ** 2).mean((1, 2, 3)) for prototype in prototypes]
    )
    pred_labels = torch.argmin(distances, dim=0).numpy().astype(np.int64)
    pred_labels[0] = gt_labels[0]

    gt_rgb_u8 = tensor_video_to_uint8(gt_rgb)
    pred_rgb_u8 = tensor_video_to_uint8(pred_rgb)
    motion_u8 = tensor_video_to_uint8(motion)
    gt_state_u8 = tensor_video_to_uint8(gt_state)
    pred_state_u8 = tensor_video_to_uint8(pred_state)
    prototype_u8 = tensor_video_to_uint8(prototypes)
    colors = {
        index: tuple(int(value) for value in prototype_u8[index].mean((0, 1)).tolist())
        for index in (0, 1)
    }

    gt_visual_parts = split_visual_canvas(gt_rgb_u8, args.visual_canvas_mode)
    pred_visual_parts = dict(split_visual_canvas(pred_rgb_u8, args.visual_canvas_mode))
    motion_parts_u8 = split_visual_canvas(motion_u8, args.visual_canvas_mode)
    motion_display = motion_parts_u8[0][1]

    flow_points = gt_flow_xyz = pred_flow_xyz = None
    if args.visual_canvas_mode == "full":
        if args.flow_encoding not in ("xyz", "xyz_linear", "linear_xyz", "physical_xyz"):
            raise ValueError(
                "Marker+arrow visualization currently requires an XYZ linear flow encoding, "
                f"got {args.flow_encoding!r}"
            )
        if "tactile_episode0_points" not in batch or "tactile_flow_xyz_grids" not in batch:
            raise KeyError(
                "Full tactile-flow visualization requires tactile_episode0_points and "
                "tactile_flow_xyz_grids from the unified cache"
            )
        flow_points = batch["tactile_episode0_points"][0].detach().float().cpu()
        gt_xyz_grids = batch["tactile_flow_xyz_grids"][0, :true_frames]
        side_count = int(gt_xyz_grids.shape[1])
        grid_hw = (int(gt_xyz_grids.shape[2]), int(gt_xyz_grids.shape[3]))
        gt_flow_xyz = gt_xyz_grids.detach().float().cpu().reshape(true_frames, side_count, -1, 3)
        layer_h = int(pred_rgb.shape[-2] // 3)
        pred_flow_layer = pred_rgb[:, :, 2 * layer_h : 3 * layer_h, :]
        pred_flow_xyz = decode_xyz_flow_panel(
            pred_flow_layer, side_count, grid_hw, args.flow_max_disp
        ).detach().float().cpu()

    metric_result = None
    if args.metrics:
        args.metric_accumulator.root = Path(args.output_dir)
        decoded_gt_flow = decode_xyz_flow_panel(
            gt_rgb[:, :, 448:672, :], side_count, grid_hw, args.flow_max_disp
        ).cpu()
        metric_result = args.metric_accumulator.evaluate(
            record, args, gt_rgb, pred_rgb, gt_labels, pred_labels,
            distances, gt_flow_xyz, pred_flow_xyz, flow_points, decoded_gt_flow,
        )
        if true_frames < source_frames and "task_success" in metric_result:
            metric_result["task_success"]["source_correct"] = None
            metric_result["task_success"]["source_success"] = None
            metric_result["task_success"]["prediction_scope"] = f"prefix {true_frames} frames"

    reward_path = None
    if reward_scorer is not None:
        if gt_flow_xyz is None or pred_flow_xyz is None:
            raise RuntimeError("Multimodal Reward scoring requires full tactile-flow tensors")
        raw_layer_h = int(gt_rgb.shape[-2] // 4)
        gt_probability = reward_scorer.score(
            gt_rgb[:, :, :raw_layer_h, :],
            gt_rgb[:, :, raw_layer_h : 2 * raw_layer_h, :],
            gt_flow_xyz,
        )
        pred_probability = reward_scorer.score(
            pred_rgb[:, :, :raw_layer_h, :],
            pred_rgb[:, :, raw_layer_h : 2 * raw_layer_h, :],
            pred_flow_xyz,
        )
        safe_reward = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["episode_id"]))
        reward_path = (
            Path(args.output_dir) / args.split / "reward_scores" / f"{safe_reward}.json"
        )
        reward_scorer.write_episode(
            reward_path,
            episode_id=str(record["episode_id"]),
            task=args.task,
            split=args.split,
            model_name=args.reward_model_name,
            gt_probabilities=gt_probability,
            pred_probabilities=pred_probability,
            fps=args.fps,
        )

    frames = []
    for index in range(0 if args.reward_only else true_frames):
        visual_rows = []
        for part_index, (name, gt_part) in enumerate(gt_visual_parts):
            condition_panel = (
                title_panel(
                    resize_panel(motion_display[index], 672, 224),
                    "Motion condition (shared by all modalities)",
                )
                if part_index == 0
                else blank_panel(672, 224, "Shared motion condition shown in RGB row")
            )
            if name.startswith("Tactile flow") and flow_points is not None:
                gt_panel = flow_panel_for_sides(
                    flow_points,
                    gt_flow_xyz[index, ..., :2],
                    672,
                    224,
                    (30, 135, 105),
                    args.flow_vis_scale,
                    stat_label="gt",
                )
                pred_panel = flow_panel_for_sides(
                    flow_points,
                    pred_flow_xyz[index, ..., :2],
                    672,
                    224,
                    (215, 70, 55),
                    args.flow_vis_scale,
                    stat_label="pred",
                )
                left_panel = title_panel(gt_panel, f"GT Tactile flow marker+arrow  frame={index:04d}")
                right_panel = title_panel(pred_panel, "Pred Tactile flow marker+arrow  fixed episode-0 markers")
            else:
                resize = (lambda panel, w, h: resize_tactile_diff_panel(panel, w, h, 4)) if name == 'Tactile diff' else resize_panel
                left_panel = title_panel(
                    resize(gt_part[index], 672, 224),
                    f"GT {name}  frame={index:04d}",
                )
                right_panel = title_panel(
                    resize(pred_visual_parts[name][index], 672, 224),
                    f"Pred {name}  frame0=GT reference",
                )
            visual_rows.append(np.concatenate([left_panel, right_panel, condition_panel], axis=1))
        bottom = np.concatenate(
            [
                title_panel(resize_panel(gt_state_u8[index], 672, 96, nearest=True), f"GT state={gt_labels[index]}"),
                title_panel(resize_panel(pred_state_u8[index], 672, 96, nearest=True), f"Pred state={pred_labels[index]} (raw)"),
                timeline_panel(gt_labels, pred_labels, index, 672, 96, colors),
            ],
            axis=1,
        )
        frames.append(np.concatenate([*visual_rows, bottom], axis=0))

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["episode_id"]))
    video_path = Path(args.output_dir) / args.split / "videos" / f"{safe}_{true_frames}f.mp4"
    if not args.reward_only:
        write_mp4(video_path, frames, args.fps)
    clean_rgb_path = None
    clean_rgb_gt_path = None
    if args.export_clean_rgb and not args.reward_only:
        clean_rgb_path = Path(args.output_dir) / args.split / "clean_rgb" / f"{safe}_{true_frames}f.mp4"
        clean_rgb_gt_path = Path(args.output_dir) / args.split / "clean_rgb_gt" / f"{safe}_{true_frames}f.mp4"
        write_mp4(clean_rgb_path, pred_visual_parts["RGB"], args.fps)
        write_mp4(clean_rgb_gt_path, dict(gt_visual_parts)["RGB"], args.fps)
    switches = int(np.count_nonzero(pred_labels[1:] != pred_labels[:-1]))
    meta = {
        "episode_id": record["episode_id"],
        "task": args.task,
        "split": args.split,
        "frames": true_frames,
        "source_episode_frames": source_frames,
        "rollout_chunks": rollout_steps,
        "num_inference_steps": args.num_inference_steps,
        "visual_canvas_mode": args.visual_canvas_mode,
        "visual_modalities": [name for name, _ in gt_visual_parts],
        "tactile_flow_encoding": args.flow_encoding if args.visual_canvas_mode == "full" else None,
        "tactile_flow_max_disp": args.flow_max_disp if args.visual_canvas_mode == "full" else None,
        "tactile_flow_visualization": (
            "episode0 fixed 18x10 markers + physical XY arrows"
            if args.visual_canvas_mode == "full" else None
        ),
        "tactile_flow_visual_scale": args.flow_vis_scale if args.visual_canvas_mode == "full" else None,
        "tactile_flow_side_order": (
            ["left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1"]
            if args.visual_canvas_mode == "full" else None
        ),
        "checkpoint": str(args.checkpoint),
        "frame0": "GT RGB/state reference copied into prediction panels",
        "future_gt_conditioned": False,
        "rollout_feedback": "predicted RGB and predicted state",
        "gt_transition_frame": first_transition(gt_labels),
        "pred_transition_frame": first_transition(pred_labels),
        "pred_state_switch_count": switches,
        "source_success": record.get("eval_source_success"),
        "metrics": metric_result,
        "mask_state_replay": True,
        "smoke_only": bool(args.smoke_frames),
        "video": None if args.reward_only else str(video_path),
        "clean_rgb_video": str(clean_rgb_path) if clean_rgb_path is not None else None,
        "clean_rgb_gt_video": str(clean_rgb_gt_path) if clean_rgb_gt_path is not None else None,
        "multimodal_reward_scores": str(reward_path) if reward_path is not None else None,
    }
    write_json(video_path.with_suffix(".json"), meta)
    return meta


def main():
    args = parse_args()
    if args.max_frames and args.max_frames < 9:
        raise ValueError("max-frames must be at least 9")
    if args.metrics and args.metric_profile == "all" and not args.threshold_json:
        raise ValueError("all metrics require --threshold-json")
    if args.metrics and args.metric_profile != "reconstruction" and not args.i3d_weights:
        raise ValueError("distribution metrics require --i3d-weights")
    if args.metrics and args.visual_canvas_mode != "full":
        raise ValueError("metrics require the full RGB/tactile/flow canvas")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    os.environ["PHYSCANVAS_DUAL_REFERENCE_MODE"] = "dual_memory"
    os.environ["PHYSCANVAS_UNIFIED_WAN_TACTILE_FLOW_ENCODING"] = args.flow_encoding
    os.environ["PHYSCANVAS_UNIFIED_WAN_TACTILE_FLOW_LINEAR_MAX_DISP"] = str(args.flow_max_disp)
    os.environ["PHYSCANVAS_SAMPLE_INIT_TRANSFORMER_CHECKPOINT"] = str(checkpoint_transformer_dir(args.checkpoint))
    os.environ["PHYSCANVAS_OBJECT_STATE_LABEL_MANIFEST"] = str(args.selection_manifest)
    os.environ['PHYSCANVAS_OBJECT_STATE_ONLINE_REWARDS'] = '0'
    os.environ['PHYSCANVAS_OBJECT_STATE_MASK_MOTION'] = '1'
    def restore(value):
        if isinstance(value, list):
            if value and value[0] == '__tuple__':
                return tuple(restore(x) for x in value[1:])
            return [restore(x) for x in value]
        if isinstance(value, dict):
            return {k: restore(v) for k, v in value.items()}
        return value
    config = restore(read_json(args.config_json)['training_config'])

    payload = read_json(args.selection_manifest)
    records = list(payload.get("records", payload))
    records = records[args.shard_index :: args.num_shards]
    if args.smoke_frames:
        records = records[:1]
    pretrained = config["models"]["pretrained"]
    vae = AutoencoderKLWan.from_pretrained(
        pretrained, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    scale = (1.0 / torch.tensor(vae.config.latents_std)).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config, shift=config['models']['flow_shift'])
    transformer = load_transformer(args.checkpoint, device, dtype)
    args.metric_accumulator = (
        EpisodeMetrics(device, args.threshold_json, args.output_dir, args.i3d_weights,
                       args.contact_threshold_json, profile=args.metric_profile)
        if args.metrics else None
    )
    reward_scorer = None
    if args.reward_checkpoint:
        if not args.reward_model_source:
            raise ValueError("--reward-model-source is required with --reward-checkpoint")
        from multimodal_reward_hook import MultimodalRewardScorer
        reward_scorer = MultimodalRewardScorer(
            args.reward_checkpoint,
            args.reward_model_source,
            torch.device(args.reward_device),
            tactile_max_disp=args.reward_tactile_max_disp,
        )

    results = []
    for index, record in enumerate(records, start=1):
        args.task = record['object_state_task']
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["episode_id"]))
        frames = min(int(record['sample_num_frames']), args.max_frames) if args.max_frames else int(record['sample_num_frames'])
        video = Path(args.output_dir) / args.split / "videos" / f"{safe}_{frames}f.mp4"
        if args.resume and video.is_file() and video.stat().st_size > 0 and video.with_suffix('.json').is_file():
            meta_path = video.with_suffix(".json")
            prior = read_json(meta_path)
            if (prior.get("episode_id") == record["episode_id"]
                    and prior.get("frames") == frames
                    and prior.get("num_inference_steps") == args.num_inference_steps
                    and bool(prior.get("metrics")) == args.metrics
                    and (not args.metrics or prior["metrics"].get("metric_profile", "all") == args.metric_profile)):
                from .aggregate import verify_video
                clean_paths = [prior.get("clean_rgb_video"), prior.get("clean_rgb_gt_video")]
                clean_complete = all(path and Path(path).is_file() and Path(path).stat().st_size > 0
                                     for path in clean_paths)
                if not args.export_clean_rgb or clean_complete:
                    verify_video(video, frames)
                    if args.export_clean_rgb:
                        for path in clean_paths:
                            verify_video(Path(path), frames)
                    results.append(prior)
                    continue
        print(f"[{args.split} shard {args.shard_index}] {index}/{len(records)} {record['episode_id']}", flush=True)
        results.append(evaluate_episode(
            record, args, config, transformer, vae, scheduler, mean, scale,
            device, dtype, reward_scorer=reward_scorer,
        ))
        write_json(
            Path(args.output_dir) / args.split / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json",
            {"split": args.split, "shard_index": args.shard_index, "num_shards": args.num_shards, "results": results},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
