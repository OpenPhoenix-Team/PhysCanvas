"""Frozen Pi0.5 closed-loop evaluation, using the same backend as GRPO.

Euler Flow-ODE with fixed frame budget. No task RM, rewards, optimizer or future GT.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw

from .backend import PhysCanvasBackend
from .config import compose_rollout_config, TASK_NAMES


class AuditVideo:
    """Streaming H.264 visualization: raw RGB, motion and motion/RGB overlay."""

    def __init__(self, path: Path, fps: int = 10):
        self.path = path
        self.width, self.height = 672, 3 * 224 + 32
        self.pipe = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
             "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, rgb, motion, text):
        rgb, motion = np.asarray(rgb), np.asarray(motion)
        if rgb.shape != (224, 672, 3) or motion.shape != rgb.shape:
            raise ValueError("Audit video requires RGB and motion rows [224,672,3]")
        foreground = np.any(np.abs(motion.astype(np.int16) - 190) > 5, axis=-1)
        overlay = rgb.copy()
        overlay[foreground] = (.55 * rgb[foreground] + .45 * np.array([255, 35, 35])).astype(np.uint8)
        image = Image.new("RGB", (self.width, self.height), (245, 245, 245))
        image.paste(Image.fromarray(rgb), (0, 32))
        image.paste(Image.fromarray(motion), (0, 256))
        image.paste(Image.fromarray(overlay), (0, 480))
        ImageDraw.Draw(image).text((6, 8), text, fill=(0, 0, 0))
        self.pipe.stdin.write(np.asarray(image).tobytes())
        self.frames += 1

    def close(self):
        if self.pipe.stdin and not self.pipe.stdin.closed:
            self.pipe.stdin.close()
        if self.pipe.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {self.path}")


def observation_fingerprint(observation: dict) -> str:
    digest = hashlib.sha256()
    for key in ("observation.images.base_0", "observation.images.left_wrist_0",
                "observation.images.right_wrist_0", "observation.tactile", "observation.state"):
        digest.update(np.ascontiguousarray(observation[key]).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--indices", default="0", help="Comma-separated reset manifest indices")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--frames", type=int, help="Future frames, excluding reset (default: 256)")
    length.add_argument("--max-chunks", type=int, help="Compatibility: request this many full 16-frame chunks")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wm-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    budget = args.frames if args.frames is not None else (16 * args.max_chunks if args.max_chunks is not None else 256)
    if budget < 1:
        parser.error("requested length must be positive")
    indices = [int(x) for x in args.indices.split(",")]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        parser.error("indices must be nonnegative and unique")
    overrides = []
    if args.wm_steps is not None:
        if args.wm_steps <= 0:
            parser.error("wm-steps must be positive")
        overrides.append(f"env.train.world_model.num_inference_steps={args.wm_steps}")
    cfg = compose_rollout_config(args.task, overrides)
    from omegaconf import OmegaConf
    from .wm_runtime import PublishedWmRuntime, validate_public_contract
    identity = validate_public_contract(cfg.env.train.world_model, hash_weights=not args.dry_run)
    print(json.dumps({"wm": identity, "indices": indices,
                      "requested_future_frames": budget,
                      "policy_sampling": "eval",
                      "wm_steps": cfg.env.train.world_model.num_inference_steps}, indent=2), flush=True)
    if args.dry_run:
        return
    # Never overwrite a prior visualization or add artifacts into a source tree.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.yaml").write_text(OmegaConf.to_yaml({"task": cfg.task, "frames": budget, "policy_sampling": "flow_ode",
        "actor_model": OmegaConf.to_container(cfg.actor.model, resolve=True),
        "world_model": OmegaConf.to_container(cfg.env.train.world_model, resolve=True), "prompt": cfg.env.train.prompt}))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    from rlinf.models.embodiment.openpi_rlinf import get_model
    from rlinf.envs.world_model.jimu_wm_protocol import stack_policy_observations

    policy = get_model(cfg.actor.model).eval().requires_grad_(False).cpu()
    results = []
    shared_runtime = PublishedWmRuntime(cfg.env.train.world_model, args.device)
    for reset_index in indices:
        backend = PhysCanvasBackend(
            cfg.env.train.world_model, device=args.device, seed=args.seed,
            prompt=cfg.env.train.prompt, reset_start_index=reset_index,
            runtime=shared_runtime,
        )
        if reset_index >= len(backend.dataset):
            raise IndexError(f"reset index {reset_index} outside {len(backend.dataset)} episodes")
        reset = backend.reset_groups(num_groups=1, group_size=1, prompt=cfg.env.train.prompt)
        sid, observation = reset["session_ids"][0], reset["observations"][0]
        output = args.output_dir / f"sample_{reset_index:03d}"
        output.mkdir()
        video = AuditVideo(output / "rollout.mp4")
        initial_rgb = np.concatenate([observation[key].transpose(1, 2, 0) for key in (
            "observation.images.base_0", "observation.images.left_wrist_0", "observation.images.right_wrist_0")], axis=1)
        Image.fromarray(initial_rgb).save(output / "reset.png")
        initial_motion = backend.renderer.frames(np.repeat(observation["observation.state"][None], 9, axis=0), sid)[0]
        video.write(initial_rgb, initial_motion, "RESET condition | RGB / motion / overlay | frame 0")
        chunks, action_parts = [], []
        try:
            for start in range(0, budget, 16):
                chunk_index, take = start // 16, min(16, budget - start)
                policy_obs = stack_policy_observations(
                    [observation], prompt=cfg.env.train.prompt, policy_chunk_index=chunk_index
                )
                before = observation_fingerprint(observation)
                # Explicit phase offload permits a bounded single-GPU audit.
                backend.offload()
                policy.to(args.device)
                with torch.inference_mode():
                    actions, policy_info = policy.predict_action_batch(
                        policy_obs, mode="eval", compute_values=False
                    )
                if torch.is_tensor(actions):
                    actions = actions.detach().float().cpu().numpy()
                actions = np.asarray(actions, dtype=np.float32)
                del policy_info
                policy.cpu()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                response = backend.rollout16([sid], actions, audit_session_id=sid, frames=take)
                panels = response["rgb_panel_video"]
                observation = response["observations"][0]
                chunks.append({"chunk_index": chunk_index, "policy_observation_sha256": before,
                               "next_observation_sha256": observation_fingerprint(observation),
                               "next_frame_index": observation["wm"]["generated_frame_index"],
                               "executed_frames": take})
                action_parts.append(actions[0, :take])
                for t in range(take):
                    frame = chunk_index * 16 + t + 1
                    video.write(panels[0, t], response["audit_motion_panel_video"][t],
                                f"frame={frame:03d} chunk={chunk_index:02d} Pi0.5 ODE")
                print(f"[rollout] reset={reset_index} frames={start + take}/{budget}", flush=True)
        finally:
            video.close()
            backend.close()
        np.save(output / "actions.npy", np.concatenate(action_parts))
        result = {"reset_index": reset_index, "episode_id": observation["wm"]["episode_id"],
                  "generated_frames": video.frames - 1, "wm_computed_frames": ((budget + 7) // 8) * 8, "video_frames": video.frames, "fps": 10,
                  "rm_scoring": False, "termination": "frame_budget",
                  "policy_sampling": "eval", "sampler": "flow_ode",
                  "wm_steps": int(cfg.env.train.world_model.num_inference_steps), "wm_identity": identity,
                  "chunks": chunks,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        (output / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
