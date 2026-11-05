"""The common closed-loop WM backend used by inference and RLinf EnvWorker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

import numpy as np
import torch

from .wm_runtime import PublishedWmRuntime, configure_public_imports


@dataclass
class Session:
    episode_id: str
    qpos: np.ndarray
    visual: torch.Tensor
    state_canvas: torch.Tensor
    anchor_rgb: torch.Tensor
    seed: int
    frames: int = 0
    anchor_latents: torch.Tensor | None = None
    rng_state: torch.Tensor | None = None


class PhysCanvasBackend:
    """One initial observation per group, independent state for each candidate.

    The 16 actions are never cropped/padded: 0:8 then 8:16, with a generated
    local reference between the calls. The policy replans only after both calls.
    Candidate diffusion RNGs start identically within each group (common random
    numbers); policy action sampling is independently stochastic.
    """

    def __init__(self, cfg, *, device: str, seed: int, prompt: str,
                 reset_start_index: int = 0, reset_stride: int = 1,
                 runtime=None, dataset=None, renderer=None):
        self.cfg, self.device = cfg, str(device)
        self.seed, self.prompt = int(seed), str(prompt)
        self.reset_index, self.reset_stride = int(reset_start_index), int(reset_stride)
        if self.reset_index < 0 or self.reset_stride < 1:
            raise ValueError("Invalid reset worker index/stride")
        self.task = str(cfg["task"])
        if self.task not in ("jimu", "shiguan"):
            raise ValueError("Only Jimu and double-arm Shiguan are supported")
        if runtime is None:
            configure_public_imports(cfg)
        from .reset_data import ResetDataset
        from .motion import MotionRenderer
        self.dataset = dataset if dataset is not None else ResetDataset(
            cfg["reset_cache"], cfg["selection_manifest"]
        )
        self.runtime = runtime if runtime is not None else PublishedWmRuntime(cfg, device)
        self.renderer = renderer if renderer is not None else MotionRenderer(cfg["renderer"])
        self.sessions: dict[str, Session] = {}

    def metadata(self) -> dict[str, Any]:
        return {
            "transport": "inprocess", "protocol": "physcanvas-pi05-v1",
            "sub_frames": 8, "state_dim": 14, "action_horizon": 16,
            "views": ["base", "left_wrist", "right_wrist"],
            "pi05_tactile_shape": [4, 700, 3], "policy_tactile_at_reset": True,
            "generated_policy_tactile": True,
            "generated_policy_tactile_third_channel": "xyz",
            "tactile_flow_encoding": "xyz_linear", "tactile_flow_linear_max_disp": .06,
            "include_depth": False, "visual_canvas_shape": [672, 672],
            "state_canvas_shape": [32, 672], "reset_count": len(self.dataset),
            "checkpoint": str(self.cfg.get("checkpoint", "injected-test-runtime")),
            "num_inference_steps": int(self.cfg.get("num_inference_steps", 30)),
            "future_gt_conditioned": False, "session_count": len(self.sessions),
        }

    @staticmethod
    def _observation(session: Session) -> dict[str, Any]:
        from .canvas import decode_policy_visual
        rgb, tactile = decode_policy_visual(session.visual)
        observation = {
            f"observation.images.{camera}": np.ascontiguousarray(
                rgb[-1, :, i * 224:(i + 1) * 224].transpose(2, 0, 1)
            )
            for i, camera in enumerate(("base_0", "left_wrist_0", "right_wrist_0"))
        }
        observation.update({
            "observation.state": session.qpos.copy(),
            "observation.tactile": np.ascontiguousarray(tactile[-1]),
            "observation.tactile_mask": np.ones(4, dtype=np.bool_),
            "wm": {"episode_id": session.episode_id, "generated_frame_index": session.frames,
                   "tactile_source": "wm_xyz_linear" if session.frames else "real_reset_xyz",
                   "state_source": "executed_absolute_action" if session.frames else "real_reset_qpos"},
        })
        return observation

    def reset_groups(self, *, num_groups: int, group_size: int, prompt: str):
        from .canvas import build_state, build_visual
        if num_groups < 1 or group_size < 1:
            raise ValueError("num_groups and group_size must be positive")
        if str(prompt) != self.prompt:
            raise ValueError("Changing task prompt requires a new backend")
        self.discard_sessions()
        result = {"session_ids": [], "group_ids": [], "observations": []}
        for group in range(num_groups):
            reset = self.dataset.read(self.reset_index % len(self.dataset))
            visual = build_visual(reset)
            state = build_state(label=int(reset.get("initial_label", 0)), task=self.task)
            for candidate in range(group_size):
                session_id = f"g{group:04d}-c{candidate:04d}-{uuid.uuid4().hex[:12]}"
                session = Session(
                    episode_id=reset["episode_id"], qpos=reset["qpos"].copy(),
                    visual=visual.clone(), state_canvas=state.clone(),
                    anchor_rgb=visual[:, :, :, :224].clone(), seed=self.seed + self.reset_index,
                )
                self.sessions[session_id] = session
                result["session_ids"].append(session_id)
                result["group_ids"].append(group)
                observation = self._observation(session)
                # Preserve exact real reset pixels; normalize/de-normalize can
                # otherwise change an uint8 value by one due to float rounding.
                for view, camera in enumerate(("base_0", "left_wrist_0", "right_wrist_0")):
                    observation[f"observation.images.{camera}"] = np.ascontiguousarray(
                        reset["rgb"][view].transpose(2, 0, 1)
                    )
                observation["observation.tactile"] = np.array(reset["tactile_xyz"], copy=True)
                result["observations"].append(observation)
            self.reset_index += self.reset_stride
        return result

    def rollout16(self, session_ids, actions, *, audit_session_id=None,
                  return_policy_tactile_video=False):
        from .canvas import build_motion_replay, decode_policy_visual
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (len(session_ids), 16, 14) or not np.isfinite(actions).all():
            raise ValueError("Expected finite absolute actions [B,16,14]")
        if not session_ids or len(set(session_ids)) != len(session_ids):
            raise ValueError("Expected nonempty unique session IDs")
        if audit_session_id is not None and audit_session_id not in session_ids:
            raise ValueError("audit_session_id not in batch")
        if any(sid not in self.sessions for sid in session_ids):
            raise KeyError("Unknown/expired session ID")
        result = {"session_ids": list(session_ids), "observations": [],
                  "segments": 2, "frames_per_segment": 8}
        videos, tactile_videos = [], []
        for index, sid in enumerate(session_ids):
            session = self.sessions[sid]
            rgb_parts, tactile_parts, motion_parts, diff_parts = [], [], [], []
            for offset in (0, 8):
                action8 = actions[index, offset:offset + 8]
                qpos9 = np.concatenate([session.qpos[None], action8], axis=0)
                motion = self.renderer.frames(qpos9, f"{sid}-frame{session.frames}")
                generated, generated_state = self.runtime.predict(session, build_motion_replay(motion))
                if tuple(generated.shape) != (1, 9, 3, 672, 672):
                    raise RuntimeError(f"WM must return reference + 8 frames: {generated.shape}")
                rgb, tactile = decode_policy_visual(generated[:, 1:])
                rgb_parts.append(rgb)
                tactile_parts.append(tactile)
                motion_parts.append(motion[1:])
                diff = generated[0, 1:, :, 224:448].permute(0, 2, 3, 1)
                diff_parts.append(((diff + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy())
                # No reset-frame reuse, VAE re-encoding of RGB uint8, or state thresholding.
                session.visual = generated[:, -1:].clone()
                session.state_canvas = generated_state[:, -1:].clone()
                session.qpos = action8[-1].copy()
                session.frames += 8
            videos.append(np.concatenate(rgb_parts))
            tactile_videos.append(np.concatenate(tactile_parts))
            result["observations"].append(self._observation(session))
            if sid == audit_session_id:
                result.update(audit_session_id=sid,
                              audit_motion_panel_video=np.concatenate(motion_parts),
                              audit_tactile_panel_video=np.concatenate(diff_parts))
        result["rgb_panel_video"] = np.ascontiguousarray(np.stack(videos))
        if return_policy_tactile_video:
            result["policy_tactile_video"] = np.ascontiguousarray(np.stack(tactile_videos))
        return result

    def discard_sessions(self):
        self.sessions.clear()

    def onload(self):
        self.runtime.onload()

    def offload(self):
        self.runtime.offload()

    def close(self):
        self.discard_sessions()
        self.offload()
