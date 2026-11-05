# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""In-process Jimu world-model backend owned by an RLinf EnvWorker."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import torch

from rlinf.envs.world_model.jimu_wm_protocol import (
    TACTILE_FLOW_LINEAR_MAX_DISP,
    validate_action_batch,
    validate_wm_action_batch,
)


def _cfg_get(cfg, key: str, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LocalJimuWmBackend:
    """Expose the former ZMQ service contract as direct Python calls.

    Keeping this narrow contract preserves the already-tested grouped candidate
    state manager while moving model placement and offload control under the
    owning RLinf EnvWorker.
    """

    def __init__(
        self,
        cfg,
        *,
        device: torch.device | str,
        seed: int,
        prompt: str,
        owner_id: str | None = None,
        reset_start_index: int = 0,
        reset_stride: int = 1,
    ) -> None:
        history_manifest = _cfg_get(cfg, "fm_reset_history_manifest")
        self.fm_reset_history_store = None
        if history_manifest:
            from rlinf.envs.world_model.fm_reset_history_store import FmResetHistoryStore

            self.fm_reset_history_store = FmResetHistoryStore(
                history_manifest,
                expected_count=int(_cfg_get(cfg, "reset_cache_count", 128)),
            )
        # Import lazily: unit tests can inject a lightweight backend without
        # importing the iMac/DiffSynth runtime.
        from services.jimu_wm_rollout_server import (
            GroupedTwoStageRolloutService,
            _load_existing_engine,
        )

        args = SimpleNamespace(
            reset_cache_root=str(_cfg_get(cfg, "reset_cache_root")),
            reset_cache_count=int(_cfg_get(cfg, "reset_cache_count", 100)),
            imac_server_file=str(_cfg_get(cfg, "imac_server_file")),
            device=str(device),
            dtype=str(_cfg_get(cfg, "dtype", "bfloat16")),
            include_flow=bool(_cfg_get(cfg, "include_flow", True)),
            num_workers=int(_cfg_get(cfg, "num_workers", 0)),
            num_inference_steps=int(_cfg_get(cfg, "num_inference_steps", 30)),
            fixed_clip_count=int(_cfg_get(cfg, "fixed_clip_count", 0)),
            tactile_flow_linear_max_disp=float(
                _cfg_get(
                    cfg,
                    "tactile_flow_linear_max_disp",
                    TACTILE_FLOW_LINEAR_MAX_DISP,
                )
            ),
            tactile_flow_encoding=str(
                _cfg_get(cfg, "tactile_flow_encoding", "linear")
            ),
            tactile_third_channel=str(
                _cfg_get(cfg, "tactile_third_channel", "zero")
            ),
            motion_calibration_root=str(
                _cfg_get(cfg, "motion_calibration_root")
            ),
            motion_renderer_version=str(
                _cfg_get(cfg, "motion_renderer_version")
            ),
            motion_renderer_implementation_version=str(
                _cfg_get(cfg, "motion_renderer_implementation_version")
            ),
            motion_renderer_git_repository=str(
                _cfg_get(cfg, "motion_renderer_git_repository")
            ),
            motion_renderer_git_commit=str(
                _cfg_get(cfg, "motion_renderer_git_commit")
            ),
            motion_renderer_git_path=str(
                _cfg_get(cfg, "motion_renderer_git_path")
            ),
            motion_renderer_sha256=str(
                _cfg_get(cfg, "motion_renderer_sha256")
            ),
            motion_gray_background=int(
                _cfg_get(cfg, "motion_gray_background", 190)
            ),
            motion_gripper_gray=int(
                _cfg_get(cfg, "motion_gripper_gray", 55)
            ),
            motion_mesh_max_faces=int(
                _cfg_get(cfg, "motion_mesh_max_faces", 1000)
            ),
            motion_outline_max_faces=int(
                _cfg_get(cfg, "motion_outline_max_faces", 0)
            ),
            motion_line_thickness=int(
                _cfg_get(cfg, "motion_line_thickness", 1)
            ),
            motion_wrist_tool_offset_x=float(
                _cfg_get(cfg, "motion_wrist_tool_offset_x", 0.0)
            ),
            motion_wrist_tool_offset_y=float(
                _cfg_get(cfg, "motion_wrist_tool_offset_y", 0.0)
            ),
            motion_wrist_tool_offset_z=float(
                _cfg_get(cfg, "motion_wrist_tool_offset_z", 0.0)
            ),
            motion_output_height=int(
                _cfg_get(cfg, "motion_output_height", 224)
            ),
            motion_output_width=int(
                _cfg_get(cfg, "motion_output_width", 224)
            ),
            seed=int(seed),
            default_prompt=str(prompt),
            output_dir=str(_cfg_get(cfg, "output_dir")),
            ckpt_dir=str(_cfg_get(cfg, "ckpt_dir")),
        )
        required_paths = {
            "reset_cache_root": args.reset_cache_root,
            "imac_server_file": args.imac_server_file,
            "output_dir": args.output_dir,
            "ckpt_dir": args.ckpt_dir,
            "motion_calibration_root": args.motion_calibration_root,
            "motion_renderer_git_repository": (
                args.motion_renderer_git_repository
            ),
        }
        missing = [name for name, value in required_paths.items() if not value]
        if missing:
            raise ValueError(f"Missing in-process Jimu WM configuration: {missing}")

        engine = _load_existing_engine(args)
        if self.fm_reset_history_store is not None:
            from rlinf.envs.world_model.fm_reset_history_store import (
                install_fm_reset_stream_cursor,
            )

            install_fm_reset_stream_cursor(engine)
        self.manager = GroupedTwoStageRolloutService(
            engine,
            tactile_flow_linear_max_disp=args.tactile_flow_linear_max_disp,
            tactile_third_channel=args.tactile_third_channel,
            tactile_flow_encoding=args.tactile_flow_encoding,
            reset_start_index=reset_start_index,
            reset_stride=reset_stride,
            return_torch_video=True,
        )
        self.owner_id = owner_id or uuid.uuid4().hex

    @property
    def is_offloaded(self) -> bool:
        return bool(self.manager._is_offloaded)

    def metadata(self) -> dict[str, Any]:
        metadata = self.manager.metadata({})
        metadata["transport"] = "inprocess"
        return metadata

    def reset_groups(
        self, *, num_groups: int, group_size: int, prompt: str
    ) -> dict[str, Any]:
        response = self.manager.reset_groups(
            {
                "owner_id": self.owner_id,
                "num_groups": int(num_groups),
                "group_size": int(group_size),
                "prompt": str(prompt),
            }
        )
        if self.fm_reset_history_store is not None:
            response["observations"] = [
                self.fm_reset_history_store.attach(
                    observation,
                    self.manager.sessions[session_id].motion_episode_id,
                )
                for session_id, observation in zip(
                    response["session_ids"], response["observations"], strict=True
                )
            ]
        return response

    def rollout16(
        self,
        session_ids: list[str],
        actions,
        *,
        audit_session_id: str | None = None,
        return_policy_tactile_video: bool = False,
    ) -> dict[str, Any]:
        action_batch = validate_action_batch(actions)
        return self.manager.rollout16(
            {
                "owner_id": self.owner_id,
                "session_ids": list(session_ids),
                "actions": action_batch,
                "audit_session_id": audit_session_id,
                "return_policy_tactile_video": bool(
                    return_policy_tactile_video
                ),
            }
        )

    def rollout8(self, session_ids: list[str], actions) -> dict[str, Any]:
        action_batch = validate_wm_action_batch(actions)
        return self.manager.rollout8(
            {
                "owner_id": self.owner_id,
                "session_ids": list(session_ids),
                "actions": action_batch,
            }
        )

    def discard_sessions(self) -> None:
        self.manager.release_owner({"owner_id": self.owner_id})

    def offload(self) -> None:
        self.manager.offload()

    def onload(self) -> None:
        self.manager.onload()

    def close(self) -> None:
        self.discard_sessions()
