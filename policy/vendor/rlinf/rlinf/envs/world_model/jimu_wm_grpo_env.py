# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf environment for closed-loop Pi0.5 rollouts in the Jimu world model.

Each RLinf EnvWorker owns one in-process Jimu WM replica. One policy call
produces 16 absolute 14-D joint targets; the environment advances each local
candidate twice (actions 0:8, then 8:16) and scores the 16 aligned three-view
RGB frames without socket serialization. The frozen reward model thresholds
every frame using the task configuration. Success normally uses any frame;
Bottle's relative-final-frame mode uses only the final frame. A binary
success signal is placed at the chunk's last action and that candidate stops.
The adapter supports the original relative-binary Jimu mode and direct
trajectory-binary reward; both give each successful trajectory return 1 and
each failure 0 when success termination is enabled.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.envs.world_model.fm_wm_history import (
    next_fm_histories,
    select_wm_action_endpoints,
    validate_fm_reset_history,
    wm_tactile_to_fm_grid,
)
from rlinf.envs.world_model.jimu_reward_model import ThreeViewRewardScorer
from rlinf.envs.world_model.jimu_tactile_ablation import (
    mask_generated_tactile_for_policy,
)
from rlinf.envs.world_model.jimu_rollout_visualizer import (
    JimuRolloutVideoWriter,
)
from rlinf.envs.world_model.jimu_wm_local_backend import LocalJimuWmBackend
from rlinf.envs.world_model.jimu_wm_protocol import (
    ACTION_DIM,
    ACTION_HORIZON,
    VIEW_ORDER,
    WM_SUB_FRAMES,
    stack_policy_observations,
    validate_action_batch,
    validate_group_layout,
    validate_panel_video,
)

__all__ = ["JimuWmGrpoEnv"]


class _LocalResetDataset:
    """Marker object satisfying ``BaseWorldEnv``'s dataset hook."""

    def __len__(self) -> int:
        return 2**31 - 1


def _cfg_get(cfg, key: str, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _merge_policy_observations(
    current: dict[str, Any],
    updates: dict[str, Any],
    indices: torch.Tensor,
) -> dict[str, Any]:
    """Replace active rows without mutating observations already in flight."""

    merged: dict[str, Any] = {}
    row_indices = indices.detach().cpu().tolist()
    for key, current_value in current.items():
        update_value = updates[key]
        if torch.is_tensor(current_value):
            value = current_value.clone()
            value[indices.to(value.device)] = update_value.to(value.device)
            merged[key] = value
        elif isinstance(current_value, list):
            value = list(current_value)
            for source_index, target_index in enumerate(row_indices):
                value[target_index] = update_value[source_index]
            merged[key] = value
        else:
            raise TypeError(
                f"Unsupported policy observation type for {key!r}: "
                f"{type(current_value).__name__}"
            )
    return merged


class JimuWmGrpoEnv(BaseWorldEnv):
    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        record_metrics: bool = True,
        worker_info=None,
        *,
        client: Any | None = None,
        reward_scorer: ThreeViewRewardScorer | None = None,
        rollout_visualizer: Any | None = None,
    ) -> None:
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
            record_metrics,
        )
        self.group_size = int(cfg.group_size)
        self.num_groups = validate_group_layout(self.num_envs, self.group_size)
        self.action_horizon = int(_cfg_get(cfg, "action_horizon", ACTION_HORIZON))
        self.action_dim = int(_cfg_get(cfg, "action_dim", ACTION_DIM))
        self.wm_sub_frames = int(_cfg_get(cfg, "wm_sub_frames", WM_SUB_FRAMES))
        self.max_policy_chunks = int(_cfg_get(cfg, "max_policy_chunks", 16))
        self.reward_coef = float(_cfg_get(cfg, "reward_coef", 1.0))
        self.terminate_on_rm_success = bool(
            _cfg_get(cfg, "terminate_on_rm_success", True)
        )
        self.policy_tactile_mode = str(
            _cfg_get(cfg, "policy_tactile_mode", "enabled")
        ).lower()
        if self.policy_tactile_mode not in {"enabled", "disabled"}:
            raise ValueError(
                "policy_tactile_mode must be 'enabled' or 'disabled', "
                f"got {self.policy_tactile_mode!r}"
            )
        self.policy_temporal_mode = str(
            _cfg_get(cfg, "policy_temporal_mode", "pi05_single_frame")
        ).lower()
        if self.policy_temporal_mode not in {
            "pi05_single_frame",
            "fm_30hz_history",
        }:
            raise ValueError(
                "policy_temporal_mode must be 'pi05_single_frame' or "
                f"'fm_30hz_history', got {self.policy_temporal_mode!r}"
            )
        self.policy_tactile_third_channel = str(
            _cfg_get(
                _cfg_get(cfg, "world_model", {}),
                "tactile_third_channel",
                "zero",
            )
        ).lower()
        if self.policy_tactile_third_channel not in {
            "zero",
            "magnitude",
            "xyz",
        }:
            raise ValueError(
                "world_model.tactile_third_channel must be 'zero' or "
                "'magnitude' or 'xyz', got "
                f"{self.policy_tactile_third_channel!r}"
            )
        if self.policy_tactile_third_channel == "xyz":
            encoding = str(
                _cfg_get(
                    _cfg_get(cfg, "world_model", {}),
                    "tactile_flow_encoding",
                    "",
                )
            ).lower()
            if encoding != "xyz_linear":
                raise ValueError(
                    "Policy [dx,dy,dz] requires WM xyz_linear encoding"
                )
        if self.policy_temporal_mode == "fm_30hz_history":
            if self.policy_tactile_mode != "enabled":
                raise ValueError("FM 30 Hz history requires policy tactile")
            if self.policy_tactile_third_channel not in {"zero", "xyz"}:
                raise ValueError(
                    "The current FM rollout contract requires WM tactile "
                    "channels [dx,dy,0] or signed [dx,dy,dz]"
                )
        if (self.action_horizon, self.action_dim, self.wm_sub_frames) != (
            ACTION_HORIZON,
            ACTION_DIM,
            WM_SUB_FRAMES,
        ):
            raise ValueError(
                "Jimu integration is defined for action_horizon=16, action_dim=14, "
                "wm_sub_frames=8"
            )
        if self.max_policy_chunks <= 0:
            raise ValueError("max_policy_chunks must be positive")
        if self.reward_coef != 1.0:
            raise ValueError("Jimu GRPO requires reward_coef=1.0")
        if not self.terminate_on_rm_success:
            raise ValueError("Jimu GRPO requires terminate_on_rm_success=True")
        expected_episode_steps = self.max_policy_chunks * ACTION_HORIZON
        if int(cfg.max_episode_steps) != expected_episode_steps:
            raise ValueError(
                "max_episode_steps must equal max_policy_chunks * action_horizon: "
                f"expected {expected_episode_steps}, got {cfg.max_episode_steps}"
            )
        if bool(_cfg_get(cfg, "auto_reset", False)):
            raise ValueError("JimuWmGrpoEnv requires auto_reset=False")

        self.prompt = str(cfg.prompt)
        self.session_ids: list[str] = []
        self.current_obs: dict[str, Any] | None = None
        self._is_offloaded = False
        self._raw_probability_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._generated_frame_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._policy_chunks_executed = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.env_worker_rank = int(
            getattr(worker_info, "rank", seed_offset)
            if worker_info is not None
            else seed_offset
        )
        visualization_cfg = _cfg_get(cfg, "rollout_visualization", {})
        self._visualization_rollout_epoch = int(_cfg_get(cfg, "rollout_epoch", 1))
        self._visualization_every_updates = int(
            _cfg_get(visualization_cfg, "every_updates", 1)
        )
        self._visualization_sample_env_index = int(
            _cfg_get(visualization_cfg, "sample_env_index", 0)
        )
        self._visualization_strict = bool(
            _cfg_get(visualization_cfg, "strict", False)
        )
        if self._visualization_rollout_epoch <= 0:
            raise ValueError("rollout_epoch must be positive")
        if self._visualization_every_updates <= 0:
            raise ValueError("visualization every_updates must be positive")
        if not 0 <= self._visualization_sample_env_index < self.num_envs:
            raise ValueError(
                "visualization sample_env_index must address a local environment"
            )
        configured_worker_ranks = _cfg_get(
            visualization_cfg, "worker_ranks", None
        )
        if configured_worker_ranks is None:
            configured_worker_ranks = [
                int(_cfg_get(visualization_cfg, "worker_rank", 0))
            ]
        self._visualization_worker_ranks = tuple(
            int(rank) for rank in configured_worker_ranks
        )
        if not self._visualization_worker_ranks:
            raise ValueError("visualization worker_ranks must not be empty")
        if len(set(self._visualization_worker_ranks)) != len(
            self._visualization_worker_ranks
        ):
            raise ValueError("visualization worker_ranks must be unique")
        if any(rank < 0 for rank in self._visualization_worker_ranks):
            raise ValueError("visualization worker_ranks must be non-negative")
        self._visualization_enabled = bool(
            _cfg_get(visualization_cfg, "enabled", False)
        ) and not bool(_cfg_get(cfg, "is_eval", False))
        self._visualization_enabled &= (
            self.env_worker_rank in self._visualization_worker_ranks
        )
        self._visualization_reset_count = 0
        self._visualization_active = False
        self._visualizer: JimuRolloutVideoWriter | None = None
        if self._visualization_enabled:
            visualization_output_dir = Path(
                str(_cfg_get(visualization_cfg, "output_dir"))
            ) / f"worker_{self.env_worker_rank:03d}"
            self._visualizer = rollout_visualizer or JimuRolloutVideoWriter(
                visualization_output_dir,
                fps=int(_cfg_get(visualization_cfg, "fps", 10)),
                motion_background=int(
                    _cfg_get(
                        visualization_cfg,
                        "motion_background",
                        _cfg_get(cfg.world_model, "motion_gray_background", 190),
                    )
                ),
            )
        backend_cls = LocalJimuWmBackend
        if str(_cfg_get(cfg.world_model, "backend", "inprocess")) == "physcanvas":
            from physcanvas_pi05.backend import PhysCanvasBackend

            backend_cls = PhysCanvasBackend
        self.client = client or backend_cls(
            cfg.world_model,
            device=self._get_runtime_device_str(),
            seed=self.seed,
            prompt=self.prompt,
            reset_start_index=self.env_worker_rank,
            reset_stride=self.total_num_processes,
        )
        metadata = self.client.metadata()
        self._validate_server_metadata(metadata)
        if client is None and metadata.get("transport") != "inprocess":
            raise RuntimeError(
                "Production Jimu GRPO requires an in-process WM backend"
            )

        reward_cfg = cfg.reward_model
        reward_output_type = str(
            _cfg_get(reward_cfg, "output_type", "label")
        ).lower()
        if reward_output_type not in {"label", "binary"}:
            raise ValueError(
                "Jimu GRPO requires binary reward labels, "
                f"got {reward_output_type!r}"
            )
        self.trajectory_reward_aggregation = str(
            _cfg_get(cfg, "trajectory_reward_aggregation", "relative_sum")
        )
        relative_modes = {
            "relative_sum",
            "relative_binary_chunk_last",
            "relative_binary_final_frame",
        }
        direct_modes = {"binary_any_success", "trajectory_binary_any_success"}
        if self.trajectory_reward_aggregation not in relative_modes | direct_modes:
            raise ValueError(
                "Unsupported Jimu trajectory reward aggregation: "
                f"{self.trajectory_reward_aggregation!r}"
            )
        if self.use_rel_reward != (
            self.trajectory_reward_aggregation in relative_modes
        ):
            raise ValueError(
                "use_rel_reward must match trajectory_reward_aggregation: "
                f"use_rel_reward={self.use_rel_reward}, "
                f"aggregation={self.trajectory_reward_aggregation!r}"
            )
        if not bool(_cfg_get(cfg, "skip_terminated_wm_rollouts", True)):
            raise ValueError(
                "Jimu GRPO requires skip_terminated_wm_rollouts=true"
            )
        self.reward_scorer = reward_scorer or ThreeViewRewardScorer(
            reward_cfg.checkpoint,
            device=self.device,
            threshold=float(_cfg_get(reward_cfg, "threshold", 0.5)),
            batch_size=int(_cfg_get(reward_cfg, "batch_size", 256)),
        )
        configured_threshold = float(_cfg_get(reward_cfg, "threshold", 0.5))
        if float(self.reward_scorer.threshold) != configured_threshold:
            raise ValueError(
                "Reward scorer threshold does not match configuration: "
                f"scorer={self.reward_scorer.threshold}, "
                f"configured={configured_threshold}"
            )

    def _build_dataset(self, cfg):
        del cfg
        return _LocalResetDataset()

    def _prepare_policy_observations(
        self, observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.policy_tactile_mode == "disabled":
            return [mask_generated_tactile_for_policy(obs) for obs in observations]

        prepared_observations = []
        for observation in observations:
            prepared = dict(observation)
            tactile = np.asarray(
                observation.get("observation.tactile"), dtype=np.float32
            ).copy()
            if tactile.shape != (4, 700, 3):
                raise ValueError(
                    "Policy tactile must have shape (4,700,3), "
                    f"got {tactile.shape}"
                )
            if self.policy_tactile_third_channel == "zero":
                tactile[..., 2] = 0.0
            elif self.policy_tactile_third_channel == "magnitude":
                tactile[..., 2] = np.linalg.norm(tactile[..., :2], axis=-1)
            # xyz: preserve signed dz from reset and generated observations.
            prepared["observation.tactile"] = np.ascontiguousarray(tactile)
            prepared_observations.append(prepared)
        return prepared_observations

    def _attach_fm_reset_histories(
        self,
        stacked: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        if self.policy_temporal_mode != "fm_30hz_history":
            return
        state_histories = []
        tactile_histories = []
        for index, observation in enumerate(observations):
            if "fm.state_history_30hz" not in observation:
                raise KeyError(
                    f"Reset observation {index} has no real FM state history; "
                    "the Pi0.5 first-frame expansion cache is not valid for FM"
                )
            if "fm.tactile_history_30hz" not in observation:
                raise KeyError(
                    f"Reset observation {index} has no real FM tactile history; "
                    "the Pi0.5 first-frame expansion cache is not valid for FM"
                )
            state = np.asarray(
                observation["fm.state_history_30hz"], dtype=np.float32
            )
            tactile = np.asarray(
                observation["fm.tactile_history_30hz"], dtype=np.float32
            )
            validate_fm_reset_history(state, tactile)
            state_histories.append(state)
            tactile_histories.append(tactile)
        stacked["state_history_30hz"] = torch.from_numpy(
            np.ascontiguousarray(np.stack(state_histories, axis=0))
        )
        stacked["tactile_history_30hz"] = torch.from_numpy(
            np.ascontiguousarray(np.stack(tactile_histories, axis=0))
        )

    def _attach_fm_next_histories(
        self,
        active_obs: dict[str, Any],
        active_indices: torch.Tensor,
        executed_actions: np.ndarray,
        policy_tactile_video: Any,
    ) -> None:
        if self.policy_temporal_mode != "fm_30hz_history":
            return
        tactile_video = np.asarray(policy_tactile_video, dtype=np.float32)
        expected = (len(executed_actions), ACTION_HORIZON, 4, 700, 3)
        if tactile_video.shape != expected:
            raise RuntimeError(
                f"FM requires numeric WM tactile video {expected}, "
                f"got {tactile_video.shape}"
            )
        state_histories = []
        tactile_histories = []
        rows = active_indices.detach().cpu().tolist()
        for source_index, target_index in enumerate(rows):
            previous_tactile = (
                self.current_obs["tactile_history_30hz"][target_index, -1]
                .cpu()
                .numpy()
            )
            anchors = wm_tactile_to_fm_grid(tactile_video[source_index])
            state, tactile = next_fm_histories(
                executed_actions[source_index], previous_tactile, anchors
            )
            state_histories.append(state)
            tactile_histories.append(tactile)
        active_obs["state_history_30hz"] = torch.from_numpy(
            np.ascontiguousarray(np.stack(state_histories, axis=0))
        )
        active_obs["tactile_history_30hz"] = torch.from_numpy(
            np.ascontiguousarray(np.stack(tactile_histories, axis=0))
        )

    def _validate_server_metadata(self, metadata: dict[str, Any]) -> None:
        if int(metadata.get("sub_frames", -1)) != WM_SUB_FRAMES:
            raise ValueError(
                f"WM server must expose sub_frames={WM_SUB_FRAMES}: {metadata}"
            )
        views = tuple(metadata.get("views", ()))
        accepted = {
            VIEW_ORDER,
            (
                "observation.images.base_0",
                "observation.images.left_wrist_0",
                "observation.images.right_wrist_0",
            ),
        }
        if views not in accepted:
            raise ValueError(f"WM server view order is incompatible: {views}")
        if int(metadata.get("state_dim", -1)) != ACTION_DIM:
            raise ValueError(f"WM server state_dim must be {ACTION_DIM}: {metadata}")
        if tuple(metadata.get("pi05_tactile_shape", ())) != (4, 700, 3):
            raise ValueError(
                f"WM server must expose reset tactile shape [4,700,3]: {metadata}"
            )
        if not bool(metadata.get("policy_tactile_at_reset", False)):
            raise ValueError(f"WM server does not expose reset tactile: {metadata}")
        if self.policy_tactile_mode == "enabled" and not bool(
            metadata.get("generated_policy_tactile", False)
        ):
            raise ValueError(
                "Multi-chunk Jimu GRPO requires generated policy tactile: "
                f"{metadata}"
            )
        expected_third_channel = str(
            _cfg_get(
                _cfg_get(self.cfg, "world_model", {}),
                "tactile_third_channel",
                "zero",
            )
        ).lower()
        actual_third_channel = str(
            metadata.get("generated_policy_tactile_third_channel", "zero")
        ).lower()
        if actual_third_channel != expected_third_channel:
            raise ValueError(
                "WM policy tactile third-channel mode is incompatible: "
                f"expected={expected_third_channel!r}, "
                f"actual={actual_third_channel!r}"
            )
        if (
            expected_third_channel == "xyz"
            and metadata.get("tactile_flow_encoding") != "xyz_linear"
        ):
            raise ValueError(
                "WM metadata must confirm xyz_linear for signed policy tactile"
            )

    @property
    def info_logging_keys(self):
        return [
            "success_rate",
            "success_once",
            "return",
            "reward_probability_mean",
            "reward_probability_group_std",
            "valid_group_fraction",
            "policy_chunks_executed",
            "wm_calls_executed",
        ]

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
        episode_indices=None,
    ):
        del seed, options, episode_indices
        if self._is_offloaded and hasattr(self.client, "discard_sessions"):
            self.client.discard_sessions()
        self.onload()
        response = self.client.reset_groups(
            num_groups=self.num_groups,
            group_size=self.group_size,
            prompt=self.prompt,
        )
        session_ids = list(response.get("session_ids", []))
        group_ids = list(response.get("group_ids", []))
        observations = self._prepare_policy_observations(
            list(response.get("observations", []))
        )
        if len(session_ids) != self.num_envs or len(observations) != self.num_envs:
            raise RuntimeError(
                "WM reset response does not match num_envs: "
                f"sessions={len(session_ids)} observations={len(observations)} "
                f"num_envs={self.num_envs}"
            )
        if len(set(session_ids)) != len(session_ids):
            raise RuntimeError("WM reset returned duplicate session ids")
        expected_group_ids = [
            group_index
            for group_index in range(self.num_groups)
            for _ in range(self.group_size)
        ]
        if group_ids != expected_group_ids:
            raise RuntimeError(
                "WM reset did not return contiguous GRPO groups: "
                f"expected={expected_group_ids} received={group_ids}"
            )
        self.session_ids = session_ids
        self.current_obs = stack_policy_observations(
            observations,
            prompt=self.prompt,
            policy_chunk_index=0,
        )
        self._attach_fm_reset_histories(self.current_obs, observations)
        self.elapsed_steps = 0
        self._is_start = False
        self._reset_metrics()
        self._raw_probability_sum.zero_()
        self._generated_frame_count.zero_()
        self._policy_chunks_executed.zero_()
        self._done.zero_()
        self._start_rollout_visualization(observations)
        return self.current_obs, {"session_ids": tuple(session_ids)}

    def _visualization_failure(self, exc: Exception) -> None:
        if self._visualizer is not None:
            self._visualizer.abort()
        self._visualization_active = False
        if self._visualization_strict:
            raise exc
        warnings.warn(
            f"Disabling the current Shiguan rollout visualization: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _start_rollout_visualization(
        self, observations: list[dict[str, Any]]
    ) -> None:
        reset_index = self._visualization_reset_count
        self._visualization_reset_count += 1
        if not self._visualization_enabled or self._visualizer is None:
            return
        if self._visualizer.active:
            self._visualization_failure(
                RuntimeError("Previous update visualization did not finish")
            )
            return
        epoch_in_update = reset_index % self._visualization_rollout_epoch
        update_index = reset_index // self._visualization_rollout_epoch + 1
        if epoch_in_update != 0 or update_index % self._visualization_every_updates:
            self._visualization_active = False
            return
        sample_observation = observations[self._visualization_sample_env_index]
        try:
            self._visualizer.begin(
                update_index=update_index,
                sample_env_index=self._visualization_sample_env_index,
                reset_metadata={
                    **sample_observation.get("wm", {}),
                    "tactile_flow_encoding": str(
                        _cfg_get(
                            _cfg_get(self.cfg, "world_model", {}),
                            "tactile_flow_encoding",
                            "linear",
                        )
                    ),
                    "policy_tactile_third_channel": (
                        self.policy_tactile_third_channel
                    ),
                },
            )
            self._visualization_active = True
        except Exception as exc:
            self._visualization_failure(exc)

    @staticmethod
    def _video_to_numpy(video: Any) -> np.ndarray:
        if torch.is_tensor(video):
            video = video.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(video, dtype=np.uint8))

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        del actions, auto_reset
        raise NotImplementedError(
            "JimuWmGrpoEnv advances one full chunk via chunk_step"
        )

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        self.onload()
        if not self.session_ids:
            raise RuntimeError("Call reset before chunk_step")
        if isinstance(policy_output_action, dict):
            policy_output_action = policy_output_action.get(
                "actions", policy_output_action.get("raw_actions")
            )
        if torch.is_tensor(policy_output_action):
            policy_output_action = policy_output_action.detach().float().cpu().numpy()
        if self.policy_temporal_mode == "fm_30hz_history":
            executed_actions = np.asarray(policy_output_action, dtype=np.float32)
            if executed_actions.ndim != 3 or executed_actions.shape[1:] != (48, 14):
                raise ValueError(
                    "FM policy must execute [B,48,14] absolute 30 Hz targets, "
                    f"got {executed_actions.shape}"
                )
            if not np.isfinite(executed_actions).all():
                raise ValueError("FM actions contain NaN or Inf")
            executed_actions = np.ascontiguousarray(executed_actions)
            wm_actions = select_wm_action_endpoints(executed_actions)
        else:
            wm_actions = validate_action_batch(policy_output_action)
            executed_actions = wm_actions
        if executed_actions.shape[0] != self.num_envs:
            raise ValueError(
                "Expected "
                f"{self.num_envs} env actions, got {executed_actions.shape[0]}"
            )

        active_indices = torch.nonzero(~self._done, as_tuple=False).flatten()
        active_count = int(active_indices.numel())
        raw_probabilities = torch.zeros(
            (self.num_envs, ACTION_HORIZON),
            dtype=torch.float32,
            device=self.device,
        )
        labels = torch.zeros_like(raw_probabilities)
        chunk_successes = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        visualization_chunk: dict[str, np.ndarray] | None = None

        if active_count:
            active_rows = active_indices.detach().cpu().numpy()
            active_session_ids = [self.session_ids[index] for index in active_rows]
            audit_session_id = None
            if self._visualization_active:
                sample_index = self._visualization_sample_env_index
                if sample_index in active_rows:
                    audit_session_id = self.session_ids[sample_index]
            rollout_kwargs = {}
            if audit_session_id is not None:
                rollout_kwargs["audit_session_id"] = audit_session_id
            if self.policy_temporal_mode == "fm_30hz_history":
                rollout_kwargs["return_policy_tactile_video"] = True
            active_executed_actions = executed_actions[active_rows]
            response = self.client.rollout16(
                active_session_ids,
                wm_actions[active_rows],
                **rollout_kwargs,
            )
            returned_sessions = list(response.get("session_ids", []))
            if returned_sessions != active_session_ids:
                raise RuntimeError("WM response changed active session order")
            panel_video = validate_panel_video(response.get("rgb_panel_video"))
            if panel_video.shape[0] != active_count:
                raise RuntimeError(
                    f"WM returned {panel_video.shape[0]} videos for "
                    f"{active_count} active envs"
                )
            if audit_session_id is not None:
                if response.get("audit_session_id") != audit_session_id:
                    raise RuntimeError("WM response changed audit session identity")
                audit_batch_index = active_session_ids.index(audit_session_id)
                visualization_chunk = {
                    "rgb": self._video_to_numpy(panel_video[audit_batch_index]),
                    "motion": self._video_to_numpy(
                        response.get("audit_motion_panel_video")
                    ),
                    "tactile": self._video_to_numpy(
                        response.get("audit_tactile_panel_video")
                    ),
                }

            # Keep reward inference in fp32, matching the reward model's
            # validation pipeline and configured decision threshold.
            active_probabilities, active_labels = (
                self.reward_scorer.score_panel_video(panel_video)
            )
            active_probabilities = active_probabilities.to(
                self.device, dtype=torch.float32
            )
            active_labels = active_labels.to(self.device, dtype=torch.float32)
            expected_shape = (active_count, ACTION_HORIZON)
            if tuple(active_probabilities.shape) != expected_shape:
                raise RuntimeError(
                    "Unexpected active reward shape: "
                    f"{tuple(active_probabilities.shape)}"
                )
            if tuple(active_labels.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected active label shape: {tuple(active_labels.shape)}"
                )
            raw_probabilities[active_indices] = active_probabilities
            labels[active_indices] = active_labels
            if self.trajectory_reward_aggregation == "relative_binary_final_frame":
                # Bottle succeeds only if the plan's final generated frame is
                # positive. A transient positive earlier in the plan must not
                # emit reward or terminate the candidate.
                chunk_successes[active_indices] = active_labels[:, -1].bool()
            else:
                chunk_successes[active_indices] = active_labels.bool().any(dim=1)

            observations = self._prepare_policy_observations(
                list(response.get("observations", []))
            )
            active_obs = stack_policy_observations(
                observations,
                prompt=self.prompt,
                policy_chunk_index=(self.elapsed_steps // ACTION_HORIZON) + 1,
            )
            self._attach_fm_next_histories(
                active_obs,
                active_indices,
                active_executed_actions,
                response.get("policy_tactile_video"),
            )
            if self.policy_tactile_mode == "enabled":
                if not bool(active_obs["tactile_masks"].all()):
                    raise RuntimeError(
                        "Generated tactile must be valid for replanning"
                    )
            else:
                if bool(active_obs["tactile_masks"].any()):
                    raise RuntimeError("Disabled policy tactile has a true mask bit")
                if bool(active_obs["tactiles"].ne(0).any()):
                    raise RuntimeError("Disabled policy tactile contains nonzero values")
            self.current_obs = _merge_policy_observations(
                self.current_obs, active_obs, active_indices
            )
            self._raw_probability_sum[active_indices] += (
                active_probabilities.sum(dim=1)
            )
            self._generated_frame_count[active_indices] += ACTION_HORIZON
            self._policy_chunks_executed[active_indices] += 1

        current_binary_reward = chunk_successes.to(torch.float32) * self.reward_coef
        emitted_reward = torch.zeros_like(current_binary_reward)
        if self.use_rel_reward:
            emitted_reward[active_indices] = (
                current_binary_reward[active_indices]
                - self.prev_step_reward[active_indices]
            )
            self.prev_step_reward[active_indices] = current_binary_reward[active_indices]
        else:
            # Active candidates can succeed at most once because success ends
            # the trajectory and later WM calls skip terminated candidates.
            emitted_reward[active_indices] = current_binary_reward[active_indices]
        chunk_rewards = torch.zeros_like(raw_probabilities)
        chunk_rewards[:, -1] = emitted_reward

        self.elapsed_steps += ACTION_HORIZON

        terminations = torch.zeros_like(chunk_rewards, dtype=torch.bool)
        terminations[:, -1] = chunk_successes
        truncations = torch.zeros_like(terminations)
        if self.elapsed_steps >= int(self.cfg.max_episode_steps):
            truncations[~self._done & ~chunk_successes, -1] = True
        self._done |= terminations.any(dim=1) | truncations.any(dim=1)

        if self.record_metrics:
            self.success_once |= chunk_successes
            self.returns += chunk_rewards.sum(dim=1)
            running_probability_mean = self._raw_probability_sum / (
                self._generated_frame_count.clamp_min(1).to(torch.float32)
            )
            group_probability_std = (
                running_probability_mean.view(self.num_groups, self.group_size)
                .std(dim=1)
                .repeat_interleave(self.group_size)
            )
            group_returns = self.returns.view(self.num_groups, self.group_size)
            group_means = group_returns.mean(dim=1)
            valid_groups = (group_means > 0.0) & (group_means < 1.0)
            episode = {
                "success_rate": self.success_once.clone(),
                "success_once": self.success_once.clone(),
                "return": self.returns.clone(),
                "episode_len": self._policy_chunks_executed.to(torch.float32)
                * ACTION_HORIZON,
                "reward_probability_mean": running_probability_mean,
                "reward_probability_group_std": group_probability_std,
                "reward_probability_chunk_mean": raw_probabilities.mean(dim=1),
                "valid_group_fraction": valid_groups.to(torch.float32)
                .mean()
                .expand(self.num_envs),
                "policy_chunks_executed": self._policy_chunks_executed.to(
                    torch.float32
                ),
                "wm_calls_executed": self._policy_chunks_executed.to(torch.float32)
                * (ACTION_HORIZON // WM_SUB_FRAMES),
            }
        else:
            episode = {}
        infos = {
            "episode": episode,
            "reward_probabilities": raw_probabilities,
            "reward_labels": labels,
            "wm_segment_count": 2 if active_count else 0,
            "wm_segment_frames": WM_SUB_FRAMES,
            "policy_chunk_index": self.elapsed_steps // ACTION_HORIZON,
            "trajectory_reward_aggregation": self.trajectory_reward_aggregation,
            "active_env_count": active_count,
        }
        if visualization_chunk is not None and self._visualizer is not None:
            sample_index = self._visualization_sample_env_index
            try:
                self._visualizer.append(
                    rgb_panel_video=visualization_chunk["rgb"],
                    tactile_panel_video=visualization_chunk["tactile"],
                    motion_panel_video=visualization_chunk["motion"],
                    probabilities=raw_probabilities[sample_index]
                    .detach()
                    .cpu()
                    .numpy(),
                    labels=labels[sample_index].detach().cpu().numpy(),
                    frame_rewards=chunk_rewards[sample_index]
                    .detach()
                    .cpu()
                    .numpy(),
                )
                if bool(self._done[sample_index]):
                    sample_success = bool(chunk_successes[sample_index])
                    video_path = self._visualizer.finish(
                        success=sample_success,
                        trajectory_return=float(sample_success),
                    )
                    print(
                        f"[Shiguan rollout visualization] saved {video_path}",
                        flush=True,
                    )
                    self._visualization_active = False
            except Exception as exc:
                self._visualization_failure(exc)
        return [self.current_obs], chunk_rewards, terminations, truncations, [infos]

    def offload(self) -> None:
        if self._is_offloaded:
            return
        if hasattr(self.client, "offload"):
            self.client.offload()
        self.reward_scorer.to("cpu")
        if self.current_obs is not None:
            self.current_obs = {
                key: value.cpu() if torch.is_tensor(value) else value
                for key, value in self.current_obs.items()
            }
        self.prev_step_reward = self.prev_step_reward.cpu()
        self._raw_probability_sum = self._raw_probability_sum.cpu()
        self._generated_frame_count = self._generated_frame_count.cpu()
        self._policy_chunks_executed = self._policy_chunks_executed.cpu()
        self._done = self._done.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        self._clear_accelerator_cache()
        self._is_offloaded = True

    def onload(self) -> None:
        if not self._is_offloaded:
            return
        if hasattr(self.client, "onload"):
            self.client.onload()
        self.reward_scorer.to(self.device)
        if self.current_obs is not None:
            self.current_obs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in self.current_obs.items()
            }
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self._raw_probability_sum = self._raw_probability_sum.to(self.device)
        self._generated_frame_count = self._generated_frame_count.to(self.device)
        self._policy_chunks_executed = self._policy_chunks_executed.to(self.device)
        self._done = self._done.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False

    def close(self) -> None:
        if self._visualizer is not None:
            self._visualizer.abort()
        self.client.close()

    def get_state(self) -> dict[str, Any]:
        """Return compact diagnostics; authoritative WM tensors stay local."""

        return {
            "session_ids": tuple(self.session_ids),
            "elapsed_steps": self.elapsed_steps,
            "prompt": self.prompt,
            "env_worker_rank": self.env_worker_rank,
            "wm_transport": "inprocess",
            "wm_offloaded": bool(getattr(self.client, "is_offloaded", False)),
        }
