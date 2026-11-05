"""Exercise the actual vendored RLinf environment with a synthetic WM/RM.

These CPU tests verify reward/termination/group contracts; they are not neural
WM quality or GPU training tests.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from physcanvas_pi05.config import compose_config


def observation(value=0):
    return {
        "observation.images.base_0": np.full((3, 2, 2), value, np.uint8),
        "observation.images.left_wrist_0": np.full((3, 2, 2), value, np.uint8),
        "observation.images.right_wrist_0": np.full((3, 2, 2), value, np.uint8),
        "observation.state": np.full(14, value, np.float32),
        "observation.tactile": np.full((4, 700, 3), value, np.float32),
        "observation.tactile_mask": np.ones(4, bool),
    }


class FakeBackend:
    def __init__(self):
        self.calls = []

    def metadata(self):
        return {
            "transport": "inprocess", "sub_frames": 8, "state_dim": 14,
            "views": ["base", "left_wrist", "right_wrist"],
            "pi05_tactile_shape": [4, 700, 3], "policy_tactile_at_reset": True,
            "generated_policy_tactile": True,
            "generated_policy_tactile_third_channel": "xyz",
            "tactile_flow_encoding": "xyz_linear",
        }

    def reset_groups(self, *, num_groups, group_size, prompt):
        count = num_groups * group_size
        return {
            "session_ids": [str(i) for i in range(count)],
            "group_ids": [i // group_size for i in range(count)],
            "observations": [observation() for _ in range(count)],
        }

    def rollout16(self, session_ids, actions, **kwargs):
        self.calls.append((list(session_ids), actions.copy()))
        frame_value = len(self.calls)
        return {
            "session_ids": session_ids,
            "rgb_panel_video": np.full((len(session_ids), 16, 2, 6, 3), frame_value, np.uint8),
            "observations": [observation(frame_value) for _ in session_ids],
        }


class FakeScorer:
    threshold = .7

    def __init__(self):
        self.calls = 0

    def score_panel_video(self, video):
        probabilities = torch.full((len(video), 16), .1)
        if self.calls == 0:
            # Transient success in frame 3 suffices; emitted reward is step16.
            probabilities[0, 2] = .9
        self.calls += 1
        return probabilities, (probabilities >= self.threshold).float()


class EnvironmentTests(unittest.TestCase):
    def test_latest_feedback_and_chunk_end_success(self):
        from rlinf.envs.world_model.jimu_wm_grpo_env import JimuWmGrpoEnv

        placeholders = {name: "/placeholder" for name in [
            "PHYSCANVAS_ROOT", "PHYSCANVAS_ASSETS", "PI05_CHECKPOINT", "PI05_NORM_STATS",
            "RESET_CACHE", "RESET_SELECTION", "RM_CHECKPOINT", "VAE_PATH",
            "PROMPT_EMBEDDING", "MOTION_SETTINGS",
        ]}
        with patch.dict(os.environ, placeholders):
            for task in ["jimu", "shiguan"]:
                cfg = compose_config(task, [
                    "env.train.rollout_visualization.enabled=false",
                    "env.train.max_policy_chunks=2", "env.train.max_episode_steps=32",
                    "env.train.max_steps_per_rollout_epoch=32",
                ])
                backend = FakeBackend()
                env = JimuWmGrpoEnv(
                    cfg.env.train, num_envs=8, seed_offset=0, total_num_processes=1,
                    client=backend, reward_scorer=FakeScorer(),
                )
                initial, _ = env.reset()
                self.assertTrue(initial["main_images"].eq(0).all())
                actions = np.arange(8 * 16 * 14, dtype=np.float32).reshape(8, 16, 14)
                observations, rewards, terminal, truncated, _ = env.chunk_step(actions)
                self.assertEqual(rewards[0, -1].item(), 1)
                self.assertEqual(rewards[:, :-1].sum().item(), 0)
                self.assertTrue(terminal[0, -1])
                self.assertTrue(observations[0]["main_images"].eq(1).all())
                self.assertTrue(observations[0]["tactiles"][..., 2].eq(1).all())
                np.testing.assert_array_equal(backend.calls[0][1], actions)
                observations, rewards, terminal, truncated, _ = env.chunk_step(actions)
                self.assertEqual(backend.calls[1][0], [str(i) for i in range(1, 8)])
                self.assertEqual(rewards.sum().item(), 0)
                self.assertTrue(truncated[1:, -1].all())
                self.assertTrue(observations[0]["main_images"][1:].eq(2).all())
                self.assertTrue(observations[0]["main_images"][0].eq(1).all())
                self.assertEqual(env.returns.sum().item(), 1)


if __name__ == "__main__":
    unittest.main()
