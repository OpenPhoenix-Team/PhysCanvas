"""Exercise the complete standalone driver with tiny CPU-only model doubles."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from physcanvas_pi05 import rollout, wm_runtime


class RolloutTests(unittest.TestCase):
    def test_300_future_frames_no_rm(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as st:
            out = Path(tmp) / "new"
            env = {k: tmp for k in ("PHYSCANVAS_ROOT", "PHYSCANVAS_ASSETS", "PI05_CHECKPOINT",
                   "PI05_NORM_STATS", "RESET_CACHE", "RESET_SELECTION", "VAE_PATH",
                   "PROMPT_EMBEDDING", "MOTION_SETTINGS")}
            st.enter_context(patch.dict(os.environ, env))
            os.environ.pop("RM_CHECKPOINT", None)
            obs = {"wm": {"episode_id": "episode", "generated_frame_index": 0},
                   "observation.state": np.zeros(14), "observation.tactile": np.zeros(1)}
            for v in ("base_0", "left_wrist_0", "right_wrist_0"):
                obs["observation.images." + v] = np.zeros((3, 224, 224), np.uint8)
            backend = MagicMock()
            backend.dataset = [None]
            backend.reset_groups.return_value = {"session_ids": ["s"], "observations": [obs]}
            def advance(ids, actions, *, frames, audit_session_id):
                obs["wm"]["generated_frame_index"] += frames
                obs["observation.state"] = np.full(14, obs["wm"]["generated_frame_index"])
                panels = np.broadcast_to(np.uint8(0), (1, frames, 224, 672, 3))
                return {"observations": [obs], "rgb_panel_video": panels, "audit_motion_panel_video": panels[0]}
            backend.rollout16.side_effect = advance
            policy = MagicMock()
            policy.eval.return_value = policy.requires_grad_.return_value = policy.cpu.return_value = policy
            policy.predict_action_batch.return_value = (torch.zeros(1, 16, 14), {})
            st.enter_context(patch.dict(sys.modules, {
                "rlinf.models.embodiment.openpi_rlinf": SimpleNamespace(get_model=lambda cfg: policy),
                "rlinf.envs.world_model.jimu_wm_protocol": SimpleNamespace(stack_policy_observations=lambda *a, **kw: obs),
                "rlinf.envs.world_model.jimu_reward_model": None,
            }))
            video = SimpleNamespace(frames=0, close=lambda: None)
            video.write = lambda *a: setattr(video, "frames", video.frames + 1)
            st.enter_context(patch.object(rollout, "AuditVideo", return_value=video))
            st.enter_context(patch.object(rollout, "PhysCanvasBackend", return_value=backend))
            st.enter_context(patch.object(wm_runtime, "PublishedWmRuntime"))
            st.enter_context(patch.object(wm_runtime, "validate_public_contract", return_value={}))
            st.enter_context(patch.object(sys, "argv", ["rollout", "--task", "test_tube", "--frames", "300",
                                                      "--device", "cpu", "--output-dir", str(out)]))
            with contextlib.redirect_stdout(io.StringIO()):
                rollout.main()
            sample = out / "sample_000"
            obj = json.loads((sample / "result.json").read_text())
            self.assertEqual((obj["generated_frames"], obj["video_frames"], obj["wm_computed_frames"]), (300, 301, 304))
            self.assertEqual(np.load(sample / "actions.npy").shape, (300, 14))
            self.assertEqual([c["executed_frames"] for c in obj["chunks"]], [16]*18+[12])
            self.assertEqual(policy.predict_action_batch.call_count, 19)
            self.assertTrue(all(c.kwargs["mode"] == "eval" for c in policy.predict_action_batch.call_args_list))
            self.assertFalse(obj["rm_scoring"])
            self.assertNotIn("success", obj)
            self.assertFalse((sample / "rewards.csv").exists())
