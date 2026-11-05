"""No GPU/model required: verify shared inference/GRPO closed-loop semantics."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from physcanvas_pi05.backend import PhysCanvasBackend
from physcanvas_pi05.motion import MotionRenderer


class FakeDataset:
    def __len__(self):
        return 3

    def read(self, index):
        return {
            "episode_id": f"sample_{index}",
            "rgb": np.full((3, 224, 224, 3), 20 + index, np.uint8),
            "tactile_diff": np.full((4, 350, 200, 3), 127, np.uint8),
            "tactile_xyz": np.zeros((4, 700, 3), np.float32),
            "qpos": np.arange(14, dtype=np.float32),
            "initial_label": 0,
        }


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def frames(self, qpos, episode_id):
        self.calls.append((qpos.copy(), episode_id))
        return np.full((9, 224, 672, 3), 190, np.uint8)


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def predict(self, session, replay_visual):
        self.calls.append({
            "local": session.visual.clone(),
            "state": session.state_canvas.clone(),
            "anchor": session.anchor_rgb.clone(),
            "frames": session.frames, "seed": session.seed,
            "replay_shape": tuple(replay_visual.shape),
        })
        result = session.visual.expand(1, 9, 3, 672, 672).clone()
        state = session.state_canvas.expand(1, 9, 3, 32, 672).clone()
        for frame in range(1, 9):
            absolute = session.frames + frame
            result[:, frame, :, :224] = absolute * .01
            result[:, frame, 0, 448:] = .1
            result[:, frame, 1, 448:] = -.2
            result[:, frame, 2, 448:] = .3 + absolute * .001
            state[:, frame] = .123456 + absolute * .001
        self.calls[-1]["last_visual"] = result[:, -1:].clone()
        self.calls[-1]["last_state"] = state[:, -1:].clone()
        return result, state

    def onload(self):
        pass

    def offload(self):
        pass


class BackendTests(unittest.TestCase):
    def make(self):
        runtime, renderer = FakeRuntime(), FakeRenderer()
        backend = PhysCanvasBackend({"task": "jimu"}, device="cpu", seed=42,
                                    prompt="sort blocks", runtime=runtime,
                                    renderer=renderer, dataset=FakeDataset())
        return backend, runtime, renderer

    def test_two_chunks_four_wm_calls_use_generated_feedback(self):
        backend, runtime, renderer = self.make()
        reset = backend.reset_groups(num_groups=1, group_size=1, prompt="sort blocks")
        sid = reset["session_ids"][0]
        reset_rgb = reset["observations"][0]["observation.images.base_0"].copy()
        actions = np.arange(32 * 14, dtype=np.float32).reshape(32, 14) / 100
        outputs = []
        for start in (0, 16):
            outputs.append(backend.rollout16([sid], actions[None, start:start + 16],
                                            audit_session_id=sid,
                                            return_policy_tactile_video=True))
        self.assertEqual(len(runtime.calls), 4)
        self.assertEqual([x["frames"] for x in runtime.calls], [0, 8, 16, 24])
        for previous, current in zip(runtime.calls, runtime.calls[1:]):
            torch.testing.assert_close(current["local"], previous["last_visual"], rtol=0, atol=0)
            torch.testing.assert_close(current["state"], previous["last_state"], rtol=0, atol=0)
        for call in runtime.calls:
            torch.testing.assert_close(call["anchor"], runtime.calls[0]["anchor"], rtol=0, atol=0)
            self.assertEqual(call["replay_shape"], (1, 9, 3, 672, 672))
        # A fractional state must be forwarded exactly, not reclassified/rendered.
        torch.testing.assert_close(runtime.calls[1]["state"], torch.full((1, 1, 3, 32, 672), .131456))
        for call_index, (qpos9, _) in enumerate(renderer.calls):
            first = np.arange(14, dtype=np.float32) if call_index == 0 else actions[call_index * 8 - 1]
            np.testing.assert_array_equal(qpos9[0], first)
            np.testing.assert_array_equal(qpos9[1:], actions[call_index * 8:call_index * 8 + 8])
        for chunk_index, output in enumerate(outputs):
            self.assertEqual(output["rgb_panel_video"].shape, (1, 16, 224, 672, 3))
            self.assertEqual(output["policy_tactile_video"].shape, (1, 16, 4, 700, 3))
            self.assertEqual(output["audit_motion_panel_video"].shape[0], 16)
            self.assertFalse(np.array_equal(output["rgb_panel_video"][0, 0, :, :224].transpose(2, 0, 1), reset_rgb))
            expected_dz = (.3 + (chunk_index * 16 + 16) * .001) * .06
            np.testing.assert_allclose(output["observations"][0]["observation.tactile"][..., 2], expected_dz, atol=1e-7)
        np.testing.assert_array_equal(backend.sessions[sid].qpos, actions[-1])
        self.assertEqual(backend.sessions[sid].frames, 32)

    def test_group_same_seed_independent_state(self):
        backend, _, _ = self.make()
        reset = backend.reset_groups(num_groups=2, group_size=2, prompt="sort blocks")
        s0, s1, s2, _ = [backend.sessions[sid] for sid in reset["session_ids"]]
        self.assertEqual(s0.episode_id, s1.episode_id)
        self.assertEqual(s0.seed, s1.seed)
        self.assertNotEqual(s0.seed, s2.seed)
        self.assertNotEqual(s0.visual.data_ptr(), s1.visual.data_ptr())
        self.assertNotEqual(s0.state_canvas.data_ptr(), s1.state_canvas.data_ptr())
        before = s1.visual.clone()
        before_state = s1.state_canvas.clone()
        backend.rollout16(reset["session_ids"][:1], np.zeros((1, 16, 14), np.float32))
        torch.testing.assert_close(s1.visual, before, rtol=0, atol=0)
        torch.testing.assert_close(s1.state_canvas, before_state, rtol=0, atol=0)
        self.assertEqual(s1.frames, 0)

    def test_real_reset_pixels_are_not_roundtrip_quantized(self):
        backend, _, _ = self.make()
        original = FakeDataset().read(0)
        original["rgb"][:] = np.arange(224, dtype=np.uint8)[None, :, None, None]
        with mock.patch.object(backend.dataset, "read", return_value=original):
            reset = backend.reset_groups(num_groups=1, group_size=1, prompt="sort blocks")
        for i, camera in enumerate(("base_0", "left_wrist_0", "right_wrist_0")):
            np.testing.assert_array_equal(
                reset["observations"][0][f"observation.images.{camera}"],
                original["rgb"][i].transpose(2, 0, 1),
            )

    def test_unknown_id_nan_and_wrong_shapes_fail_before_wm(self):
        backend, runtime, _ = self.make()
        sid = backend.reset_groups(num_groups=1, group_size=1, prompt="sort blocks")["session_ids"][0]
        actions = np.zeros((1, 16, 14), np.float32)
        with self.assertRaises(KeyError):
            backend.rollout16(["unknown"], actions)
        bad = actions.copy()
        bad[0, 7, 3] = np.nan
        with self.assertRaises(ValueError):
            backend.rollout16([sid], bad)
        with self.assertRaises(ValueError):
            backend.rollout16([sid], actions[:, :8])
        with self.assertRaises(ValueError):
            backend.rollout16([sid, sid], np.zeros((2, 16, 14), np.float32))
        self.assertEqual(runtime.calls, [])
        old = sid
        backend.reset_groups(num_groups=1, group_size=1, prompt="sort blocks")
        with self.assertRaises(KeyError):
            backend.rollout16([old], actions)


class MotionSettingsTests(unittest.TestCase):
    def test_settings_file_loaded_and_old_cache_forcibly_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = {key: tmp for key in ("debug_dir", "x5a_root", "urdf_path", "fk_offset_path", "left_calib", "right_calib")}
            settings.update(render_size_hw=[224, 224], use_cache=True, cache_dir=tmp)
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps(settings))
            with mock.patch("PhysCanvas.motion.arxx5_link678_gray_renderer.ARXLink678GrayMotionRenderer") as factory:
                renderer = MotionRenderer({"settings_file": str(path)})
            called = factory.call_args.args[0]
            self.assertFalse(called["use_cache"])
            self.assertIsNone(called["cache_dir"])
            self.assertNotIn("settings_file", called)
            self.assertEqual(renderer.render_size_hw, (224, 224))


if __name__ == "__main__":
    unittest.main()
