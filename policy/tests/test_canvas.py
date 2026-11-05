import unittest

import numpy as np
import torch

from physcanvas_pi05.canvas import (
    build_motion_replay, build_state, build_state_replay, build_visual,
    decode_policy_visual, normalize_rgb,
)


class CanvasTests(unittest.TestCase):
    @staticmethod
    def reset():
        rgb = np.stack([np.full((224, 224, 3), value, np.uint8) for value in (0, 127, 255)])
        diff = np.stack([np.full((350, 200, 3), value, np.uint8) for value in (20, 60, 120, 220)])
        xyz = np.zeros((4, 700, 3), np.float32)
        xyz[:, :, 0] = np.array([.01, .02, .03, .04])[:, None]
        xyz[:, :, 1] = -.005
        xyz[:, :, 2] = .015
        return {"rgb": rgb, "tactile_diff": diff, "tactile_xyz": xyz}

    def test_visual_layout_and_xyz_roundtrip(self):
        reset = self.reset()
        visual = build_visual(reset)
        self.assertEqual(tuple(visual.shape), (1, 1, 3, 672, 672))
        self.assertEqual(visual.dtype, torch.float32)
        rgb, xyz = decode_policy_visual(visual)
        self.assertEqual(rgb.shape, (1, 224, 672, 3))
        self.assertEqual(xyz.shape, (1, 4, 700, 3))
        for view in range(3):
            np.testing.assert_allclose(rgb[0, :, 224 * view:224 * (view + 1)], reset["rgb"][view], atol=1)
        # Packing/resizing twice slightly mixes markers on side boundaries;
        # interior marker rows/columns must retain sensor and XYZ order.
        np.testing.assert_allclose(xyz.reshape(1, 4, 35, 20, 3)[0, :, :, 2:-2],
                                   reset["tactile_xyz"].reshape(4, 35, 20, 3)[:, :, 2:-2], atol=1e-7)
        self.assertTrue(np.all(xyz[..., 2] > 0), "dz must survive, not be zero-filled")

    def test_predicted_flow_is_not_clipped(self):
        visual = torch.zeros(1, 2, 3, 672, 672)
        visual[:, :, 2, 448:] = 2.0
        _, xyz = decode_policy_visual(visual)
        np.testing.assert_allclose(xyz[..., 2], .12)

    def test_motion_repeated_three_times(self):
        motion = np.full((9, 224, 672, 3), 190, np.uint8)
        motion[:, 80:120, 40:140] = 55
        replay = build_motion_replay(motion)
        self.assertEqual(tuple(replay.shape), (1, 9, 3, 672, 672))
        torch.testing.assert_close(replay[..., :224, :], replay[..., 224:448, :])
        torch.testing.assert_close(replay[..., :224, :], replay[..., 448:672, :])
        self.assertEqual(tuple(build_state_replay(motion).shape), (1, 9, 3, 32, 672))

    def test_rgb_uses_padding_and_no_double_padding(self):
        wide = np.full((1, 480, 640, 3), 255, np.uint8)
        normalized = normalize_rgb(wide)
        self.assertTrue(torch.all(normalized[:, :, :28] == -1))
        self.assertTrue(torch.all(normalized[:, :, 28:196] == 1))
        small = np.full((1, 224, 224, 3), 255, np.uint8)
        self.assertTrue(torch.all(normalize_rgb(small) == 1))

    def test_task_state_canvas_is_public_implementation(self):
        from PhysCanvas.object_state_canvas import CanvasSpec, render_state_sequence
        for task in ("jimu", "shiguan"):
            for state in (0, 1):
                expected = render_state_sequence([state], task, CanvasSpec())[None]
                torch.testing.assert_close(build_state(state, task), expected)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            build_motion_replay(np.zeros((8, 224, 672, 3), np.uint8))
        with self.assertRaises(ValueError):
            decode_policy_visual(torch.zeros(2, 1, 3, 672, 672))
        with self.assertRaises(ValueError):
            normalize_rgb(np.zeros((1, 224, 224, 3), np.float32))


if __name__ == "__main__":
    unittest.main()
