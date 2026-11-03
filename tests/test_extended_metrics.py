import unittest

import numpy as np

from evaluation.aggregate import marker_displacement_summary
from evaluation.extended_metrics import marker_displacement_errors


class MarkerDisplacementTests(unittest.TestCase):
    def test_active_idle_and_amplitude_errors(self):
        gt = np.zeros((2, 4, 2, 3), dtype=np.float32)
        pred = np.zeros_like(gt)
        points = np.ones((4, 2, 2), dtype=np.float32)
        gt[:, :, 0, 0] = 0.01
        pred[:, :, 0, 0] = 0.02
        pred[:, :, 1, 1] = 0.003
        result = marker_displacement_errors(gt, pred, points, [0.005] * 4)
        self.assertAlmostEqual(result["active_mae"], 0.01 / 3)
        self.assertAlmostEqual(result["idle_fp_mae"], 0.003 / 3)
        self.assertAlmostEqual(result["active_amplitude_mae"], 0.01)
        self.assertEqual(sum(result["active_point_counts_per_side"]), 8)

    def test_invalid_gt_markers_are_excluded_and_nonfinite_prediction_fails(self):
        gt = np.zeros((1, 4, 2, 3), dtype=np.float32)
        pred = np.zeros_like(gt)
        points = np.ones((4, 2, 2), dtype=np.float32)
        points[0, 0] = 0
        result = marker_displacement_errors(gt, pred, points, [0.005] * 4)
        self.assertEqual(sum(result["idle_point_counts_per_side"]), 7)
        pred[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            marker_displacement_errors(gt, pred, points, [0.005] * 4)

    def test_micro_and_episode_macro_are_distinct(self):
        def episode(frames, error):
            gt = np.zeros((frames, 4, 1, 3), dtype=np.float32)
            pred = np.zeros_like(gt)
            gt[..., 0] = 0.01
            pred[..., 0] = 0.01 + error
            return {"metrics": {"marker_displacement": marker_displacement_errors(
                gt, pred, np.ones((4, 1, 2)), [0.005] * 4
            )}}

        result = marker_displacement_summary([episode(1, 0.003), episode(3, 0.009)])
        self.assertAlmostEqual(result["active_mae_micro"], 0.0075 / 3)
        self.assertAlmostEqual(result["active_mae_episode_macro"], 0.006 / 3)

    def test_z_displacement_is_scored_with_xy_marker_positions(self):
        gt = np.zeros((1, 4, 1, 3), dtype=np.float32)
        pred = np.zeros_like(gt)
        points = np.ones((4, 1, 2), dtype=np.float32)
        gt[..., 2] = 0.01
        pred[..., 2] = 0.02
        result = marker_displacement_errors(gt, pred, points, [0.005] * 4)
        self.assertAlmostEqual(result["active_mae"], 0.01 / 3)
        self.assertGreater(result["active_abs_sum_per_side_axis"][0][2], 0)


if __name__ == "__main__":
    unittest.main()
