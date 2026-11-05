import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.cli import run_extra_metrics
from evaluation.tactile_events import event_metrics, tactile_event_metrics
from evaluation.tracking_eval import point_error
from evaluation.vbench_eval import DIMENSIONS, command, validate_inputs


class OptionalMetricTests(unittest.TestCase):
    def test_event_matching_and_undefined_case(self):
        gt = [False, True, True, True, False]
        pred = [False, False, True, True, False]
        result = event_metrics(gt, pred, fps=10)
        self.assertEqual(result["event_f1"], 1)
        self.assertAlmostEqual(result["onset_error_sec"], 0.1)
        self.assertEqual(result["offset_error_sec"], 0)
        self.assertIsNone(event_metrics([False] * 5, [False] * 5)["event_f1"])

    def test_contact_threshold_is_required_and_model_space_score(self):
        gt = np.zeros((5, 3, 2, 8), dtype=np.float32)
        pred = gt.copy()
        gt[1:4] = 0.4
        pred[2:4] = 0.4
        flow_gt = np.zeros((5, 4, 2, 3), dtype=np.float32)
        flow_pred = flow_gt.copy()
        flow_gt[1:4, :, 0, 0] = 0.01
        flow_pred[2:4, :, 0, 0] = 0.01
        points = np.ones((4, 2, 2), dtype=np.float32)
        thresholds = {"frozen_from_gt_only": True, "image_space": "model_space_01", "fps": 10,
                      "diff_threshold_per_side": [0.1] * 4,
                      "diff_event_fraction_per_side": [0.1] * 4,
                      "flow_event_fraction_per_side": [0.1] * 4}
        result = tactile_event_metrics(gt, pred, flow_gt, flow_pred, points, [0.005] * 4, thresholds)
        side = result["sides"]["left_wrist_0"]
        self.assertAlmostEqual(side["contact_mae_01"], 0.2 / 3)
        self.assertEqual(side["contact_events"]["event_f1"], 1)
        self.assertEqual(side["flow_events"]["event_f1"], 1)
        with self.assertRaises(ValueError):
            tactile_event_metrics(gt, pred, flow_gt, flow_pred, points, [0.005] * 4, {})

    def test_paired_track_error_uses_mutual_visibility(self):
        gt = np.zeros((3, 2, 2))
        pred = gt.copy()
        pred[1, 0, 0] = 3
        pred[2, 1, 1] = 4
        gv = np.ones((3, 2), dtype=bool)
        pv = gv.copy()
        pv[2, 1] = False
        result = point_error(gt, pred, gv, pv, ["eef", "object"])
        self.assertEqual(result["eef"]["mean_error_px"], 1.5)
        self.assertEqual(result["object"]["mean_error_px"], 0)
        self.assertEqual(result["object"]["coverage"], 0.5)

    def test_vbench_requires_clean_fixed_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_spec.json").write_text(json.dumps({"export_clean_rgb": True, "horizon": "n128"}))
            (root / "COMPLETED.json").write_text("{}")
            (root / "selected_records.json").write_text(json.dumps({"records": [
                {"episode_id": "example", "sample_num_frames": 200}
            ]}))
            videos = root / "evaluation" / "clean_rgb"
            videos.mkdir(parents=True)
            (videos / "example_128f.mp4").touch()
            _, records, selected = validate_inputs(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(selected, videos.resolve())
            (root / "evaluate.py").touch()
            with patch.dict(os.environ, {"VBENCH_CACHE_DIR": ""}):
                args = command(root, videos, root / "out")
            self.assertEqual(args[-3:], list(DIMENSIONS))
            (root / "evaluate.py").unlink()
            nested = root / "vbench" / "launch" / "evaluate.py"
            nested.parent.mkdir(parents=True)
            nested.touch()
            self.assertEqual(command(root, videos, root / "out")[1], str(nested))
            with patch.dict(os.environ, {"VBENCH_CACHE_DIR": str(root)}):
                self.assertEqual(command(root, videos, root / "out")[-2:],
                                 ["--load_ckpt_from_local", "True"])

    def test_unified_summary_reuses_matching_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            spec = {"episode_ids": ["example"], "horizon": "n128", "export_clean_rgb": True}
            (run_dir / "COMPLETED.json").write_text("{}")
            (run_dir / "metrics_aggregate.json").write_text(json.dumps({"overall": {"episodes": 1}}))
            vbench = run_dir / "vbench"
            vbench.mkdir()
            (vbench / "physcanvas_summary.json").write_text(json.dumps({
                "source_run_spec": spec, "episodes": 1, "dimensions": {"subject_consistency_x100": 80.0}
            }))
            checkpoint = run_dir / "cotracker.pth"
            queries = run_dir / "queries.json"
            checkpoint.write_bytes(b"weights")
            queries.write_text("{}")
            (run_dir / "tracking_summary.json").write_text(json.dumps({
                "source_run_spec": spec, "episodes": 1,
                "queries_sha256": hashlib.sha256(queries.read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "aggregate": {"eef": {"coverage": 1.0}}
            }))
            args = argparse.Namespace(mode="metrics-128", vbench_repo=run_dir,
                                      cotracker_repo=run_dir, cotracker_checkpoint=checkpoint,
                                      queries_json=queries, device="cpu")
            run_extra_metrics(args, run_dir, spec)
            combined = json.loads((run_dir / "evaluation_suite.json").read_text())
            self.assertEqual(combined["base"]["overall"]["episodes"], 1)
            self.assertEqual(combined["vbench"]["dimensions"]["subject_consistency_x100"], 80.0)
            self.assertEqual(combined["tracking"]["aggregate"]["eef"]["coverage"], 1.0)
            with self.assertRaises(ValueError):
                run_extra_metrics(args, run_dir, {**spec, "episode_ids": ["different"]})


if __name__ == "__main__":
    unittest.main()
