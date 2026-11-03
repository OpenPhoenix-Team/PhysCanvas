import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.aggregate import frechet_distance, weighted_image_summary
from evaluation.cli import parse_horizons, run, select_cohort
from tools.build_example_cohort import build
from tools.cohort_fixture import TASK_SLUGS, archive_items, archive_name, archive_path
from tools.raw_fixture import nearest_indices


TASKS = ("bottle", "shiguan", "jimu", "mofang")


class CohortTests(unittest.TestCase):
    def test_fixed_public_cohort(self):
        path = Path(__file__).parents[1] / "evaluation" / "data" / "fixed_20_ids.json"
        cohort = json.loads(path.read_text())["episodes"]
        self.assertEqual(len(cohort), 20)
        self.assertEqual(len({row["episode_id"] for row in cohort}), 20)
        self.assertEqual({task: sum(row["task"] == task for row in cohort) for task in TASKS},
                         dict.fromkeys(TASKS, 5))
        self.assertTrue(all(row["source_frames"] >= 128 for row in cohort))
        self.assertEqual(set().union(*(row.keys() for row in cohort)),
                         {"episode_id", "task", "source_frames"})

    def test_selection_matches_ids_and_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"episode_id": f"{task}-{index}", "task": task,
                     "source_frames": 128 if index < 3 else 256}
                    for task in TASKS for index in range(5)]
            records = [{"episode_id": row["episode_id"], "object_state_task": row["task"],
                        "sample_num_frames": row["source_frames"]} for row in rows]
            cohort_path, records_path = root / "cohort.json", root / "records.json"
            cohort_path.write_text(json.dumps({"episodes": rows}))
            records_path.write_text(json.dumps({"records": list(reversed(records))}))
            self.assertEqual(len(select_cohort(cohort_path, records_path, 128)), 20)
            self.assertEqual(len(select_cohort(cohort_path, records_path, 256)), 8)
            self.assertEqual(len(select_cohort(cohort_path, records_path, None)), 20)
            records[0]["sample_num_frames"] = 129
            records_path.write_text(json.dumps({"records": records}))
            with self.assertRaisesRegex(ValueError, "frame count mismatch"):
                select_cohort(cohort_path, records_path, 128)

    def test_horizons(self):
        self.assertEqual(parse_horizons("128,192,256,full"), [128, 192, 256, None])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_horizons("128,128")

    def test_archived_example_ids_match(self):
        manifest_name = os.environ.get("PHYSCANVAS_EVAL_MANIFEST")
        if not manifest_name:
            self.skipTest("set PHYSCANVAS_EVAL_MANIFEST for an optional ID audit")
        index_file = os.environ.get("PHYSCANVAS_EVAL_INDICES")
        if not index_file:
            self.skipTest("set PHYSCANVAS_EVAL_INDICES for the optional private selection audit")
        expected = build(Path(manifest_name), Path(index_file))
        cohort = json.loads((Path(__file__).parents[1] / "evaluation" / "data" / "fixed_20_ids.json").read_text())
        self.assertEqual(cohort, expected)


class AggregateTests(unittest.TestCase):
    def test_identical_frechet_is_zero(self):
        features = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 2.0]])
        self.assertAlmostEqual(frechet_distance(features, features), 0.0, places=6)
        self.assertIsNone(frechet_distance(features[:1], features[:1]))

    def test_image_weighting_excludes_frame_zero(self):
        records = [
            {"metrics": {"images": {"rgb": {"frames": 127, "mse_01": 0.1,
                                               "psnr": 10.0, "ssim": 0.5, "lpips": 0.2}}}},
            {"metrics": {"images": {"rgb": {"frames": 127, "mse_01": 0.2,
                                               "psnr": 20.0, "ssim": 0.7, "lpips": 0.3}}}},
        ]
        summary = weighted_image_summary(records, "rgb")
        self.assertEqual(summary["future_frames"], 254)
        self.assertAlmostEqual(summary["psnr"], 15.0)
        self.assertAlmostEqual(summary["mse_01"], 0.15)


class RawFixtureTests(unittest.TestCase):
    def test_nearest_timestamp_alignment(self):
        source = np.array([100, 200, 300], dtype=np.int64)
        target = np.array([100, 149, 150, 251, 500], dtype=np.int64)
        np.testing.assert_array_equal(nearest_indices(source, target), [0, 0, 0, 2, 2])

    def test_public_config_has_no_private_paths(self):
        path = Path(__file__).parents[1] / "evaluation" / "data" / "public_eval_config.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["training_config"]["models"]["flow_shift"], 5.0)
        self.assertEqual(config["training_config"]["dataloaders"]["test"]["data_or_config"]["require_depth"], False)
        self.assertNotIn("/mnt/", path.read_text())


class ArchiveOnlyTests(unittest.TestCase):
    def test_named_metric_suites_fail_before_inference_when_inputs_are_incomplete(self):
        common = ["--mode", "metrics-128", "--checkpoint", "/tmp/model",
                  "--config-json", "/tmp/config", "--output-dir", "/tmp/output"]
        with self.assertRaisesRegex(ValueError, "does not accept optional metric assets"):
            run([*common, "--evaluation-suite", "reconstruction",
                 "--vbench-repo", "/tmp/vbench"])
        with self.assertRaisesRegex(ValueError, "requires contact thresholds"):
            run([*common, "--evaluation-suite", "all"])

    def test_public_task_names_are_neutral(self):
        self.assertEqual(set(TASK_SLUGS.values()),
                         {"bottle-cap", "test-tube", "building-blocks", "rubiks-cube"})
        self.assertEqual(archive_name(5, "shiguan"), "06_shiguan_n128.tar.gz")

    def test_archive_path_rejects_traversal(self):
        with self.assertRaisesRegex(ValueError, "unsafe archive path"):
            archive_path(Path("/tmp/data"), "example.tar.gz", {"path": "../outside.tar.gz"})

    def test_archive_path_accepts_downloaded_full_episode_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "full-episode"
            archive = source / "bottle-cap" / "01_bottle-cap_full.tar.gz"
            archive.parent.mkdir(parents=True)
            archive.touch()
            self.assertEqual(
                archive_path(source, "01_bottle_full.tar.gz", {
                    "path": "evaluation-data/full-episode/bottle-cap/01_bottle-cap_full.tar.gz"
                }), archive)

    def test_archive_mode_rejects_external_records(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            run(["--mode", "metrics-128", "--archives-dir", "/tmp/archives",
                 "--records-json", "/tmp/external.json", "--checkpoint", "/tmp/model",
                 "--config-json", "/tmp/config", "--output-dir", "/tmp/output"])

    def test_public_mode_requires_archives(self):
        with self.assertRaisesRegex(ValueError, "external records require"):
            run(["--mode", "metrics-128", "--records-json", "/tmp/external.json",
                 "--checkpoint", "/tmp/model", "--config-json", "/tmp/config",
                 "--output-dir", "/tmp/output"])

    def test_archive_mode_rejects_longer_horizon(self):
        with patch("evaluation.cli.verify_archives", return_value={"schema": "physcanvas_cohort_n128_v1"}):
            with self.assertRaisesRegex(ValueError, "128 prepared frames"):
                run(["--mode", "infer-all-n", "--archives-dir", "/tmp/archives",
                     "--horizons", "192", "--checkpoint", "/tmp/model",
                     "--config-json", "/tmp/config", "--output-dir", "/tmp/output"])

    def test_full_episode_index_preserves_variable_lengths(self):
        index = {"schema": "physcanvas_full_episode_cohort_v1", "archives": {
            "01_bottle_full.tar.gz": {"episode_id": "one", "task": "bottle", "frames": 249},
            "02_shiguan_full.tar.gz": {"episode_id": "two", "task": "shiguan", "frames": 188},
        }}
        items = archive_items(index)
        self.assertEqual([item[1]["source_frames"] for item in items], [249, 188])


if __name__ == "__main__":
    unittest.main()
