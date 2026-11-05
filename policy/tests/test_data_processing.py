"""Synthetic tests execute the preserved algorithms, not reimplementations."""
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest

import cv2
import numpy as np

from data_processing.convert import ensure_new_output, execute, load_manifest, prepare
from data_processing.original import algorithms, file_sha256
from data_processing.validate import validate_bundle


def ramp(n, start, stop):
    q = np.zeros((n, 7), np.float32)
    q[start:stop, 0] = np.linspace(0, 1, stop - start)
    q[stop:, 0] = 1
    return q


def fixture(root, *, with_tactile=True, variants=("raw",)):
    episode = root / "raw" / "000001"
    for part in ("camera", "robot", "tactile"):
        (episode / part).mkdir(parents=True)
    n = 240
    ts = np.arange(n, dtype=np.int64) * 10 + 1000
    left, right = ramp(n, 45, 130), ramp(n, 100, 195)
    right[:, 1] = 5
    camera_ts = {name: ts.copy() for name in ("base_0", "left_wrist_0", "right_wrist_0")}
    robot = {"left": {"joint": left, "timestamps": ts}, "right": {"joint": right, "timestamps": ts}}
    for path, data in ((episode / "camera/timestamp.pkl", camera_ts), (episode / "robot/robot_state_dict.pkl", robot)):
        with path.open("wb") as handle:
            pickle.dump(data, handle)
    for name in ("base_0_rgb.mp4", "left_wrist_0_rgb.mp4", "right_wrist_0_rgb.mp4", "base_0_rgb_remove_hand.mp4"):
        writer = cv2.VideoWriter(str(episode / "camera" / name), cv2.VideoWriter_fourcc(*"mp4v"), 100, (64, 48))
        if not writer.isOpened():
            raise RuntimeError("Synthetic MP4 writer unavailable")
        for i in range(n):
            image = np.full((48, 64, 3), i % 200 + 20, np.uint8)
            writer.write(image)
        writer.release()
    if with_tactile:
        points, times = {}, {}
        for sensor_index, sensor in enumerate(("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")):
            stamps = ts if sensor_index < 3 else ts[::80]
            values = np.zeros((len(stamps), 700, 6), np.float32)
            values[..., :3] = 123  # Must not be used as displacement.
            for axis, factor in enumerate((1, 2, 3)):
                values[..., axis + 3] = ((stamps - 1000) * factor / 100000)[:, None] + sensor_index
            points[sensor] = {"data": values}
            times[sensor] = stamps
        for path, data in ((episode / "tactile/tactile_pointcloud_dict.pkl", points), (episode / "tactile/timestamp.pkl", times)):
            with path.open("wb") as handle:
                pickle.dump(data, handle)
    row = {"episode_dir": str(episode), "physical_episode_id": "synthetic-000001",
           "task": "shiguan_shuang", "prompt": "Use both arms collaboratively to pick up the test tube and place it on the test tube rack.",
           "arm_mode": "bimanual", "swap_robot_sides": False, "head_variants": list(variants)}
    manifest = root / "sources.json"
    manifest.write_text(json.dumps({"version": 1, "repo_id": "test/synthetic", "episodes": [row]}))
    return manifest


class DataProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner, cls.writer, cls.tactile = algorithms()

    def test_trim_bimanual_union_and_static_fallback(self):
        left, right = ramp(300, 45, 145), ramp(300, 125, 245)
        both = self.planner.compute_task_aware_trim_bounds(np.concatenate([left, right], 1), arm_mode="bimanual")
        only = self.planner.compute_task_aware_trim_bounds(np.concatenate([left, right], 1), arm_mode="right")
        self.assertLess(both.trim_start, only.trim_start)
        self.assertGreater(both.trim_end, 220)
        still = self.planner.compute_task_aware_trim_bounds(np.zeros((300, 14)), arm_mode="bimanual")
        self.assertEqual((still.trim_start, still.trim_end, still.reason), (0, 300, "no_active_run"))

    def test_alignment_tie_and_50ms_filter(self):
        np.testing.assert_array_equal(self.planner.nearest_indices(np.array([0, 100]), np.array([50, 51])), [0, 1])
        target = np.array([0, 100, 200, 300], np.int64)
        streams = {name: target for name in ("base_0", "left_wrist_0", "right_wrist_0")}
        streams["right_wrist_0"] = np.array([0, 200, 300])
        kept, *_ = self.planner._filter_alignment_by_slop(target, streams, target, target, 50)
        np.testing.assert_array_equal(kept, [0, 200, 300])

    def test_output_paths_and_explicit_manifest_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            rows = [{"episode_dir": str(source)}]
            for target in (source, root, source / "new"):
                with self.assertRaises((ValueError, FileExistsError)):
                    ensure_new_output(target, rows)
            self.assertEqual(ensure_new_output(root / "new", rows), root / "new")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"version": 1, "repo_id": "x", "episodes": [{"episode_dir": str(source)}]}))
            with self.assertRaises(ValueError):
                load_manifest(manifest)

    def test_plan_next_state_and_explicit_swap_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = fixture(root)
            payload = load_manifest(manifest)
            with self.assertRaises(PermissionError):
                prepare(payload, trusted_pickle=False)
            plans, _ = prepare(payload, trusted_pickle=True)
            plan = plans[0]
            self.assertGreater(plan.timestamps_ms[0], plan.pre_crop_start_ms)
            np.testing.assert_array_equal(plan.actions[:-1], plan.states[1:])
            swapped = self.planner.make_episode_plan(plan.episode_dir, target_hz=10, arm_mode="bimanual", swap_robot_sides=True)
            np.testing.assert_array_equal(swapped.states[:, :7], plan.states[:, 7:])
            np.testing.assert_array_equal(swapped.states[:, 7:], plan.states[:, :7])
            self.assertFalse((root / "output").exists())

    def test_original_sidecar_xyz_precrop_reference_and_stale_mask(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = fixture(root)
            payload = load_manifest(manifest)
            plan = prepare(payload, trusted_pickle=True)[0][0]
            output = root / "sidecar"
            output.mkdir()
            row = {"global_episode_index": 0, "frames_after_filtering": len(plan.states),
                   "trim": self.planner.trim_to_dict(plan.trim), "arm_mode": "bimanual", "swap_robot_sides": False}
            job = {"source_id": "fixture", "source_index": 0, "episode_dir": str(plan.episode_dir),
                   "lerobot_root": str(root / "unused"), "output_root": str(output), "variant_rows": [row],
                   "target_hz": 10, "max_tactile_skew_ms": 60, "state_atol": 1e-6}
            with mock.patch.object(self.tactile, "verify_plan_against_cache"):
                result = self.tactile.process_source(job)
            xyz = np.load(output / result["tactile_relpath"])
            mask = np.load(output / result["tactile_mask_relpath"])
            self.assertEqual(result["reference_time_ms"], plan.pre_crop_start_ms)
            self.assertTrue(np.any(xyz[0, 0, :, 2] != 0), "Do not re-zero at trimmed frame")
            np.testing.assert_allclose(xyz[:, 0, 0, 2], xyz[:, 0, 0, 0] * 3, atol=1e-6)
            self.assertTrue(mask[:, :3].all())
            self.assertTrue((~mask[:, 3]).any())
            # Missing source tactile is a documented all-masked fallback, not
            # invented valid zero-touch observations.
            with mock.patch.object(self.tactile, "verify_plan_against_cache"), mock.patch.object(self.tactile, "load_tactile", side_effect=FileNotFoundError("fixture")):
                result2 = self.tactile.process_source({**job, "source_id": "missing"})
            self.assertFalse(np.load(output / result2["tactile_mask_relpath"]).any())
            self.assertFalse(np.load(output / result2["tactile_relpath"]).any())

    @unittest.skipUnless(os.environ.get("PHYSCANVAS_TEST_DATA_WRITER") == "1", "opt-in synthetic LeRobot video writer")
    def test_synthetic_full_writer_and_sidecar(self):
        from unittest import mock
        import data_processing.convert as converter
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = fixture(root, variants=("raw", "remove_hand"))
            payload = load_manifest(manifest)
            originals = {str(path): file_sha256(path) for path in (root / "raw").rglob("*") if path.is_file()}
            output = root / "output"
            video_encoder = self.writer.lerobot_dataset_module.encode_video_frames
            report = execute(payload, output, trusted_pickle=True, manifest_sha256=file_sha256(manifest))
            self.assertIs(self.writer.lerobot_dataset_module.encode_video_frames, video_encoder)
            self.assertEqual(report["episodes"], 2)
            self.assertTrue(validate_bundle(output)["valid"])
            self.assertTrue((output / "COMPLETE.json").is_file())
            for name, digest in originals.items():
                self.assertEqual(file_sha256(name), digest)
            with self.assertRaises(FileExistsError):
                execute(payload, output, trusted_pickle=True)
            # A second conversion in this same Python process must not stack
            # codec wrappers or leave completion true if final commit fails.
            failing = root / "completion-failure"
            real_json = converter._json

            def injected_failure(path, value):
                if Path(path).name == "COMPLETE.json":
                    raise OSError("synthetic completion write failure")
                return real_json(path, value)

            with mock.patch.object(converter, "_json", side_effect=injected_failure):
                with self.assertRaisesRegex(OSError, "synthetic completion"):
                    execute(payload, failing, trusted_pickle=True)
            self.assertFalse(json.loads((failing / "tactile/summary.json").read_text())["complete"])
            self.assertTrue((failing / "FAILED.json").is_file())
            self.assertFalse((failing / "COMPLETE.json").exists())
            self.assertIs(self.writer.lerobot_dataset_module.encode_video_frames, video_encoder)


if __name__ == "__main__":
    unittest.main()
