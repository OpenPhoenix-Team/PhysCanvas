import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from physcanvas_pi05.canvas import SIDE_ORDER, VIEW_ORDER
from physcanvas_pi05.reset_data import ResetDataset


class ResetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "episodes").mkdir()
        self.manifest = self.root / "selection_manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def manifest_rows(self, rows):
        self.manifest.write_text(json.dumps({"episodes": rows}))

    def test_manifest_order_and_read_only_first_index(self):
        self.manifest_rows([{"cache_episode_id": "second"}, {"cache_episode_id": "first"}])
        dataset = ResetDataset(self.root, self.manifest)
        self.assertEqual(dataset.episode_ids, ("second", "first"))
        self.assertEqual(len(dataset), 2)
        accessed = []

        class FirstOnlyArray:
            ndim = 1
            shape = (9,)

            def __init__(self, frame):
                self.frame = frame

            def __getitem__(self, key):
                if key != 0:
                    raise AssertionError(f"Future frame access: {key}")
                return self.frame

        def load(path, mmap_mode=None, allow_pickle=None):
            self.assertEqual(mmap_mode, "r")
            self.assertFalse(allow_pickle)
            self.assertEqual(Path(path).parts[-3] if "tactile" in Path(path).parts else Path(path).parts[-2], "second")
            accessed.append(Path(path).name)
            if str(path).endswith("_rgb.npy"):
                value = np.full((224, 224, 3), 127, np.uint8)
            elif str(path).endswith(".difference.npy"):
                value = np.full((400, 700, 3), 21, np.uint8)
            elif str(path).endswith(".pointcloud.npy"):
                value = np.arange(35 * 20 * 6, dtype=np.float32).reshape(35, 20, 6)
            elif Path(path).name == "qpos.npy":
                value = np.arange(14, dtype=np.float32)
            else:
                raise AssertionError(f"Unexpected file accessed: {path}")
            return FirstOnlyArray(value)

        with mock.patch("physcanvas_pi05.reset_data.np.load", side_effect=load):
            reset = dataset.read(0)
        self.assertEqual(reset["episode_id"], "second")
        self.assertEqual(reset["initial_label"], 0)
        self.assertEqual(reset["rgb"].shape, (3, 224, 224, 3))
        self.assertEqual(reset["tactile_diff"].shape, (4, 400, 700, 3))
        np.testing.assert_array_equal(reset["tactile_xyz"], np.zeros((4, 700, 3), np.float32))
        self.assertEqual(len(accessed), 12)
        self.assertFalse(any("motion" in x or "depth" in x or "label" in x for x in accessed))

    def test_duplicate_traversal_and_nonzero_start_rejected(self):
        for rows in (
            [{"cache_episode_id": "a"}, {"cache_episode_id": "a"}],
            [{"cache_episode_id": "../elsewhere"}],
            [{"episode_id": "a", "start": 1}],
        ):
            self.manifest_rows(rows)
            with self.assertRaises(ValueError):
                ResetDataset(self.root)

    def test_invalid_index_rejected(self):
        self.manifest_rows([{"cache_episode_id": "a"}])
        dataset = ResetDataset(self.root)
        for index in (-1, 1, 0.5):
            with self.assertRaises(IndexError):
                dataset.read(index)


if __name__ == "__main__":
    unittest.main()
