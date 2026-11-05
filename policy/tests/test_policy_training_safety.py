import contextlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from policy_training.plan import Plan, asset_preflight, check_norm_stats, require_write_authority
from policy_training.run import main
from policy_training.validate_torch import read_header, tactile_shapes, validate_checkpoint


class PolicyTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.config = self.root / "config.json"
        self.values = dict(name="pi05_custom", exp_name="test", lerobot_root=str(self.root / "bundle/lerobot"),
            tactile_sidecar_root=str(self.root / "bundle/tactile"), initial_params=str(self.root / "initial/params"),
            tokenizer_file=str(self.root / "tokenizer.model"), output_root=str(self.root / "new_output"), asset_id="custom")
        self.save_config()

    def tearDown(self):
        self.temp.cleanup()

    def save_config(self):
        self.config.write_text(json.dumps(self.values))

    def stats(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"norm_stats": {key: {"mean": [0.] * 14, "std": [1.] * 14,
            "q01": [-1.] * 14, "q99": [1.] * 14} for key in ("state", "actions")}}))

    def test_plan_has_no_side_effect_and_resolves_explicit_root(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["plan", "--config", str(self.config)]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(Path(report["HF_LEROBOT_HOME"]) / report["repo_id"], Path(self.values["lerobot_root"]))
        self.assertFalse(Path(self.values["output_root"]).exists())
        self.assertEqual(report["model"]["tactile_shape"], [4, 700, 3])

    def test_default_train_is_only_plan(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["train-jax", "--config", str(self.config)]), 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "plan_only")
        self.assertFalse(Path(self.values["output_root"]).exists())

    def test_double_authority_and_torch_refusal(self):
        with patch.dict(os.environ, {"POLICY_TRAINING_ALLOW_WRITE": "0"}):
            with self.assertRaisesRegex(ValueError, "requires both"):
                require_write_authority(True)
        with patch.dict(os.environ, {"POLICY_TRAINING_ALLOW_WRITE": "1"}):
            require_write_authority(True)
            with self.assertRaises(ValueError):
                require_write_authority(False)
        with self.assertRaisesRegex(ValueError, "NOT supported"):
            main(["train-torch", "--config", str(self.config)])

    def test_path_traversal_and_overlap_rejected(self):
        for key in ("name", "exp_name", "asset_id"):
            values = dict(self.values)
            values[key] = "../../escape"
            self.config.write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                Plan.load(self.config)
        self.values["output_root"] = str(self.root / "bundle")
        self.save_config()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            Plan.load(self.config)

    def test_missing_assets_fail_without_writes(self):
        report = asset_preflight(Plan.load(self.config))
        self.assertFalse(report["ok"])
        self.assertGreater(len(report["errors"]), 5)
        self.assertFalse(Path(self.values["output_root"]).exists())

    def test_new_bundle_complete_gate(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "source_manifest.json").write_text("{}")
        report = asset_preflight(Plan.load(self.config), "norm-stats")
        self.assertTrue(any("COMPLETE gate" in x for x in report["errors"]))
        (bundle / "COMPLETE.json").write_text('{"complete": true}')
        (bundle / "FAILED.json").write_text("{}")
        report = asset_preflight(Plan.load(self.config), "norm-stats")
        self.assertTrue(any("not successfully complete" in x for x in report["errors"]))

    def test_stats_finite_14d_and_quantile_order(self):
        path = self.root / "stats.json"
        self.stats(path)
        self.assertEqual(check_norm_stats(path)["dimensions"], 14)
        value = json.loads(path.read_text())
        value["norm_stats"]["state"]["mean"][0] = float("nan")
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "finite"):
            check_norm_stats(path)

    def write_checkpoint(self, root, *, missing_tactile=False):
        root.mkdir()
        config = dict(pi05=True, use_tactile=True, action_dim=32, action_horizon=16, max_token_len=200,
            tactile_num_sensors=4, tactile_points_per_sensor=700, tactile_input_channels=3,
            tactile_grid_height=35, tactile_grid_width=20, tactile_hidden_dim=1024,
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
        (root / "config.json").write_text(json.dumps(config))
        shapes = tactile_shapes()
        if missing_tactile:
            del shapes["tactile_encoder.gate"]
        shapes.update({key: [1] for key in ("img.stem.weight", "llm.embedder.embedding.weight", "action_in_proj.weight", "action_out_proj.weight")})
        header, offset = {}, 0
        for key, shape in shapes.items():
            size = math.prod(shape) * 4
            header[key] = dict(dtype="F32", shape=shape, data_offsets=[offset, offset + size])
            offset += size
        raw = json.dumps(header).encode()
        with (root / "model.safetensors").open("wb") as stream:
            stream.write(struct.pack("<Q", len(raw)))
            stream.write(raw)
            stream.truncate(8 + len(raw) + offset)  # Sparse test fixture; no real weights.
        self.stats(root / "assets/custom/norm_stats.json")

    def test_header_only_tactile_complete_and_downstream_stats(self):
        checkpoint = self.root / "torch"
        self.write_checkpoint(checkpoint)
        result = validate_checkpoint(checkpoint, "custom")
        self.assertEqual(result["tactile_tensor_count"], 23)
        self.assertEqual(result["PI05_NORM_STATS"], str(checkpoint / "assets/custom/norm_stats.json"))
        self.assertEqual(result["PI05_CHECKPOINT"], str(checkpoint))

    def test_missing_tactile_and_truncated_header_rejected(self):
        checkpoint = self.root / "torch"
        self.write_checkpoint(checkpoint, missing_tactile=True)
        with self.assertRaisesRegex(ValueError, "Tactile keys mismatch"):
            validate_checkpoint(checkpoint, "custom")
        with (checkpoint / "model.safetensors").open("wb") as stream:
            stream.write(b"bad")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            read_header(checkpoint / "model.safetensors")

    def test_conversion_plan_requires_matching_checkpoint_stats(self):
        params = Path(self.values["initial_params"])
        params.mkdir(parents=True)
        stats = params.parent / "assets/custom/norm_stats.json"
        self.stats(stats)
        output = self.root / "converted"
        argv = ["convert", "--input-params", str(params), "--norm-stats", str(stats),
                "--output", str(output), "--asset-id", "custom"]
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(argv), 0)
        self.assertEqual(json.loads(stdout.getvalue())["mode"], "conversion_plan")
        self.assertFalse(output.exists())
        outside = self.root / "old_stats.json"
        self.stats(outside)
        argv[4] = str(outside)
        with self.assertRaisesRegex(ValueError, "SAME JAX checkpoint"):
            main(argv)


if __name__ == "__main__":
    unittest.main()
