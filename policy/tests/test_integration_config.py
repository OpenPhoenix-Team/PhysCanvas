"""CPU-only config checks; no Ray, CUDA or source assets required."""

import os
import unittest
from unittest.mock import patch

from physcanvas_pi05.config import compose_config
from physcanvas_pi05.preflight import validate_experiment


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            name: "/placeholder" for name in [
                "PHYSCANVAS_ROOT", "PHYSCANVAS_ASSETS", "PI05_CHECKPOINT",
                "PI05_NORM_STATS", "RESET_CACHE", "RESET_SELECTION", "RM_CHECKPOINT",
                "VAE_PATH", "PROMPT_EMBEDDING", "MOTION_SETTINGS",
            ]
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_two_tasks_share_public_wm_contract(self):
        for task in ["jimu", "shiguan"]:
            cfg = compose_config(task)
            result = validate_experiment(cfg, check_assets=False)
            self.assertEqual(result["trajectories_per_update"], 512)
            self.assertEqual(result["groups_per_update"], 64)
            self.assertEqual(result["wm_steps"], 30)
            self.assertEqual(cfg.env.train.world_model.tactile_third_channel, "xyz")
            self.assertEqual(cfg.env.train.prompt, cfg.task_prompt)

    def test_overrides_are_shared_by_eval_without_source_edits(self):
        cfg = compose_config("shiguan", ["env.train.world_model.num_inference_steps=5"])
        self.assertEqual(cfg.env.eval.world_model.num_inference_steps, 5)
        validate_experiment(cfg, check_assets=False)

    def test_reject_old_tactile_or_backend(self):
        for override in ["env.train.world_model.tactile_third_channel=zero",
                         "env.train.world_model.backend=inprocess"]:
            with self.assertRaises(ValueError):
                validate_experiment(compose_config("jimu", [override]), check_assets=False)

    def test_importing_preflight_does_not_import_ray(self):
        # module can be imported in a fresh process without touching RLinf;
        # all expensive runtime imports are inside opt-in validation.
        import inspect
        import physcanvas_pi05.preflight as module
        self.assertNotIn("from rlinf.config import", inspect.getsource(module))


if __name__ == "__main__":
    unittest.main()
