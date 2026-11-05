"""CPU topology checks, independent of Ray initialization."""
import unittest
from omegaconf import OmegaConf
from physcanvas_pi05.topology import validate_actor_env_layout

class LayoutTests(unittest.TestCase):
    def config(self):
        return OmegaConf.create({
            "actor": {"global_batch_size": 128, "micro_batch_size": 8},
            "rollout": {"pipeline_stage_num": 1},
            "env": {mode: {"total_num_envs": 128, "group_size": 8} for mode in ("train", "eval")},
        })

    def test_default_two_node_sixteen_gpu_layout(self):
        result = validate_actor_env_layout(self.config(), actor_world_size=16, env_world_size=16)
        self.assertEqual(result["gradient_accumulation"], 1)
        self.assertEqual(result["local_train_envs"], 8)

    def test_fewer_gpus_cannot_silently_accumulate(self):
        for count in (1, 2, 8):
            with self.assertRaisesRegex(ValueError, "No-gradient-accumulation"):
                validate_actor_env_layout(self.config(), actor_world_size=count, env_world_size=count)

    def test_reject_partial_groups_and_unsupported_pipeline(self):
        with self.assertRaisesRegex(ValueError, "complete GRPO groups"):
            validate_actor_env_layout(self.config(), actor_world_size=16, env_world_size=32)
        cfg = self.config()
        cfg.rollout.pipeline_stage_num = 2
        with self.assertRaisesRegex(ValueError, "pipeline_stage_num"):
            validate_actor_env_layout(cfg, actor_world_size=16, env_world_size=16)

if __name__ == "__main__":
    unittest.main()
