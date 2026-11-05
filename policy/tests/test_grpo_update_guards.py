"""Actual actor-loop CPU tests; no Ray, CUDA or large-model loading."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "vendor/rlinf/rlinf/workers/actor/embodied_fsdp_actor_worker.py"
GUARD = ROOT / "vendor/rlinf/rlinf/algorithms/grpo_update_guard.py"

def load_guard():
    spec = importlib.util.spec_from_file_location("standalone_grpo_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.has_global_valid_samples

def _gloo_worker(rank, init_file, queue):
    from datetime import timedelta
    dist.init_process_group("gloo", rank=rank, world_size=2,
        init_method=f"file://{init_file}", timeout=timedelta(seconds=25))
    try:
        guard = load_guard()
        empty = guard(torch.zeros(2, dtype=torch.bool), local_batch_size=2, device="cpu")
        mixed = guard(torch.tensor([False, rank == 1]), local_batch_size=2, device="cpu")
        queue.put((rank, empty, mixed))
    finally:
        dist.destroy_process_group()

def append(metrics, values):
    for key, value in values.items():
        metrics.setdefault(key, []).append(value)

def chunks(batch, count):
    split = {key: torch.chunk(value, count, dim=0) for key, value in batch.items()}
    return [{key: value[i] for key, value in split.items()} for i in range(count)]

def actor_loop():
    """Compile the actual run_training body, replacing only infrastructure."""
    tree = ast.parse(WORKER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EmbodiedFSDPActor")
    function = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_training"))
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "torch": torch, "np": np, "has_global_valid_samples": load_guard(),
        "process_nested_dict_for_train": lambda data, order: {
            key: value.flatten(0, 1) for key, value in data.items()},
        "split_dict_to_chunk": chunks, "append_to_dict": append,
        "clear_memory": lambda: None, "pop_critic_explained_variance_stats": lambda _: {},
        "all_reduce_dict": lambda data, **kwargs: data,
    }
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace["run_training"]

class FakeActor:
    """Real AdamW with primed momentum; no distributed/model infrastructure."""
    def __init__(self, mask):
        self.cfg = OmegaConf.create({
            "algorithm": {"loss_type": "actor", "update_epoch": 1},
            "actor": {"seed": 42, "global_batch_size": 8, "micro_batch_size": 8}})
        self.is_weight_offloaded = self.is_optimizer_offloaded = False
        self._rank, self._world_size, self.gradient_accumulation = 0, 1, 1
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01, weight_decay=.1)
        self.model(torch.ones(1, 1)).sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_count = self.backward_count = self.scheduler_count = 0
        self.lr_scheduler = types.SimpleNamespace(step=self.scheduler_step)
        self.torch_platform = types.SimpleNamespace(empty_cache=lambda: None)
        self.rollout_batch = {"prev_logprobs": torch.zeros(1, 16, 16, 14),
                              "loss_mask": mask.reshape(1, 16, 1)}

    def scheduler_step(self):
        self.scheduler_count += 1

    def optimizer_step(self):
        self.optimizer_count += 1
        self.optimizer.step()
        return 0.0, [.01]

    def train_micro_batch(self, *, micro_batch, metrics, is_last):
        self.backward_count += 1
        self.model(torch.ones(8, 1)).mul(micro_batch["loss_mask"]).mean().backward()
        append(metrics, {"actor/policy_loss": 0.0})

class GuardTests(unittest.TestCase):
    def test_all_empty_preserves_weights_and_adam_state(self):
        actor = FakeActor(torch.zeros(16, dtype=torch.bool))
        weights = copy.deepcopy(actor.model.state_dict())
        state = copy.deepcopy(actor.optimizer.state_dict())
        with patch.object(dist, "get_world_size", return_value=1):
            metrics = actor_loop()(actor)
        self.assertEqual((actor.backward_count, actor.optimizer_count, actor.scheduler_count), (0, 0, 0))
        self.assertEqual(metrics["actor/skipped_empty_update"], 1)
        self.assertEqual(metrics["actor/optimizer_minibatches"], 0)
        for key, value in weights.items():
            torch.testing.assert_close(value, actor.model.state_dict()[key], rtol=0, atol=0)
        for index, values in state["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, actor.optimizer.state_dict()["state"][index][name], rtol=0, atol=0)

    def test_skip_only_empty_minibatch_not_entire_update(self):
        actor = FakeActor(torch.tensor([False] * 8 + [True] * 8))
        with patch.object(dist, "get_world_size", return_value=1):
            metrics = actor_loop()(actor)
        self.assertEqual((actor.backward_count, actor.optimizer_count, actor.scheduler_count), (1, 1, 1))
        self.assertEqual(metrics["actor/skipped_empty_minibatch"], .5)
        self.assertEqual(metrics["actor/skipped_empty_update"], 0)

    def test_empty_local_rank_participates_when_other_rank_valid(self):
        context = mp.get_context("spawn")
        queue = context.Queue()
        with tempfile.TemporaryDirectory(prefix="physcanvas-gloo-") as directory:
            children = [context.Process(target=_gloo_worker, args=(i, str(Path(directory) / "init"), queue)) for i in range(2)]
            for child in children:
                child.start()
            try:
                result = sorted(queue.get(timeout=40) for _ in children)
                self.assertEqual(result, [(0, False, True), (1, False, True)])
                for child in children:
                    child.join(timeout=10)
                    self.assertEqual(child.exitcode, 0)
            finally:
                for child in children:
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=5)

    def test_openpi_passes_configured_dual_clip(self):
        conditions = [n for n in ast.walk(ast.parse(WORKER.read_text()))
                      if isinstance(n, ast.If) and "SupportedModel.OPENPI_RLINF" in ast.unparse(n.test)]
        self.assertEqual(len(conditions), 1)
        cfg = OmegaConf.create({"algorithm": {"clip_ratio_c": 3.0}})
        namespace = {"self": types.SimpleNamespace(cfg=cfg), "loss_kwargs": {}}
        branch = ast.fix_missing_locations(ast.Module(body=conditions[0].body, type_ignores=[]))
        exec(compile(branch, str(WORKER), "exec"), namespace)
        self.assertEqual(namespace["loss_kwargs"]["clip_ratio_c"], 3.0)

if __name__ == "__main__":
    unittest.main()
