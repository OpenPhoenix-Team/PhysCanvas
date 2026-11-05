"""Pi0.5 GRPO using RLinf workers and the shared PhysCanvas environment."""

import argparse
import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from physcanvas_pi05.config import compose_config, TASK_NAMES
from physcanvas_pi05.preflight import validate_experiment
from physcanvas_pi05.topology import validate_actor_env_layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    if os.environ.get("RUN_TRAINING") != "1":
        raise SystemExit("Set RUN_TRAINING=1 explicitly to start training")
    cfg = compose_config(args.task, args.overrides)
    validate_experiment(cfg, check_assets=True, check_imports=True)

    # validate_cfg initializes Cluster/Ray: never call it in preflight.
    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    # Every launch gets an independent run; resume_dir stays explicit.
    log_root = Path(cfg.runner.logger.log_path)
    cfg.runner.logger.log_path = str(
        log_root / f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.task}"
    )
    cfg.env.train.rollout_visualization.output_dir = str(
        Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name
        / "rollout_visualizations"
    )
    Path(cfg.runner.logger.log_path).mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, Path(cfg.runner.logger.log_path) / "resolved_input.yaml")
    cfg = validate_cfg(cfg)
    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    placement = HybridComponentPlacement(cfg, cluster)
    layout = validate_actor_env_layout(
        cfg, actor_world_size=placement.get_world_size("actor"),
        env_world_size=placement.get_world_size("env"),
    )
    print(f"Verified actual GRPO placement: {layout}", flush=True)
    actor = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=placement.get_strategy("actor")
    )
    rollout = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=placement.get_strategy("env")
    )
    runner = EmbodiedRunner(cfg=cfg, actor=actor, rollout=rollout, env=env, reward=None)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
