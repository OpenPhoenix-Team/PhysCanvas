"""Pure checks of the actual placement before any model workers are launched."""


def validate_actor_env_layout(cfg, *, actor_world_size: int, env_world_size: int) -> dict:
    """Reject hidden accumulation or GRPO groups split across EnvWorkers."""
    actor_world_size, env_world_size = int(actor_world_size), int(env_world_size)
    if actor_world_size < 1 or env_world_size < 1:
        raise ValueError("Actor and environment placements must each have at least one worker")
    expected_global_batch = int(cfg.actor.micro_batch_size) * actor_world_size
    if int(cfg.actor.global_batch_size) != expected_global_batch:
        raise ValueError(
            "No-gradient-accumulation contract violated by actual placement: "
            f"global_batch_size={cfg.actor.global_batch_size}, "
            f"micro_batch_size={cfg.actor.micro_batch_size}, actor_workers={actor_world_size}. "
            f"Need global_batch_size == micro_batch_size * actor_workers = {expected_global_batch}. "
            "Default production config requires 16 actor GPUs (2 nodes x 8); "
            "do not spoof Ray GPU resources on a smaller machine."
        )
    stages = int(cfg.rollout.pipeline_stage_num)
    if stages != 1:
        raise ValueError("This PhysCanvas integration is validated for pipeline_stage_num=1 only")
    for mode in ("train", "eval"):
        env = cfg.env[mode]
        divisor = env_world_size * int(env.group_size) * stages
        if int(env.total_num_envs) % divisor:
            raise ValueError(
                f"env.{mode}.total_num_envs={env.total_num_envs} must be divisible by "
                f"env_workers({env_world_size}) * group_size({env.group_size}) * stages({stages}) "
                f"= {divisor}; every local environment must own complete GRPO groups."
            )
    return {
        "actor_workers": actor_world_size, "env_workers": env_world_size,
        "gradient_accumulation": 1,
        "local_train_envs": int(cfg.env.train.total_num_envs) // env_world_size,
    }
