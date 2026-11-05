# RLinf integration boundary

This project uses an independent source copy at `vendor/rlinf/rlinf`, copied from
the deployed Jimu Reward-WM-v2 RLinf tree. It does not edit the original Jimu,
Shiguan, OpenPI, or public PhysCanvas checkouts. The local source tree was not a
clean upstream release: preserve its Apache license and review provenance before
publishing the combined distribution.

The environment integration change is `rlinf_physcanvas_backend.patch`, applied
to this new copy. It selects `physcanvas_pi05.backend.PhysCanvasBackend` when
`env.train.world_model.backend=physcanvas`. Keeping `env_type=jimu_wm` reuses the
existing registered action, worker and environment contracts without global
module monkey-patching. Both task configs use this backend; neither references
historical task-specific WM checkpoints.

Original environment SHA-256 (read on 2026-11-04):
`92a6f7a7ffe51b03035cadd9d9ea4724a88fbac0f38af8b946b4d46a8afaf14f`.
Patched independent-copy SHA-256:
`fee7b78a74e371d43b3de206719f153bad7a5abaa353b42c2dc606c2ea1de34d`.

## Backend contract

- Constructor: `(cfg, *, device, seed, prompt, reset_start_index=0, reset_stride=1)`.
- `metadata()` confirms 8 WM frames, 14 state dimensions, view order
  `base/left_wrist/right_wrist`, `(4,700,3)` policy tactile, generated signed XYZ,
  `xyz_linear`, and `transport=inprocess`.
- `reset_groups(num_groups, group_size, prompt)` returns unique `session_ids`,
  contiguous `group_ids` (each group shares the same initial condition), and
  independent observation dictionaries.
- `rollout16(session_ids, actions, audit_session_id=None)` accepts `[B,16,14]`
  absolute joint targets and returns same-order session IDs, `[B,16,H,3H,3]`
  uint8 RGB panels, and latest generated observations. Optional audit returns
  one sample's motion and tactile panel videos with the same 16 frames.
- Raw observation dictionary: three `observation.images.*_0` CHW uint8 arrays,
  `observation.state [14]`, `observation.tactile [4,700,3]`,
  `observation.tactile_mask [4]`. RLinf converts these to batched tensors.
- `offload/onload/close/discard_sessions` manage only this backend's state and
  models, never global processes or other experiments.

## Training execution

`physcanvas_pi05.train_grpo` builds the same RLinf `EmbodiedFSDPActor`,
`MultiStepRolloutWorker`, `EnvWorker` and `EmbodiedRunner` chain. The frozen RM
stays inside EnvWorker. The policy's existing flow-SDE sampler stores the sampled
transition and its old log probability; the FSDP actor recomputes that transition
for the clipped GRPO objective. There is no new, approximate action-MSE RL loss.

The public WM and policy/RM run in alternating collocated phases with offload.
Default production layout is 2 nodes / 16 GPUs: 128 environments × 4 rollout
epochs = 512 trajectories (64 groups) per update. No training or Ray cluster is
started by preflight. `RUN_TRAINING=1` is required to enter the training path.

Cluster setup remains an operator step: source identical runtime environment on
each node, set unique `RLINF_NODE_RANK` before starting Ray, join workers to the
head, then launch the training command only on the head. A successful local
preflight does not certify a 2-node NCCL/Ray deployment.

Before model-worker launch the driver checks the **actual** actor count:
`global_batch_size == micro_batch_size * actor_workers`. Merely setting
`enable_gradient_accumulation=false` does not enforce this in upstream RLinf.
It also checks that each EnvWorker receives complete groups, and rejects
unsupported pipeline stages. Default settings therefore reject this debug
server's single visible GPU instead of silently enabling accumulation.

Example cluster startup (operator commands only; not run during this work):

```bash
# Node 0, with eight real GPUs visible:
cd /path/to/physcanvas_pi05_experiments
source scripts/runtime_env.sh jimu   # use shiguan on BOTH nodes for that task
export RLINF_NODE_RANK=0
"$PHYSCANVAS_PYTHON" -m ray.scripts.scripts start --head \
  --node-ip-address=<HEAD_PRIVATE_IP> --port=6379

# Node 1, same code/assets/dependency paths and eight real GPUs:
cd /path/to/physcanvas_pi05_experiments
source scripts/runtime_env.sh jimu
export RLINF_NODE_RANK=1
"$PHYSCANVAS_PYTHON" -m ray.scripts.scripts start \
  --address=<HEAD_PRIVATE_IP>:6379 --node-ip-address=<WORKER_PRIVATE_IP>

# Node 0 ONLY, after both Ray nodes are ready:
RUN_TRAINING=1 bash scripts/train_grpo.sh jimu
```

Replace the IP placeholders; do not paste them literally. Ray must start with
the selected interpreter and environment on **each** node, not with an unrelated
`ray` executable from PATH. Do not set a fictitious `--num-gpus=8` on a 1-GPU
debug machine. A stale pre-existing Ray cluster must be handled by its operator,
not silently stopped by these scripts.

## Important semantic differences from the historical runs

- Both tasks use the final public `epoch30_step300` WM and its no-depth contract.
- Default WM denoising is 30 steps, matching the public inference release;
  explicitly override `env.train.world_model.num_inference_steps` for speed studies.
- Both tasks preserve generated signed dz. Historical Shiguan policy feedback
  discarded dz; that behavior is deliberately not carried into this release.
- Jimu's `relative_sum` is a difference of **chunk success indicators**, not a
  difference of per-frame probabilities. Any positive RM frame triggers success,
  placed at action16; subsequent chunks for that candidate are skipped. Shiguan
  directly emits binary success. With success termination, both trajectory
  returns are 0 or 1.
- Existing validation metrics are online rollout metrics, not an independent
  held-out success rate. The task RM weights remain private dependencies.

## Actor corrections in the independent copy (2026-11-04)

`rlinf_grpo_update_guards.patch` plus `rlinf/algorithms/grpo_update_guard.py` fix
two issues in the copied actor; historical experiment code and results are not
modified. Every global minibatch collectively counts valid loss-mask elements.
Only a zero count across **all actor ranks** skips backward and AdamW. A rank
with no local valid elements still participates when another rank has data.
This prevents momentum/weight-decay updates from all-invalid groups without
diverging FSDP collective order. The LR scheduler advances once per update only
if at least one minibatch performed an optimizer step. Logging always includes
`actor/skipped_empty_minibatch`, `actor/optimizer_minibatches`, and
`actor/skipped_empty_update`, even for entirely empty updates.

The OpenPI actor now forwards `algorithm.clip_ratio_c` to the PPO-style loss.
The copied source only forwarded it for GR00T, so the configured value 3 had
previously been ignored for Pi0.5 (`compute_ppo_actor_loss` defaults to None).
This correction makes the declared configuration effective; do not claim the
old results were computed with dual clipping merely because their YAML says 3.
