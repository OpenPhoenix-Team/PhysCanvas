# Pi0.5: tactile SFT, closed-loop PhysCanvas rollout, and GRPO

This directory adds policy code without changing the existing PhysCanvas WM
implementation. Tasks are **building blocks** (`building_blocks` in CLI/configs) and
**both-arm test tube** (`test_tube`). Raw demonstrations, reset datasets and
trained task reward models are **not distributed**. For GRPO, train your own RM. For both modes,
provide your own calibrated robot assets and reset observations.

For **WM-only inference or evaluation**, use the existing
[PhysCanvas setup and evaluation tutorial](../README.md#setup). Nothing in
this directory replaces those commands. For **policy rollout or GRPO**, use
the policy environment below: it imports the same repository's `PhysCanvas/`
and `evaluation/` modules. Do not install a second copy of the WM source.

## Released policy weights

Download from [ArteMe/PhysCanvas_model](https://modelscope.cn/models/ArteMe/PhysCanvas_model/files):

| Policy | Repository file | Format |
|---|---|---|
| Base | `policy/pi05_base/model.safetensors` | converted tactile Pi0.5 |
| Test tube GRPO | `policy/pi05_test_tube_wm_grpo/full_weights.pt` | RLinf PyTorch wrapper |
| Building blocks GRPO | `policy/pi05_building_blocks_wm_grpo/full_weights.pt` | RLinf PyTorch wrapper |

These are **weights**, not self-contained robot deployments. The model is
Pi0.5 (Gemma-2B/SigLIP prefix, Gemma-300M action expert) with a four-sensor
tactile CNN, three RGB cameras, 14 physical action dimensions padded to 32,
and 16-action chunks. Use this code's `openpi_rlinf` loader, not the JAX
checkpoint loader. Tokenizer and matching state/action normalization are
required separately. Do not substitute unrelated normalization statistics.
No hardware-control service is started by the scripts here.

```bash
python -c 'from modelscope.hub.snapshot_download import snapshot_download; snapshot_download("ArteMe/PhysCanvas_model", local_dir="/absolute/path/to/downloads", allow_patterns=["policy/*"])'
```

## Environments

Use Linux, Python 3.11, an NVIDIA CUDA-capable driver, system `ffmpeg`, and EGL
libraries for rendering. Keep SFT/data and Torch rollout environments separate.
Do not stack other environments' `site-packages` on `PYTHONPATH`.

| What you want to run | Environment to configure |
|---|---|
| Existing WM evaluation | Follow the root README unchanged (its reference versions are Torch 2.7.0 and Transformers 4.53.3). |
| Policy rollout, GRPO, or three-view RM | Separate Torch environment. Policy recipe: Torch 2.7.1+cu128, TorchVision 0.22.1+cu128, Transformers 4.53.2; the WM dependencies Diffusers 0.38.0 and giga-datasets/giga-train 1.0.0 are shared with the root recipe. GRPO also uses Ray and Hydra. |
| Data conversion or tactile policy SFT | Separate OpenPI environment: use the included lock for JAX, Flax and the legacy LeRobot v2.1 writer. Do not replace it with the latest LeRobot. |

```bash
cd policy
# Requires uv. Creates a new directory; never upgrades an old environment.
bash scripts/setup_runtime.sh /absolute/path/to/new_runtime
export PHYSCANVAS_PYTHON=/absolute/path/to/new_runtime/bin/python
# Subsequent Python commands in this section use this interpreter.
source /absolute/path/to/new_runtime/bin/activate
```

`requirements-runtime.txt` is the Torch/WM/RM runtime recipe; the setup script
installs CUDA 12.8 Torch 2.7.1 and TorchVision 0.22.1 first. See `VALIDATION.md`
for what was actually validated; an installation recipe is not evidence of a
completed production distributed training run.

The historical JAX SFT/data environment is defined by
`policy_training/openpi/{pyproject.toml,uv.lock}`. From that directory, use
`uv sync --frozen` on a supported GPU system. Set `SFT_PYTHON` and `DATA_PYTHON`
explicitly. Historical JAX 0.5.3 CUDA compatibility must be checked on your
GPU before SFT; Torch Blackwell support does not imply JAX GPU support.

## Private inputs and asset bindings

Copy `scripts/local_paths.example.sh` outside the repository, edit all paths,
then set `PHYSCANVAS_PATHS_FILE=/absolute/path/to/your_paths.sh`. No credentials
belong in this file. Download the final WM using the existing repository's
`tools.download_model`; use its pinned `weights/epoch30_step300/transformer`.
Also supply the Wan VAE/scheduler, prompt embedding, tokenizer, calibration
and URDF/meshes. Motion settings must name `debug_dir`, `x5a_root`, `urdf_path`,
`fk_offset_path`, `left_calib`, `right_calib`, and `render_size_hw`. Do not use
calibration from another robot/camera arrangement.

From the **repository root**, the WM download is the same command as in the
root README (choose an external asset directory):

```bash
export PHYSCANVAS_STORE=/absolute/path/to/physcanvas-assets
python -m tools.download_model --source modelscope --download-dir "$PHYSCANVAS_STORE"
```

Set `PHYSCANVAS_ASSETS` in your private paths file to that same directory.
The 20-episode WM evaluation cohort is not the task-specific policy reset
cache. The renderer's `debug_dir` must supply the FK, projection and camera
helper modules imported by the [existing renderer](../PhysCanvas/motion/arxx5_link678_gray_renderer.py);
calibration JSON alone is insufficient. For the following commands, return to
`policy/`.

Reset cache layout is `cache/episodes/<episode_id>/` containing `qpos.npy`,
`base_rgb.npy`, `left_wrist_rgb.npy`, `right_wrist_rgb.npy`, and four sensor
`tactile/<side>.{pointcloud,difference}.npy` arrays. Exact sensor order and array
contracts are in `physcanvas_pi05/{canvas,reset_data}.py`. The selection JSON
contains an `episodes` list of `{"episode_id":"...", "start":0}`. Only frame
zero is loaded. Production preflight expects 128 distinct resets; both-arm
test-tube IDs use the `use_both_arms_collaboratively_` prefix. No future RGB,
action or success annotation is used as a rollout condition.

## Closed-loop rollout

```bash
bash scripts/preflight.sh building_blocks --rollout --imports
bash scripts/preflight.sh test_tube --rollout --imports
bash scripts/rollout.sh building_blocks --indices 0 --frames 300 \
  --output-dir /absolute/path/to/NEW_blocks_rollout
bash scripts/rollout.sh test_tube --indices 0 --frames 300 \
  --output-dir /absolute/path/to/NEW_test_tube_rollout
```

Pi0.5 plans 16 actions; WM generates 8+8 new frames; the next policy call uses
the last generated observation. The initial RGB anchor remains fixed while
local visual/state conditions advance. The final WM uses signed XYZ tactile
flow (`xyz_linear`, displacement scale 0.06); all three generated components
are fed back. Proprioception is the last commanded absolute joint target,
not an independently predicted robot state. Initial motion gate: .001,
at most four policy resamples; the accepted chunk is retained up to the requested frame budget.

Default WM denoising is 30 steps; `--wm-steps 2` is a functional smoke override,
not a quality evaluation. Standalone inference always uses Euler Flow-ODE and never loads an RM or stops for predicted success. `--frames N` counts future frames only (default 256); the reset is one extra display frame. For 300 frames there are 19 policy chunks, 38 fixed eight-frame WM calls and 301 video frames. Only the requested prefix of the final chunk is committed to the observation, video and actions. No reward CSV or success label is produced. `--max-chunks K` remains shorthand for `--frames 16*K`. The GRPO horizon and sampling mode are independent.

## GRPO

```bash
# Requires the real Ray cluster and all private assets; not a one-GPU demo.
bash scripts/preflight.sh building_blocks --imports  # includes private RM checks
RUN_TRAINING=1 bash scripts/train_grpo.sh building_blocks
# Alternative task (not simultaneously on the same GPUs):
RUN_TRAINING=1 bash scripts/train_grpo.sh test_tube
```

Defaults: 2 nodes/16 GPUs, collocated phase-offloaded actor/env/rollout,
group size 8, 128 environments × 4 rollout epochs = 512 trajectories/update,
at most 16 policy chunks/256 frames, actor LR 5e-6, global BS128, micro BS8/GPU,
no gradient accumulation, one actor pass, ratio clipping [.8,1.28], dual clip3,
no KL/entropy/critic objective. Save every 4 updates and record three worker
rollouts per update. Building-block flow-SDE noise=.5; test-tube=.3.

Task RM is shared-ResNet18 three-view classification. Sigmoid ≥.7 confirms
success in any generated frame; reward is emitted at the chunk's last action,
and later WM calls stop. Test-tube uses binary trajectory success. Building
blocks uses chunk-relative binary reward; with success termination, total
return is also 0/1. All-equal groups are excluded; globally empty minibatches
skip backward/AdamW/scheduler advancement. No reward gradients flow into WM/RM.

All nodes must have identical paths/code/environment. Start the RLinf Ray
cluster before the head-node command. `RLINF_NODE_RANK` identifies each node.
The launcher validates actual placement and refuses a hidden change to
gradient accumulation. Multi-node launch instructions and verified scope
are recorded in `VALIDATION.md`; do not claim a 2-node reproduction from a
single-device smoke.

Example cluster bootstrap (private, trusted training network only; do not
expose Ray ports to the internet). Choose an unused port and the reachable
private address of each node; exactly eight real visible GPUs per node:

```bash
# Both nodes, from the same absolute policy directory:
export PHYSCANVAS_PATHS_FILE=/absolute/path/to/your_paths.sh
source scripts/runtime_env.sh test_tube

# Head only:
export RLINF_NODE_RANK=0
ray start --head --node-ip-address="$HEAD_PRIVATE_IP" --port=6379 \
  --num-gpus=8 --include-dashboard=false

# Worker only:
export RLINF_NODE_RANK=1
ray start --address="$HEAD_PRIVATE_IP:6379" --node-ip-address="$WORKER_PRIVATE_IP" \
  --num-gpus=8

# Back on head after ray status reports both nodes and 16 GPUs:
export RAY_ADDRESS="$HEAD_PRIVATE_IP:6379"
ray status
RUN_TRAINING=1 bash scripts/train_grpo.sh test_tube
```

Do not use `--num-gpus` to advertise GPUs that do not physically exist. Shared
filesystem paths do not replace network/NCCL validation. Do not stop or reuse
another job's Ray cluster. These are launch instructions, not a claim that this
release has been rerun on two nodes.

## SFT and data conversion

See [data processing](docs/DATA_PROCESSING.md),
[SFT / conversion](docs/POLICY_TRAINING.md), and
[end-to-end workflow](docs/PI05_WORKFLOW.md).

`data_processing/run.sh` aligns timestamps, trims static head/tail frames,
resamples to 10Hz, writes LeRobot v2.1 and full XYZ tactile sidecars. Inputs
stay read-only; output must be a new, disjoint directory. Default is plan-only.
SFT uses the preserved tactile-aware JAX training code. The original generic
Torch SFT trainer is **not** the tactile SFT entry. Exact historical weights
cannot be regenerated without the private dataset and original initializer.

## Train a private three-view RM

See [RM training](docs/REWARD_TRAINING.md). No trained RM weights are included.

## Tests

```bash
source scripts/runtime_env.sh building_blocks
export PYTHONPATH="$PHYSCANVAS_ROOT/tests:$PYTHONPATH"
"$PHYSCANVAS_PYTHON" -m unittest -v test_backend test_canvas test_reset_data \
  test_integration_config test_integration_env test_grpo_update_guards \
  test_integration_layout test_reward_training test_standalone_rollout
"$PHYSCANVAS_PYTHON" -m reward_training.smoke --output /absolute/path/to/NEW_rm_smoke
"$PHYSCANVAS_PYTHON" -m physcanvas_pi05.smoke_optimizer --task building_blocks \
  --output /absolute/path/to/NEW_optimizer_smoke
```

The optimizer smoke uses **fixed diagnostic rewards**, not task success, and
sequential replay to bound memory; it verifies real policy sampling/logprob/
backward/update but is not the distributed training benchmark.

## Licenses

Retain the repository license and the vendored OpenPI/RLinf licenses. Gemma
terms are included under `policy_training/openpi/LICENSE_GEMMA.txt`. Model,
dataset, robot mesh and tokenizer permissions are independent of code licensing.
