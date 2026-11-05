# Tactile π0.5 supervised training and JAX → PyTorch conversion

This is an **independent source snapshot and guarded entrypoint**, not a new completed SFT run. It does not modify the original OpenPI repository, datasets, checkpoints, or GRPO runtime. No training or full checkpoint conversion was launched during preparation.

## What is included

- `policy_training/openpi/src/openpi/`: the audited original OpenPI fork, including JAX π0.5, the four-sensor tactile encoder, dataset transforms/sidecar loader, optimizers and checkpoint handling.
- `policy_training/openpi/scripts/train.py` and `compute_norm_stats.py`: historical JAX SFT and vectorized state/action normalizer.
- `policy_training/openpi/packages/openpi-client/`, `pyproject.toml`, `uv.lock`, `LICENSE`, and `LICENSE_GEMMA.txt`.
- `vendor/rlinf/rlinf/utils/ckpt_convertor/openpi/jax_to_openpi_rlinf.py`: the tactile-aware converter already included in this independent project. This is the converter used by the new wrapper.

The training core matches the historical 77-source arm-fix 59999 implementation. Original source revision: `b804a3a02556e3acb346c5e28631b327d29e92d0`. Training data, JAX initialization weights, and gated tokenizer/model assets are **not included**. OpenPI's Apache-2.0 license and Gemma terms remain applicable; possessing this code does not grant redistribution rights to every pretrained asset.

## Model and historical recipe

The historical reference config is `pi05_arx_multitask_nofail_77src_armfix_tactile_60k_finetune`. The new wrapper creates a fresh config from it and replaces all dataset, output, tokenizer and initialization paths; it does not alter the old registry.

| Item | Setting |
| --- | --- |
| Policy | π0.5, PaliGemma/Gemma-2B vision-language backbone + Gemma-300M action expert |
| RGB | Base + left wrist + right wrist; direct resize 640×480 → 224×224 |
| Actions | Absolute robot actions 14D, padded to model 32D; horizon 16 at 10 Hz |
| Tactile | 4 × 700 × 3 **physical displacement XYZ**; 4 validity masks |
| Tactile encoder | Shared 3 stride-2 CNNs (64/128/256), 5×3×256 flatten → 1024 → 2048; LayerNorm, sensor/modality embeddings, learned gate initially 0.01 |
| SFT objective | Flow-matching MSE, full-parameter JAX training; this is not GRPO |
| Normalization | Raw state/action q01–q99; tactile is not normalized by dataset norm stats |
| Batch / steps | Global 64 / 60,000 steps |
| LR | 5e-5, warmup 1,000 steps; final LR equals peak LR |
| AdamW | β=(0.9,0.95), ε=1e-8, weight decay 1e-10, gradient clip 1.0 |
| EMA / FSDP | 0.999 / 8 devices in one JAX process |
| Checkpoints | Save every 1,000, keep every 5,000; no wrapper overwrite or resume |
| Validation | Historical trainer does not implement an independent validation split or early stopping |

The original 59999 run initialized from a packphone checkpoint at step 28000. Reproducing its exact weights also requires that original initialization and 77-source dataset; the new example paths do not claim these assets are public. Supplying another authorized JAX initializer defines a new SFT experiment. A JAX `params` initializer is not an optimizer-state resume.

## Environment: separate from GRPO

Do **not** source `scripts/runtime_env.sh` before SFT or reuse its `PYTHONPATH`: the GRPO Torch runtime deliberately selects `JAX_PLATFORMS=cpu`. `policy_training/run.sh` replaces the import path with the independent SFT snapshot and selects CUDA only for explicitly executed `train-jax`.

Use a dedicated Python 3.11+ environment. The historical lock pins JAX CUDA12 **0.5.3**, Flax 0.10.2, Orbax 0.11.13, Torch 2.7.1 and the LeRobot Git revision **0cf864870cf29f4738d3ade893e6fd13fbd7cdb5**; do not `pip install -U lerobot`. Preserve the provided `pyproject.toml`/`uv.lock` when preparing an environment, e.g. `uv sync --frozen` from `policy_training/openpi`. These installation commands are documentation only and were not run here.

**Current pro5000 limitation:** the original OpenPI venv supports CPU imports and a CPU BF16 test, but its JAX CUDA backend failed device enumeration. The shared `pro5000_env` Torch/GRPO CUDA success does not fix that JAX failure. Therefore SFT **is not GPU-ready on this machine yet**. Before training, prepare a validated compatible JAX CUDA environment and pass the explicit `SFT_PYTHON`; the launcher refuses actual training if CUDA/device-count checks fail. H20's historical SFT interpreter is not usable either. No GPU SFT/collective smoke test has been claimed.

## Dataset handoff

The companion `data_processing` converter produces:

```text
NEW_BUNDLE/
  COMPLETE.json
  lerobot/   # LeRobot v2.1, 10 Hz, three aligned RGB videos and 14D state/action
  tactile/   # summary.json + episodes.jsonl + [T,4,700,3] XYZ and [T,4] masks
```

Configure `lerobot_root` and `tactile_sidecar_root` explicitly. The wrapper sets `HF_LEROBOT_HOME=parent(lerobot_root)` and `repo_id=basename(lerobot_root)` **before importing LeRobot**. Thus `NEW_BUNDLE/lerobot` is resolved exactly, even if the metadata's descriptive repo ID is `physcanvas/custom`. A separate safe `asset_id` prevents dataset IDs from changing output paths.

For new converter bundles, preflight requires `COMPLETE.json.complete=true`, no `FAILED.json`, sidecar completeness, aligned episode/frame counts, and all parquet/video/sidecar files. It does not decode videos. Run the companion full bundle validator for pixel/frame/tactile validation. Legacy complete datasets are accepted without requiring the new bundle marker.

The converter's action contract is next retained measured absolute `q[t+1]`, not a freshly reconstructed robot command. Prompt variants share physical-source tactile data. SFT retains the original action-tail clamping. This distinction matters when comparing to action-command/WM rollouts.

## Safe command sequence

Start from project root; make your own JSON using `policy_training/example.json`. It requires absolute local paths for inputs and a **new, disjoint output root**. `name`, `exp_name`, and `asset_id` are safe single components. Unknown config keys, path traversal and overlapping input/output roots are rejected. Initialization/tokenizer assets are never downloaded implicitly.

```bash
# Dependency-free plan: no writes, no model imports, no dataset decode.
bash policy_training/run.sh plan --config /path/to/sft.json

# Set this to an independently validated environment, not a generic GRPO alias.
export SFT_PYTHON=/path/to/jax_environment/bin/python

# Check data and CPU import/config plumbing. No weight load or training.
bash policy_training/run.sh preflight --config /path/to/sft.json \
  --stage norm-stats --check-imports

# Explicitly create NEW normalization assets. Existing output is never overwritten.
POLICY_TRAINING_ALLOW_WRITE=1 bash policy_training/run.sh norm-stats \
  --config /path/to/sft.json --execute

# This fails closed on the current uncorrected pro5000 JAX CUDA environment.
bash policy_training/run.sh preflight --config /path/to/sft.json \
  --stage train-jax --check-imports --check-gpu

# Only after data, norms, initialization, tokenizer and CUDA checks pass:
POLICY_TRAINING_ALLOW_WRITE=1 bash policy_training/run.sh train-jax \
  --config /path/to/sft.json --execute
```

Running `norm-stats` or `train-jax` without `--execute` prints a plan only. Execution requires **both** `--execute` and `POLICY_TRAINING_ALLOW_WRITE=1`. Torch tactile SFT is rejected even if these flags are supplied. No `--overwrite`/resume interface is provided. W&B is disabled by default, and if enabled is forced offline with a new-output-local directory. Dataset Arrow cache, JAX compile cache, run metadata and checkpoints are routed to the new output root; no old project is a write target.

Normalization output:
`OUTPUT_ROOT/assets/NAME/ASSET_ID/norm_stats.json` plus `provenance.json`.

Checkpoint output:
`OUTPUT_ROOT/checkpoints/NAME/EXP_NAME/STEP/` containing EMA `params`, training state, and copied `assets/ASSET_ID/norm_stats.json`.

The normalizer follows the original all-frames sequential, 16-action-window, episode-tail-clamp, **drop-last** algorithm. `batch_size` must not exceed the number of frames. No tactile/RGB decode is required for this calculation; no arbitrary downsample option is added.

## Convert a newly trained JAX checkpoint

The active converter preserves SigLIP, both Gemma experts, projections, and **all 23 tactile tensors**, including learned gate. It writes FP32 `model.safetensors` and a BF16 inference-hint `config.json`. This is the **OpenPI_RLinf** format consumed by this project's policy, not a generic Hugging Face `PI0Pytorch` checkpoint.

```bash
# First print a conversion plan; does not restore weights.
bash policy_training/run.sh convert \
  --input-params /path/to/SFT_STEP/params \
  --norm-stats /path/to/SFT_STEP/assets/my_asset/norm_stats.json \
  --asset-id my_asset --output /path/to/NEW_TORCH_CHECKPOINT

# Explicit heavy conversion, CPU Orbax restore; not performed during this preparation.
POLICY_TRAINING_ALLOW_WRITE=1 bash policy_training/run.sh convert \
  --input-params /path/to/SFT_STEP/params \
  --norm-stats /path/to/SFT_STEP/assets/my_asset/norm_stats.json \
  --asset-id my_asset --output /path/to/NEW_TORCH_CHECKPOINT --execute

bash policy_training/run.sh validate-torch \
  --checkpoint /path/to/NEW_TORCH_CHECKPOINT --asset-id my_asset
```

The wrapper requires norms carried by the **same JAX checkpoint**, verifies 14 finite state/action stats and ordered quantiles, refuses existing outputs, and records norm-stat SHA256. Output validation reads only safetensors headers and file extents: 23 tactile keys/shapes, required backbone/projection keys and model config. It does not load multi-GB tensors and does **not** claim numerical parity or tensor-value finiteness. Failed conversions are not marked complete; inspect their new output before using a new path for retry.

### Feed the correct pair to closed-loop rollout / GRPO

Always override **both** variables before `scripts/local_paths.sh`:

```bash
export PI05_CHECKPOINT=/path/to/NEW_TORCH_CHECKPOINT
export PI05_NORM_STATS=/path/to/NEW_TORCH_CHECKPOINT/assets/my_asset/norm_stats.json
source scripts/local_paths.sh jimu  # or shiguan; preserves explicit overrides
bash scripts/preflight.sh jimu
```

Changing just `PI05_CHECKPOINT` can leave historical 59999 norm stats active. The validator prints both values; use them as a pair. WM, prompt, reset cache and RM selection remain governed by the separate rollout/GRPO config. This SFT package does not silently launch either stage.

## Verification scope

CPU unit tests cover plan-only behavior, dual execution authorization, unsafe paths/overlap, incomplete bundle markers, finite 14D norms, safetensors header/truncation, tactile key preservation, and checkpoint/norm pairing (10/10 locally and remotely). A synthetic 34-frame, two-variant bundle passed the real LeRobot loader and historical normalizer: four full batches of eight, H16 window/tail comparisons, and finite 14D state/action statistics. The existing 59999 header was inspected successfully: 690 tensors, including all 23 tactile tensors. The tactile-aware converter module imports successfully without restoring a checkpoint. These checks are not a new-weight-conversion or GPU SFT test. See [release validation](../VALIDATION.md) for the current checks and environment limitations.
