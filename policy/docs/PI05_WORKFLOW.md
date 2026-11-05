# Pi0.5: data → supervised training → PhysCanvas rollout → GRPO

This independent project prepares the code for both supervised Pi0.5 training
and the Jimu / double-arm Shiguan world-model experiments. Existing repositories,
source demonstrations, model weights and historical experiments are read-only.
Nothing in this preparation starts formal training or publishes code/weights.

## Which code is used for each stage?

| Stage | Code in this project | Contract |
|---|---|---|
| Align, trim idle head/tail, resample | `data_processing/` | Explicit episode manifest; historical ARX alignment/cropping algorithms |
| Write training dataset | `data_processing/` | LeRobot v2.1, three RGB views, 14-dimensional state/action, tactile sidecar |
| Compute normalization, supervised fine-tune | `policy_training/` | Original tactile-aware JAX OpenPI implementation used for arm-fix 59999 |
| Convert inference parameters | `policy_training/` | Tactile-aware JAX → RLinf PyTorch conversion, with matching norm stats |
| Closed-loop inference | `scripts/rollout.sh` | Converted policy, current published PhysCanvas WM (no RM) |
| GRPO | `scripts/train_grpo.sh` | Existing RLinf flow-SDE policy update with the new public WM backend |

The standalone SFT source snapshot is deliberately separate from
`vendor/openpi`, which serves the rollout/GRPO runtime. Do not source the GRPO
runtime as a JAX GPU training environment: it intentionally selects JAX CPU
for PyTorch inference and contains different dependency overlays.

## Training-data semantics

The packaged historical SFT recipe uses:

- Three timestamp-aligned RGB cameras, nominally 10 Hz, directly resized to
  224×224 by the training transform.
- State: measured absolute left-arm 7 + right-arm 7 joint/gripper values.
- Action label: the **next retained measured joint state**, not the recorded
  sent-command stream. The 16-action training window does not cross episodes.
- Four tactile sensors in fixed order: left wrist 0/1, right wrist 0/1. Each
  contributes `(700,3)` full signed XYZ displacement, relative to the selected
  **pre-trim** reference. Invalid tactile samples use a separate boolean mask.
- Quantile normalization for state/actions; the historical policy does not
  apply the world-model image encoding scale to its tactile input.

An explicit manifest determines task prompt, arm mode, robot-side swapping,
and head-video variants. Historical correction lists are not inferred for
new recordings. The converter does not generate remove-hand videos.
Legacy nested ARX recordings are the implemented source contract; newer
flat V3 recordings and actual `action_traces` labels require a separately
reviewed adapter, not a silent change of label semantics.

Raw/remove-hand variants of one physical episode must stay in the same split
if you add train/validation splitting. The historical 59999 recipe used all
selected episodes for training and did not have independent validation or
early stopping; this packaging does not invent historical validation results.

See [data processing](DATA_PROCESSING.md) for the manifest, commands, output
validation and source-protection checks. See [policy training](POLICY_TRAINING.md)
for the parameterized training configuration, environment, normalization and
conversion commands.

## Connect a newly converted policy to rollout / GRPO

Configure the converted checkpoint and its matching normalization statistics
in your private paths file as described in the
[policy training guide](POLICY_TRAINING.md#feed-the-correct-pair-to-closed-loop-rollout--grpo).
From the `policy/` directory, run:

```bash
export PHYSCANVAS_PATHS_FILE="$HOME/.config/physcanvas/paths.sh"
bash scripts/preflight.sh building_blocks --rollout --imports  # use test_tube for the both-arm test-tube task
```

After model/normalizer contract checks, the same environment variables select
that policy for `scripts/rollout.sh building_blocks` or `scripts/rollout.sh test_tube`.
Only after a satisfactory rollout and a prepared cluster should the explicit
`RUN_TRAINING=1` GRPO entry be used. No such training is launched here.

The public WM pin, its XYZ encoding, task RMs and reset caches do not change
when preparing SFT code. A new policy must retain the current adapter's
three-view / 14-state / 16-action / four-XYZ-sensor contract, or the adapter
needs a separate explicit change.

## Preparation versus execution

This is a server-side code handoff, not a completed public release or a claim
that every target machine can train. The original OpenPI training scripts and
dependency lock are retained with their license. The original non-tactile
PyTorch SFT script is not presented as a trainer for the tactile policy.

Current pro5000 checks found that the old OpenPI environment supports CPU
JAX operations but fails CUDA backend initialization. It can be used for the
validated CPU data-preparation checks, not claimed as a verified JAX GPU
training environment. No shared environment was modified to work around this.
The existing PyTorch rollout/GRPO runtime remains separate and unchanged.

No real dataset was rebuilt, no 60k supervised training was run, and no new
full-size checkpoint conversion or formal GRPO was started by this preparation.
Trained reward-model weights remain external/private; raw demonstrations and
historical training artifacts are not included in the code package.
