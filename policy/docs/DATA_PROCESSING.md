# Pi0.5 supervised-data preparation

This package prepares **supervised Pi0.5 data**, not WM reset caches and not
GRPO replay. It preserves the algorithms used for the 77-source armfix,
7-prompt, checkpoint-59999 experiment. The original source is unchanged;
`data_processing/vendor/` contains pinned implementations verified by `data_processing/original.py`. Only the new manifest-driven orchestrator
is a supported entry point. Do **not** execute historical vendor scripts:
their old CLIs contain obsolete paths, overwrite options and dataset assumptions.

The final H20 armfix spec was inspected but its old Excel/folder corrections
are not imported. No automatic `SWAP_PATHS`, `dan/shuang` guessing, or
`shiguan_other` prompt is applied to new data.

## Runtime

Use Python 3.11 with the original LeRobot **v2.1 API**, pinned upstream revision
`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`, not a current LeRobot v3 writer.
Dependencies include numpy, torch, datasets, pyarrow, OpenCV, PyAV/FFmpeg and
tqdm. In the current server deployment the existing
`/mnt/workspace/xzb/workspace/openpi/.venv/bin/python` supplies these dependencies;
the shared `pro5000_env` is a different GRPO runtime and is not a conversion
environment. Nothing here reinstalls or changes either environment.
`data_processing/run.sh` resets `PYTHONPATH` to this project and disables the
user site, so sourcing a GRPO runtime first does not inject its incompatible
torch/Hugging Face overlays into conversion. If necessary, supply deliberate
extra import roots through `DATA_EXTRA_PYTHONPATH`; the inherited `PYTHONPATH`
is not reused.

Example portable setup uses the matching OpenPI lockfile supplied with the
supervised-training code; do not replace that lock with latest package versions.
Source pixels are kept at their native common shape (historically 480×640),
stored as 10 Hz H.264; Pi0.5 applies its own 224×224 **direct resize** later.
This differs from WM's aspect-ratio-preserving canvas preprocessing.

## Explicit source manifest

Copy and edit `data_processing/example_manifest.json`. Each physical episode
must declare an existing absolute `episode_dir`, unique `physical_episode_id`,
`task`, nonempty `prompt`, `arm_mode` (`left`, `right`, `bimanual`), boolean
`swap_robot_sides`, and `head_variants` (`raw`, `remove_hand`, or both).
The example contains placeholders, not private data paths. `remove_hand` is an
existing input video; this tool does not run a hand-removal model.

Required raw structure:

```text
episode/
  camera/timestamp.pkl
  camera/base_0_rgb.mp4
  camera/base_0_rgb_remove_hand.mp4  # only if explicitly selected
  camera/left_wrist_0_rgb.mp4
  camera/right_wrist_0_rgb.mp4
  robot/robot_state_dict.pkl
  tactile/timestamp.pkl
  tactile/tactile_pointcloud_dict.pkl
```

Pickle can execute arbitrary code. `--trusted-pickle` is required even for a
plan, since time alignment reads pickles. Use only records from a trusted
source. JSON does not make the referenced pickles safe. Missing tactile files
use the original all-zero/**mask=False** fallback; malformed/nonfinite tactile
is an error, not silently accepted.

## Commands and safety

From the independent experiment project root:

```bash
export DATA_PYTHON=/path/to/matching-openpi-env/bin/python

# Default is plan-only; no output directory is created.
bash data_processing/run.sh \
  --manifest /data/manifests/my_sources.json \
  --output /data/derived/my_pi05_bundle_v1 \
  --trusted-pickle

# Explicitly execute after checking the plan.
bash data_processing/run.sh \
  --manifest /data/manifests/my_sources.json \
  --output /data/derived/my_pi05_bundle_v1 \
  --trusted-pickle --execute

# Independent read-only validation; no source pickles are loaded.
"$DATA_PYTHON" -m data_processing.validate \
  --output /data/derived/my_pi05_bundle_v1
```

The output must be new. Existing outputs, symlinks, source directories, source
descendants and ancestors are rejected. There is no overwrite or automatic
resume; a failed new build remains for inspection, with `FAILED.json` and an
incomplete sidecar. Do not point the output at an original dataset.

Source files are never written. All consumed videos/pickles are hashed before
conversion and after validation; changed source bytes prevent completion.
This extra hashing intentionally costs sequential I/O on large datasets.
No automatic download, upload, Hugging Face publication, cache symlink or
source-list discovery occurs.

## Preserved data semantics

- Camera/robot nearest-neighbor alignment: ≤50 ms; base camera is the
  high-rate time axis. Ties choose the earlier sample.
- Remove only leading/trailing static ranges, not pauses inside an action.
  Original per-arm detector uses p95-normalized joint differences, smoothing15,
  minimum active run10, margin10, minimum retained length60 (high-rate frames).
  Threshold is `max(0.05, 3*p15)`. Bimanual trim uses the union of active arms.
  No motion/too-short trim preserves the original valid interval.
- Resample to nominal10 Hz, repeat nearest alignment filtering. Actual retained
  source intervals can exceed100 ms after filtering; output video clock is10 Hz.
- `state[t] = q[t]`; **`action[t] = q[t+1]` at the next retained sample**.
  This is measured absolute14-D joint state, not delta, velocity, pose or
  `action_traces` sent commands. The last observation is dropped because it
  has no next state. Arm order remains left7 then right7.
- Explicit side swap exchanges robot dictionaries only; RGB and tactile are
  not swapped. Arm mode selects trimming, not the14-D output width.
- Four tactile sides are ordered `left_wrist_0,left_wrist_1,right_wrist_0,right_wrist_1`.
  Raw `(700,6)` takes **columns3:6 = dx,dy,dz**.
- Tactile is `D(t)-D(t_ref)`, with reference anchored at common validity **before
  leading-static trimming**. Do not zero it again at the first trimmed frame.
  Nearest tactile sample skew ≤60 ms yields mask=True; stale samples are masked.
  Nearest sampling can choose a slightly later timestamp, not strictly causal.
- No WM `0.06` image encoding, flow modulus, or dataset tactile normalization is
  applied to SFT sidecars. All3 displacement axes remain intact.
- Raw/remove-hand visual variants share action/state and one tactile array.
  Split train/validation by physical episode to avoid leakage; this converter
  does not invent a split. Compute training statistics on the training split.

## Outputs and training handoff

```text
my_pi05_bundle_v1/
  lerobot/
    meta/{info,tasks,episodes,episodes_stats,...}
    meta/episodes_multitask_manifest.jsonl
    meta/conversion_summary.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.*/episode_000000.mp4
  tactile/
    summary.json
    episodes.jsonl
    sources.jsonl
    episodes/chunk-000/<source_id>.npy       # T,4,700,3 float32
    masks/chunk-000/<source_id>.npy          # T,4 bool
    references/chunk-000/<source_id>.npy     # 4,700,3 float32
  source_manifest.json
  source_hashes.json
  validation.json
  COMPLETE.json
```

Set the supervised-training config `lerobot_root` to the `lerobot` child and
`tactile_sidecar_root` to the `tactile` child. Both `COMPLETE.json` and
`tactile/summary.json` must report complete. Completion follows decoded-video,
Parquet, episode/stat counts, task mappings, next-state actions, full-XYZ,
mask/reference lengths, shared-variant checks and source-hash verification.
An incomplete bundle is not training-ready even if some Parquet files exist.

Validation does not measure real-robot task success or prove camera calibration.
It verifies this data representation and index alignment. New hardware still
requires explicit unit/order/action-interface review.

## Tests

```bash
"$DATA_PYTHON" -m unittest discover -s tests -p 'test_data_processing.py' -v

# Only generated fixtures, not any real demonstration:
PHYSCANVAS_TEST_DATA_WRITER=1 "$DATA_PYTHON" -m unittest discover \
  -s tests -p 'test_data_processing.py' -v
```

Synthetic tests cover original trim bounds/static fallback, alignment tolerance,
next-state actions, explicit side swap, trusted-pickle gate, source/output
isolation, pre-trim XYZ reference, stale/missing tactile masks, optional real
LeRobot v2.1 video/Parquet writing, deduplicated visual variants, source hashes,
and refusal to overwrite. Test temporary files are generated fixtures only.

Verified on pro5000 on 2026-11-04: all six tests passed, including the optional
real writer test. One synthetic physical episode produced two visual variants,
17 frames each, with six decodable H.264 videos and a shared full-XYZ sidecar.
The test also injected a failed final completion write and checked the sidecar
returned to incomplete; source hashes stayed unchanged. This is not a full
real-data conversion or a model-training result. The historical PyAV/LeRobot
encoder emitted `ResourceWarning` for its first PNG reader; output decode and
validation succeeded. The original dependency was not modified.
