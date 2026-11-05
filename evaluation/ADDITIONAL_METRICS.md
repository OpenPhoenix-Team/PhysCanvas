# Evaluation metrics

## Entry points

- `physcanvas-evaluate-reconstruction`: RGB and tactile-image MSE, PSNR, SSIM, and LPIPS only; no I3D, VBench, or flow thresholds required.
- `physcanvas-evaluate-visual`: reconstruction metrics plus FID/FVD and RGB VBench; requires `--i3d-weights` and `--vbench-repo`.
- `physcanvas-evaluate-all`: all visual, tracking, marker-displacement, and contact/event metrics; additionally requires `--threshold-json`, `--contact-threshold-json`, `--cotracker-repo`, `--cotracker-checkpoint`, and `--queries-json`.

The new entry points fail on missing suite assets rather than silently omitting a requested family. Tactile flow is always visualized but is scored only numerically in the full suite. Reconstruction and visual profiles do not compute flow, marker, or contact/event scores. The legacy `physcanvas-evaluate-n128` command remains available with optional VBench, tracking, and contact/event families.

For visual and full evaluation, install the official [VBench](https://github.com/Vchitect/VBench) package and download its model assets; supply the repository path and a TorchScript I3D detector. Full evaluation also requires the official [CoTracker3](https://github.com/facebookresearch/co-tracker) code and offline checkpoint. Contact thresholds and semantic queries must be prepared as described below; they are not generic defaults.

## Metric definitions

The N=128 evaluation already computes paired RGB and tactile-diff MSE, PSNR,
SSIM, LPIPS, FID, and FVD. Physical tactile flow is scored separately with
overall, active, and idle false-positive MSE. These are distinct from the
additional metrics below; no score from another checkpoint or cohort is reused.

| Family | Metric | Definition and required input | Current status |
| --- | --- | --- | --- |
| RGB / VBench | Subject consistency | Official VBench DINO frame-feature consistency on clean predicted RGB video, score x100 | Unified entry with `--vbench-repo` and official assets |
| RGB / VBench | Background consistency | Official VBench CLIP frame-feature consistency on the same clean RGB video, score x100 | Same unified entry |
| RGB / VBench | Motion smoothness | Official VBench AMT-S midpoint interpolation, score x100 | Same unified entry; requires AMT-S weights |
| RGB / tracking | End-effector and object error (px) | Paired GT/predicted CoTracker3 trajectories from manually reviewed semantic frame-0 points | Unified entry with CoTracker3 source, checkpoint, and query points |
| Tactile image | Contact MAE | Mean absolute tactile-diff pixel error under a GT-derived contact mask | Optional `--contact-threshold-json` in N=128 evaluation; model-space threshold evidence required |
| Tactile image | Onset/offset error (s), event F1 | Match GT/predicted contact events; 10 Hz, minimum two frames, bridge gaps of at most one frame, temporal IoU >= 0.1 | Same optional threshold file; undefined events remain null |
| Marker displacement | Active MAE, idle FP-MAE, active amplitude MAE | GT valid-marker mask; GT activity from frozen per-side physical-flow tau; no prediction clipping | Implemented in `extended_metrics.py`, scored on future frames and aggregated as pooled micro and episode macro |
| Marker displacement | Onset/offset error (s), event F1 | Apply a frozen fraction-of-active-markers threshold per side before matching events | Optional `--contact-threshold-json`; the pointwise tau alone is insufficient |

The marker MAEs are stored in native physical XYZ displacement units. Multiply
by 1000 only for a table labelled `MAE x 10^3`; the JSON retains unscaled
values. All metric reports must include the episode IDs, frame/clip counts,
checkpoint hash, sampling rate, threshold evidence, and model-asset versions.

The `physcanvas-evaluate-n128` entry invokes the
[official VBench custom-input implementation](https://github.com/Vchitect/VBench)
on **clean, unannotated RGB predictions**, not the composite visualization.
Supplying VBench or CoTracker3 assets automatically exports clean GT and
predicted RGB videos. The CoTracker3 runner requires a JSON file of
reviewed points: `{"episodes": {"episode_id": [{"kind": "eef", "frame": 0,
"x": 100, "y": 100}, {"kind": "object", "frame": 0, "x": 200, "y": 100}]}}`.
It reports tracker-estimated paired-video error, not human-annotated object
trajectory ground truth. The demo archives do not include these queries.

Optional contact/event thresholds are a four-side JSON with
`frozen_from_gt_only: true`, `image_space: "model_space_01"`, `fps: 10`, and
arrays `diff_threshold_per_side`, `diff_event_fraction_per_side`, and
`flow_event_fraction_per_side`. They must be derived and frozen outside the
test set; no default thresholds are provided. These tactile-diff contact
scores use the generated 224-pixel model canvas, not the original sensor
resolution, so they are not directly interchangeable with older tables.
Never present an unavailable metric as zero or copy scores from a different run.

The same command accepts `--contact-threshold-json`, `--vbench-repo`,
`--cotracker-repo`, `--cotracker-checkpoint`, and `--queries-json`. It writes
`evaluation_suite.json` alongside the standard aggregate. Existing completed
inference may be reused only when its run specification matches and clean RGB
was already exported. These optional score families have not been run on the
demo samples merely by adding their code paths.
