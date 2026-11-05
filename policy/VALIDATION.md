# Reproduction status (2026-11-05)

This table distinguishes execution checks from task-performance reproduction.
Neither private training data nor trained task-RM weights are released. New
training therefore requires your own inputs; this code release cannot by
itself regenerate historical reward curves or exact SFT weights.

| Check | Evidence/status |
|---|---|
| Published Pi0.5 base | PyTorch loader instantiated 3.36B parameters, four XYZ tactile sensors, H16; this is not a JAX checkpoint |
| Public WM | Fixed epoch30_step300; strict load missing=0/unexpected=0; published SHA256 verified |
| Frozen closed-loop, both tasks | Previous server smoke: 2 replans, 4 WM calls, 32 new frames and 33 video frames each; generated feedback fingerprints match |
| Published-layout test-tube rollout | Rerun from the release tree: 32 generated/33 displayed frames, 10Hz, 2 policy replans/4 WM calls at 2 denoising steps; generated feedback equality verified; return=0 (not a performance benchmark) |
| Real Pi0.5 optimizer | Eight flow-SDE samples; recomputed logprob max error=0 for all; finite gradient norm .4453125; one AdamW step changed 190 tensors; peak allocated 10,196,990,976 bytes |
| Optimizer smoke reward | Fixed diagnostic [0,0,0,0,1,1,1,1], not WM/RM returns; sequential replay, not production distributed BS |
| Three-view RM | One GPU optimizer epoch on generated images, strict reload, [1,16] label output passed |
| RM distributed | Two CPU/Gloo ranks completed one epoch, balanced and full validation; explicit CPU device selected |
| Published-layout tests | 28/28 tests passed from the new code tree in 78.123 seconds using the existing server runtime |
| Data and SFT plumbing | Previous real LeRobot synthetic conversion 34 frames, tactile sidecars and CPU normalization passed; 16 safety/contract tests passed |
| Published-layout data/SFT tests | 15 default tests plus the explicitly enabled real LeRobot/video/XYZ-sidecar writer test passed; no real demonstrations used |
| Historical JAX SFT | Code and lock provided; current available legacy interpreter has CPU only; GPU SFT is NOT verified |
| New clean runtime | Clean installation was not fully validated; do not label the recipe fully reproduced until dependencies/imports/rollout pass |
| Production 2-node/16-GPU run | NOT rerun for this release; no formal new training or success-rate claim |

Source weights, raw robot data and original repositories were not modified.
Only disposable in-memory policy parameters were updated during the optimizer
diagnostic. No derived RM/policy smoke weights are part of this release.

## Weight integrity

The three ModelScope files have source-matching sizes and SHA256:

* `policy/pi05_base/model.safetensors`: 13,439,494,196 bytes;
  `abfb805d367d8cb5b2326e8fbab6e3621fd3e2d2d6a70586406e19023539784d`.
* `policy/pi05_test_tube_wm_grpo/full_weights.pt`: 6,725,464,831 bytes;
  `3a528352b48605f9995e713cd820812b7df3164b742f03b42694a9b257140ac1`.
* `policy/pi05_building_blocks_wm_grpo/full_weights.pt`: 6,725,464,831 bytes;
  `d1dfae4dc2f9e31472016de1361f60619bbf5fb0fe2c82110e3fd1264c15b49f`.
