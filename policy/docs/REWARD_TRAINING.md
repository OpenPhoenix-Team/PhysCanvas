# Three-view binary reward model

Inputs are simultaneous base/left-wrist/right-wrist RGB images. Directly resize
each view to 224×224. A shared ImageNet-pretrained ResNet18 encodes each view;
concatenate 3×512 features, then Linear(1536,256), ReLU, Dropout(.1), Linear(256,1).
Training uses BCE-with-logits; probability is sigmoid(logit), already in [0,1].
The threshold is used for classification metrics and rollout, not in BCE.

Prepare privately:

* `cache/<sample_id>.npy`: uint8 `[T,3 views,3 RGB channels,224,224]`.
* `frames.jsonl`: `{"sample_id":"example","aligned_frame_index":0,"label":0}`.
* `split.json`: disjoint `train_episodes` and `val_test_episodes` lists covering
  every episode. Split whole trajectories, never neighboring frames randomly.

Failed trajectories contain only zero labels. For successful trajectories,
frames before the human onset are zero; onset and later frames are one. Ensure
all views are timestamp-aligned before annotating. No input data are modified.

Copy `reward_training/example.json` and set absolute private input paths and a
new output directory. Training batches use exactly 1:1 positive/negative frames
across trajectories, without replacement inside an epoch; excess majority
frames and incomplete batches are dropped. Validation reports both a fixed
balanced subset and all held-out frames. Sharing validation/test is supported
for compatibility but does **not** provide an independent final test estimate.

```bash
python -m reward_training.train --config /absolute/path/rm.json  # plan only
RM_ALLOW_TRAINING=1 torchrun --standalone --nproc_per_node=4 \
  -m reward_training.train --config /absolute/path/rm.json --execute
```

Defaults: 200 epochs, global batch128, no accumulation, LR1e-4 for all parameters,
AdamW weight_decay1e-5, linear warmup3epochs then constant LR, save every20epochs,
no early stopping, metric threshold.5. Each rank receives32 examples with four
GPUs. `global_batch_size` must be divisible by `2*world_size`.

Outputs: `metrics.jsonl`, `best.pt` (lowest balanced validation BCE), periodic
`epoch_XXXX.pt`. These contain `model_state_dict`, threshold and view order;
they load into the GRPO environment scorer. Optimizer resume is not implemented. Training inputs, logs and trained RM weights are not distributed.

GPU smoke with generated images only:

```bash
python -m reward_training.smoke --output /absolute/path/to/NEW_rm_smoke
```

This uses `pretrained=False` to avoid implicit downloads and checks a real
optimizer step, strict checkpoint reload and sixteen-frame prediction.
