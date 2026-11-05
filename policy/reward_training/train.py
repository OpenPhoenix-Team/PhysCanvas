"""Code-only RM training. Inputs stay private and read-only; default is plan-only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def configuration(path, *, check_output=True):
    cfg = json.loads(Path(path).read_text())
    for key in ("frames", "cache", "split", "output"):
        p = Path(cfg[key]).expanduser()
        if not p.is_absolute():
            raise ValueError(f"{key} must be absolute")
        cfg[key] = str(p.resolve())
    out = Path(cfg["output"])
    for key in ("frames", "cache", "split"):
        src = Path(cfg[key])
        if out == src or src in out.parents or out in src.parents:
            raise ValueError("Output must be disjoint from inputs")
    if check_output and out.exists():
        raise FileExistsError("Output exists; overwrite/resume is not supported")
    for key, default in (("epochs", 200), ("global_batch_size", 128), ("warmup_epochs", 3),
                         ("save_every", 20), ("seed", 42), ("workers", 4)):
        cfg.setdefault(key, default)
        if type(cfg[key]) is not int or cfg[key] < (0 if key in ("warmup_epochs", "workers", "seed") else 1):
            raise ValueError(f"Invalid {key}")
    cfg.setdefault("lr", 1e-4)
    cfg.setdefault("weight_decay", 1e-5)
    cfg.setdefault("threshold", .5)
    cfg.setdefault("pretrained", True)
    cfg.setdefault("device", "auto")
    if cfg["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    import math
    if not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0 or not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("Invalid optimizer parameters")
    if not 0 <= cfg["threshold"] <= 1 or type(cfg["pretrained"]) is not bool:
        raise ValueError("Invalid threshold/pretrained")
    return cfg


def partitions(rows, split):
    train, val = set(split["train_episodes"]), set(split["val_test_episodes"])
    known = {r["sample_id"] for r in rows}
    if train & val or train | val != known or not train or not val:
        raise ValueError("Split must cover all episodes once, with nonempty disjoint train/val")
    for row in rows:
        sid = row["sample_id"]
        if not isinstance(sid, str) or not sid or Path(sid).name != sid or sid in (".", ".."):
            raise ValueError("sample_id must be one safe filename component")
        if type(row["label"]) is not int or row["label"] not in (0, 1) or type(row["aligned_frame_index"]) is not int or row["aligned_frame_index"] < 0:
            raise ValueError("Invalid frame index or binary label")
    keys = [(r["sample_id"], r["aligned_frame_index"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate labeled frames")
    return ([i for i, r in enumerate(rows) if r["sample_id"] in train],
            [i for i, r in enumerate(rows) if r["sample_id"] in val])


def balanced_batches(rows, indices, batch_size, rank, world_size, seed):
    import numpy as np
    if batch_size % (2 * world_size):
        raise ValueError("global_batch_size must be divisible by 2 * world_size; no accumulation")
    rng = np.random.default_rng(seed)
    pos = rng.permutation([i for i in indices if rows[i]["label"] == 1])
    neg = rng.permutation([i for i in indices if rows[i]["label"] == 0])
    half = batch_size // 2
    count = min(len(pos), len(neg)) // half
    if not count:
        raise ValueError("Not enough examples of both classes for one global batch")
    per_rank = half // world_size
    result = []
    for start in range(0, count * half, half):
        offset = start + rank * per_rank
        batch = np.concatenate((pos[offset:offset + per_rank], neg[offset:offset + per_rank]))
        rng.shuffle(batch)
        result.append(batch.tolist())
    return result


def train(cfg):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from .core import ThreeViewResNet18, ThreeViewFrameDataset, evaluate
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    use_cuda = cfg["device"] != "cpu" and torch.cuda.is_available()
    if cfg["device"] == "cuda" and not use_cuda:
        raise RuntimeError("Requested CUDA but it is unavailable")
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # Every rank checks its read-only inputs before distributed setup.
    rows = [json.loads(line) for line in Path(cfg["frames"]).read_text().splitlines() if line.strip()]
    split = json.loads(Path(cfg["split"]).read_text())
    tr, va = partitions(rows, split)
    arrays = {}
    cache = Path(cfg["cache"])
    for row in rows:
        sid = row["sample_id"]
        if sid not in arrays:
            p = (cache / (sid + ".npy")).resolve()
            if cache not in p.parents:
                raise ValueError("Cache path escapes root")
            arrays[sid] = np.load(p, mmap_mode="r", allow_pickle=False)
        a = arrays[sid]
        if a.dtype != np.uint8 or a.shape[1:] != (3, 3, 224, 224) or row["aligned_frame_index"] >= len(a):
            raise ValueError("Cache must be aligned uint8 [T,view=3,RGB=3,224,224]")
    del arrays
    balanced_batches(rows, tr, cfg["global_batch_size"], rank, world, cfg["seed"])
    p = [i for i in va if rows[i]["label"] == 1]
    n = [i for i in va if rows[i]["label"] == 0]
    m = min(len(p), len(n))
    if not m:
        raise ValueError("Validation needs both labels")
    rng = np.random.default_rng(cfg["seed"])
    balanced_val = np.concatenate((rng.permutation(p)[:m], rng.permutation(n)[:m])).tolist()
    out = Path(cfg["output"])
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        setup_error = [None]
        if rank == 0:
            try:
                out.mkdir(parents=True, exist_ok=False)
                (out / "config.json").write_text(json.dumps(cfg, indent=2))
            except Exception as exc:
                setup_error[0] = f"{type(exc).__name__}: {exc}"
        if world > 1:
            dist.broadcast_object_list(setup_error, src=0)
        if setup_error[0]:
            raise RuntimeError(setup_error[0])
        torch.manual_seed(cfg["seed"])
        model = ThreeViewResNet18(pretrained=cfg["pretrained"]).to(device)
        wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        base = ThreeViewFrameDataset(rows, cache)
        best = float("inf")
        for epoch in range(cfg["epochs"]):
            torch.manual_seed(cfg["seed"] + epoch * world + rank)
            batches = balanced_batches(rows, tr, cfg["global_batch_size"], rank, world, cfg["seed"] + epoch)
            loader = DataLoader(base, batch_sampler=batches, num_workers=cfg["workers"])
            wrapped.train()
            sums = torch.zeros(2, device=device, dtype=torch.float64)
            for step, (images, labels) in enumerate(loader):
                progress = epoch + (step + 1) / len(batches)
                lr = cfg["lr"] * min(1., progress / max(1e-12, cfg["warmup_epochs"]))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(wrapped(images.to(device)), labels.to(device))
                finite = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                optimizer.step()
                sums += torch.tensor([loss.item() * len(labels), len(labels)], device=device)
            if world > 1:
                dist.all_reduce(sums)
            # DDP buffer broadcast before each forward aligns BN starting state;
            # rank-zero evaluation/checkpoint use rank-zero final running statistics.
            if rank == 0:
                def ev(indices):
                    ds = ThreeViewFrameDataset(rows, cache, np.asarray(indices))
                    return evaluate(model, DataLoader(ds, batch_size=cfg["global_batch_size"], num_workers=cfg["workers"]), device, cfg["threshold"])
                metrics = {"epoch": epoch + 1, "train_loss": (sums[0] / sums[1]).item(),
                           "val_balanced": ev(balanced_val), "val_full": ev(va), "learning_rate": lr}
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                state = {"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                         "threshold": cfg["threshold"], "view_order": ["base", "left_wrist", "right_wrist"],
                         "config": cfg, "metrics": metrics}
                if metrics["val_balanced"]["loss"] < best:
                    best = metrics["val_balanced"]["loss"]
                    torch.save(state, out / "best.pt")
                if (epoch + 1) % cfg["save_every"] == 0 or epoch + 1 == cfg["epochs"]:
                    torch.save(state, out / f"epoch_{epoch + 1:04d}.pt")
                print(json.dumps(metrics), flush=True)
            if world > 1:
                dist.barrier()
    finally:
        if world > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    # Rank zero creates the directory only after every rank reaches the group.
    # Checking existence independently here races with a faster rank zero.
    cfg = configuration(args.config, check_output=not args.execute)
    if not args.execute:
        print(json.dumps({"mode": "plan_only", **cfg}, indent=2))
        return
    if os.getenv("RM_ALLOW_TRAINING") != "1":
        raise PermissionError("Requires --execute and RM_ALLOW_TRAINING=1")
    train(cfg)


if __name__ == "__main__":
    main()
