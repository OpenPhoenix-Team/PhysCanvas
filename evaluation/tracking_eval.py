"""CoTracker3 paired-video point error with reviewed semantic queries."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .vbench_eval import sha256, validate_inputs


def point_error(gt_xy, pred_xy, gt_visible, pred_visible, kinds):
    gt_xy, pred_xy = np.asarray(gt_xy), np.asarray(pred_xy)
    gt_visible, pred_visible = np.asarray(gt_visible, bool), np.asarray(pred_visible, bool)
    if gt_xy.shape != pred_xy.shape or gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
        raise ValueError("tracks must match [T,point,XY]")
    if gt_visible.shape != gt_xy.shape[:2] or pred_visible.shape != gt_visible.shape:
        raise ValueError("track visibility shapes do not match")
    if len(kinds) != gt_xy.shape[1] or not np.isfinite(gt_xy).all() or not np.isfinite(pred_xy).all():
        raise ValueError("invalid semantic points or nonfinite tracks")
    distances = np.linalg.norm(pred_xy[1:] - gt_xy[1:], axis=-1)
    both = gt_visible[1:] & pred_visible[1:]
    result = {}
    for kind in ("eef", "object"):
        columns = np.asarray([value == kind for value in kinds], dtype=bool)
        if not columns.any():
            raise ValueError(f"no reviewed {kind} query points")
        mask = both[:, columns]
        gt_count = int(gt_visible[1:, columns].sum())
        result[kind] = {
            "mean_error_px": float(distances[:, columns][mask].mean()) if mask.any() else None,
            "matched_visible_points": int(mask.sum()),
            "gt_visible_points": gt_count,
            "coverage": float(mask.sum() / gt_count) if gt_count else None,
        }
    return result


def read_video(path):
    import cv2

    stream = cv2.VideoCapture(str(path))
    if not stream.isOpened():
        raise ValueError(f"cannot decode video: {path}")
    frames = []
    try:
        while True:
            ok, frame = stream.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        stream.release()
    if not frames:
        raise ValueError(f"empty video: {path}")
    return np.stack(frames)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--queries-json", type=Path, required=True)
    parser.add_argument("--cotracker-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    spec, records, pred_dir = validate_inputs(args.run_dir)
    query_data = json.loads(args.queries_json.read_text())
    query_map = query_data["episodes"]
    if set(query_map) != {row["episode_id"] for row in records}:
        raise ValueError("reviewed query episodes differ from the fixed selection")
    if not (args.cotracker_repo / "hubconf.py").is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("CoTracker3 source or checkpoint is missing")
    import torch

    predictor = torch.hub.load(str(args.cotracker_repo), "cotracker3_offline",
                               source="local", pretrained=False).to(args.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    predictor.model.load_state_dict(state, strict=True)
    rows = []
    for record in records:
        episode_id = record["episode_id"]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id))
        horizon = None if spec["horizon"] == "full" else int(spec["horizon"].removeprefix("n"))
        count = min(int(record["sample_num_frames"]), horizon) if horizon else int(record["sample_num_frames"])
        name = f"{safe}_{count}f.mp4"
        pred = read_video(pred_dir / name)
        gt = read_video(pred_dir.parent / "clean_rgb_gt" / name)
        if pred.shape != gt.shape or len(pred) != count:
            raise ValueError(f"paired video shape/frame mismatch: {episode_id}")
        queries = query_map[episode_id]
        if not queries or any(point.get("kind") not in ("eef", "object") or point.get("frame") != 0
                              for point in queries):
            raise ValueError("each reviewed query needs kind=eef/object and frame=0")
        height, width = pred.shape[1:3]
        if any(not (0 <= point["x"] < width and 0 <= point["y"] < height) for point in queries):
            raise ValueError(f"query outside RGB frame: {episode_id}")
        query_tensor = torch.tensor([[0, point["x"], point["y"]] for point in queries],
                                    dtype=torch.float32, device=args.device)
        query_tensor = query_tensor.unsqueeze(0)
        tracks = []
        with torch.inference_mode():
            for video in (gt, pred):
                tensor = torch.from_numpy(video.copy()).permute(0, 3, 1, 2)[None].float().to(args.device)
                xy, visible = predictor(tensor, queries=query_tensor)
                tracks.append((xy[0].cpu().numpy(), visible[0].cpu().numpy()))
                del tensor
        scored = point_error(tracks[0][0], tracks[1][0], tracks[0][1], tracks[1][1],
                             [point["kind"] for point in queries])
        rows.append({"episode_id": episode_id, "frames": count, "error": scored})
    summary = {"source_run_spec": spec, "episodes": len(rows),
               "checkpoint_sha256": sha256(args.checkpoint),
               "queries_sha256": sha256(args.queries_json),
               "definition": "CoTracker3 offline paired GT/predicted RGB tracks from reviewed frame-0 semantic points; future mutually visible frames only",
               "rows": rows, "aggregate": {}}
    for kind in ("eef", "object"):
        values = [row["error"][kind] for row in rows]
        matched = sum(row["matched_visible_points"] for row in values)
        gt_visible = sum(row["gt_visible_points"] for row in values)
        defined = [row["mean_error_px"] for row in values if row["mean_error_px"] is not None]
        summary["aggregate"][kind] = {
            "episode_macro_error_px": float(np.mean(defined)) if defined else None,
            "coverage": matched / gt_visible if gt_visible else None,
            "defined_episodes": len(defined),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
