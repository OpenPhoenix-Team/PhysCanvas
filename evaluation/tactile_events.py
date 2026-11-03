"""Optional tactile contact and activity-event metrics with frozen thresholds."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _events(mask, min_frames=2, max_gap=1):
    mask = np.asarray(mask, bool).copy()
    for begin, end in _runs(~mask):
        if begin > 0 and end < len(mask) and end - begin <= max_gap:
            mask[begin:end] = True
    return [(int(begin), int(end)) for begin, end in _runs(mask) if end - begin >= min_frames]


def event_metrics(gt, pred, fps=10):
    """Match events by temporal IoU >= 0.1, prioritizing match count."""
    gt, pred = np.asarray(gt, bool), np.asarray(pred, bool)
    if gt.ndim != 1 or gt.shape != pred.shape or fps <= 0:
        raise ValueError("event masks must be matching 1-D sequences with positive fps")
    gs, ps = _events(gt), _events(pred)
    iou = np.zeros((len(gs), len(ps)), dtype=np.float64)
    for i, (a, b) in enumerate(gs):
        for j, (c, d) in enumerate(ps):
            iou[i, j] = max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
    matches = []
    if iou.size:
        eligible = iou >= 0.1
        rows, cols = linear_sum_assignment(-(eligible * (1 + min(iou.shape)) + iou * eligible))
        matches = [(i, j) for i, j in zip(rows, cols) if eligible[i, j]]
    onsets = [abs(gs[i][0] - ps[j][0]) / fps for i, j in matches
              if gs[i][0] > 0 and ps[j][0] > 0]
    offsets = [abs(gs[i][1] - ps[j][1]) / fps for i, j in matches
               if gs[i][1] < len(gt) and ps[j][1] < len(pred)]
    return {
        "gt_events": len(gs), "pred_events": len(ps), "matched_events": len(matches),
        "event_f1": 2 * len(matches) / (len(gs) + len(ps)) if gs or ps else None,
        "onset_error_sec": float(np.mean(onsets)) if onsets else None,
        "offset_error_sec": float(np.mean(offsets)) if offsets else None,
        "onset_count": len(onsets), "offset_count": len(offsets),
    }


def tactile_event_metrics(gt_image, pred_image, gt_flow, pred_flow, points, tau, thresholds):
    """Score model-space tactile-diff images and physical XYZ marker events.

    Images are [T,C,H,W] in [-1,1], with four tactile sides concatenated in
    width. Flow is [T,4,marker,3]; frame zero provides event context only.
    """
    if thresholds.get("frozen_from_gt_only") is not True or thresholds.get("image_space") != "model_space_01":
        raise ValueError("contact/event thresholds must be frozen from GT in model-space [0,1]")
    gt_image = np.asarray(gt_image, dtype=np.float32)
    pred_image = np.asarray(pred_image, dtype=np.float32)
    gt_flow = np.asarray(gt_flow, dtype=np.float32)
    pred_flow = np.asarray(pred_flow, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if gt_image.shape != pred_image.shape or gt_image.ndim != 4:
        raise ValueError("tactile image shapes must match [T,C,H,W]")
    if gt_flow.shape != pred_flow.shape or gt_flow.shape[:2] != (len(gt_image), 4):
        raise ValueError("flow shapes must match [T,4,marker,3]")
    if points.shape != gt_flow.shape[1:3] + (2,) or gt_image.shape[-1] % 4:
        raise ValueError("reference marker points or four-side image packing invalid")
    if not np.isfinite(pred_image).all() or not np.isfinite(pred_flow).all():
        raise ValueError("nonfinite predictions are not filtered")
    image_tau = np.asarray(thresholds["diff_threshold_per_side"], dtype=np.float64)
    image_fraction = np.asarray(thresholds["diff_event_fraction_per_side"], dtype=np.float64)
    flow_fraction = np.asarray(thresholds["flow_event_fraction_per_side"], dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if any(x.shape != (4,) for x in (image_tau, image_fraction, flow_fraction, tau)):
        raise ValueError("all thresholds must have four side values")
    if not (np.isfinite(image_tau).all() and (image_tau >= 0).all()
            and ((image_fraction >= 0) & (image_fraction <= 1)).all()
            and ((flow_fraction >= 0) & (flow_fraction <= 1)).all()):
        raise ValueError("invalid frozen contact/event thresholds")
    gt_image = np.moveaxis(gt_image, 1, -1)
    pred_image = np.moveaxis(pred_image, 1, -1)
    gt_sides = np.stack(np.split(gt_image, 4, axis=2), axis=1)
    pred_sides = np.stack(np.split(pred_image, 4, axis=2), axis=1)
    sides = {}
    for side, name in enumerate(("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")):
        reference = gt_sides[0, side]
        gt_activity = np.abs(gt_sides[:, side] - reference).mean(-1) / 2
        pred_activity = np.abs(pred_sides[:, side] - reference).mean(-1) / 2
        gt_contact = gt_activity > image_tau[side]
        pred_contact = pred_activity > image_tau[side]
        contact_mask = gt_contact[1:] & np.isfinite(gt_sides[1:, side]).all(-1)
        contact_error = np.abs(gt_sides[1:, side] - pred_sides[1:, side]).mean(-1) / 2
        image_events = event_metrics(gt_contact.mean((1, 2)) > image_fraction[side],
                                      pred_contact.mean((1, 2)) > image_fraction[side],
                                      fps=thresholds.get("fps", 10))
        valid = np.isfinite(gt_flow[:, side]).all(-1) & np.isfinite(points[side]).all(-1)[None]
        valid &= (np.linalg.norm(points[side], axis=-1) > 0)[None]
        denominator = np.maximum(valid.sum(-1), 1)
        gt_fraction = ((np.linalg.norm(gt_flow[:, side], axis=-1) > tau[side]) & valid).sum(-1) / denominator
        pred_fraction = ((np.linalg.norm(pred_flow[:, side], axis=-1) > tau[side]) & valid).sum(-1) / denominator
        flow_events = event_metrics(gt_fraction > flow_fraction[side],
                                    pred_fraction > flow_fraction[side], fps=thresholds.get("fps", 10))
        sides[name] = {
            "contact_mae_01": float(contact_error[contact_mask].mean()) if contact_mask.any() else None,
            "contact_pixels": int(contact_mask.sum()),
            "contact_events": image_events,
            "flow_events": flow_events,
        }
    return {"sides": sides, "frame_zero_for_event_context_only": True,
            "image_space": "model-space tactile diff, normalized [0,1] error; not raw-sensor resolution"}
