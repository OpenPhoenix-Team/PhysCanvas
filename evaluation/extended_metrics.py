"""Additional physical marker-displacement errors for the public evaluation."""

import numpy as np


def marker_displacement_errors(gt, pred, points, tau):
    """Score future frames in native physical XYZ units without clipping predictions."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 4 or gt.shape[-1] != 3:
        raise ValueError("GT and prediction must both be [T, side, marker, XYZ]")
    if points.shape != gt.shape[1:3] + (2,) or tau.shape != (gt.shape[1],):
        raise ValueError("Marker reference points or side thresholds do not match flow")
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predicted flow; no prediction filtering permitted")
    valid = np.isfinite(gt).all(-1) & np.isfinite(points).all(-1)[None]
    valid &= (np.linalg.norm(points, axis=-1) > 0)[None]
    active = valid & (np.linalg.norm(gt, axis=-1) > tau[None, :, None])
    idle = valid & ~active
    absolute_error = np.abs(pred - gt)
    amplitude_error = np.abs(np.linalg.norm(pred, axis=-1) - np.linalg.norm(gt, axis=-1))
    active_count = active.sum(axis=(0, 2)).astype(np.int64)
    idle_count = idle.sum(axis=(0, 2)).astype(np.int64)
    active_abs = np.where(active[..., None], absolute_error, 0).sum(axis=(0, 2))
    idle_abs = np.where(idle[..., None], np.abs(pred), 0).sum(axis=(0, 2))
    amplitude_abs = np.where(active, amplitude_error, 0).sum(axis=(0, 2))
    return {
        "side_order": ["left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1"],
        "axis_order": ["x", "y", "z"],
        "active_point_counts_per_side": active_count.tolist(),
        "idle_point_counts_per_side": idle_count.tolist(),
        "active_abs_sum_per_side_axis": active_abs.tolist(),
        "idle_pred_abs_sum_per_side_axis": idle_abs.tolist(),
        "active_amplitude_abs_sum_per_side": amplitude_abs.tolist(),
        "active_mae": float(active_abs.sum() / (3 * active_count.sum())) if active_count.sum() else None,
        "idle_fp_mae": float(idle_abs.sum() / (3 * idle_count.sum())) if idle_count.sum() else None,
        "active_amplitude_mae": float(amplitude_abs.sum() / active_count.sum()) if active_count.sum() else None,
        "display_scale": 1000,
        "definition": "Future frames only; GT valid-marker mask and frozen per-side tau; native physical XYZ units; no prediction clipping",
    }
