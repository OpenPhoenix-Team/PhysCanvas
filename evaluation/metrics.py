"""Compact episode metrics; temporary distribution features are removed on finalize."""
import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import uuid
import os

import numpy as np
import torch
from .image_metrics import ImageMetricAccumulator
from .extended_metrics import marker_displacement_errors
from .tactile_events import tactile_event_metrics


def flow_statistics(gt, pred, points, tau):
    gt, pred, points = map(np.asarray, (gt, pred, points))
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite predicted flow; no prediction filtering permitted')
    valid = np.isfinite(gt).all(-1) & np.isfinite(points).all(-1)[None]
    valid &= (np.linalg.norm(points, axis=-1) > 0)[None]
    active = np.linalg.norm(gt, axis=-1) > np.asarray(tau)[None, :, None]
    out = {}
    for name, mask, values in (
        ('overall', valid, (pred - gt) ** 2),
        ('active', valid & active, (pred - gt) ** 2),
        ('idle_fp', valid & ~active, pred ** 2),
    ):
        counts = mask.sum(axis=(0, 2)).astype(np.int64)
        sse = np.where(mask[..., None], values, 0).sum(axis=(0, 2), dtype=np.float64)
        per_axis = np.divide(sse, counts[:, None], out=np.full_like(sse, np.nan), where=counts[:, None] > 0)
        out[name] = {'point_counts_per_side': counts.tolist(), 'sse_per_side_axis': sse.tolist(),
                     'mse': float(sse.sum() / (3 * counts.sum())) if counts.sum() else None,
                     'axis_mse': (sse.sum(0) / counts.sum()).tolist() if counts.sum() else None,
                     'side_axis_mse': [[None if not np.isfinite(x) else float(x) for x in row] for row in per_axis]}
    n = sum(out['overall']['point_counts_per_side'])
    out['active_coverage'] = sum(out['active']['point_counts_per_side']) / n if n else None
    out['tau_per_side'] = list(tau)
    out['side_order'] = ['left_wrist_0', 'left_wrist_1', 'right_wrist_0', 'right_wrist_1']
    out['axis_order'] = ['x', 'y', 'z']
    out['definition'] = 'Physical XYZ displacement, scale 0.06, GT finite/nonzero-marker mask, no predicted clipping/filtering, frame 0 excluded'
    return out


class EpisodeMetrics:
    def __init__(self, device, threshold_path, root, i3d_weights, contact_threshold_path=None, profile="all"):
        self.device, self.root = device, Path(root)
        if profile not in ("reconstruction", "visual", "all"):
            raise ValueError(f"unknown metric profile: {profile}")
        self.profile = profile
        self.tau = json.loads(Path(threshold_path).read_text())['tau_per_side'] if profile == "all" else None
        self.contact_thresholds = (json.loads(Path(contact_threshold_path).read_text())
                                   if contact_threshold_path else None)
        self.contact_threshold_sha256 = (hashlib.sha256(Path(contact_threshold_path).read_bytes()).hexdigest()
                                         if contact_threshold_path else None)
        self.images = ImageMetricAccumulator(['rgb', 'tactile_diff'], device, frame_dir=None)
        if self.images._ensure_lpips() is None:
            raise RuntimeError(self.images.lpips_error)
        if profile != "reconstruction":
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.inception = FrechetInceptionDistance(feature=2048, normalize=False).to(device).inception.eval()
            self.i3d = torch.jit.load(str(i3d_weights), map_location=device).eval()

    @torch.inference_mode()
    def evaluate(self, record, args, gt, pred, labels, pred_labels, distances, gt_flow, pred_flow, points, decoded_gt_flow):
        self.images.records.clear()
        features = {}
        for name, offset in [('rgb', 0), ('tactile_diff', 224)]:
            a, b = gt[1:, :, offset:offset+224], pred[1:, :, offset:offset+224]
            au = ((a.float() + 1) * 127.5).clamp(0, 255).byte()
            bu = ((b.float() + 1) * 127.5).clamp(0, 255).byte()
            for start in range(0, len(a), 8):
                self.images.add(name, au[start:start+8].permute(0, 2, 3, 1).numpy(),
                                bu[start:start+8].permute(0, 2, 3, 1).numpy(),
                                a[start:start+8], b[start:start+8], 0, 0, start+1)
            if self.profile == "reconstruction":
                continue
            for role, data in [('gt', au), ('pred', bu)]:
                feats = [self.inception(data[i:i+8].to(self.device)).float().flatten(1).cpu().numpy()
                         for i in range(0, len(data), 8)]
                features[f'{name}_fid_{role}'] = np.concatenate(feats)
                clips = []
                for i in range(0, len(data)-15, 8):
                    clip = data[i:i+16].permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(self.device)
                    clips.append(self.i3d(clip, rescale=True, resize=True, return_features=True).float().flatten(1).cpu().numpy())
                features[f'{name}_fvd_{role}'] = np.concatenate(clips) if clips else np.empty((0, 400), np.float32)
        if self.images.lpips_error:
            raise RuntimeError(self.images.lpips_error)
        image_result = {'images': self.images.summary()['overall'], 'frame_zero_excluded': True,
                        'metric_profile': self.profile}
        if self.profile == "reconstruction":
            return image_result
        feature_dir = self.root / 'temporary_features' / args.split
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_file = feature_dir / (record['episode_id'] + '.npz')
        staged = feature_file.with_name(feature_file.name + '.partial.' + uuid.uuid4().hex)
        try:
            staging_dir = os.environ.get('PHYS_CANVAS_TMPDIR') or (
                '/dev/shm' if Path('/dev/shm').is_dir() else tempfile.gettempdir()
            )
            with tempfile.TemporaryDirectory(prefix='physcanvas_features_', dir=staging_dir) as local_dir:
                local_file = Path(local_dir) / feature_file.name
                np.savez_compressed(local_file, **features)
                shutil.copyfile(local_file, staged)
            with np.load(staged) as saved:
                assert set(saved.files) == set(features)
                assert all(saved[name].shape == value.shape for name, value in features.items())
            staged.replace(feature_file)
        finally:
            staged.unlink(missing_ok=True)
        image_result.update(fid_feature_counts={k: len(v) for k, v in features.items()},
                            fvd_clip_length=16, fvd_stride=8, temporary_features=str(feature_file))
        if self.profile == "visual":
            return image_result
        source = record.get('eval_source_success')
        predicted = bool(pred_labels[-1])
        d = distances[:, -1].numpy()
        margin = float((d[0] - d[1]) / max(float(d.sum()), 1e-12))
        result = {
            'metric_profile': self.profile,
            'images': self.images.summary()['overall'],
            'flow': flow_statistics(gt_flow.numpy()[1:], pred_flow.numpy()[1:], points.numpy(), self.tau),
            'marker_displacement': marker_displacement_errors(
                gt_flow.numpy()[1:], pred_flow.numpy()[1:], points.numpy(), self.tau
            ),
            'flow_encoded_gt_reference': flow_statistics(decoded_gt_flow.numpy()[1:], pred_flow.numpy()[1:], points.numpy(), self.tau),
            'flow_encoding_reconstruction': flow_statistics(gt_flow.numpy()[1:], decoded_gt_flow.numpy()[1:], points.numpy(), self.tau),
            'flow_reference_note': 'Primary reference is raw physical GT. Additional metrics decode both GT and prediction using identical bilinear four-side xyz_linear packing and scale 0.06; encoding baseline exposes GT clipping/resizing error.',
            'task_success': {
                'predicted_success': predicted,
                'prediction_rule': 'Nearest binary state-canvas prototype at final frame; no monotonic postprocessing',
                'prototype_margin_not_probability': margin,
                'source_success': source, 'source_annotation': record.get('eval_source_record_manifest'),
                'source_correct': None if source is None or args.smoke_frames else predicted == source,
                'prediction_scope': 'smoke prefix only' if args.smoke_frames else 'full episode final state',
                'reward_proxy_success': bool(labels[-1]),
                'reward_proxy_correct': predicted == bool(labels[-1]),
                'future_state_correct_frames': int((pred_labels[1:] == labels[1:]).sum()),
                'future_state_frames': len(labels)-1,
                'initial_state': int(labels[0]),
                'initial_state_already_success': bool(labels[0]),
            },
            'fid_feature_counts': {k: len(v) for k, v in features.items()},
            'frame_zero_excluded': True,
            'fvd_clip_length': 16, 'fvd_stride': 8,
            'temporary_features': str(feature_file),
        }
        if self.contact_thresholds is not None:
            result['tactile_events'] = tactile_event_metrics(
                gt[:, :, 224:448, :].float().numpy(),
                pred[:, :, 224:448, :].float().numpy(),
                gt_flow.numpy(), pred_flow.numpy(), points.numpy(), self.tau,
                self.contact_thresholds,
            )
            result['tactile_events']['threshold_sha256'] = self.contact_threshold_sha256
        return result
