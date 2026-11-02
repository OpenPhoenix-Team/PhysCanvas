"""Shared, VAE-friendly discrete object-state canvas encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


TASK_NAMES = ("bottle", "shiguan", "jimu", "mofang")
TASK_TO_ID = {name: index for index, name in enumerate(TASK_NAMES)}


@dataclass(frozen=True)
class CanvasSpec:
    height: int = 32
    width: int = 672
    header_height: int = 8
    encoding: str = "lowfreq_bands_v1"


STATE_COLORS = {
    "mofang": {
        0: np.asarray((80, 42, 12), dtype=np.uint8),
        1: np.asarray((245, 192, 52), dtype=np.uint8),
    },
    "bottle": {
        0: np.asarray((15, 35, 88), dtype=np.uint8),
        1: np.asarray((38, 188, 242), dtype=np.uint8),
    },
    "shiguan": {
        0: np.asarray((18, 72, 45), dtype=np.uint8),
        1: np.asarray((91, 224, 104), dtype=np.uint8),
    },
    "jimu": {
        0: np.asarray((70, 25, 78), dtype=np.uint8),
        1: np.asarray((235, 86, 204), dtype=np.uint8),
        2: np.asarray((244, 157, 55), dtype=np.uint8),
    },
}


def normalize_task_name(task: str | int) -> str:
    if isinstance(task, (int, np.integer)):
        return TASK_NAMES[int(task)]
    value = str(task).lower()
    if "bottle" in value or "pinggai" in value:
        return "bottle"
    if "shiguan" in value or "saizi" in value:
        return "shiguan"
    if "jimu" in value or "building_block" in value:
        return "jimu"
    if "mofang" in value or "rubik" in value:
        return "mofang"
    raise ValueError(f"Cannot resolve object-state task from {task!r}")


def _task_header(canvas: np.ndarray, task_id: int, header_height: int) -> None:
    """Encode task identity independently from binary state using broad gray cells."""
    width = canvas.shape[1]
    canvas[:header_height] = (52, 52, 52)
    # Preserve the exact three-cell encoding used by existing checkpoints.
    cells = 3 if task_id < 3 else 4
    edges = np.linspace(0, width, cells + 1, dtype=np.int64)
    for index in range(cells):
        value = 196 if index == task_id else 76
        canvas[:header_height, edges[index] : edges[index + 1]] = value
    separator = max(1, header_height // 12)
    canvas[header_height - separator : header_height] = 126


def render_state_canvas(
    state: int | bool,
    task: str | int,
    spec: CanvasSpec | None = None,
) -> np.ndarray:
    """Return an RGB uint8 canvas whose state semantics are shared by all tasks."""
    spec = spec or CanvasSpec()
    state = int(state)
    task_name = normalize_task_name(task)
    valid_states = tuple(sorted(STATE_COLORS[task_name]))
    if state not in valid_states:
        raise ValueError(
            f"state must be one of {valid_states} for task={task_name}, got {state}"
        )
    task_id = TASK_TO_ID[task_name]
    if spec.height <= spec.header_height or spec.width < 16:
        raise ValueError(f"Invalid canvas shape {(spec.height, spec.width)}")

    canvas = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    canvas[:] = STATE_COLORS[task_name][state]
    _task_header(canvas, task_id, spec.header_height)

    y0 = spec.header_height
    body_h = spec.height - y0
    if spec.encoding == "solid_v1":
        return canvas
    if spec.encoding == "lowfreq_bands_v1":
        # Redundant geometry is state-specific but task-independent. Its broad
        # frequency content survives the Wan VAE much better than text/thin lines.
        if state == 0:
            band_w = max(8, spec.width // 10)
            shade = (STATE_COLORS[task_name][state].astype(np.float32) * 0.52).round().astype(np.uint8)
            for center in (spec.width // 3, 2 * spec.width // 3):
                canvas[y0:, center - band_w // 2 : center + band_w // 2] = shade
        elif state == 1:
            band_h = max(2, body_h // 6)
            shade = (STATE_COLORS[task_name][state].astype(np.float32) * 0.66).round().astype(np.uint8)
            for center in (y0 + body_h // 3, y0 + 2 * body_h // 3):
                canvas[center - band_h // 2 : center + band_h // 2] = shade
        else:
            # Three-class jimu uses a broad cross for partial completion. This
            # remains low frequency after Wan-VAE spatial compression.
            shade = (STATE_COLORS[task_name][state].astype(np.float32) * 0.58).round().astype(np.uint8)
            band_w = max(8, spec.width // 8)
            band_h = max(2, body_h // 5)
            center_x = spec.width // 2
            center_y = y0 + body_h // 2
            canvas[y0:, center_x - band_w // 2 : center_x + band_w // 2] = shade
            canvas[center_y - band_h // 2 : center_y + band_h // 2] = shade
        return canvas
    if spec.encoding == "lowfreq_split_v1":
        factor = {0: 0.52, 1: 0.66, 2: 0.58}[state]
        shade = (STATE_COLORS[task_name][state].astype(np.float32) * factor).round().astype(np.uint8)
        canvas[y0:, spec.width // 2 :] = shade
        return canvas
    raise ValueError(f"Unsupported canvas encoding {spec.encoding!r}")


def render_state_sequence(
    states: np.ndarray | torch.Tensor | list[int],
    task: str | int,
    spec: CanvasSpec | None = None,
) -> torch.Tensor:
    """Render [T] labels as normalized [T, C, H, W] tensors in [-1, 1]."""
    labels = np.asarray(states, dtype=np.int64).reshape(-1)
    frames = np.stack([render_state_canvas(int(label), task, spec) for label in labels])
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return tensor.div_(127.5).sub_(1.0)


def classify_state_canvas(images: np.ndarray, spec: CanvasSpec | None = None) -> np.ndarray:
    """Nearest-prototype state readout used only for audit/evaluation."""
    spec = spec or CanvasSpec()
    array = np.asarray(images)
    if array.ndim == 3:
        array = array[None]
    body_rgb = array[:, spec.header_height :].astype(np.float32).mean(axis=(1, 2))
    task_states = [
        (task, state)
        for task in TASK_NAMES
        for state in sorted(STATE_COLORS[task])
    ]
    prototypes = np.stack([
        STATE_COLORS[task][state] for task, state in task_states
    ]).astype(np.float32)
    prototype_states = np.asarray([state for _task, state in task_states])
    distances = ((body_rgb[:, None] - prototypes[None]) ** 2).sum(axis=-1)
    return prototype_states[distances.argmin(axis=1)].astype(np.int64)


def classify_state_canvas_for_task(
    images: np.ndarray,
    task: str | int,
    spec: CanvasSpec | None = None,
) -> np.ndarray:
    """Read binary state using only the known task's two color prototypes."""
    spec = spec or CanvasSpec()
    array = np.asarray(images)
    if array.ndim == 3:
        array = array[None]
    task_name = normalize_task_name(task)
    body_rgb = array[:, spec.header_height :].astype(np.float32).mean(axis=(1, 2))
    states = np.asarray(sorted(STATE_COLORS[task_name]), dtype=np.int64)
    prototypes = np.stack([STATE_COLORS[task_name][int(state)] for state in states]).astype(np.float32)
    distances = ((body_rgb[:, None] - prototypes[None]) ** 2).sum(axis=-1)
    return states[distances.argmin(axis=1)]


def classify_task_canvas(images: np.ndarray, spec: CanvasSpec | None = None) -> np.ndarray:
    """Read the state-independent three-cell task marker for audit/evaluation."""
    spec = spec or CanvasSpec()
    array = np.asarray(images)
    if array.ndim == 3:
        array = array[None]
    header = array[:, : spec.header_height - max(2, spec.header_height // 12)].astype(np.float32)
    edges = np.linspace(0, spec.width, len(TASK_NAMES) + 1, dtype=np.int64)
    scores = np.stack([header[:, :, edges[i] : edges[i + 1]].mean(axis=(1, 2, 3)) for i in range(3)], axis=1)
    return scores.argmax(axis=1).astype(np.int64)
