"""Read safetensors headers, never tensor payloads or model allocations."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from .plan import check_norm_stats, identifier


def tactile_shapes() -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for name, output, input_ in (("conv1", 64, 3), ("conv2", 128, 64), ("conv3", 256, 128)):
        shapes[f"{name}.weight"] = [output, input_, 3, 3]
        shapes[f"{name}.bias"] = [output]
    for name, size in (("norm1", 64), ("norm2", 128), ("norm3", 256), ("hidden_norm", 1024), ("output_norm", 2048)):
        shapes[f"{name}.weight"] = [size]
        shapes[f"{name}.bias"] = [size]
    for name, output, input_ in (("spatial_projection", 1024, 3840), ("output_projection", 2048, 1024)):
        shapes[f"{name}.weight"] = [output, input_]
        shapes[f"{name}.bias"] = [output]
    shapes.update({"sensor_embedding.weight": [4, 2048], "modality_embedding": [1, 1, 2048], "gate": []})
    return {f"tactile_encoder.{key}": value for key, value in shapes.items()}


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 32 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header size {length}")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("Truncated safetensors header")
    header = json.loads(raw)
    payload_size = path.stat().st_size - 8 - length
    ranges = []
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        shape, offsets, dtype = value["shape"], value["data_offsets"], value["dtype"]
        if dtype not in widths or not all(type(n) is int and n >= 0 for n in shape):
            raise ValueError(f"Unexpected weight dtype/shape: {key}")
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] > payload_size:
            raise ValueError(f"Invalid payload extent: {key}")
        if offsets[1] - offsets[0] != math.prod(shape) * widths[dtype]:
            raise ValueError(f"Incorrect tensor byte extent: {key}")
        ranges.append((offsets[0], offsets[1]))
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError("Overlapping/gapped safetensors payload")
        end = stop
    if end != payload_size:
        raise ValueError("Unaccounted safetensors payload bytes")
    return header


def validate_checkpoint(root: Path, asset_id: str) -> dict:
    identifier(asset_id)
    config = json.loads((root / "config.json").read_text())
    expected = {"pi05": True, "use_tactile": True, "action_dim": 32, "action_horizon": 16,
                "max_token_len": 200, "tactile_num_sensors": 4, "tactile_points_per_sensor": 700,
                "tactile_input_channels": 3, "tactile_grid_height": 35, "tactile_grid_width": 20,
                "tactile_hidden_dim": 1024, "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected config {key}={config.get(key)!r}; expected {value!r}")
    header = read_header(root / "model.safetensors")
    required = tactile_shapes()
    actual = {key for key in header if key.startswith("tactile_encoder.")}
    if actual != set(required):
        raise ValueError(f"Tactile keys mismatch: missing={sorted(set(required)-actual)}, extra={sorted(actual-set(required))}")
    for key, shape in required.items():
        if header[key]["shape"] != shape:
            raise ValueError(f"Bad shape {key}: {header[key]['shape']} != {shape}")
    for key in ("img.stem.weight", "llm.embedder.embedding.weight", "action_in_proj.weight", "action_out_proj.weight"):
        if key not in header:
            raise ValueError(f"Missing backbone/projection key: {key}")
    stats = check_norm_stats(root / "assets" / asset_id / "norm_stats.json")
    return {"ok": True, "tensor_count": len(header) - int("__metadata__" in header),
            "tactile_tensor_count": len(required), "norm_stats": stats,
            "PI05_CHECKPOINT": str(root.resolve()), "PI05_NORM_STATS": stats["path"],
            "validation_scope": "header/key/shape/extent and finite norm stats; not weight finiteness or numerical inference parity"}
