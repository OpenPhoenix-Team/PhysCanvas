"""Bounded CPU-only check of online motion against one reset-cache first frame.

One candidate per process prevents calibration modules leaking via sys.modules.
No cache or source file is written. The report goes under the new experiment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("building_blocks", "test_cube", "jimu", "shiguan"), required=True)
    parser.add_argument("--candidate", choices=("original_jimu", "standard0822"), required=True)
    parser.add_argument("--root", type=Path, default=Path(os.environ["PHYSCANVAS_EXPERIMENT_ROOT"]))
    args = parser.parse_args()
    workspace = Path("/mnt/workspace/xzb/workspace")
    cache_roots = {
        "jimu": workspace / "jimu_pi05_wm_grpo/artifacts/jimu_reset128_rmdata/cache",
        "shiguan": workspace / "shiguan_saizi_pi05_wm_grpo/artifacts/shiguan_saizi_reset128_0822_standard_v2/cache",
    }
    debug_paths = {
        "original_jimu": workspace / "iMac_sample_fastfs/scripts/debug/check_traj_map_for_wujie_data",
        "standard0822": workspace / "shiguan_saizi_pi05_wm_grpo/assets/motion_calibration/0822_shiguan_standard",
    }
    module_path = args.root / "vendor/PhysCanvas/PhysCanvas/motion/arxx5_link678_gray_renderer.py"
    spec = importlib.util.spec_from_file_location("public_motion_renderer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    debug = debug_paths[args.candidate]
    settings = {
        "debug_dir": str(debug),
        "x5a_root": str(workspace / "ARX_Model/X5_M2/X5A"),
        "urdf_path": str(workspace / "ARX_Model/X5_M2/X5A/urdf/X5A.urdf"),
        "fk_offset_path": str(debug / "assets/arxx5_fk_offsets_jimu1_0625_157.json"),
        "left_calib": str(debug / "assets/calib_eye_in_hand_left_M/result_eye_in_hand.json"),
        "right_calib": str(debug / "assets/calib_eye_in_hand_right_M/result_eye_in_hand.json"),
        "use_cache": False,
        "mesh_max_faces": 1000,
        "outline_max_faces": 0,
        "gray_background": 190,
        "gripper_gray": 55,
        "wrist_mesh_tool_offset_x": 0.0,
        "wrist_mesh_tool_offset_y": 0.0,
        "wrist_mesh_tool_offset_z": 0.0,
    }
    episode = sorted((cache_roots[{"building_blocks": "jimu", "test_cube": "shiguan"}.get(args.task, args.task)] / "episodes").glob("*/qpos.npy"))[0].parent
    qpos = np.asarray(np.load(episode / "qpos.npy", mmap_mode="r")[:1])
    renderer = module.ARXLink678GrayMotionRenderer(settings)
    predicted = renderer.frames({"qpos": qpos, "episode_id": episode.name}, np.array([0]), (224, 224))
    metrics = {}
    for output, camera in (("high", "base_0"), ("left_wrist", "left_wrist_0"), ("right_wrist", "right_wrist_0")):
        reference = np.asarray(np.load(episode / f"{camera}_motion.npy", mmap_mode="r")[0])
        actual = predicted[output][0]
        if reference.shape != actual.shape:
            metrics[camera] = {"cache_shape": list(reference.shape), "render_shape": list(actual.shape)}
            continue
        difference = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
        metrics[camera] = {"mae": float(difference.mean()), "max": float(difference.max()), "pixel_fraction_different": float(np.any(difference != 0, axis=-1).mean())}
    report = {"task": args.task, "candidate": args.candidate, "episode": str(episode), "settings": settings, "metrics": metrics, "source_modified": False}
    output = args.root / "reports" / f"motion_asset_{args.task}_{args.candidate}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
