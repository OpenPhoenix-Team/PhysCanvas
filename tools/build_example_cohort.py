"""Derive a path-free example cohort from a manifest and 1-based index list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("bottle", "shiguan", "jimu", "mofang")


def build(manifest_path: Path, index_path: Path) -> dict:
    indices = json.loads(index_path.read_text())["manifest_indices_1based"]
    records = json.loads(manifest_path.read_text())["records"]
    if set(indices) != set(TASKS):
        raise ValueError("expected four task index lists")
    selected = []
    used = set()
    for task in TASKS:
        positions = indices[task]
        if not positions or len(set(positions)) != len(positions):
            raise ValueError(f"{task} needs unique positions")
        for position in positions:
            if not 1 <= position <= len(records):
                raise ValueError(f"position {position} is out of range")
            row = records[position - 1]
            if row["object_state_task"] != task:
                raise ValueError(f"position {position} is not {task}")
            episode_id = str(row["episode_id"])
            if episode_id in used or int(row["sample_num_frames"]) < 128:
                raise ValueError(f"duplicate or too-short episode: {episode_id}")
            used.add(episode_id)
            selected.append({"episode_id": episode_id, "task": task,
                             "source_frames": int(row["sample_num_frames"])})
    return {"episodes": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.manifest, args.indices), indent=2) + "\n")


if __name__ == "__main__":
    main()
