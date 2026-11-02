"""Expand the fixed public N=128 fixtures to their original episode lengths.

This reads the already released raw input archives and the original, read-only
aligned motion/label caches. It refuses any mismatch in the first 128 frames.
"""

from __future__ import annotations

import argparse
import collections
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from tools.raw_fixture import CAMERAS, FRAMES, sha256, write_json


SCHEMA = "physcanvas_raw_full_episode_v1"
PUBLIC_DIR = "evaluation-data/full-episode"
TASK_NAMES = {"bottle": "bottle-cap", "shiguan": "test-tube",
              "jimu": "building-blocks", "mofang": "rubiks-cube"}


def checked_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if any((member.name != "fixture" and not member.name.startswith("fixture/"))
               or (not member.isfile() and not member.isdir()) for member in members):
            raise ValueError(f"unexpected archive member: {archive}")
        source.extractall(destination, filter="data")
    return destination / "fixture"


def check_existing(fixture: Path, entry: dict) -> dict:
    manifest = json.loads((fixture / "manifest.json").read_text())
    if (manifest.get("schema") != "physcanvas_raw_n128_v1"
            or manifest.get("episode_id") != entry["episode_id"]
            or manifest.get("task") != entry["task"]
            or manifest.get("frames") != FRAMES):
        raise ValueError(f"old fixture metadata differs: {entry['episode_id']}")
    for name, info in manifest["files"].items():
        path = fixture / name
        if not path.is_file() or path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError(f"old fixture file differs: {name}")
    return manifest


def expand_one(archive: Path, entry: dict, record: dict, output: Path) -> dict:
    if archive.stat().st_size != entry["bytes"] or sha256(archive) != entry["sha256"]:
        raise ValueError(f"original archive differs: {archive}")
    frames = int(entry["source_frames"])
    if frames <= FRAMES or record["episode_id"] != entry["episode_id"]:
        raise ValueError(f"episode record differs: {entry['episode_id']}")
    if int(record["sample_num_frames"]) != frames:
        raise ValueError(f"record length differs: {entry['episode_id']}")
    cache = Path(record["sample_cache_episode_dir"])
    label = Path(record["object_state_label_path"])
    meta = json.loads((cache / "meta.json").read_text())
    if int(meta["num_frames"]) != frames or meta["episode_id"] != entry["episode_id"]:
        raise ValueError(f"cache metadata differs: {entry['episode_id']}")

    with tempfile.TemporaryDirectory(prefix="physcanvas-full-") as temporary:
        fixture = checked_extract(archive, Path(temporary))
        old = check_existing(fixture, entry)
        conditioning = fixture / "conditioning"
        timeline = np.load(cache / "timestamps.npy", mmap_mode="r", allow_pickle=False)
        old_timeline = np.load(conditioning / "timeline.npy", allow_pickle=False)
        if len(timeline) != frames or not np.array_equal(timeline[:FRAMES], old_timeline):
            raise ValueError(f"timeline prefix differs: {entry['episode_id']}")
        np.save(conditioning / "timeline.npy", np.asarray(timeline))
        with np.load(conditioning / "motion.npz", allow_pickle=False) as old_motion:
            full_motion = {}
            for camera in CAMERAS:
                values = np.load(cache / f"{camera}_motion.npy", mmap_mode="r", allow_pickle=False)
                if len(values) != frames or not np.array_equal(values[:FRAMES], old_motion[camera]):
                    raise ValueError(f"motion prefix differs: {entry['episode_id']} {camera}")
                full_motion[camera] = np.asarray(values)
        np.savez_compressed(conditioning / "motion.npz", **full_motion)
        with np.load(conditioning / "labels.npz", allow_pickle=False) as old_labels:
            with np.load(label, allow_pickle=False) as source_labels:
                full_labels = {}
                for name in ("labels", "probabilities"):
                    values = source_labels[name]
                    if len(values) != frames or not np.array_equal(values[:FRAMES], old_labels[name]):
                        raise ValueError(f"label prefix differs: {entry['episode_id']} {name}")
                    full_labels[name] = values
        np.savez_compressed(conditioning / "labels.npz", **full_labels)

        new_manifest = {key: value for key, value in old.items() if key not in {"files", "note"}}
        new_manifest.update(schema=SCHEMA, frames=frames)
        new_manifest["files"] = {
            str(path.relative_to(fixture)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(fixture.rglob("*")) if path.is_file() and path.name != "manifest.json"
        }
        write_json(fixture / "manifest.json", new_manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"partial output already exists: {partial}")
        with tarfile.open(partial, "w:gz", compresslevel=1) as target:
            target.add(fixture, arcname="fixture", recursive=True)
        partial.rename(output)
    return {"bytes": output.stat().st_size, "sha256": sha256(output), "frames": frames}


def build(index_path: Path, archive_root: Path, records_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    source_index = json.loads(index_path.read_text())
    entries = source_index["archives"]
    records = {row["episode_id"]: row for row in json.loads(records_path.read_text())["records"]}
    if len(entries) != 20 or len(records) != 20 or {e["episode_id"] for e in entries.values()} != set(records):
        raise ValueError("expected the same fixed 20 episodes in the index and records")
    counts = collections.Counter(e["task"] for e in entries.values())
    if counts != {"bottle": 5, "shiguan": 5, "jimu": 5, "mofang": 5}:
        raise ValueError("expected five episodes per task")
    output.mkdir(parents=True)
    destination_entries = {}
    for number, (name, entry) in enumerate(sorted(entries.items()), start=1):
        if Path(name).name != name or entry["frames"] != FRAMES:
            raise ValueError(f"unexpected original archive entry: {name}")
        archive_name = name.replace("_n128.tar.gz", "_full.tar.gz")
        if archive_name == name:
            raise ValueError(f"unexpected original archive name: {name}")
        task = entry["task"]
        public_name = archive_name.replace(f"_{task}_", f"_{TASK_NAMES[task]}_")
        public_path = f"{PUBLIC_DIR}/{TASK_NAMES[task]}/{public_name}"
        result = expand_one(archive_root / name, entry, records[entry["episode_id"]],
                            output / "archives" / archive_name)
        destination_entries[archive_name] = {
            "episode_id": entry["episode_id"], "task": entry["task"],
            "path": public_path, **result,
        }
        print(f"{number}/20 {archive_name}: {result['frames']} frames, {result['bytes']} bytes", flush=True)
    summary = {"schema": "physcanvas_full_episode_cohort_v1", "archives": destination_entries,
               "episodes": 20, "frames": sum(item["frames"] for item in destination_entries.values()),
               "note": "The original fixed episode IDs and all original raw inputs are retained."}
    write_json(output / "index.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.index, args.archive_root, args.records, args.output), indent=2))


if __name__ == "__main__":
    main()
