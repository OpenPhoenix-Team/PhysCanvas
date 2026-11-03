"""Pack and prepare a fixed 20-episode cohort without the source cache."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

from tools.raw_fixture import CAMERAS, SIDES, export_fixture, prepare_fixture, sha256, write_json


TASK_SLUGS = {"bottle": "bottle-cap", "shiguan": "test-tube",
              "jimu": "building-blocks", "mofang": "rubiks-cube"}


def selected_records(path: Path, cohort_path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    cohort = json.loads(cohort_path.read_text())["episodes"]
    by_id = {row["episode_id"]: row for row in records}
    if len(by_id) != len(records) or len(cohort) != 20:
        raise ValueError("expected 20 unique fixed cohort records")
    ordered = []
    for item in cohort:
        row = by_id[item["episode_id"]]
        if row["object_state_task"] != item["task"] or int(row["sample_num_frames"]) != item["source_frames"]:
            raise ValueError(f"cohort mismatch: {item['episode_id']}")
        if item["source_frames"] < 128:
            raise ValueError(f"episode shorter than 128 frames: {item['episode_id']}")
        ordered.append(row)
    if len(by_id) != len(ordered) or Counter(row["object_state_task"] for row in ordered) != dict.fromkeys(("bottle", "shiguan", "jimu", "mofang"), 5):
        raise ValueError("expected exactly five records per task")
    return ordered, cohort


def archive_name(index: int, task: str) -> str:
    return f"{index + 1:02d}_{task}_n128.tar.gz"


def archive_path(source: Path, name: str, info: dict) -> Path:
    relative = Path(info.get("path", f"archives/{name}"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive path: {relative}")
    candidates = [source / relative, source / "archives" / name]
    if relative.parts[:2] == ("evaluation-data", "full-episode"):
        candidates.append(source / Path(*relative.parts[2:]))
    for path in candidates:
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"archive path escapes source directory: {relative}")
        if path.is_file():
            return path
    return candidates[0]


def archive_items(index: dict) -> list[tuple[str, dict, dict]]:
    archives = index["archives"]
    if index.get("schema") == "physcanvas_cohort_n128_v1":
        return [(archive_name(number, item["task"]), item,
                 archives[archive_name(number, item["task"])])
                for number, item in enumerate(index["cohort"])]
    if index.get("schema") == "physcanvas_full_episode_cohort_v1":
        return [(name, {"episode_id": info["episode_id"], "task": info["task"],
                        "source_frames": info["frames"]}, info)
                for name, info in sorted(archives.items())]
    raise ValueError("unsupported cohort archive schema")


def verify_archives(source: Path) -> dict:
    index = json.loads((source / "index.json").read_text())
    archives = index.get("archives", {})
    if len(archives) != 20:
        raise ValueError("expected a complete 20-episode archive index")
    items = archive_items(index)
    cohort = [item for _, item, _ in items]
    if len(cohort) != 20:
        raise ValueError("expected 20 cohort episodes")
    if len({item["episode_id"] for item in cohort}) != 20 or Counter(item["task"] for item in cohort) != {task: 5 for task in ("bottle", "shiguan", "jimu", "mofang")}:
        raise ValueError("archive index has an invalid cohort")
    for name, item, info in items:
        path = archive_path(source, name, info)
        frames = 128 if index["schema"] == "physcanvas_cohort_n128_v1" else item["source_frames"]
        if (info["episode_id"] != item["episode_id"] or info["task"] != item["task"]
                or info["frames"] != frames or item["source_frames"] < 128
                or not path.is_file() or path.stat().st_size != info["bytes"]
                or sha256(path) != info["sha256"]):
            raise ValueError(f"archive mismatch: {name}")
    if index["schema"] == "physcanvas_full_episode_cohort_v1" and sum(info["frames"] for _, _, info in items) != index["frames"]:
        raise ValueError("full-episode frame total differs")
    return index


def verify_prepared(source: Path, output: Path) -> tuple[Path, Path]:
    index = verify_archives(source)
    prepared_manifest = json.loads((output / "prepared_manifest.json").read_text())
    items = archive_items(index)
    total_frames = sum(info["frames"] for _, _, info in items)
    prepared_frames = prepared_manifest.get("total_frames")
    if prepared_frames is None and index["schema"] == "physcanvas_cohort_n128_v1":
        prepared_frames = prepared_manifest.get("frames_per_episode", 0) * 20
    if (prepared_manifest.get("source_index_sha256") != sha256(source / "index.json")
            or prepared_manifest.get("episodes") != 20
            or prepared_frames != total_frames):
        raise ValueError("prepared inputs do not match the compressed archive index")
    records_path, cohort_path = output / "records.json", output / "cohort.json"
    records = json.loads(records_path.read_text())["records"]
    cohort = json.loads(cohort_path.read_text())["episodes"]
    if len(records) != 20 or len(cohort) != 20:
        raise ValueError("prepared inputs do not contain 20 episodes")
    root = output.resolve()

    def nested_strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from nested_strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from nested_strings(child)

    for number, ((_, item, info), record, selected) in enumerate(zip(items, records, cohort)):
        frames = info["frames"]
        expected = {"episode_id": item["episode_id"], "task": item["task"], "source_frames": frames}
        if selected != expected or record["episode_id"] != item["episode_id"] or record["object_state_task"] != item["task"] or record["sample_num_frames"] != frames:
            raise ValueError(f"prepared episode mismatch at index {number}")
        paths = [Path(record[key]).resolve() for key in
                 ("sample_cache_root", "sample_cache_episode_dir", "object_state_label_path")]
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError(f"prepared episode escapes archive-derived inputs: {item['episode_id']}")
        if any(Path(text).is_absolute() and not Path(text).resolve().is_relative_to(root)
               for text in nested_strings(record)):
            raise ValueError(f"prepared record contains an external path: {item['episode_id']}")
        if any(".." in Path(text).parts for text in nested_strings(record)):
            raise ValueError(f"prepared record contains path traversal: {item['episode_id']}")
        episode = paths[1]
        expected_files = ([episode / f"{camera}_{kind}.npy" for camera in CAMERAS for kind in ("rgb", "motion")]
                          + [episode / "timestamps.npy", paths[2]]
                          + [episode / "tactile" / f"{side}.{kind}.npy"
                             for side in SIDES for kind in ("difference", "pointcloud")])
        if any(not path.is_file() or not path.resolve().is_relative_to(root)
               for path in expected_files):
            raise ValueError(f"missing archive-derived input: {item['episode_id']}")
    return records_path, cohort_path


def export_cohort(records_path: Path, cohort_path: Path, output: Path) -> None:
    records, cohort = selected_records(records_path, cohort_path)
    output.mkdir(parents=True, exist_ok=True)
    archive_dir = output / "archives"
    archive_dir.mkdir(exist_ok=True)
    index_path = output / "index.json"
    old = json.loads(index_path.read_text()) if index_path.exists() else None
    if old and old["cohort"] != cohort:
        raise ValueError("existing index belongs to a different cohort")
    entries = old["archives"] if old else {}
    for index, row in enumerate(records):
        name = archive_name(index, row["object_state_task"])
        destination = archive_dir / name
        if name in entries:
            info = entries[name]
            if info["episode_id"] != row["episode_id"] or not destination.is_file() or sha256(destination) != info["sha256"]:
                raise ValueError(f"existing archive mismatch: {name}")
            print(f"verified {index + 1}/20: {name}", flush=True)
            continue
        if destination.exists():
            raise FileExistsError(f"unindexed archive: {destination}")
        with tempfile.TemporaryDirectory(dir=output, prefix="packing_") as temp:
            fixture = Path(temp) / "fixture"
            export_fixture(Path(row["sample_cache_meta"]["episode_path"]),
                           Path(row["sample_cache_episode_dir"]),
                           Path(row["object_state_label_path"]), fixture,
                           row["object_state_task"])
            partial = Path(temp) / name
            with tarfile.open(partial, "w:gz") as archive:
                archive.add(fixture, arcname="fixture")
            partial.replace(destination)
        entries[name] = {"episode_id": row["episode_id"], "task": row["object_state_task"],
                         "source_frames": row["sample_num_frames"], "frames": 128,
                         "bytes": destination.stat().st_size, "sha256": sha256(destination)}
        write_json(index_path, {"schema": "physcanvas_cohort_n128_v1", "cohort": cohort,
                                "archives": entries})
        print(f"packed {index + 1}/20: {name} ({entries[name]['bytes']} bytes)", flush=True)
    print(f"total archive bytes: {sum(item['bytes'] for item in entries.values())}", flush=True)


def prepare_cohort(source: Path, output: Path) -> None:
    index = verify_archives(source)
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    output.mkdir(parents=True)
    records = []
    prepared = []
    items = archive_items(index)
    for number, (name, item, info) in enumerate(items):
        file_path = archive_path(source, name, info)
        with tempfile.TemporaryDirectory(dir=output, prefix="extracting_") as temp:
            with tarfile.open(file_path, "r:gz") as archive:
                for member in archive.getmembers():
                    path = Path(member.name)
                    if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                        raise ValueError(f"unsafe archive member: {member.name}")
                archive.extractall(temp)
            destination = output / "episodes" / f"{number + 1:02d}_{item['task']}"
            prepare_fixture(Path(temp) / "fixture", destination)
        record = json.loads((destination / "records.json").read_text())["records"][0]
        record["data_index"] = number
        records.append(record)
        prepared.append({"episode_id": item["episode_id"], "task": item["task"],
                         "source_frames": info["frames"]})
        print(f"prepared {number + 1}/20: {name}", flush=True)
    write_json(output / "records.json", {"records": records})
    write_json(output / "cohort.json", {"episodes": prepared})
    write_json(output / "prepared_manifest.json", {
        "source_index_sha256": sha256(source / "index.json"), "episodes": 20,
        "total_frames": sum(info["frames"] for _, _, info in items),
        "record": str((output / "records.json").resolve()),
        "cohort": str((output / "cohort.json").resolve())})
    verify_prepared(source, output)


def stage_release(source: Path, output: Path) -> None:
    index = verify_archives(source)
    if index["schema"] != "physcanvas_cohort_n128_v1":
        raise ValueError("stage-release supports the original N=128 cohort only")
    if output.exists():
        raise FileExistsError(f"release destination must not exist: {output}")
    output.mkdir(parents=True)
    for number, item in enumerate(index["cohort"]):
        name = archive_name(number, item["task"])
        info = index["archives"][name]
        task_name = TASK_SLUGS[item["task"]]
        public_name = f"{number % 5 + 1:02d}_{task_name}_n128.tar.gz"
        relative = Path("evaluation-data") / "n128" / task_name / public_name
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_path(source, name, info), destination)
        info["path"] = relative.as_posix()
    write_json(output / "index.json", index)
    verify_archives(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    export = actions.add_parser("export")
    export.add_argument("--records-json", type=Path, required=True)
    export.add_argument("--cohort", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--source-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    stage = actions.add_parser("stage-release")
    stage.add_argument("--source-dir", type=Path, required=True)
    stage.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "export":
        export_cohort(args.records_json, args.cohort, args.output_dir)
    elif args.action == "prepare":
        prepare_cohort(args.source_dir, args.output_dir)
    else:
        stage_release(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
