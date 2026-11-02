"""Download and prepare the public full-episode evaluation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from tools.cohort_fixture import archive_items, prepare_cohort, verify_archives, verify_prepared


SOURCES = {
    "modelscope": "https://modelscope.cn/models/ArteMe/PhysCanvas_model/resolve/master/",
    "huggingface": "https://huggingface.co/XXXXing/PhysCanvas_model/resolve/main/",
}
PREFIX = Path("evaluation-data/full-episode")
INDEX_LIMIT = 2 * 1024 * 1024


def url_for(base: str, relative: Path) -> str:
    return base + quote(relative.as_posix(), safe="/")


def open_remote(url: str):
    return urlopen(Request(url, headers={"User-Agent": "PhysCanvas-evaluation/0.1"}), timeout=60)


def fetch_index(base: str) -> tuple[bytes, list[tuple[str, dict, dict]]]:
    with open_remote(url_for(base, PREFIX / "index.json")) as response:
        data = response.read(INDEX_LIMIT + 1)
    if len(data) > INDEX_LIMIT:
        raise ValueError("evaluation index exceeds size limit")
    index = json.loads(data)
    if index.get("schema") != "physcanvas_full_episode_cohort_v1":
        raise ValueError("unexpected evaluation index schema")
    items = archive_items(index)
    if len(items) != 20 or len({item[1]["episode_id"] for item in items}) != 20:
        raise ValueError("expected 20 unique episode archives")
    for _, _, info in items:
        relative = Path(info["path"])
        if (relative.is_absolute() or relative.parts[:2] != PREFIX.parts
                or ".." in relative.parts or len(relative.parts) != 4
                or not isinstance(info["bytes"], int) or info["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])):
            raise ValueError(f"unsafe or incomplete archive entry: {info}")
    return data, items


def matches(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == digest


def download(base: str, relative: Path, destination: Path, size: int, digest: str) -> None:
    if destination.exists():
        if not matches(destination, size, digest):
            raise ValueError(f"existing file differs from the public index: {destination}")
        print(f"verified existing: {relative}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part",
                                             dir=destination.parent)
    try:
        hasher = hashlib.sha256()
        received = 0
        with os.fdopen(descriptor, "wb") as output, open_remote(url_for(base, relative)) as response:
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(chunk)
                hasher.update(chunk)
                received += len(chunk)
        if received != size or hasher.hexdigest() != digest:
            raise ValueError(f"download failed size/SHA-256 verification: {relative}")
        os.replace(temporary, destination)
        print(f"downloaded: {relative}", flush=True)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="modelscope")
    parser.add_argument("--download-dir", type=Path, help="Parent directory for evaluation-data/")
    parser.add_argument("--prepared-dir", type=Path, help="Model-ready inputs derived from the archives")
    parser.add_argument("--list-only", action="store_true", help="Read the public index without downloading archives")
    args = parser.parse_args()
    data, items = fetch_index(SOURCES[args.source])
    print(f"{len(items)} archives, {sum(info['frames'] for _, _, info in items)} frames, "
          f"{sum(info['bytes'] for _, _, info in items) / 1024**3:.2f} GiB compressed", flush=True)
    if args.list_only:
        return
    if args.download_dir is None or args.prepared_dir is None:
        parser.error("--download-dir and --prepared-dir are required unless --list-only is used")
    root = args.download_dir / PREFIX
    index_path = root / "index.json"
    if index_path.exists():
        if index_path.read_bytes() != data:
            raise ValueError(f"existing index differs from the public source: {index_path}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        index_path.write_bytes(data)
    for _, _, info in items:
        relative = Path(info["path"])
        download(SOURCES[args.source], relative, args.download_dir / relative,
                 info["bytes"], info["sha256"])
    verify_archives(root)
    if args.prepared_dir.exists():
        verify_prepared(root, args.prepared_dir)
        print(f"verified existing prepared inputs: {args.prepared_dir}", flush=True)
    else:
        prepare_cohort(root, args.prepared_dir)
        print(f"prepared inputs: {args.prepared_dir}", flush=True)


if __name__ == "__main__":
    main()
