"""Build a checksummed, frame-counted review list for qualitative MP4s."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_count(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.videos_dir.glob("*.mp4"))
    if len(paths) != 20:
        raise ValueError(f"expected 20 videos, found {len(paths)}")
    files = []
    for path in paths:
        frames = frame_count(path)
        if frames != 128:
            raise ValueError(f"unexpected frame count in {path.name}: {frames}")
        files.append({"destination": f"examples/videos/{path.name}",
                      "source": str(path), "bytes": path.stat().st_size,
                      "frames": frames, "sha256": hash_file(path)})
    payload = {"status": "review_only_not_uploaded", "video_count": len(files),
               "total_bytes": sum(item["bytes"] for item in files), "files": files}
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
