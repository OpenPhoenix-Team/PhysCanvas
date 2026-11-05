"""Download and assemble the public epoch-30 PhysCanvas transformer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from tools.download_eval_inputs import SOURCES, download, open_remote, url_for


PREFIX = Path("weights/epoch30_step300/transformer")
MANIFEST_SHA256 = "72cd121e50d40c874df7716e6086d06c98af6c956754e1d3193dd8b00aef70ea"
SMALL_FILES = {
    "manifest.json": (1268, MANIFEST_SHA256),
    "config.json": (670, "fc00b8544ef9f4ccde2de02a383422ac7ac4fd431d7b4504fdcaefe750ff1b83"),
    "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    "NOTICE": (97, "1b1422c2945fc61520653953a152c339b05a9a523e3f027c1248e25099b4080f"),
}


def read_manifest(base: str) -> dict:
    with open_remote(url_for(base, PREFIX / "manifest.json")) as response:
        data = response.read(4096)
    if len(data) != SMALL_FILES["manifest.json"][0] or hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError("public model manifest differs from the verified release")
    manifest = json.loads(data)
    if manifest.get("format") != "physcanvas-byte-shards-v1" or manifest.get("source_filename") != "diffusion_pytorch_model.bin":
        raise ValueError("unexpected model manifest format")
    parts = manifest.get("parts", [])
    if len(parts) != 5 or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("source_sha256", "")):
        raise ValueError("incomplete model manifest")
    offset = 0
    for index, part in enumerate(parts):
        if (part.get("name") != f"diffusion_pytorch_model.bin.part-{index:03d}"
                or part.get("offset") != offset or not isinstance(part.get("size"), int)
                or part["size"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", part.get("sha256", ""))):
            raise ValueError("invalid model part entry")
        offset += part["size"]
    if offset != manifest.get("source_size"):
        raise ValueError("model part sizes do not match")
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(directory: Path, manifest: dict) -> Path:
    output = directory / manifest["source_filename"]
    if output.exists():
        if output.stat().st_size != manifest["source_size"] or sha256(output) != manifest["source_sha256"]:
            raise ValueError(f"existing assembled model differs from the public manifest: {output}")
        print(f"verified existing model: {output}", flush=True)
        return output
    descriptor, temporary = tempfile.mkstemp(prefix=".model.", suffix=".part", dir=directory)
    try:
        total = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as target:
            for part in manifest["parts"]:
                with (directory / part["name"]).open("rb") as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        total.update(chunk)
                        target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if os.stat(temporary).st_size != manifest["source_size"] or total.hexdigest() != manifest["source_sha256"]:
            raise ValueError("assembled model failed size/SHA-256 verification")
        os.replace(temporary, output)
        print(f"assembled model: {output}", flush=True)
        return output
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="modelscope")
    parser.add_argument("--download-dir", type=Path, help="Parent directory for weights/")
    parser.add_argument("--list-only", action="store_true", help="Verify the model manifest without downloading weights")
    args = parser.parse_args()
    base = SOURCES[args.source]
    manifest = read_manifest(base)
    print(f"epoch30_step300 transformer: {len(manifest['parts'])} parts, "
          f"{manifest['source_size'] / 1024**3:.2f} GiB", flush=True)
    if args.list_only:
        return
    if args.download_dir is None:
        parser.error("--download-dir is required unless --list-only is used")
    directory = args.download_dir / PREFIX
    for name, (size, digest) in SMALL_FILES.items():
        download(base, PREFIX / name, directory / name, size, digest)
    for part in manifest["parts"]:
        download(base, PREFIX / part["name"], directory / part["name"],
                 part["size"], part["sha256"])
    assemble(directory, manifest)


if __name__ == "__main__":
    main()
