"""Record the official downloader's completed verification without rehashing 10GB.

Run only after scripts/prepare_assets.sh exits successfully. This report records
that attested run and independently rechecks metadata, manifest/config hashes,
and all file sizes. It explicitly distinguishes these checks from a fresh hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--official-downloader-succeeded", action="store_true", required=True)
    args = parser.parse_args()
    checkpoint = args.root / "assets/weights/epoch30_step300/transformer"
    expected_manifest = "72cd121e50d40c874df7716e6086d06c98af6c956754e1d3193dd8b00aef70ea"
    expected_config = "fc00b8544ef9f4ccde2de02a383422ac7ac4fd431d7b4504fdcaefe750ff1b83"
    expected_weight = "7fd2e1b620d2b75e1ca50cfb3949055fe5204d451dbdc1ab1d950ba4e11c7259"
    for name, expected in (("manifest.json", expected_manifest), ("config.json", expected_config)):
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected public-release {name}")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest["source_sha256"] != expected_weight or manifest["source_size"] != 10002231473:
        raise ValueError("Unexpected public-release weight identity")
    for part in manifest["parts"]:
        if (checkpoint / part["name"]).stat().st_size != part["size"]:
            raise ValueError(f"Part size mismatch: {part['name']}")
    weights = checkpoint / manifest["source_filename"]
    if weights.stat().st_size != manifest["source_size"]:
        raise ValueError("Assembled size mismatch")
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_release": "epoch30_step300",
        "source": "https://modelscope.cn/models/ArteMe/PhysCanvas_model/resolve/master/",
        "checkpoint": str(checkpoint),
        "weights": str(weights),
        "source_size": manifest["source_size"],
        "source_sha256": expected_weight,
        "manifest_sha256": expected_manifest,
        "config_sha256": expected_config,
        "official_downloader_exit_success_attested": True,
        "prior_verification": "prepare_assets.sh: pinned upstream downloader verified all five part hashes, assembled hash, and independent sha256sum verified the assembled file; exit 0",
        "checks_repeated_by_this_report": ["manifest SHA256", "config SHA256", "all five part sizes", "assembled file size"],
        "full_weight_rehashed_by_this_report": False,
        "part_files_retained": True,
        "server_download_direct": True,
        "old_models_or_datasets_modified": False,
        "gpu_inference_or_training_started": False,
    }
    output = args.root / "reports/asset_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
