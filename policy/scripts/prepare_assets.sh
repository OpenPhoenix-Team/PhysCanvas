#!/usr/bin/env bash
# Download the *public final* WM, never fall back to a historical task WM.
# Does not modify Git repositories, source datasets or shared environments.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/dependency_env.sh"
SOURCE=${1:-modelscope}
case "$SOURCE" in modelscope|huggingface) ;; *) echo "Use modelscope or huggingface" >&2; exit 2 ;; esac
ASSET_ROOT=${PHYSCANVAS_ASSET_ROOT:-${PHYSCANVAS_EXPERIMENT_ROOT}/assets}

# The upstream downloader verifies a pinned manifest, each existing/downloaded
# part, and the assembled 10,002,231,473-byte file. A mismatching existing file
# is a hard failure, not overwritten. Part files are intentionally retained.
mkdir -p "$ASSET_ROOT"
cd "$PHYSCANVAS_VENDOR/PhysCanvas"
"$PHYSCANVAS_PYTHON" -m tools.download_model --source "$SOURCE" --list-only
"$PHYSCANVAS_PYTHON" -m tools.download_model --source "$SOURCE" --download-dir "$ASSET_ROOT"

MODEL="$ASSET_ROOT/weights/epoch30_step300/transformer/diffusion_pytorch_model.bin"
EXPECTED=7fd2e1b620d2b75e1ca50cfb3949055fe5204d451dbdc1ab1d950ba4e11c7259
ACTUAL=$(sha256sum "$MODEL")
[[ "${ACTUAL%% *}" == "$EXPECTED" ]] || { echo "Public final WM hash mismatch" >&2; exit 3; }
echo "Verified public final WM: $MODEL"
echo "SHA256: $EXPECTED"
