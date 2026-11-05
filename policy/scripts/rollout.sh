#!/usr/bin/env bash
set -euo pipefail
TASK="${1:?Usage: rollout.sh building_blocks|test_tube --output-dir PATH [options]}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/runtime_env.sh" "$TASK"
exec "$PHYSCANVAS_PYTHON" -m physcanvas_pi05.rollout --task "$TASK" "$@"
