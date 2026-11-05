#!/usr/bin/env bash
set -euo pipefail
TASK="${1:-jimu}"
if [[ $# -gt 0 ]]; then shift; fi
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh" "$TASK"
exec "$PHYSCANVAS_PYTHON" -m physcanvas_pi05.preflight --task "$TASK" "$@"
