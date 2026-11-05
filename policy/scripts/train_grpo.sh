#!/usr/bin/env bash
set -euo pipefail
TASK="${1:-jimu}"
if [[ $# -gt 0 ]]; then shift; fi
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh" "$TASK"
if [[ "${RUN_TRAINING:-0}" != 1 ]]; then
  echo "Training is disabled. Run preflight, then explicitly set RUN_TRAINING=1." >&2
  exit 2
fi
# Only run on the Ray head after all nodes have joined with the same env/code.
# Pass Hydra overrides as separate arguments; no eval or string word splitting.
exec "$PHYSCANVAS_PYTHON" -m physcanvas_pi05.train_grpo --task "$TASK" "$@"
