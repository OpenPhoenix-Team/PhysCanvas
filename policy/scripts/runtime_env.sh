#!/usr/bin/env bash
set -euo pipefail
export PHYSCANVAS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK=${1:-building_blocks}
case "$TASK" in building_blocks|test_tube|jimu|shiguan) ;; *) echo 'Expected building_blocks or test_tube (aliases: jimu, shiguan)' >&2; return 2 ;; esac
if [[ -n ${PHYSCANVAS_PATHS_FILE:-} ]]; then
  source "$PHYSCANVAS_PATHS_FILE" "$TASK"
elif [[ -f "$PHYSCANVAS_ROOT/scripts/local_paths.sh" ]]; then
  source "$PHYSCANVAS_ROOT/scripts/local_paths.sh" "$TASK"
fi
export PHYSCANVAS_PYTHON=${PHYSCANVAS_PYTHON:-$PHYSCANVAS_ROOT/.venv_runtime/bin/python}
[[ -x "$PHYSCANVAS_PYTHON" ]] || { echo 'Set PHYSCANVAS_PYTHON to the prepared Python interpreter' >&2; return 2; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export ROBOT_PLATFORM=ALOHA PYOPENGL_PLATFORM=egl MUJOCO_GL=egl
export PHYSCANVAS_EVAL_RUNTIME_SOURCE="$PHYSCANVAS_ROOT/vendor/PhysCanvas"
export PYTHONPATH="$PHYSCANVAS_ROOT:$PHYSCANVAS_ROOT/vendor/rlinf:$PHYSCANVAS_ROOT/vendor/openpi/src:$PHYSCANVAS_ROOT/vendor/openpi-client/src:$PHYSCANVAS_ROOT/vendor/PhysCanvas"
export PATH="$(dirname "$PHYSCANVAS_PYTHON"):$PATH"
