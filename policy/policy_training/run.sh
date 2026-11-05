#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$POLICY_DIR/.." && pwd)
operation=${1:-plan}
if [[ $# -eq 0 ]]; then
  set -- plan --config "$POLICY_DIR/example.json"
fi
# This intentionally does NOT source scripts/runtime_env.sh (GRPO's JAX=CPU).
# SFT_PYTHON must select an independently verified JAX CUDA environment.
POLICY_PYTHON=${SFT_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PROJECT_DIR:$POLICY_DIR/openpi/src:$POLICY_DIR/openpi/packages/openpi-client/src:$PROJECT_DIR/vendor/rlinf${SFT_EXTRA_PYTHONPATH:+:$SFT_EXTRA_PYTHONPATH}"
export JAX_PLATFORMS=cpu
if [[ "$operation" == train-jax && " $* " == *" --execute "* ]]; then
  if [[ -z ${SFT_PYTHON:-} ]]; then
    echo 'Actual SFT requires explicit SFT_PYTHON; the GRPO environment is not a JAX CUDA guarantee.' >&2
    exit 2
  fi
  export JAX_PLATFORMS=cuda
fi
exec "$POLICY_PYTHON" -m policy_training.run "$@"
