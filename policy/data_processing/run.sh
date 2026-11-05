#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT${DATA_EXTRA_PYTHONPATH:+:$DATA_EXTRA_PYTHONPATH}"
exec "${DATA_PYTHON:-python3}" -m data_processing.convert "$@"
