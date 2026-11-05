#!/usr/bin/env bash
# New isolated environment only; never upgrades a training environment in place.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${1:?Usage: setup_runtime.sh /absolute/path/to/NEW_environment}
[[ "$ENV_DIR" = /* && ! -e "$ENV_DIR" ]] || { echo 'Need a new absolute environment path' >&2; exit 2; }
uv venv --python 3.11 --seed "$ENV_DIR"
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120}
uv pip install --python "$ENV_DIR/bin/python" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' numpy==1.26.4
uv pip install --python "$ENV_DIR/bin/python" -r "$ROOT/requirements-runtime.txt"
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.ones(16,16,device="cuda",dtype=torch.bfloat16) @ torch.ones(16,16,device="cuda",dtype=torch.bfloat16))'
