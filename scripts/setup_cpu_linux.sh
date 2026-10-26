#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Same Torch release as uv.lock, official CPU wheel instead of CUDA dependencies.
if [[ ! -x .venv-linux/bin/python ]]; then
    uv venv --python 3.11 .venv-linux
fi
mkdir -p cache/setup
uv export --locked --extra gpu --no-dev --no-emit-project --prune torch --output-file cache/setup/cpu-locked.txt
uv pip install --python .venv-linux/bin/python -r cache/setup/cpu-locked.txt
.venv-linux/bin/python scripts/export_cpu_deps.py > cache/setup/cpu-torch-deps.txt
uv pip install --python .venv-linux/bin/python --require-hashes -r cache/setup/cpu-torch-deps.txt
uv pip install --python .venv-linux/bin/python --require-hashes --no-deps -r requirements-cpu-torch.txt --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-linux/bin/python --no-deps -e .
