#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-/workspace/sam3-video}"
PORT="${PORT:-8000}"

cd "$APP_DIR"

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<82" wheel
python -m pip install -r requirements.txt

mkdir -p "${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "${RUNS_DIR:-$APP_DIR/runs}"

exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port "$PORT"
