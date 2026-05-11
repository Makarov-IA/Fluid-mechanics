#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/alex/torch_cavity_601"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CONFIG_PATH="${1:?usage: run_one.sh CONFIG_PATH OUTPUT_DIR [DEVICE]}"
OUTPUT_DIR="${2:?usage: run_one.sh CONFIG_PATH OUTPUT_DIR [DEVICE]}"
DEVICE="${3:-cuda:0}"
DTYPE="${DTYPE:-float64}"

cd "$ROOT_DIR"

"$PYTHON_BIN" "$EXP_DIR/torch_cavity_solver.py" \
  "$CONFIG_PATH" \
  "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE"
