#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/alex/torch_cavity_601"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float64}"

cd "$ROOT_DIR"

python3 "$EXP_DIR/torch_cavity_solver.py" \
  "$EXP_DIR/configs/smoke.cfg" \
  "$EXP_DIR/results/smoke" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

bash "$EXP_DIR/render_one.sh" \
  "$EXP_DIR/results/smoke" \
  "$EXP_DIR/configs/smoke.cfg" \
  "$EXP_DIR/analysis/smoke"

echo "[smoke] OK"
