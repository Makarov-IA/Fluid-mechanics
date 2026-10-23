#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="gpu" \
EXP_NAME="gpu_smoke" \
CONFIG_PATH="$ALEX_DIR/gpu/configs/smoke.cfg" \
DEVICE="${DEVICE:-cuda:0}" \
DTYPE="${DTYPE:-float64}" \
bash "$ALEX_DIR/scripts/run.sh"

BACKEND="gpu" \
EXP_NAME="gpu_smoke" \
CONFIG_PATH="$ALEX_DIR/gpu/configs/smoke.cfg" \
bash "$ALEX_DIR/scripts/render.sh"

BACKEND="gpu" \
EXP_NAME="gpu_smoke" \
CONFIG_PATH="$ALEX_DIR/gpu/configs/smoke.cfg" \
bash "$ALEX_DIR/scripts/video.sh"

echo "[smoke] OK"
