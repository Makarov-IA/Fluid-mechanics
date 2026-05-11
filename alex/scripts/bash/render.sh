#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"

mkdir -p "$FIG_DIR" "$TABLE_DIR"

BACKEND="$BACKEND" \
EXP_NAME="$EXP_NAME" \
CONFIG_PATH="$CONFIG_PATH" \
BIN_DIR="$BIN_DIR" \
FIG_DIR="$FIG_DIR" \
VIDEO_DIR="$VIDEO_DIR" \
TABLE_DIR="$TABLE_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$ALEX_DIR/scripts/bash/plot.sh" "$BIN_DIR" "$FIG_DIR"

if [[ "$BACKEND" == "gpu" ]]; then
  echo "[render] cavity profiles: $FIG_DIR, $TABLE_DIR"
  "$PYTHON_BIN" "$ALEX_DIR/gpu/profile_cavity.py" \
    "$BIN_DIR" \
    "$FIG_DIR" \
    --tables-dir "$TABLE_DIR"
fi

echo "[render] Done."
