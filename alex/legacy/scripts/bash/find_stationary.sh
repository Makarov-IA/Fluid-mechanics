#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"

mkdir -p "$STATIONARY_DIR"

echo "[stationary] exp_name: $EXP_NAME"
echo "[stationary] input   : $BIN_DIR"
echo "[stationary] output  : $STATIONARY_DIR"

"$PYTHON_BIN" "$ALEX_DIR/scripts/python/find_stationary.py" \
  --results-dir "$BIN_DIR" \
  --config "$CONFIG_PATH" \
  --metrics-csv "$STATIONARY_DIR/consecutive_difference_norm.csv" \
  --plot-png "$STATIONARY_DIR/consecutive_difference_norm.png" \
  --stationary-snapshot "$STATIONARY_DIR/stationary_state.bin" \
  --streamplot-png "$STATIONARY_DIR/stationary_streamplot.png" \
  --field state \
  --skip-newest 1
