#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"

OUT_DIR="$STABILITY_DIR/filtered_detection"
SNAPSHOT_INDEX="$STABILITY_DIR/filtered_snapshots.csv"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" "$ALEX_DIR/scripts/python/find_stationary.py" \
  --results-dir "$FILTERED_BIN_DIR" \
  --config "$CONFIG_PATH" \
  --snapshot-index "$SNAPSHOT_INDEX" \
  --metrics-csv "$OUT_DIR/consecutive_difference_norm.csv" \
  --plot-png "$OUT_DIR/consecutive_difference_norm.png" \
  --stationary-snapshot "$OUT_DIR/stationary_state.bin" \
  --streamplot-png "$OUT_DIR/stationary_streamplot.png" \
  --field state \
  --skip-newest 1
