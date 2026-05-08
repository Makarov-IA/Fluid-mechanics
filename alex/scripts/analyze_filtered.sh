#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/linear_stability/filtered_detection"

mkdir -p "$OUT_DIR"

python3 "$ROOT_DIR/scripts/find_stationary.py" \
  --results-dir "$ROOT_DIR/linear_stability/filtered_results" \
  --config "$ROOT_DIR/configs/config.cfg" \
  --snapshot-index "$ROOT_DIR/linear_stability/filtered_snapshots.csv" \
  --metrics-csv "$OUT_DIR/consecutive_difference_norm.csv" \
  --plot-png "$OUT_DIR/consecutive_difference_norm.png" \
  --stationary-csv "$OUT_DIR/stationary_state.csv" \
  --streamplot-png "$OUT_DIR/stationary_streamplot.png" \
  --field state \
  --skip-newest 1
