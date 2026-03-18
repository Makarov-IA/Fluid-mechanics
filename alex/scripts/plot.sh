#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/data/results}"
PLOT_ROOT="${2:-$ROOT_DIR/plots}"
FRAMES_DIR="$PLOT_ROOT/frames"

mkdir -p "$FRAMES_DIR"

rm -f "$FRAMES_DIR"/*.png 2>/dev/null || true
rm -f "$PLOT_ROOT"/*.png 2>/dev/null || true

RESULTS_ARG="$OUT_DIR"
FRAMES_ARG="$FRAMES_DIR"
if command -v cygpath >/dev/null 2>&1; then
  RESULTS_ARG="$(cygpath -m "$OUT_DIR")"
  FRAMES_ARG="$(cygpath -m "$FRAMES_DIR")"
fi

echo "[plot] Rendering PNG frames from $OUT_DIR"
python "$ROOT_DIR/scripts/plot_fields.py" "$RESULTS_ARG" "$FRAMES_ARG"

echo "[plot] Done."
