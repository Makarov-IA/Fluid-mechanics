#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/data/results}"
PLOT_ROOT="${2:-$ROOT_DIR/plots}"
FRAMES_DIR="$PLOT_ROOT/frames"
GIFS_DIR="$PLOT_ROOT/gifs"

mkdir -p "$FRAMES_DIR" "$GIFS_DIR"

# Clear previous render artifacts
rm -f "$FRAMES_DIR"/*.png 2>/dev/null || true
rm -f "$GIFS_DIR"/*.gif 2>/dev/null || true
rm -f "$PLOT_ROOT"/*.png 2>/dev/null || true
rm -f "$PLOT_ROOT"/*.gif 2>/dev/null || true

if ! ls -1 "$OUT_DIR"/result_*.csv >/dev/null 2>&1; then
  echo "[plot] No result_*.csv found in $OUT_DIR"
  exit 1
fi

PYTHON_CMD=()
if command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
else
  echo "[plot] Python is not found (python3/python/py)"
  exit 1
fi

RESULTS_ARG="$OUT_DIR"
FRAMES_ARG="$FRAMES_DIR"
GIFS_ARG="$GIFS_DIR"
if command -v cygpath >/dev/null 2>&1; then
  RESULTS_ARG="$(cygpath -m "$OUT_DIR")"
  FRAMES_ARG="$(cygpath -m "$FRAMES_DIR")"
  GIFS_ARG="$(cygpath -m "$GIFS_DIR")"
fi

echo "[plot] Rendering PNG/GIF from $OUT_DIR"
"${PYTHON_CMD[@]}" "$ROOT_DIR/scripts/plot_fields.py" "$RESULTS_ARG" "$FRAMES_ARG" "$GIFS_ARG"

echo "[plot] Done."
