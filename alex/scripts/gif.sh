#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOT_ROOT="$ROOT_DIR/plots"
FRAMES_DIR="$PLOT_ROOT/frames"
GIFS_DIR="$PLOT_ROOT/gifs"
DURATION=50

mkdir -p "$GIFS_DIR"
rm -f "$GIFS_DIR"/*.gif 2>/dev/null || true

FRAMES_ARG="$FRAMES_DIR"
GIFS_ARG="$GIFS_DIR"
if command -v cygpath >/dev/null 2>&1; then
  FRAMES_ARG="$(cygpath -m "$FRAMES_DIR")"
  GIFS_ARG="$(cygpath -m "$GIFS_DIR")"
fi

python "$ROOT_DIR/scripts/make_gifs.py" \
  "$FRAMES_ARG" \
  "$GIFS_ARG" \
  --duration "$DURATION"
