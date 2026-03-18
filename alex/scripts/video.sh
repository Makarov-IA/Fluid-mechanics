#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOT_ROOT="$ROOT_DIR/plots"
FRAMES_DIR="$PLOT_ROOT/frames"
VIDEOS_DIR="$PLOT_ROOT/videos"
FPS=20

mkdir -p "$VIDEOS_DIR"
rm -f "$VIDEOS_DIR"/*.mp4 2>/dev/null || true

FRAMES_ARG="$FRAMES_DIR"
VIDEOS_ARG="$VIDEOS_DIR"
if command -v cygpath >/dev/null 2>&1; then
  FRAMES_ARG="$(cygpath -m "$FRAMES_DIR")"
  VIDEOS_ARG="$(cygpath -m "$VIDEOS_DIR")"
fi

python "$ROOT_DIR/scripts/make_videos.py" \
  "$FRAMES_ARG" \
  "$VIDEOS_ARG" \
  --fps "$FPS"
