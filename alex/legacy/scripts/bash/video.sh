#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"

FRAMES_DIR="${1:-$FIG_DIR/frames}"
OUT_VIDEO_DIR="${2:-$VIDEO_DIR}"

mkdir -p "$OUT_VIDEO_DIR"
rm -f "$OUT_VIDEO_DIR"/*.mp4 2>/dev/null || true

FRAMES_ARG="$FRAMES_DIR"
VIDEOS_ARG="$OUT_VIDEO_DIR"
if command -v cygpath >/dev/null 2>&1; then
  FRAMES_ARG="$(cygpath -m "$FRAMES_DIR")"
  VIDEOS_ARG="$(cygpath -m "$OUT_VIDEO_DIR")"
fi

echo "[video] exp_name: $EXP_NAME"
echo "[video] frames  : $FRAMES_DIR"
echo "[video] output  : $OUT_VIDEO_DIR"

"$PYTHON_BIN" "$ALEX_DIR/scripts/python/make_videos.py" \
  "$FRAMES_ARG" \
  "$VIDEOS_ARG" \
  --fps "$FPS"
