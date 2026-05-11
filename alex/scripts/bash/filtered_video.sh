#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"

SNAPSHOT_INDEX="$STABILITY_DIR/filtered_snapshots.csv"
FRAMES_DIR="$FILTERED_FIG_DIR/frames"

if [[ ! -d "$FILTERED_BIN_DIR" ]]; then
  echo "[filtered-video] Filtered snapshot directory not found: $FILTERED_BIN_DIR" >&2
  echo "[filtered-video] Run linear stability first:" >&2
  echo "[filtered-video]   bash $ALEX_DIR/scripts/bash/linear_stability.sh" >&2
  exit 1
fi

echo "[filtered-video] Rendering filtered frames"
bash "$ALEX_DIR/scripts/bash/plot.sh" "$FILTERED_BIN_DIR" "$FILTERED_FIG_DIR" "$SNAPSHOT_INDEX"

FRAME_COUNT="$(find "$FRAMES_DIR" -maxdepth 1 -type f -name '*_streamplot.png' | wc -l | tr -d ' ')"
if [[ "$FRAME_COUNT" -lt 2 ]]; then
  echo "[filtered-video] Need at least two streamplot frames in: $FRAMES_DIR" >&2
  echo "[filtered-video] Found: $FRAME_COUNT" >&2
  exit 1
fi

mkdir -p "$FILTERED_VIDEO_DIR"
rm -f "$FILTERED_VIDEO_DIR"/*.mp4 2>/dev/null || true

"$PYTHON_BIN" "$ALEX_DIR/scripts/python/make_videos.py" \
  "$FRAMES_DIR" \
  "$FILTERED_VIDEO_DIR" \
  --fps "$FPS"

for kind in streamplot psi omega; do
  if [[ -f "$FILTERED_VIDEO_DIR/$kind.mp4" ]]; then
    mv "$FILTERED_VIDEO_DIR/$kind.mp4" "$FILTERED_VIDEO_DIR/filtered_$kind.mp4"
  fi
done
