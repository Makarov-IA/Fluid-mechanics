#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILTERED_DIR="$ROOT_DIR/linear_stability/filtered_results"
SNAPSHOT_INDEX="$ROOT_DIR/linear_stability/filtered_snapshots.csv"
PLOT_ROOT="$ROOT_DIR/linear_stability/plots"
FRAMES_DIR="$PLOT_ROOT/frames"
VIDEOS_DIR="$ROOT_DIR/linear_stability/videos"
FPS=30

if [[ ! -d "$FILTERED_DIR" ]]; then
  echo "[filtered-video] Filtered CSV directory not found: $FILTERED_DIR" >&2
  echo "[filtered-video] Run linear stability first:" >&2
  echo "[filtered-video]   bash $ROOT_DIR/scripts/linear_stability.sh" >&2
  exit 1
fi

echo "[filtered-video] Rendering filtered frames with the common plot script"
bash "$ROOT_DIR/scripts/plot.sh" "$FILTERED_DIR" "$PLOT_ROOT" "$SNAPSHOT_INDEX"

FRAME_COUNT="$(find "$FRAMES_DIR" -maxdepth 1 -type f -name '*_streamplot.png' | wc -l | tr -d ' ')"
if [[ "$FRAME_COUNT" -lt 2 ]]; then
  echo "[filtered-video] Need at least two streamplot frames in: $FRAMES_DIR" >&2
  echo "[filtered-video] Found: $FRAME_COUNT" >&2
  exit 1
fi

mkdir -p "$VIDEOS_DIR"
rm -f "$VIDEOS_DIR"/*.mp4 2>/dev/null || true

FRAMES_ARG="$FRAMES_DIR"
VIDEOS_ARG="$VIDEOS_DIR"
if command -v cygpath >/dev/null 2>&1; then
  FRAMES_ARG="$(cygpath -m "$FRAMES_DIR")"
  VIDEOS_ARG="$(cygpath -m "$VIDEOS_DIR")"
fi

python3 "$ROOT_DIR/scripts/make_videos.py" \
  "$FRAMES_ARG" \
  "$VIDEOS_ARG" \
  --fps "$FPS"

if [[ -f "$VIDEOS_DIR/streamplot.mp4" ]]; then
  mv "$VIDEOS_DIR/streamplot.mp4" "$VIDEOS_DIR/filtered_streamplot.mp4"
  echo "[filtered-video] filtered streamplot: $VIDEOS_DIR/filtered_streamplot.mp4"
fi

if [[ -f "$VIDEOS_DIR/psi.mp4" ]]; then
  mv "$VIDEOS_DIR/psi.mp4" "$VIDEOS_DIR/filtered_psi.mp4"
  echo "[filtered-video] filtered psi: $VIDEOS_DIR/filtered_psi.mp4"
fi

if [[ -f "$VIDEOS_DIR/omega.mp4" ]]; then
  mv "$VIDEOS_DIR/omega.mp4" "$VIDEOS_DIR/filtered_omega.mp4"
  echo "[filtered-video] filtered omega: $VIDEOS_DIR/filtered_omega.mp4"
fi
