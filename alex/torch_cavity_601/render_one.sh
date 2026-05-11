#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/alex/torch_cavity_601"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RESULTS_DIR="${1:?usage: render_one.sh RESULTS_DIR CONFIG_PATH ANALYSIS_DIR}"
CONFIG_PATH="${2:?usage: render_one.sh RESULTS_DIR CONFIG_PATH ANALYSIS_DIR}"
ANALYSIS_DIR="${3:?usage: render_one.sh RESULTS_DIR CONFIG_PATH ANALYSIS_DIR}"

PLOT_ROOT="$ANALYSIS_DIR/plots"
FRAMES_DIR="$PLOT_ROOT/frames"
VIDEOS_DIR="$PLOT_ROOT/videos"
PROFILES_DIR="$ANALYSIS_DIR/profiles"

mkdir -p "$FRAMES_DIR" "$VIDEOS_DIR" "$PROFILES_DIR"

cd "$ROOT_DIR"

echo "[render-one] frames: $RESULTS_DIR"
bash "$ROOT_DIR/alex/pipeline/01_frames.sh" "$RESULTS_DIR" "$PLOT_ROOT"

echo "[render-one] videos: $FRAMES_DIR"
"$PYTHON_BIN" "$ROOT_DIR/alex/scripts/make_videos.py" "$FRAMES_DIR" "$VIDEOS_DIR" --fps 30

echo "[render-one] final plots and centerline profiles"
"$PYTHON_BIN" "$EXP_DIR/profile_cavity.py" "$RESULTS_DIR" "$PROFILES_DIR"

echo "[render-one] done: $ANALYSIS_DIR"
