#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN_DIR="${1:?usage: render_one.sh BIN_DIR FIG_DIR TABLE_DIR [VIDEO_DIR]}" \
FIG_DIR="${2:?usage: render_one.sh BIN_DIR FIG_DIR TABLE_DIR [VIDEO_DIR]}" \
TABLE_DIR="${3:?usage: render_one.sh BIN_DIR FIG_DIR TABLE_DIR [VIDEO_DIR]}" \
VIDEO_DIR="${4:-${VIDEO_DIR:-$ALEX_DIR/results/videos/manual_gpu_render}}" \
BACKEND="gpu" \
bash "$ALEX_DIR/scripts/render.sh"

BIN_DIR="${1}" \
FIG_DIR="${2}" \
TABLE_DIR="${3}" \
VIDEO_DIR="${4:-${VIDEO_DIR:-$ALEX_DIR/results/videos/manual_gpu_render}}" \
BACKEND="gpu" \
bash "$ALEX_DIR/scripts/video.sh"
