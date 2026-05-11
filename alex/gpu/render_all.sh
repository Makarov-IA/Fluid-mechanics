#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CASES=(
  "gpu_cavity_re1000:$ALEX_DIR/gpu/configs/cavity_re1000.cfg"
  "gpu_cavity_re5000:$ALEX_DIR/gpu/configs/cavity_re5000.cfg"
  "gpu_cavity_re10000:$ALEX_DIR/gpu/configs/cavity_re10000.cfg"
)

for item in "${CASES[@]}"; do
  IFS=':' read -r exp_name config_path <<< "$item"
  bin_dir="$ALEX_DIR/results/binaries/$exp_name"
  if [[ ! -d "$bin_dir" ]]; then
    echo "[gpu-render-all] missing binaries, skipped: $bin_dir" >&2
    continue
  fi
  echo "[gpu-render-all] exp=$exp_name"
  BACKEND="gpu" \
  EXP_NAME="$exp_name" \
  CONFIG_PATH="$config_path" \
  bash "$ALEX_DIR/scripts/render.sh"
  BACKEND="gpu" \
  EXP_NAME="$exp_name" \
  CONFIG_PATH="$config_path" \
  bash "$ALEX_DIR/scripts/video.sh"
done
