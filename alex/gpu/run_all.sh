#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LIST="${GPU_LIST:-0}"
RUN_PARALLEL="${RUN_PARALLEL:-0}"
DTYPE="${DTYPE:-float64}"

CASES=(
  "gpu_cavity_re1000:$ALEX_DIR/gpu/configs/cavity_re1000.cfg"
  "gpu_cavity_re5000:$ALEX_DIR/gpu/configs/cavity_re5000.cfg"
  "gpu_cavity_re10000:$ALEX_DIR/gpu/configs/cavity_re10000.cfg"
)

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  GPUS=(0)
fi

run_case() {
  local idx="$1"
  local item="${CASES[$idx]}"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local exp_name config_path
  IFS=':' read -r exp_name config_path <<< "$item"
  echo "[gpu-run-all] exp=$exp_name config=$config_path gpu=$gpu dtype=$DTYPE"
  BACKEND="gpu" \
  EXP_NAME="$exp_name" \
  CONFIG_PATH="$config_path" \
  DEVICE="cuda:0" \
  DTYPE="$DTYPE" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  bash "$ALEX_DIR/scripts/run.sh"
}

if [[ "$RUN_PARALLEL" == "1" ]]; then
  pids=()
  for idx in "${!CASES[@]}"; do
    run_case "$idx" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
else
  for idx in "${!CASES[@]}"; do
    run_case "$idx"
  done
fi
