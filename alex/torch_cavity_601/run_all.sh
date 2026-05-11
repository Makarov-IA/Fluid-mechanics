#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/alex/torch_cavity_601"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_LIST="${GPU_LIST:-0}"
RUN_PARALLEL="${RUN_PARALLEL:-0}"
DTYPE="${DTYPE:-float64}"

CONFIGS=(
  "$EXP_DIR/configs/cavity_re1000.cfg"
  "$EXP_DIR/configs/cavity_re5000.cfg"
  "$EXP_DIR/configs/cavity_re10000.cfg"
)

OUT_DIRS=(
  "$EXP_DIR/results/re1000"
  "$EXP_DIR/results/re5000"
  "$EXP_DIR/results/re10000"
)

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  GPUS=(0)
fi

cd "$ROOT_DIR"

run_case() {
  local idx="$1"
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local config="${CONFIGS[$idx]}"
  local out_dir="${OUT_DIRS[$idx]}"
  echo "[run-all] config=$config out=$out_dir gpu=$gpu dtype=$DTYPE"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$EXP_DIR/torch_cavity_solver.py" \
    "$config" \
    "$out_dir" \
    --device cuda:0 \
    --dtype "$DTYPE"
}

if [[ "$RUN_PARALLEL" == "1" ]]; then
  pids=()
  for idx in "${!CONFIGS[@]}"; do
    run_case "$idx" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
else
  for idx in "${!CONFIGS[@]}"; do
    run_case "$idx"
  done
fi
