#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${1:-${CONFIG_PATH:-$ALEX_DIR/gpu/configs/cavity_re1000.cfg}}" \
BIN_DIR="${2:-${BIN_DIR:-$ALEX_DIR/results/binaries/gpu_cavity_re1000}}" \
BACKEND="gpu" \
DEVICE="${3:-${DEVICE:-cuda:0}}" \
bash "$ALEX_DIR/scripts/run.sh"
