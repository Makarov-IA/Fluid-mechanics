#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(cd "$ALEX_DIR/.." && pwd)"
source "$ALEX_DIR/RUN"

cd "$PROJECT_DIR"

if [[ "$CLEAN_BINARIES" == "true" ]]; then
  rm -rf "$BIN_DIR"
fi
mkdir -p "$BIN_DIR"

echo "[run] backend : $BACKEND"
echo "[run] exp_name: $EXP_NAME"
echo "[run] config  : $CONFIG_PATH"
echo "[run] output  : $BIN_DIR"

case "$BACKEND" in
  cpu)
    make -C "$ALEX_DIR" all
    CPU_BIN="$ALEX_DIR/cpu/build/solver_app.exe"
    if [[ ! -x "$CPU_BIN" ]]; then
      echo "[run] CPU binary not found: $CPU_BIN" >&2
      exit 1
    fi
    "$CPU_BIN" "$CONFIG_PATH" "$BIN_DIR"
    ;;
  gpu)
    CMD=(
      "$PYTHON_BIN" "$ALEX_DIR/gpu/torch_cavity_solver.py"
      "$CONFIG_PATH"
      "$BIN_DIR"
      --device "$DEVICE"
      --dtype "$DTYPE"
    )
    if [[ "$CLEAN_BINARIES" != "true" ]]; then
      CMD+=(--no-clean)
    fi
    "${CMD[@]}"
    ;;
  *)
    echo "[run] Unknown BACKEND=$BACKEND; expected cpu or gpu" >&2
    exit 1
    ;;
esac
