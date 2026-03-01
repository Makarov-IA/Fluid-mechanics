#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
BIN="$BUILD_DIR/solver_app"

CONFIG_PATH="${1:-$ROOT_DIR/configs/config.cfg}"
OUT_DIR="${2:-$ROOT_DIR/data/results}"

mkdir -p "$BUILD_DIR"
mkdir -p "$OUT_DIR"

echo "[run] Building..."
g++ -std=c++17 \
  "$ROOT_DIR/main.cpp" \
  "$ROOT_DIR/solver/v1/solver.cpp" \
  -I"$ROOT_DIR" \
  -I"$ROOT_DIR/../External_libs" \
  -O2 \
  -o "$BIN"

echo "[run] Running..."
"$BIN" "$CONFIG_PATH" "$OUT_DIR"

echo "[run] Finished."
