#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
BIN="$BUILD_DIR/solver_app.exe"

CONFIG_PATH="${1:-$ROOT_DIR/configs/config.cfg}"
OUT_DIR="${2:-$ROOT_DIR/data/results}"

rm -rf "$OUT_DIR"

mkdir -p "$OUT_DIR"

make -C "$ROOT_DIR" all

if [[ ! -x "$BIN" ]]; then
  echo "[run] Build failed: binary not found at $BIN" >&2
  exit 1
fi

"$BIN" "$CONFIG_PATH" "$OUT_DIR"
