#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ALEX_DIR/RUN"

CONFIG_PATH="${1:-$CONFIG_PATH}" \
BIN_DIR="${2:-$BIN_DIR}" \
BACKEND="${BACKEND:-cpu}" \
bash "$ALEX_DIR/scripts/bash/run.sh"
