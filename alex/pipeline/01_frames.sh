#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ALEX_DIR/RUN"

BIN_DIR="${1:-$BIN_DIR}" \
FIG_DIR="${2:-$FIG_DIR}" \
SNAPSHOT_INDEX="${3:-${SNAPSHOT_INDEX:-}}" \
bash "$ALEX_DIR/scripts/bash/plot.sh"
