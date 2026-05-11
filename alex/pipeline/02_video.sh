#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ALEX_DIR/RUN"

bash "$ALEX_DIR/scripts/bash/video.sh" "${1:-$FIG_DIR/frames}" "${2:-$VIDEO_DIR}"
