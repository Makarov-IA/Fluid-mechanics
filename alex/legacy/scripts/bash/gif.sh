#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"

FRAMES_DIR="${1:-$FIG_DIR/frames}"
GIFS_DIR="${2:-$FIG_DIR/gifs}"
DURATION="${DURATION:-50}"

mkdir -p "$GIFS_DIR"
rm -f "$GIFS_DIR"/*.gif 2>/dev/null || true

FRAMES_ARG="$FRAMES_DIR"
GIFS_ARG="$GIFS_DIR"
if command -v cygpath >/dev/null 2>&1; then
  FRAMES_ARG="$(cygpath -m "$FRAMES_DIR")"
  GIFS_ARG="$(cygpath -m "$GIFS_DIR")"
fi

"$PYTHON_BIN" "$ALEX_DIR/scripts/python/make_gifs.py" \
  "$FRAMES_ARG" \
  "$GIFS_ARG" \
  --duration "$DURATION"
