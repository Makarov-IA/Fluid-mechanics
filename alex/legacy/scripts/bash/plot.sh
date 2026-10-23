#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"

OUT_DIR="${1:-$BIN_DIR}"
PLOT_ROOT="${2:-$FIG_DIR}"
SNAPSHOT_INDEX="${3:-${SNAPSHOT_INDEX:-}}"
FRAMES_DIR="$PLOT_ROOT/frames"

mkdir -p "$FRAMES_DIR"

rm -f "$FRAMES_DIR"/*.png 2>/dev/null || true
rm -f "$PLOT_ROOT"/*.png 2>/dev/null || true

RESULTS_ARG="$OUT_DIR"
FRAMES_ARG="$FRAMES_DIR"
if command -v cygpath >/dev/null 2>&1; then
  RESULTS_ARG="$(cygpath -m "$OUT_DIR")"
  FRAMES_ARG="$(cygpath -m "$FRAMES_DIR")"
fi

echo "[plot] exp_name: $EXP_NAME"
echo "[plot] input   : $OUT_DIR"
echo "[plot] figures : $PLOT_ROOT"

CMD=(
  "$PYTHON_BIN" "$ALEX_DIR/scripts/python/plot_fields.py"
  "$RESULTS_ARG"
  "$FRAMES_ARG"
)

if [[ -n "$SNAPSHOT_INDEX" ]]; then
  INDEX_ARG="$SNAPSHOT_INDEX"
  if command -v cygpath >/dev/null 2>&1; then
    INDEX_ARG="$(cygpath -m "$SNAPSHOT_INDEX")"
  fi
  CMD+=(--snapshot-index "$INDEX_ARG")
fi

"${CMD[@]}"

echo "[plot] Done."
