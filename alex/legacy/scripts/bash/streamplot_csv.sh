#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"
PY_SCRIPT="$ALEX_DIR/scripts/python/streamplot_csv.py"

# =========================
# USER SETTINGS
# =========================
INPUT_SNAPSHOT="${INPUT_SNAPSHOT:-$NEWTON_DIR/newton_equilibrium.bin}"
OUTPUT_PNG="${OUTPUT_PNG:-$FIG_DIR/newton_equilibrium_streamplot.png}"

DENSITY="${DENSITY:-1.6}"
DPI="${DPI:-180}"
TITLE="${TITLE:-$EXP_NAME}"
DRAW_CONTOURS="${DRAW_CONTOURS:-true}"

EXTRA_ARGS=(
)
# =========================

if [[ ! -f "$INPUT_SNAPSHOT" ]]; then
  echo "[streamplot] Snapshot not found: $INPUT_SNAPSHOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PNG")"

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  "$INPUT_SNAPSHOT"
  -o "$OUTPUT_PNG"
  --density "$DENSITY"
  --dpi "$DPI"
  --title "$TITLE"
)

if [[ "$DRAW_CONTOURS" != "true" ]]; then
  CMD+=(--no-contours)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[streamplot] input : $INPUT_SNAPSHOT"
echo "[streamplot] output: $OUTPUT_PNG"
"${CMD[@]}"
