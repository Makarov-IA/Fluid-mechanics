#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_SCRIPT="$ROOT_DIR/scripts/streamplot_csv.py"

# =========================
# USER SETTINGS
# =========================
# Edit this block and then run:
#   bash alex/pipeline/tools/streamplot_csv.sh

INPUT_SNAPSHOT="$ROOT_DIR/stationary_detection/newton_equilibrium.bin"
OUTPUT_PNG="$ROOT_DIR/stationary_snapshot/newton_equilibrium_from_stationary.png"

# INPUT_SNAPSHOT="$ROOT_DIR/stationary_detection/stationary_state.bin"
# OUTPUT_PNG="$ROOT_DIR/stationary_snapshot/before_oscillation_streamplot.png"

DENSITY="1.6"
DPI=180
TITLE="Before oscillation, t=2.32001"

# Set DRAW_CONTOURS=false to hide psi contour lines.
DRAW_CONTOURS=true

# Add raw extra arguments here if needed.
EXTRA_ARGS=(
)
# =========================

if [[ ! -f "$INPUT_SNAPSHOT" ]]; then
  echo "[streamplot] Snapshot not found: $INPUT_SNAPSHOT" >&2
  exit 1
fi

CMD=(
  python3 "$PY_SCRIPT"
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
