#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="$ROOT_DIR/scripts/newton_from_csv.py"

# =========================
# USER SETTINGS
# =========================
# Edit this block and then run:
#   bash alex/scripts/newton_from_csv.sh
#
# INPUT_CSV can be any Alex CSV with columns:
#   x,y,psi,omega,u,v
INPUT_CSV="$ROOT_DIR/stationary_detection/stationary_state.csv"
OUTPUT_CSV="$ROOT_DIR/stationary_detection/newton_equilibrium.csv"
CONFIG_PATH="$ROOT_DIR/configs/config.cfg"

# Set CHECK_ONLY=true to print the residual and skip Newton iterations.
CHECK_ONLY=false

# Newton settings.
MAX_NEWTON=10
NEWTON_TOL="1e-8"
LINEAR_TOL="1e-3"

# Inner GMRES settings for J * delta = -F.
GMRES_RESTART=30
GMRES_MAX_ITER=160

# Finite-difference Jacobian-vector product and damping.
FD_EPS="1e-9"
LINE_SEARCH_STEPS=12

# Optional override. Leave empty to use Re from CONFIG_PATH.
RE_OVERRIDE=""

# Add raw extra arguments here if needed.
EXTRA_ARGS=(
  --jacobian exact
  --preconditioner stokes
  --verify-jv
)
# =========================

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "[newton] CSV not found: $INPUT_CSV" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[newton] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

CMD=(
  python3 "$PY_SCRIPT"
  "$INPUT_CSV"
  --config "$CONFIG_PATH"
  -o "$OUTPUT_CSV"
  --max-newton "$MAX_NEWTON"
  --newton-tol "$NEWTON_TOL"
  --linear-tol "$LINEAR_TOL"
  --gmres-restart "$GMRES_RESTART"
  --gmres-max-iter "$GMRES_MAX_ITER"
  --fd-eps "$FD_EPS"
  --line-search-steps "$LINE_SEARCH_STEPS"
)

if [[ "$CHECK_ONLY" == "true" ]]; then
  CMD+=(--check-only)
fi

if [[ -n "$RE_OVERRIDE" ]]; then
  CMD+=(--re "$RE_OVERRIDE")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[newton] input : $INPUT_CSV"
echo "[newton] output: $OUTPUT_CSV"
echo "[newton] config: $CONFIG_PATH"
"${CMD[@]}"
