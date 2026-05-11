#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ALEX_DIR/RUN"

PY_SCRIPT="$ALEX_DIR/scripts/python/newton_from_csv.py"

# =========================
# USER SETTINGS
# =========================
INPUT_SNAPSHOT="${INPUT_SNAPSHOT:-$STATIONARY_DIR/stationary_state.bin}"
OUTPUT_SNAPSHOT="${OUTPUT_SNAPSHOT:-$NEWTON_DIR/newton_equilibrium.bin}"

CHECK_ONLY="${CHECK_ONLY:-false}"

MAX_NEWTON="${MAX_NEWTON:-10}"
NEWTON_TOL="${NEWTON_TOL:-1e-8}"
LINEAR_TOL="${LINEAR_TOL:-1e-3}"

GMRES_RESTART="${GMRES_RESTART:-30}"
GMRES_MAX_ITER="${GMRES_MAX_ITER:-160}"

FD_EPS="${FD_EPS:-1e-9}"
LINE_SEARCH_STEPS="${LINE_SEARCH_STEPS:-12}"

RE_OVERRIDE="${RE_OVERRIDE:-}"

EXTRA_ARGS=(
  --jacobian exact
  --preconditioner stokes
  --verify-jv
)
# =========================

mkdir -p "$NEWTON_DIR"

if [[ ! -f "$INPUT_SNAPSHOT" ]]; then
  echo "[newton] Snapshot not found: $INPUT_SNAPSHOT" >&2
  echo "[newton] Run first: bash $ALEX_DIR/scripts/bash/find_stationary.sh" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[newton] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  "$INPUT_SNAPSHOT"
  --config "$CONFIG_PATH"
  -o "$OUTPUT_SNAPSHOT"
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

echo "[newton] exp_name: $EXP_NAME"
echo "[newton] input   : $INPUT_SNAPSHOT"
echo "[newton] output  : $OUTPUT_SNAPSHOT"
echo "[newton] config  : $CONFIG_PATH"
"${CMD[@]}"
