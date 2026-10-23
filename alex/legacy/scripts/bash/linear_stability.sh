#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(cd "$ALEX_DIR/.." && pwd)"
source "$ALEX_DIR/RUN"

if [[ -x "$PROJECT_DIR/.venv/bin/python" && "$PYTHON_BIN" == "python3" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
fi

# =========================
# USER SETTINGS
# =========================
EQUILIBRIUM_SNAPSHOT="${EQUILIBRIUM_SNAPSHOT:-$NEWTON_DIR/newton_equilibrium.bin}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$BIN_DIR}"
OUT_DIR="${OUT_DIR:-$STABILITY_DIR}"
FILTERED_DIR="${FILTERED_DIR:-$FILTERED_BIN_DIR}"

EIGS_COUNT="${EIGS_COUNT:-30}"
EIGS_TOL="${EIGS_TOL:-1e-8}"
EIGS_MAX_ITER="${EIGS_MAX_ITER:-3000}"
UNSTABLE_TOL="${UNSTABLE_TOL:-1e-9}"
SNAPSHOT_LIMIT="${SNAPSHOT_LIMIT:-}"
NO_PLOTS="${NO_PLOTS:-true}"
FILTER_ONLY="${FILTER_ONLY:-false}"
MAX_UNSTABLE_MODES="${MAX_UNSTABLE_MODES:-8}"
# =========================

if [[ ! -f "$EQUILIBRIUM_SNAPSHOT" ]]; then
  echo "[stability] Equilibrium snapshot not found: $EQUILIBRIUM_SNAPSHOT" >&2
  echo "[stability] Run: bash $ALEX_DIR/scripts/bash/newton_from_csv.sh" >&2
  exit 1
fi

if [[ ! -d "$SNAPSHOTS_DIR" ]]; then
  echo "[stability] Snapshots directory not found: $SNAPSHOTS_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$FILTERED_DIR"

CMD=(
  "$PYTHON_BIN" "$ALEX_DIR/scripts/python/linear_stability.py"
  "$EQUILIBRIUM_SNAPSHOT"
  --config "$CONFIG_PATH"
  --snapshots-dir "$SNAPSHOTS_DIR"
  --out-dir "$OUT_DIR"
  --filtered-dir "$FILTERED_DIR"
  --eigs-count "$EIGS_COUNT"
  --eigs-tol "$EIGS_TOL"
  --eigs-max-iter "$EIGS_MAX_ITER"
  --unstable-tol "$UNSTABLE_TOL"
  --max-unstable-modes "$MAX_UNSTABLE_MODES"
)

if [[ -n "$SNAPSHOT_LIMIT" ]]; then
  CMD+=(--snapshot-limit "$SNAPSHOT_LIMIT")
fi

if [[ "$NO_PLOTS" == "true" ]]; then
  CMD+=(--no-plots)
fi

if [[ "$FILTER_ONLY" == "true" ]]; then
  CMD+=(--filter-only)
fi

echo "[stability] exp_name   : $EXP_NAME"
echo "[stability] equilibrium: $EQUILIBRIUM_SNAPSHOT"
echo "[stability] snapshots  : $SNAPSHOTS_DIR"
echo "[stability] output     : $OUT_DIR"
echo "[stability] filtered   : $FILTERED_DIR"
echo "[stability] filter-only: $FILTER_ONLY"
"${CMD[@]}"
