#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "$ROOT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$PROJECT_DIR/.venv/bin/python" && "$PYTHON_BIN" == "python3" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
fi

# =========================
# USER SETTINGS
# =========================
EQUILIBRIUM_SNAPSHOT="$ROOT_DIR/stationary_detection/newton_equilibrium.bin"
CONFIG_PATH="$ROOT_DIR/configs/config.cfg"
SNAPSHOTS_DIR="$ROOT_DIR/data/results"
OUT_DIR="$ROOT_DIR/linear_stability"
FILTERED_DIR="$OUT_DIR/filtered_results"

# Number of rightmost eigenpairs found by scipy.sparse.linalg.eigs.
EIGS_COUNT=30
EIGS_TOL="1e-8"
EIGS_MAX_ITER=3000

# Positive real part threshold.
UNSTABLE_TOL="1e-9"

# For quick tests set, for example, SNAPSHOT_LIMIT=5.
SNAPSHOT_LIMIT=""

# Linear stability writes CSV. Frames/videos are produced by the common plot/video scripts.
NO_PLOTS=true

# If the spectrum is already computed and unstable_modes/eig_*.csv exist,
# set FILTER_ONLY=true to redo only filtered_results without running SciPy eigs.
FILTER_ONLY="${FILTER_ONLY:-false}"

# Subtract at most this many unstable eigenmodes from each snapshot.
MAX_UNSTABLE_MODES=8

# Bad modes are removed by orthogonal projection of q(t)-q_equilibrium
# onto span(Re(v), Im(v)); there is no hand-tuned amplitude.
# =========================

if [[ ! -f "$EQUILIBRIUM_SNAPSHOT" ]]; then
  echo "[stability] Equilibrium snapshot not found: $EQUILIBRIUM_SNAPSHOT" >&2
  echo "[stability] Run: bash $ROOT_DIR/pipeline/04_newton.sh" >&2
  exit 1
fi

if [[ ! -d "$SNAPSHOTS_DIR" ]]; then
  echo "[stability] Snapshots directory not found: $SNAPSHOTS_DIR" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/linear_stability.py"
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

echo "[stability] equilibrium: $EQUILIBRIUM_SNAPSHOT"
echo "[stability] snapshots  : $SNAPSHOTS_DIR"
echo "[stability] output     : $OUT_DIR"
echo "[stability] filter-only: $FILTER_ONLY"
"${CMD[@]}"
