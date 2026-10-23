#!/usr/bin/env bash

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ALEX_DIR/../.venv/bin/python"
CONFIG="$ALEX_DIR/solver_python/config.yaml"
STEPS=10000
OUTPUT_DIR="$ALEX_DIR/results/binaries/magnets_check_$(date +%Y%m%d_%H%M%S)"

"$PYTHON_BIN" "$ALEX_DIR/solver_python/verify_magnets.py" \
  --config "$CONFIG" --steps "$STEPS" --output "$OUTPUT_DIR"
