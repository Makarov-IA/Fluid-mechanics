#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
CONFIG="$ALEX_DIR/configs/magnets_2x3_re260/magnets_2x3_re260.yaml"
STEPS=10000
OUTPUT_DIR="$ALEX_DIR/results/magnets_2x3_re260/verification/$(date +%Y%m%d_%H%M%S)"

"$PYTHON_BIN" "$ALEX_DIR/solver_python/verify_magnets.py" \
  --config "$CONFIG" --steps "$STEPS" --output "$OUTPUT_DIR"
