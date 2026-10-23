#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
BASE="$ALEX_DIR/configs/pulsation_re260_base/pulsation_re260_base.yaml"
PROJECTION="$ALEX_DIR/configs/pulsation_re260_projection/pulsation_re260_projection.yaml"
RHS="$ALEX_DIR/configs/pulsation_re260_rhs/pulsation_re260_rhs.yaml"

"$PYTHON_BIN" "$ALEX_DIR/solver_python/compare_projection.py" --base "$BASE" --projection "$PROJECTION" --rhs "$RHS"
