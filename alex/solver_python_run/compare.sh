#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
CONFIG="$ALEX_DIR/configs/pulsation_re260_stabilized/pulsation_re260_stabilized.yaml"
UNCONTROLLED=pulsation_re260

"$PYTHON_BIN" "$ALEX_DIR/solver_python/compare_stabilization.py" --config "$CONFIG" --uncontrolled "$UNCONTROLLED"
