#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
CONFIG="$ALEX_DIR/configs/pulsation_re260_projection/pulsation_re260_projection.yaml"
"$PYTHON_BIN" "$ALEX_DIR/solver_python/run.py" --config "$CONFIG"
