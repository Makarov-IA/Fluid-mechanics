#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
GAIN=1.0

"$PYTHON_BIN" "$ALEX_DIR/solver_python/stability.py" --config "$CONFIG" --gain "$GAIN"
