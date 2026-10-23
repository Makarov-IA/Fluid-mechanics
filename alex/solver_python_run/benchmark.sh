#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
STEPS=1000

"$PYTHON_BIN" "$ALEX_DIR/solver_python/run.py" --config "$CONFIG" --benchmark "$STEPS"
