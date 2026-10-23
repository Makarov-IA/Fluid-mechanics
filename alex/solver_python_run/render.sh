#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

"$PYTHON_BIN" "$ALEX_DIR/solver_python/render.py" --config "$CONFIG"
