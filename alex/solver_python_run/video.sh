#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
FPS=20

"$PYTHON_BIN" "$ALEX_DIR/solver_python/render.py" --config "$CONFIG" --video --fps "$FPS"
