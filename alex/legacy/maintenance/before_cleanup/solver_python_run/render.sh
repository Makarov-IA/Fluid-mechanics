#!/usr/bin/env bash

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ALEX_DIR/../.venv/bin/python"
CONFIG="$ALEX_DIR/solver_python/config.yaml"

"$PYTHON_BIN" "$ALEX_DIR/solver_python/render.py" --config "$CONFIG"
