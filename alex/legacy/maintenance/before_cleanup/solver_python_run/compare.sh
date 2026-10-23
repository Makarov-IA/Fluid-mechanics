#!/usr/bin/env bash

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ALEX_DIR/../.venv/bin/python"
CONFIG="$ALEX_DIR/solver_python/config.yaml"
UNCONTROLLED=pulsation_re260

"$PYTHON_BIN" "$ALEX_DIR/solver_python/compare_stabilization.py" --config "$CONFIG" --uncontrolled "$UNCONTROLLED"
