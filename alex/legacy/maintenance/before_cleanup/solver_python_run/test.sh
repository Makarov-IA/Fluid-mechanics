#!/usr/bin/env bash

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ALEX_DIR/../.venv/bin/python"
TESTS="$ALEX_DIR/solver_python/tests"

PYTHONPATH="$ALEX_DIR" "$PYTHON_BIN" -m pytest "$TESTS" -q
