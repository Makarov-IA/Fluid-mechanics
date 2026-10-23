#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
TESTS="$ALEX_DIR/solver_python/tests"

PYTHONPATH="$ALEX_DIR" "$PYTHON_BIN" -m pytest "$TESTS" -q --basetemp "$ALEX_DIR/legacy/maintenance/pytest_tmp" -o cache_dir="$ALEX_DIR/legacy/maintenance/pytest_cache"
