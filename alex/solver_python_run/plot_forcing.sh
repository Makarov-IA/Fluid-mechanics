#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
CONFIG="$ALEX_DIR/configs/magnets_2x3_re260/magnets_2x3_re260.yaml"

"$PYTHON_BIN" "$ALEX_DIR/solver_python/plot_forcing.py" --config "$CONFIG" --open
