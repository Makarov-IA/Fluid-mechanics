#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
CONFIG="$ALEX_DIR/configs/magnet_placement_array/magnet_placement_array.yaml"

"$PYTHON_BIN" "$ALEX_DIR/solver_python/magnet_design.py" --config "$CONFIG"
