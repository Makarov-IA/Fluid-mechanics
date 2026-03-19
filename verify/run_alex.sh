#!/usr/bin/env bash
# Build Alex's solver and run it with the verification config.
# Run from the repo root or from verify/:
#   cd /path/to/Fluid-mechanics/verify && bash run_alex.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALEX_DIR="$SCRIPT_DIR/../alex"
VERIFY_DIR="$SCRIPT_DIR"

echo "=== Building Alex's solver ==="
cd "$ALEX_DIR"
make -j"$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"

echo ""
echo "=== Running Alex's solver (Re=10000, 105x105, fixed_dt_steps, t=720) ==="
cd "$VERIFY_DIR"
"$ALEX_DIR/build/solver_app.exe" "$VERIFY_DIR/config_alex.cfg" "$VERIFY_DIR/data_alex"

echo ""
echo "=== Done. Results in: $VERIFY_DIR/data_alex ==="
