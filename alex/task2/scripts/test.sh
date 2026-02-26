#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TASK2_ROOT="$SCRIPT_DIR/.."

cd "$TASK2_ROOT" || exit 1

if command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
else
  echo "Python not found. Install Python 3."
  exit 1
fi

TEST_ROOT=tmp_test_runs
rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"

echo "[1/4] Build"
make all >/dev/null

echo "[2/4] zero forcing, zero lid => near-zero at t=1"
./task2 --Nx 41 --Ny 41 --Nt 80 --nu 0.02 --lid 0 --forcing zero --output-dir "$TEST_ROOT/case_zero" --quiet >/dev/null
"${PYTHON_CMD[@]}" - <<'PY'
import numpy as np
from pathlib import Path
p = Path('tmp_test_runs/case_zero')
psi = np.loadtxt(p / 'psi.txt')
omega = np.loadtxt(p / 'omega.txt')
mx = max(float(np.max(np.abs(psi))), float(np.max(np.abs(omega))))
assert mx < 1e-5, f'expected near-zero fields, got {mx}'
print(f'  ok: near-zero max={mx:.3e}')
PY

echo "[3/4] g(x,y,t) != 0 => non-trivial omega"
./task2 --Nx 41 --Ny 41 --Nt 120 --nu 0.02 --lid 0 --forcing sin --forcing-amp 1.0 --forcing-omega 6.283185307 --save-every 30 --output-dir "$TEST_ROOT/case_forcing" --quiet >/dev/null
"${PYTHON_CMD[@]}" - <<'PY'
import numpy as np
from pathlib import Path
w = np.loadtxt(Path('tmp_test_runs/case_forcing/omega.txt'))
mx = float(np.max(np.abs(w)))
assert mx > 1e-3, f'expected non-trivial omega, got {mx}'
print(f'  ok: non-trivial omega max={mx:.3e}')
PY

echo "[4/4] snapshot plotting + scheme metadata"
"${PYTHON_CMD[@]}" ./plot_solution.py --input-dir "$TEST_ROOT/case_forcing" --output-dir "$TEST_ROOT/plots" --field omega --graph contour --snapshot-index 1 --save --no-show >/dev/null

for f in psi.txt omega.txt meta.txt snapshots/index.csv; do
  test -f "$TEST_ROOT/case_forcing/$f"
done
test -f "$TEST_ROOT/plots/omega_snap001_step00030_t0.250_contour.png"
grep -q "scheme backward_euler_implicit_fd" "$TEST_ROOT/case_forcing/meta.txt"

echo "All task2 tests passed"
