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

mkdir -p test_runs

echo "[1/4] Build"
make all >/dev/null

echo "[2/4] Zero forcing + zero lid => near-zero solution"
./task2 --Nx 41 --Ny 41 --nu 0.02 --lid 0 --forcing zero --tol 1e-7 --max-iter 6000 --output-dir test_runs/case_zero >/dev/null
"${PYTHON_CMD[@]}" - <<'PY'
import numpy as np
from pathlib import Path
p = Path('test_runs/case_zero')
psi = np.loadtxt(p / 'psi.txt')
omega = np.loadtxt(p / 'omega.txt')
mx = max(float(np.max(np.abs(psi))), float(np.max(np.abs(omega))))
assert mx < 5e-6, f'expected near-zero fields, got max={mx}'
print(f'  ok: near-zero max={mx:.3e}')
PY

echo "[3/4] Sin forcing => non-trivial omega"
./task2 --Nx 41 --Ny 41 --nu 0.02 --lid 0 --forcing sin --forcing-amp 1.0 --tol 1e-6 --max-iter 7000 --output-dir test_runs/case_forcing >/dev/null
"${PYTHON_CMD[@]}" - <<'PY'
import numpy as np
from pathlib import Path
w = np.loadtxt(Path('test_runs/case_forcing/omega.txt'))
mx = float(np.max(np.abs(w)))
assert mx > 1e-3, f'expected non-trivial omega, got max={mx}'
print(f'  ok: non-trivial omega max={mx:.3e}')
PY

echo "[4/4] Lid=1 + zero forcing => files + plot"
./task2 --Nx 51 --Ny 51 --nu 0.01 --lid 1 --forcing zero --tol 1e-5 --max-iter 12000 --output-dir test_runs/case_lid >/dev/null
"${PYTHON_CMD[@]}" ./plot_solution.py --input-dir test_runs/case_lid --output-dir test_runs/case_lid_plots --field speed --save --no-show >/dev/null

for f in psi.txt omega.txt u.txt v.txt g.txt meta.txt; do
  test -f "test_runs/case_lid/$f"
done
test -f test_runs/case_lid_plots/speed.png

echo "All task2 tests passed"
