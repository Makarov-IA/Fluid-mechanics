"""Checks on the actual 201 x 101 grid and saved unstable modes."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from solver_python.config import load_config
from solver_python.solver import Solver


report = {}
for method in ('projection', 'rhs'):
    name = 'pulsation_re260_' + method
    cfg = load_config(ROOT / 'configs' / name / (name + '.yaml'))
    s = Solver(cfg)
    s.warm_up()
    with np.load(cfg.stabilization['file']) as data:
        p, w = data['psi_star'], data['omega_star']
    s.psi[:] = p
    s.omega[:] = w
    for _ in range(20):
        s.step()
    drift = [float(np.max(np.abs(s.psi-p))), float(np.max(np.abs(s.omega-w)))]
    assert max(drift) < 1e-10, drift
    equilibrium_residuals = s.last_residuals
    s = Solver(cfg)
    s.warm_up()
    started = time.perf_counter()
    for _ in range(10000):
        s.step()
    _, _, coefficients = s.state_projection.correction(s.psi, s.omega)
    if method == 'projection':
        assert np.linalg.norm(coefficients) < 1e-10
    report[method] = dict(equilibrium_20_step_drift=drift,
                          equilibrium_residuals=equilibrium_residuals,
                          zero_start_steps=s.completed_steps,
                          residuals=s.last_residuals,
                          modal_norm=float(np.linalg.norm(coefficients)),
                          seconds=time.perf_counter()-started)
    print(method, report[method], flush=True)
(ROOT/'legacy/maintenance/state_control_checks.json').write_text(json.dumps(report, indent=2)+'\n')
