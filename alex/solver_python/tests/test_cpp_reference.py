"""Compare with the user's actual C++ source, fixed ONLY in a temporary copy."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from solver_python.snapshot_io import read_snapshot
from solver_python.config import parse_config
from solver_python.solver import Solver


def test_matches_cpp_pipeline_with_only_documented_fixes(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is not available")
    root = Path(__file__).resolve().parents[2]
    source = (root/"legacy/cpu/solver/solver.cpp").read_text()
    # Compile an isolated reference; archived relative Eigen paths predate the move.
    header = (root/"legacy/cpu/solver/solver.hpp").read_text()
    header = header.replace('../../../External_libs/Eigen/Dense', str(root.parent/'External_libs/Eigen/Dense'))
    header = header.replace('../utils/config.hpp', str(root/'legacy/cpu/utils/config.hpp'))
    (tmp_path/'solver.hpp').write_text(header)
    utils = (root/'legacy/cpu/solver/solver_utils.hpp').read_text()
    utils = utils.replace('../../External_libs/Eigen/Dense', str(root.parent/'External_libs/Eigen/Dense'))
    (tmp_path/'solver_utils.hpp').write_text(utils)
    replacements = {
        '#include "solver.hpp"': f'#include "{tmp_path}/solver.hpp"',
        '#include "../forcing/omega_forcing.hpp"': f'#include "{root}/legacy/cpu/forcing/omega_forcing.hpp"',
        "- dt_ * jacobian(i, j) - dt_ * OmegaForcing": "+ dt_ * jacobian(i, j) + dt_ * OmegaForcing",
        "tmp -= jacobian(i, j);": "tmp += jacobian(i, j);",
        "tmp -= OmegaForcing(": "tmp += OmegaForcing(",
        "- 2.0 * cfg_.bc.bottom.u / dy_;": "+ 2.0 * cfg_.bc.bottom.u / dy_;",
        "- 2.0 * cfg_.bc.right.v / dx_;": "+ 2.0 * cfg_.bc.right.v / dx_;",
        "d(0) = omega_(0, j);": "d(0) = omega_(0, j) - dt_/Re_ * (omega_(0,j-1)-2*omega_(0,j)+omega_(0,j+1))/(dy_*dy_);",
        "d(nx_ - 1) = omega_(nx_ - 1, j);": "d(nx_ - 1) = omega_(nx_-1,j) - dt_/Re_ * (omega_(nx_-1,j-1)-2*omega_(nx_-1,j)+omega_(nx_-1,j+1))/(dy_*dy_);",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, f"C++ reference changed: {old}"
        source = source.replace(old, new)
    cpp = tmp_path/"solver_fixed.cpp"
    cpp.write_text(source)
    driver = tmp_path/"driver.cpp"
    driver.write_text(f'''#include "{tmp_path}/solver.hpp"
int main(int argc, char** argv) {{
    Config c; c.nx=13; c.ny=9; c.lx=1.3; c.ly=.6; c.Re=260;
    c.dt=.0002; c.n_time_steps=50; c.mode="fixed_dt_steps"; c.use_arakawa=true;
    c.bc.bottom.u=.2; c.bc.top.u=.4; c.bc.left.v=-.1; c.bc.right.v=.3;
    c.save_every_step=49; c.save_dir=argv[1]; c.log_info.enabled=false;
    Solver s(c); s.solve();
}}
''')
    exe = tmp_path/"reference"
    subprocess.run([compiler, "-std=c++17", "-O2", f"-I{root}/legacy/cpu", str(cpp), str(driver),
                    str(root/"legacy/cpu/forcing/omega_forcing.cpp"), "-o", str(exe)],
                   check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([str(exe), str(tmp_path/"cpp_output")], check=True, capture_output=True, timeout=30)
    expected = read_snapshot(tmp_path/"cpp_output/result_49.bin")
    cfg = parse_config({
        "version": 1,
        "grid": dict(nx=13, ny=9, lx=1.3, ly=.6), "physics": {"re": 260},
        "time": dict(mode="fixed_dt_steps", dt=.0002, n_time_steps=50),
        "boundaries": {"bottom": {"u": .2}, "top": {"u": .4},
                       "left": {"v": -.1}, "right": {"v": .3}},
        "forcing": {"type": "custom", "file": str(root/"solver_python/sources/harmonics.py")}}, tmp_path/"config.yaml")
    solver = Solver(cfg)
    for _ in range(50):
        solver.step()
    np.testing.assert_allclose(solver.psi, expected.psi, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(solver.omega, expected.omega, rtol=2e-12, atol=1e-12)
