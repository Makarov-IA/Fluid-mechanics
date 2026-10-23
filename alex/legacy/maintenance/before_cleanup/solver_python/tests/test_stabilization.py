"""Feedback term, steady system and exact Jacobian used by stability.py."""
import copy
from pathlib import Path

import numpy as np
import pytest

from solver_python import kernels as k
from solver_python.config import ConfigError, grid, load_config, parse_config
from solver_python.solver import Solver
from solver_python.stability import SteadyProblem, check_jacobian

CONFIG = Path(__file__).parents[1]/"config.yaml"


def config(tmp_path, **changes):
    raw = copy.deepcopy(load_config(CONFIG).resolved)
    raw.pop("stabilization", None)
    raw["physics"] = {"re": 260.}
    raw["forcing"] = {"type": "custom", "file": str(CONFIG.parent/"sources/harmonics.py")}
    raw["grid"] = dict(nx=13, ny=9, lx=1.3, ly=.6)
    raw["time"].update(mode="fixed_dt_steps", n_time_steps=8, dt=2e-4)
    raw["output"].update(root=str(tmp_path), save_every_step=3, log_every_step=3)
    raw.update(changes)
    return parse_config(raw, tmp_path/"input.yaml")


def stabilization_file(tmp_path, cfg, seed=0, modes=2):
    rng = np.random.default_rng(seed)
    xs, ys = grid(cfg)
    n = (cfg.nx-2)*(cfg.ny-2)
    basis = np.linalg.qr(rng.standard_normal((n, modes)))[0].T
    path = tmp_path/"stabilization.npz"
    np.savez(path, xs=xs, ys=ys, omega_star=rng.standard_normal((cfg.nx, cfg.ny)),
             psi_star=np.zeros((cfg.nx, cfg.ny)), basis=basis, re=cfg.re)
    return path, basis


def stabilized(tmp_path, gain, **changes):
    base = config(tmp_path, **changes)
    path, basis = stabilization_file(tmp_path, base)
    raw = copy.deepcopy(base.resolved)
    raw["stabilization"] = {"file": str(path), "gain": gain}
    return parse_config(raw, tmp_path/"input.yaml"), basis


def test_zero_gain_is_bitwise_the_uncontrolled_solver(tmp_path):
    a = Solver(config(tmp_path))
    b = Solver(stabilized(tmp_path, 0.0)[0])
    for _ in range(20):
        assert a.step() == b.step()
    assert np.array_equal(a.psi, b.psi) and np.array_equal(a.omega, b.omega)


def test_control_is_minus_gain_times_orthogonal_projection(tmp_path):
    cfg, basis = stabilized(tmp_path, 1.7)
    s = Solver(cfg)
    for _ in range(3):
        s.step()
    with np.load(cfg.stabilization["file"]) as data:
        omega_star = data["omega_star"][1:-1, 1:-1].ravel()
    deviation = s.omega[1:-1, 1:-1].ravel() - omega_star
    projection = basis.T@np.linalg.lstsq(basis.T, deviation, rcond=None)[0]
    expected = s.source[1:-1, 1:-1].ravel() - 1.7*projection
    np.testing.assert_allclose(s.stabilized_source()[1:-1, 1:-1].ravel(), expected, rtol=1e-12, atol=1e-12)


def test_control_vanishes_on_the_target_state(tmp_path):
    cfg, _ = stabilized(tmp_path, 1.0)
    s = Solver(cfg)
    with np.load(cfg.stabilization["file"]) as data:
        s.omega[:] = data["omega_star"]
    assert np.array_equal(s.stabilized_source(), s.source)
    assert s.last_control == 0.0


def test_residuals_ignore_the_control(tmp_path):
    cfg, _ = stabilized(tmp_path, 5.0)
    s, plain = Solver(cfg), Solver(config(tmp_path))
    rng = np.random.default_rng(3)
    s.psi[1:-1, 1:-1] = plain.psi[1:-1, 1:-1] = rng.standard_normal((cfg.nx-2, cfg.ny-2))
    s.omega[:] = plain.omega[:] = rng.standard_normal((cfg.nx, cfg.ny))
    assert s.compute_residuals() == plain.compute_residuals()


def test_mismatched_stabilization_file_is_rejected(tmp_path):
    cfg, _ = stabilized(tmp_path, 1.0)
    raw = copy.deepcopy(cfg.resolved)
    raw["grid"]["nx"] = 15
    with pytest.raises(ValueError, match="different grid"):
        Solver(parse_config(raw, tmp_path/"input.yaml"))
    raw = copy.deepcopy(cfg.resolved)
    raw["stabilization"]["file"] = str(tmp_path/"missing.npz")
    with pytest.raises(ConfigError):
        parse_config(raw, tmp_path/"input.yaml")


def test_steady_residual_is_what_the_solver_reports(tmp_path):
    walls = {"bottom": {"u": .2}, "top": {"u": .4}, "left": {"v": -.1}, "right": {"v": .3}}
    problem = SteadyProblem(config(tmp_path, boundaries=walls))
    z = np.random.default_rng(4).standard_normal(2*problem.n)
    assert float(np.max(np.abs(problem.residual(z)))) == max(problem.solver_residuals(z))


def test_assembled_jacobian_is_exact_with_moving_walls(tmp_path):
    walls = {"bottom": {"u": .2}, "top": {"u": .4}, "left": {"v": -.1}, "right": {"v": .3}}
    for nx, ny in ((5, 4), (13, 9), (16, 11)):
        problem = SteadyProblem(config(tmp_path, boundaries=walls, grid=dict(nx=nx, ny=ny, lx=1.3, ly=.6)))
        rng = np.random.default_rng(nx)
        z = rng.standard_normal(2*problem.n)
        assembled, fd = check_jacobian(problem, z, problem.jacobian(z), rng)
        assert assembled < 1e-13 and fd < 1e-8
        dense = np.column_stack([problem.tangent(z, e) for e in np.eye(2*problem.n)])
        np.testing.assert_allclose(problem.jacobian(z).toarray(), dense, rtol=0, atol=1e-10)
