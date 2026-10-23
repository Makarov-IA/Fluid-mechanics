"""Independent mathematical checks, including the corrected intermediate boundary."""
import copy
from pathlib import Path

import numba
import numpy as np
import pytest

from common.snapshot_io import read_snapshot
from solver_python.config import load_config, parse_config
from solver_python.forcing import build_forcing, dipole_field
from solver_python import kernels as k
from solver_python.run import run
from solver_python.solver import Solver


CONFIG = Path(__file__).parents[1] / "config.yaml"


def config(tmp_path, **changes):
    raw = copy.deepcopy(load_config(CONFIG).resolved)
    raw.pop("stabilization", None)  # tests must not depend on the working config.yaml
    raw["physics"] = {"re": 260.}
    raw["forcing"] = {"type": "custom", "file": str(CONFIG.parent/"sources/harmonics.py")}
    raw["grid"] = dict(nx=13, ny=9, lx=1.3, ly=.6)
    # Pin the mode: tests must not depend on how the working config.yaml is set.
    raw["time"].update(mode="fixed_dt_steps", n_time_steps=8, dt=2e-4)
    raw["output"].update(root=str(tmp_path), save_every_step=3, log_every_step=3)
    raw.update(changes)
    return parse_config(raw, tmp_path / "input.yaml")


def lap(q, dx, dy):
    return ((q[:-2, 1:-1]-2*q[1:-1, 1:-1]+q[2:, 1:-1])/dx**2
            + (q[1:-1, :-2]-2*q[1:-1, 1:-1]+q[1:-1, 2:])/dy**2)


def mixed(q, dx, dy):
    dyy = (q[:, :-2]-2*q[:, 1:-1]+q[:, 2:])/dy**2
    return (dyy[:-2]-2*dyy[1:-1]+dyy[2:])/dx**2


def jacobian(p, w, dx, dy):
    # Independent NumPy expression, not a call into the production scalar kernel.
    j1 = ((p[2:, 1:-1]-p[:-2, 1:-1])*(w[1:-1, 2:]-w[1:-1, :-2])
          -(p[1:-1, 2:]-p[1:-1, :-2])*(w[2:, 1:-1]-w[:-2, 1:-1]))
    j2 = (p[2:, 1:-1]*(w[2:, 2:]-w[2:, :-2])-p[:-2, 1:-1]*(w[:-2, 2:]-w[:-2, :-2])
          -p[1:-1, 2:]*(w[2:, 2:]-w[:-2, 2:])+p[1:-1, :-2]*(w[2:, :-2]-w[:-2, :-2]))
    j3 = (w[1:-1, 2:]*(p[2:, 2:]-p[:-2, 2:])-w[1:-1, :-2]*(p[2:, :-2]-p[:-2, :-2])
          -w[2:, 1:-1]*(p[2:, 2:]-p[2:, :-2])+w[:-2, 1:-1]*(p[:-2, 2:]-p[:-2, :-2]))
    return (j1+j2+j3)/(12*dx*dy)


def dense_sweeps(rhs, old, a, dx, dy):
    """Reference solves use dense numpy.linalg, not Thomas or cached factors."""
    nx, ny = rhs.shape
    rx, ry = a/dx**2, a/dy**2
    ax = np.diag(np.full(nx-2, 1+2*rx))+np.diag(np.full(nx-3, -rx), 1)+np.diag(np.full(nx-3, -rx), -1)
    ay = np.diag(np.full(ny-2, 1+2*ry))+np.diag(np.full(ny-3, -ry), 1)+np.diag(np.full(ny-3, -ry), -1)
    first = rhs[1:-1, 1:-1].copy()
    first[0] += rx*(old[0, 1:-1]-ry*(old[0, :-2]-2*old[0, 1:-1]+old[0, 2:]))
    first[-1] += rx*(old[-1, 1:-1]-ry*(old[-1, :-2]-2*old[-1, 1:-1]+old[-1, 2:]))
    second = np.linalg.solve(ax, first)
    second[:, 0] += ry*old[1:-1, 0]
    second[:, -1] += ry*old[1:-1, -1]
    out = old.copy()
    out[1:-1, 1:-1] = np.linalg.solve(ay, second.T).T
    return out


@pytest.mark.parametrize("threads", [1, 2])
def test_full_step_against_dense_reference(tmp_path, threads):
    if threads > numba.config.NUMBA_NUM_THREADS:
        pytest.skip("Only one hardware thread available")
    cfg = config(tmp_path, execution={"threads": threads},
                 boundaries={"bottom": {"u": .2}, "top": {"u": .4},
                             "left": {"v": -.1}, "right": {"v": .3}})
    s = Solver(cfg)
    rng = np.random.default_rng(291)
    s.psi[1:-1, 1:-1] = rng.normal(0, .001, (cfg.nx-2, cfg.ny-2))
    s.omega[:] = rng.normal(0, .2, s.omega.shape)
    dt, dx, dy = cfg.dt, s.dx, s.dy
    p, w = s.psi.copy(), s.omega.copy()
    rhs = np.zeros_like(p)
    rhs[1:-1, 1:-1] = p[1:-1, 1:-1]+dt*w[1:-1, 1:-1]+dt**2*mixed(p, dx, dy)
    p = dense_sweeps(rhs, p, dt, dx, dy)
    # Independent Thom formulas from Taylor expansion on each wall.
    w[1:-1, 0] = -2*p[1:-1, 1]/dy**2 + 2*.2/dy
    w[1:-1, -1] = -2*p[1:-1, -2]/dy**2 - 2*.4/dy
    w[0, 1:-1] = -2*p[1, 1:-1]/dx**2 - 2*(-.1)/dx
    w[-1, 1:-1] = -2*p[-2, 1:-1]/dx**2 + 2*.3/dx
    w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 0
    rhs[1:-1, 1:-1] = (w[1:-1, 1:-1]+dt*(jacobian(p, w, dx, dy)+s.source[1:-1, 1:-1])
                       +(dt/cfg.re)**2*mixed(w, dx, dy))
    w = dense_sweeps(rhs, w, dt/cfg.re, dx, dy)
    s.step()
    np.testing.assert_allclose(s.psi, p, atol=1e-15, rtol=2e-14)
    np.testing.assert_allclose(s.omega, w, atol=1e-14, rtol=2e-14)
    expected = (np.max(np.abs(lap(p, dx, dy)+w[1:-1, 1:-1])),
                np.max(np.abs(lap(w, dx, dy)/cfg.re+jacobian(p, w, dx, dy)+s.source[1:-1, 1:-1])))
    np.testing.assert_allclose(s.last_residuals, expected, rtol=1e-13)


@pytest.mark.parametrize("threads", [1, 2])
def test_exact_discrete_stationary_state_is_preserved(tmp_path, threads):
    if threads > numba.config.NUMBA_NUM_THREADS:
        pytest.skip("Only one hardware thread available")
    s = Solver(config(tmp_path, execution={"threads": threads}))
    x, y = s.xs[:, None], s.ys[None, :]
    s.psi[1:-1, 1:-1] = (.03*np.sin(np.pi*x/s.cfg.lx)**2*np.sin(np.pi*y/s.cfg.ly)**2)[1:-1, 1:-1]
    s.omega[1:-1, 1:-1] = -lap(s.psi, s.dx, s.dy)
    s.apply_thom_boundary()
    s.source[1:-1, 1:-1] = -lap(s.omega, s.dx, s.dy)/s.cfg.re-jacobian(s.psi, s.omega, s.dx, s.dy)
    assert max(s.compute_residuals()) < 1e-12
    p, w = s.psi.copy(), s.omega.copy()
    for _ in range(5):
        s.step()
    np.testing.assert_allclose(s.psi, p, atol=1e-15, rtol=0)
    np.testing.assert_allclose(s.omega, w, atol=3e-14, rtol=0)


def test_arakawa_sign_and_periodic_invariants():
    xs, ys = np.arange(9)*.1, np.arange(8)*.2
    p, w = np.broadcast_arrays(xs[:, None], ys[None, :])
    assert k.arakawa_at(p.copy(), w.copy(), 3, 3, 1/(4*.1*.2)) == pytest.approx(1)
    rng = np.random.default_rng(3)
    p0, w0 = rng.normal(size=(12, 10)), rng.normal(size=(12, 10))
    p, w = np.pad(p0, 1, mode="wrap"), np.pad(w0, 1, mode="wrap")
    j = np.array([[k.arakawa_at(p, w, i+1, m+1, .25) for m in range(10)] for i in range(12)])
    assert abs(j.sum()) < 1e-13
    assert abs(np.sum(p0*j)) < 1e-13  # discrete energy conservation
    assert abs(np.sum(w0*j)) < 1e-13  # discrete enstrophy conservation


def test_forcing_sign_matches_curl_of_cpp_force(tmp_path):
    cfg = config(tmp_path)
    s = Solver(cfg)
    x, y = s.xs[:, None], s.ys[None, :]
    def fx(y):
        a22 = -19*cfg.ly/(2*np.pi)
        return sum(a*np.sin(m*np.pi*x/cfg.lx)*np.cos(n*np.pi*y/cfg.ly)
                   for m, n, a in ((2, 2, a22), (4, 2, -6*cfg.ly/(2*np.pi)),
                                   (6, 2, 7*cfg.ly/(2*np.pi)), (2, 4, 14*cfg.ly/(4*np.pi)),
                                   (1, 3, a22/50), (3, 1, a22/50)))
    eps = 1e-6
    numerical_curl = -(fx(y+eps)-fx(y-eps))/(2*eps)
    np.testing.assert_allclose(s.source, numerical_curl, atol=5e-9, rtol=1e-8)


def test_dipoles_derivatives_superposition_and_polarity():
    magnets = [[.02, .03, .015, .006], [.1, .08, .02, -.004]]
    x, y = np.array([[.02], [.04], [.12]]), np.array([[.03, .06]])
    bz, dbx, dby = dipole_field(x, y, magnets, .003)
    assert dipole_field(.02, .03, [magnets[0]], .003)[0] == pytest.approx(.006)
    eps = 1e-7
    dfx = (dipole_field(x+eps, y, magnets, .003)[0]-dipole_field(x-eps, y, magnets, .003)[0])/(2*eps)
    dfy = (dipole_field(x, y+eps, magnets, .003)[0]-dipole_field(x, y-eps, magnets, .003)[0])/(2*eps)
    np.testing.assert_allclose(dbx, dfx, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(dby, dfy, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(bz, sum(dipole_field(x, y, [m], .003)[0] for m in magnets))
    reversed_magnets = [m[:3]+[-m[3]] for m in magnets]
    for original, reverse in zip((bz, dbx, dby), dipole_field(x, y, reversed_magnets, .003)):
        np.testing.assert_allclose(original, -reverse)


def test_nan_is_not_convergence(tmp_path):
    for field in ("psi", "omega", "source"):
        s = Solver(config(tmp_path))
        getattr(s, field)[0, 0] = np.nan
        with pytest.raises(FloatingPointError):
            s.compute_residuals()


def test_serial_parallel_same_trajectory(tmp_path):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("Only one hardware thread available")
    a = Solver(config(tmp_path, execution={"threads": 1}))
    b = Solver(config(tmp_path, execution={"threads": 2}))
    for _ in range(40):
        a.step()
        b.step()
    np.testing.assert_array_equal(a.psi, b.psi)
    np.testing.assert_array_equal(a.omega, b.omega)


def test_minimum_grid_and_fractional_re(tmp_path):
    s = Solver(config(tmp_path, grid=dict(nx=3, ny=3, lx=1., ly=.5), physics={"re": .5}))
    s.source.fill(1.)
    p, w = s.psi.copy(), s.omega.copy()
    rhs = np.full_like(p, s.cfg.dt)
    expected = dense_sweeps(rhs, w, s.cfg.dt/s.cfg.re, s.dx, s.dy)
    s.step()
    np.testing.assert_allclose(s.omega, expected)


def test_binary_output_and_time_index(tmp_path):
    cfg = config(tmp_path)
    summary = run(cfg)
    assert summary["status"] == "completed"
    out = cfg.output_dir
    assert sorted(p.name for p in out.glob("result_*.bin")) == ["result_0.bin", "result_3.bin", "result_6.bin", "result_7.bin"]
    first, last = read_snapshot(out/"result_0.bin"), read_snapshot(out/"result_7.bin")
    assert first.time == pytest.approx(cfg.dt)
    assert last.time == pytest.approx(8*cfg.dt)
    assert last.re == cfg.re and last.step == 7
    np.testing.assert_allclose(last.u[1:-1, 1:-1], (last.psi[1:-1, 2:]-last.psi[1:-1, :-2])/(2*(cfg.ly/(cfg.ny-1))))
    assert last.psi.shape == (cfg.nx, cfg.ny)

    # Re-running replaces the directory; a longer previous run leaves nothing behind.
    (out/"result_999.bin").write_bytes(b"stale")
    assert run(cfg)["status"] == "completed"
    assert not (out/"result_999.bin").exists()
    assert sorted(p.name for p in out.glob("result_*.bin")) == ["result_0.bin", "result_3.bin", "result_6.bin", "result_7.bin"]

    # A directory that is not ours is never erased.
    foreign = out.parent/"foreign"
    foreign.mkdir()
    (foreign/"notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        run(cfg, output_dir=foreign)
    assert (foreign/"notes.txt").read_text() == "keep me"


def test_convergence_stops_and_step_limit_is_honest(tmp_path):
    cfg = config(tmp_path, forcing={"type": "none"})
    raw = copy.deepcopy(cfg.resolved)
    raw["time"]["mode"] = "till_converges"
    zero = parse_config(raw, cfg.path)
    assert run(zero)["completed_steps"] == 1
    raw["experiment"] = "limit"
    raw["forcing"] = config(tmp_path).forcing
    forced = parse_config(raw, cfg.path)
    assert run(forced)["status"] == "step_limit"
