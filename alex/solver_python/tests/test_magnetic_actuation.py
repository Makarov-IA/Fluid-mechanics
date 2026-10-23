"""Additional magnetic RHS with an explicitly preserved modal reference problem."""
import copy

import numpy as np
import pytest

from solver_python.config import ConfigError, parse_config
from solver_python.solver import Solver
from solver_python.tests.test_math import dense_sweeps, jacobian, mixed
from solver_python.tests.test_state_projection import solver_pair


def actuator(current=.5):
    return dict(type='magnets', reference='base_forcing',
                magnets=[[.12, .04, .015, .005]], density_kg_m3=1000.,
                current_density_A_m2=[0., current], plane_height_m=0.,
                scales=dict(length_m=.24, velocity_m_s=260e-6/.24,
                            kinematic_viscosity_m2_s=1e-6))


def magnetic_solver(tmp_path, current=.5):
    _, original = solver_pair(tmp_path, gain=-1.)
    raw = copy.deepcopy(original.cfg.resolved)
    raw['magnetic_actuation'] = actuator(current)
    magnetic = Solver(parse_config(raw, tmp_path/'magnetic.yaml'))
    magnetic.psi[:] = original.psi
    magnetic.omega[:] = original.omega
    return original, magnetic


def test_zero_actuation_is_bitwise_identical_with_modal_addition(tmp_path):
    a, b = magnetic_solver(tmp_path, current=0.)
    for _ in range(30):
        assert a.step() == b.step()
    np.testing.assert_array_equal(a.psi, b.psi)
    np.testing.assert_array_equal(a.omega, b.omega)


def test_added_magnets_and_negative_gain_match_independent_dense_step(tmp_path):
    a, s = magnetic_solver(tmp_path)
    p, w = s.psi.copy(), s.omega.copy()
    dt = s.cfg.dt
    q = s.state_projection.basis
    z = np.r_[p[1:-1, 1:-1].ravel(), w[1:-1, 1:-1].ravel()]
    control = s.gain * q.T @ (q @ z)
    cp, cw = control[:77].reshape(11, 7), control[77:].reshape(11, 7)
    rhs = np.zeros_like(p)
    rhs[1:-1, 1:-1] = (p[1:-1, 1:-1] + dt*w[1:-1, 1:-1]
                         + dt**2*mixed(p, s.dx, s.dy) - dt*cp)
    p = dense_sweeps(rhs, p, dt, s.dx, s.dy)
    w[0, 1:-1] = -2*p[1, 1:-1]/s.dx**2
    w[-1, 1:-1] = -2*p[-2, 1:-1]/s.dx**2
    w[1:-1, 0] = -2*p[1:-1, 1]/s.dy**2
    w[1:-1, -1] = -2*p[1:-1, -2]/s.dy**2
    rhs[1:-1, 1:-1] = (w[1:-1, 1:-1] + dt*jacobian(p, w, s.dx, s.dy)
                         + (dt*s.nu)**2*mixed(w, s.dx, s.dy)
                         + dt*s.magnetic_source[1:-1, 1:-1] - dt*cw)
    w = dense_sweeps(rhs, w, dt*s.nu, s.dx, s.dy)
    s.step()
    a.step()
    np.testing.assert_array_equal(s.psi, a.psi)  # No direct magnetic input to psi.
    np.testing.assert_allclose(s.psi, p, atol=1e-16, rtol=1e-13)
    np.testing.assert_allclose(s.omega, w, atol=1e-14, rtol=1e-13)
    np.testing.assert_array_equal(s.omega[0], a.omega[0])
    np.testing.assert_array_equal(s.omega[:, 0], a.omega[:, 0])


def test_base_source_validation_is_not_bypassed(tmp_path):
    _, s = magnetic_solver(tmp_path)
    raw = copy.deepcopy(s.cfg.resolved)
    raw['forcing'] = {k: v for k, v in actuator().items() if k != 'reference'}
    with pytest.raises(ValueError, match='different source'):
        Solver(parse_config(raw, tmp_path/'wrong.yaml'))


def test_warmup_restores_actuation_and_zero_state(tmp_path):
    _, s = magnetic_solver(tmp_path)
    s.psi.fill(0)
    s.omega.fill(0)
    expected = s.forced_source.copy()
    s.warm_up()
    np.testing.assert_array_equal(s.forced_source, expected)
    assert not np.any(s.psi) and not np.any(s.omega)
    assert s.completed_steps == 0
    s.step()
    assert np.any(s.omega)


def test_explicit_reference_and_fixed_duration_required(tmp_path):
    _, s = magnetic_solver(tmp_path)
    raw = copy.deepcopy(s.cfg.resolved)
    del raw['magnetic_actuation']['reference']
    with pytest.raises(ConfigError, match='reference'):
        parse_config(raw, tmp_path/'wrong.yaml')
    raw['magnetic_actuation']['reference'] = 'base_forcing'
    raw['time']['mode'] = 'till_converges'
    with pytest.raises(ConfigError, match='fixed-duration'):
        parse_config(raw, tmp_path/'wrong.yaml')


def test_run_preserves_actuator_provenance(tmp_path):
    from solver_python.run import run
    _, s = magnetic_solver(tmp_path)
    raw = copy.deepcopy(s.cfg.resolved)
    raw['time']['n_time_steps'] = 3
    cfg = parse_config(raw, tmp_path/'small.yaml')
    summary = run(cfg)
    assert summary['magnetic_actuation_kind'].startswith('constant')
    assert len(summary['magnetic_actuation_sha256']) == 64
    with np.load(cfg.output_dir/'forcing.npz') as f:
        np.testing.assert_array_equal(f['source'], s.source)
    with np.load(cfg.output_dir/'magnetic_actuation.npz') as f:
        np.testing.assert_array_equal(f['source'], s.magnetic_source)
        np.testing.assert_array_equal(f['total_prescribed_source'], s.forced_source)
