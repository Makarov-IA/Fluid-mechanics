import copy
from pathlib import Path

import numpy as np
import pytest

from solver_python.config import load_config, parse_config, ConfigError
from solver_python.projection import full_state_basis
from solver_python.solver import Solver
from solver_python.tests.test_math import dense_sweeps, jacobian, lap, mixed


def solver_pair(tmp_path, method='rhs', gain=1.):
    root = Path(__file__).parents[2]
    raw = copy.deepcopy(load_config(root/'configs/pulsation_re260/pulsation_re260.yaml').resolved)
    raw['grid'] = dict(nx=13, ny=9, lx=1.3, ly=.6)
    raw['forcing'] = dict(type='none')
    raw['time'].update(dt=2e-4, n_time_steps=30)
    plain = parse_config(raw, tmp_path/'pair.yaml')
    a = Solver(plain)
    rng = np.random.default_rng(43)
    basis = np.linalg.qr(rng.normal(size=(77, 2)))[0].T
    state_basis = full_state_basis(basis, 13, 9, a.dx, a.dy)
    bundle = tmp_path/'modes.npz'
    np.savez(bundle, xs=a.xs, ys=a.ys, re=a.cfg.re, basis=basis,
             state_basis=state_basis, psi_star=np.zeros_like(a.psi), omega_star=np.zeros_like(a.omega),
             source=a.source, wall_velocities=np.zeros((4, 2)))
    raw['stabilization'] = dict(file=str(bundle), gain=gain, method=method, components='state')
    b = Solver(parse_config(raw, tmp_path/'pair.yaml'))
    for s in (a,b):
        s.psi[1:-1, 1:-1] = rng.normal(0, .001, (11, 7))
        s.omega[1:-1, 1:-1] = rng.normal(0, .2, (11, 7))
        s.apply_thom_boundary()
    b.psi[:] = a.psi
    b.omega[:] = a.omega
    return a,b


def test_pair_basis_satisfies_poisson_and_orthonormality(tmp_path):
    _,s = solver_pair(tmp_path)
    q=s.state_projection.basis
    np.testing.assert_allclose(q@q.T,np.eye(2),atol=1e-14)
    for row in q:
        phi=np.zeros_like(s.psi); phi[1:-1,1:-1]=row[:77].reshape(11,7)
        # Differentiation amplifies the QR/Poisson roundoff by inverse mesh spacing squared.
        tolerance = 32*np.finfo(float).eps*(1/s.dx**2+1/s.dy**2)*np.linalg.norm(row)
        np.testing.assert_allclose(lap(phi,s.dx,s.dy)+row[77:].reshape(11,7),0,atol=tolerance)


def test_full_projection_removes_modes_preserves_poisson_defect_and_walls(tmp_path):
    _,s=solver_pair(tmp_path,'projection')
    defect=lap(s.psi,s.dx,s.dy)+s.omega[1:-1,1:-1]
    s.project_state()
    _,_,c=s.state_projection.correction(s.psi,s.omega)
    np.testing.assert_allclose(c,0,atol=1e-15)
    np.testing.assert_allclose(lap(s.psi,s.dx,s.dy)+s.omega[1:-1,1:-1],defect,atol=1e-14)
    p,w=s.psi.copy(),s.omega.copy()
    s.apply_thom_boundary()
    np.testing.assert_array_equal(s.omega,w)
    s.project_state()
    np.testing.assert_allclose(s.psi,p,atol=1e-17)
    np.testing.assert_allclose(s.omega,w,atol=1e-15)


def test_projection_is_after_step_and_next_step_uses_corrected_state(tmp_path):
    a,b=solver_pair(tmp_path,'projection')
    q=b.state_projection.basis
    for _ in range(3):
        a.step()
        z=np.concatenate((a.psi[1:-1,1:-1].ravel(),a.omega[1:-1,1:-1].ravel()))
        z-=q.T@(q@z)
        a.psi[1:-1,1:-1]=z[:77].reshape(11,7)
        a.omega[1:-1,1:-1]=z[77:].reshape(11,7)
        a.apply_thom_boundary()
        b.step()
        np.testing.assert_allclose(b.psi,a.psi,atol=1e-16,rtol=1e-14)
        np.testing.assert_allclose(b.omega,a.omega,atol=1e-14,rtol=1e-14)
        np.testing.assert_allclose(b.last_residuals,a.compute_residuals(),atol=1e-12)


@pytest.mark.parametrize('gain', [1.7, -1.0])
def test_both_rhs_against_dense_reference_one_frozen_projection_dt_once(tmp_path, gain):
    _,s=solver_pair(tmp_path,'rhs',gain=gain)
    p,w=s.psi.copy(),s.omega.copy();dt=s.cfg.dt
    q=s.state_projection.basis
    z=np.concatenate((p[1:-1,1:-1].ravel(),w[1:-1,1:-1].ravel()))
    control=gain*q.T@(q@z)
    cp,cw=control[:77].reshape(11,7),control[77:].reshape(11,7)
    rhs=np.zeros_like(p)
    rhs[1:-1,1:-1]=p[1:-1,1:-1]+dt*w[1:-1,1:-1]+dt**2*mixed(p,s.dx,s.dy)-dt*cp
    p=dense_sweeps(rhs,p,dt,s.dx,s.dy)
    w[0,1:-1]=-2*p[1,1:-1]/s.dx**2;w[-1,1:-1]=-2*p[-2,1:-1]/s.dx**2
    w[1:-1,0]=-2*p[1:-1,1]/s.dy**2;w[1:-1,-1]=-2*p[1:-1,-2]/s.dy**2
    rhs[1:-1,1:-1]=w[1:-1,1:-1]+dt*jacobian(p,w,s.dx,s.dy)+(dt*s.nu)**2*mixed(w,s.dx,s.dy)-dt*cw
    w=dense_sweeps(rhs,w,dt*s.nu,s.dx,s.dy)
    s.step()
    np.testing.assert_allclose(s.psi,p,atol=1e-16,rtol=1e-13)
    np.testing.assert_allclose(s.omega,w,atol=1e-14,rtol=1e-13)
    np.testing.assert_allclose(s.control_psi,cp,atol=1e-17)
    np.testing.assert_allclose(s.control_omega,cw,atol=1e-15)


def test_full_rhs_zero_gain_matches_ordinary_steps(tmp_path):
    a,b=solver_pair(tmp_path,'rhs',gain=0)
    for _ in range(30):
        assert a.step()==b.step()
    np.testing.assert_array_equal(a.psi,b.psi)
    np.testing.assert_array_equal(a.omega,b.omega)


@pytest.mark.parametrize('method',['rhs','projection'])
def test_zero_equilibrium_stays_fixed_and_warmup_resets(tmp_path,method):
    _,s=solver_pair(tmp_path,method)
    s.psi.fill(0);s.omega.fill(0)
    s.warm_up()
    assert s.completed_steps==0 and s.last_control==0 and s.last_control_psi==0
    for _ in range(3):
        assert s.step()==(0.,0.)
    assert not np.any(s.psi) and not np.any(s.omega)


@pytest.mark.parametrize('gain', [.5, -1.0])
def test_full_projection_rejects_partial_gain(tmp_path, gain):
    with pytest.raises(ConfigError,match='gain: 1'):
        solver_pair(tmp_path,'projection',gain=gain)


@pytest.mark.parametrize('method', [None, 'projection', 'rhs'])
def test_yaml_resume_is_bitwise_equal_to_uninterrupted_steps(tmp_path, method):
    from solver_python.run import run
    from solver_python.snapshot_io import read_snapshot
    a, b = solver_pair(tmp_path, method or 'rhs')
    original = a.cfg if method is None else b.cfg
    raw = copy.deepcopy(original.resolved)
    # Nonzero forcing exercises a nontrivial continued trajectory even for a zero target.
    raw['forcing'] = dict(type='custom', file=str(Path(__file__).parents[1]/'sources/harmonics.py')) if method is None else raw['forcing']
    if method is not None:
        # Start from a nonzero pair while keeping the validated zero-source bundle.
        from solver_python.snapshot_io import write_snapshot
        initial = tmp_path/'initial.bin'
        write_snapshot(initial, b.xs, b.ys, b.psi, b.omega, step=0, time=0., re=b.cfg.re, bc=b.cfg.bc)
        raw['resume'] = dict(psi=str(initial), omega=str(initial))
    raw['time']['n_time_steps'] = 8
    prefix = parse_config(raw, tmp_path/'prefix.yaml')
    run(prefix)
    raw['time']['n_time_steps'] = 19
    continuous = parse_config(raw, tmp_path/'continuous.yaml')
    expected = run(continuous)
    saved = str(prefix.output_dir/'result_7.bin')
    raw['resume'] = dict(psi=saved, omega=saved)
    raw['time']['n_time_steps'] = 11
    resumed = parse_config(raw, tmp_path/'resumed.yaml')
    actual = run(resumed)
    x = read_snapshot(continuous.output_dir/'result_18.bin')
    y = read_snapshot(resumed.output_dir/'result_10.bin')
    np.testing.assert_array_equal(x.psi, y.psi)
    np.testing.assert_array_equal(x.omega, y.omega)
    assert (actual['psi_res'], actual['omega_res']) == (expected['psi_res'], expected['omega_res'])
