"""Independent surface integration and inverse-design checks."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from solver_python.finite_magnets import MU0, cuboid_field, vorticity_source
from solver_python.magnet_design import (fit_sources, load_design, valid_positions,
    greedy_selection, fit_with_amplitude_limit, candidate_positions, PlacementProblem)


def test_cuboid_against_surface_quadrature():
    nodes, weights = leggauss(50)
    size = np.array([.01, .014, .005])
    center = np.array([.013, .021])
    x = np.array([.013, .016, -.012, .043])
    y = np.array([.021, .019, .038, .018])
    z, magnetization = .009, 800000.
    integral = np.zeros((3, len(x)))
    xx = x[:, None, None] - (center[0] + nodes[None, :, None]*size[0]/2)
    yy = y[:, None, None] - (center[1] + nodes[None, None, :]*size[1]/2)
    area = weights[:, None]*weights[None, :]*size[0]*size[1]/4
    for depth, sign in ((z, 1), (z+size[2], -1)):
        r2 = xx*xx+yy*yy+depth**2
        for k, kernel in enumerate((depth/r2**1.5, -3*depth*xx/r2**2.5,
                                    -3*depth*yy/r2**2.5)):
            integral[k] += sign*np.sum(kernel*area, axis=(1, 2))*MU0*magnetization/(4*np.pi)
    actual = cuboid_field(x, y, center, size, z, magnetization)
    np.testing.assert_allclose(actual, integral, atol=5e-13, rtol=2e-11)


def test_derivatives_symmetry_polarity_and_far_field():
    args = ([0., 0.], [.01, .01, .005], .008, 800000.)
    x, y, h = .003, -.004, 1e-7
    b, bx, by = cuboid_field(x, y, *args)
    dx = (cuboid_field(x+h, y, *args)[0]-cuboid_field(x-h, y, *args)[0])/(2*h)
    dy = (cuboid_field(x, y+h, *args)[0]-cuboid_field(x, y-h, *args)[0])/(2*h)
    np.testing.assert_allclose([bx, by], [dx, dy], rtol=1e-8)
    np.testing.assert_allclose(cuboid_field(-x, -y, *args), [b, -bx, -by], atol=1e-14)
    np.testing.assert_allclose(cuboid_field(x, y, *args[:-1], -args[-1]), [-b, -bx, -by])
    field = cuboid_field(0., 0., args[0], args[1], 1., args[3])[0]
    moment = args[3]*np.prod(args[1])
    expected = MU0*moment/(2*np.pi*(1.+args[1][2]/2)**3)
    assert field > 0
    assert field == pytest.approx(expected, rel=1e-4)


def test_source_sign_units_and_zero_current():
    np.testing.assert_allclose(vorticity_source(np.array([2.]), np.array([-3.]),
        [0., .1], 1000., .24, .001), [17.28])
    np.testing.assert_array_equal(vorticity_source(np.ones(4), np.ones(4),
        [0., 0.], 1000., .24, .001), np.zeros(4))


def test_fit_recovers_known_actuation_and_is_basis_invariant():
    rng = np.random.default_rng(72)
    g = rng.normal(size=(60, 2))*[.001, 200.]
    c = np.array([[3., -2.], [.4, .7]])
    target = g@c
    coefficients, fitted, error = fit_sources(g, target)
    np.testing.assert_allclose(coefficients, c, atol=1e-9)
    np.testing.assert_allclose(fitted, target, atol=1e-12)
    assert error < 1e-9
    target += rng.normal(size=target.shape)
    rotation = np.array([[.8, -.6], [.6, .8]])
    assert fit_sources(g, target)[2] == pytest.approx(fit_sources(g, target@rotation)[2], abs=1e-12)
    # A missing independent spatial direction cannot be concealed by a good first fit.
    q = np.linalg.qr(rng.normal(size=(60, 3)))[0]
    assert fit_sources(q[:, :1], q[:, :2])[2] == pytest.approx(1.)


def test_footprints_and_spacing():
    bounds = np.array([[.01, .23], [.01, .11]])
    assert valid_positions([[.05, .05], [.063, .05]], [.01, .01, .005], .002, bounds)
    assert not valid_positions([[.05, .05], [.059, .05]], [.01, .01, .005], .002, bounds)
    assert not valid_positions([[.05, .05], [.061, .05]], [.01, .01, .005], .002, bounds)
    assert not valid_positions([[.001, .05]], [.01, .01, .005], .002, bounds)


def test_invalid_geometry_and_duplicate_yaml(tmp_path):
    with pytest.raises(ValueError):
        cuboid_field(0, 0, [0, 0], [.01, .01, .005], 0, 1)
    path = tmp_path/"bad.yaml"
    path.write_text("mode_file: one\nmode_file: two\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_design(path)


def test_dictionary_search_refits_both_modes_and_matches_svd():
    rng = np.random.default_rng(75)
    sources = rng.normal(size=(100, 12))
    # Include two almost dependent candidates to exercise the Schur updates.
    sources[:, 1] = sources[:, 0] + 1e-4*rng.normal(size=100)
    target = sources[:, [4, 9]] @ np.array([[1., .3], [-.4, 1.]])
    selected, history = greedy_selection(sources, target, 8)
    assert len(set(selected)) == 8
    assert np.all(np.diff(history) <= 1e-10)
    for count, error in enumerate(history, 1):
        assert error == pytest.approx(fit_sources(sources[:, selected[:count]], target)[2], abs=3e-8)
    assert history[-1] < 3e-8
    with pytest.raises(ValueError, match="More magnets"):
        greedy_selection(sources, target, 13)


def test_amplitude_bound_for_all_modal_directions():
    rng = np.random.default_rng(16)
    g = rng.normal(size=(80, 7)) * np.logspace(-2, 1, 7)
    target = rng.normal(size=(80, 2))
    radius, gain, limit = 5., 2., .05
    c, fitted, worst, info = fit_with_amplitude_limit(g, target, radius, gain, limit)
    assert info['regularization'] > 0
    assert info['max_amplitude'] <= limit*(1+1e-12)
    angles = np.linspace(0, 2*np.pi, 1001)
    a = radius*np.vstack((np.cos(angles), np.sin(angles)))
    assert np.max(np.abs(gain*c@a)) <= limit*(1+1e-12)
    np.testing.assert_allclose(fitted, g@c)
    assert 0 <= worst <= 1+1e-12
    # Bound inactive: preserve the original least-squares fit exactly.
    loose = fit_with_amplitude_limit(g, target, radius, gain, 1e8)
    np.testing.assert_array_equal(loose[0], fit_sources(g, target)[0])


def test_candidate_lattice_obeys_footprints_and_extended_frame():
    from pathlib import Path
    root = Path(__file__).parents[2]
    _, cfg = load_design(root/'configs/magnet_placement_array/magnet_placement_array.yaml')
    problem = PlacementProblem(cfg, np.linspace(0, 1, 21), np.linspace(0, .5, 11),
                               np.zeros((19*9, 2)))
    sites = candidate_positions(problem)
    assert len(sites) == 231
    assert valid_positions(sites, cfg['magnets']['size_m'],
                           cfg['magnets']['spacing_m']-1e-12, problem.bounds)
    assert sites[:, 0].min() < 0
    assert sites[:, 0].max() > .24
