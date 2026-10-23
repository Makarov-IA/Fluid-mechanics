import numpy as np

from solver_python.corner_vortices import corner_measurements
from solver_python.snapshot_io import BoundaryConditions, Snapshot, velocities_from_psi


def state(psi):
    xs, ys = np.linspace(0, 1, 201), np.linspace(0, .5, 101)
    bc = BoundaryConditions()
    p = psi(xs[:, None], ys[None, :])
    p[[0, -1], :] = 0
    p[:, [0, -1]] = 0
    w = np.zeros_like(p)
    w[1:-1, 1:-1] = -(p[:-2, 1:-1]+p[2:, 1:-1]+p[1:-1, :-2]
                        + p[1:-1, 2:]-4*p[1:-1, 1:-1])/.005**2
    u, v = velocities_from_psi(p, xs, ys, bc)
    return Snapshot(xs, ys, p, w, u, v, 0, 0., 260., bc)


def test_detects_four_small_positive_corner_cores():
    centers = [(x, y) for y in (.05, .45) for x in (.075, .925)]
    def field(x, y):
        p = -.001*np.sin(np.pi*x)*np.sin(2*np.pi*y)
        for cx, cy in centers:
            p += .01*np.exp(-((x-cx)/.025)**2-((y-cy)/.02)**2)
        return p
    rows = corner_measurements(state(field))
    assert all(r['detected'] and r['localized_core'] for r in rows)
    assert all(r['core_circulation'] > 0 for r in rows)
    for row, (cx, cy) in zip(rows, centers):
        assert abs(row['x']-cx) <= .005 and abs(row['y']-cy) <= .005


def test_central_vortex_and_zero_state_are_not_corner_detections():
    for field in (lambda x, y: np.zeros((len(x), y.shape[1])),
                  lambda x, y: np.sin(np.pi*x)*np.sin(2*np.pi*y)):
        rows = corner_measurements(state(field))
        assert not any(r['detected'] for r in rows)


def test_broad_core_is_not_labeled_localized():
    rows = corner_measurements(state(
        lambda x, y: np.exp(-((x-.075)/.3)**2-((y-.05)/.2)**2)))
    assert rows[0]['detected']
    assert not rows[0]['localized_core']
