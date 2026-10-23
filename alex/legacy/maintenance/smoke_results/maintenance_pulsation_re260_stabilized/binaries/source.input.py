"""Example custom source: the six harmonics of the original C++ forcing.

The return value is the nondimensional CURL of acceleration S, added with a
PLUS sign in omega_t = (1/Re)*laplacian(omega) + J_A(psi, omega) + S.
It is not f_x, Bz or a function requiring another derivative.
"""
import numpy as np


def source(x, y, lx, ly):
    """x is (nx, 1), y is (1, ny); return a scalar or broadcastable (nx, ny)."""
    result = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=np.float64)
    modes = [(2, 2, -19.0), (4, 2, -6.0), (6, 2, 7.0),
             (2, 4, 14.0), (1, 3, -0.57), (3, 1, -0.19)]
    for m, n, amplitude in modes:
        result += amplitude*np.sin(m*np.pi*x/lx)*np.sin(n*np.pi*y/ly)
    return result
