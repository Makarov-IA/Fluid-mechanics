"""Full-state modes derived from the existing physical unstable omega subspace."""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def full_state_basis(omega_basis, nx, ny, dx, dy):
    """Rows span (phi, eta), with -lap_h phi = eta and homogeneous psi walls.

    This Poisson solve constructs fixed modes only; the timestep algorithm is unchanged.
    Euclidean normalization is over the concatenated interior (psi, omega) vector.
    """
    ni, nj = nx-2, ny-2
    lx = diags([np.ones(ni-1), -2*np.ones(ni), np.ones(ni-1)], [-1, 0, 1])/dx**2
    ly = diags([np.ones(nj-1), -2*np.ones(nj), np.ones(nj-1)], [-1, 0, 1])/dy**2
    lap = (kron(lx, eye(nj)) + kron(eye(ni), ly)).tocsc()
    eta = np.asarray(omega_basis).T
    phi = splu(lap).solve(-eta)
    q, r = np.linalg.qr(np.vstack((phi, eta)))
    if np.min(np.abs(np.diag(r))) < 1e-12*np.max(np.abs(np.diag(r))):
        raise ValueError("Dependent full-state modes")
    return np.ascontiguousarray(q.T)


class StateProjection:
    def __init__(self, psi_star, omega_star, basis):
        self.shape = (psi_star.shape[0]-2, psi_star.shape[1]-2)
        self.n = self.shape[0]*self.shape[1]
        self.target = np.concatenate((psi_star[1:-1, 1:-1].ravel(),
                                      omega_star[1:-1, 1:-1].ravel()))
        self.basis = np.ascontiguousarray(basis)
        if (self.basis.ndim != 2 or self.basis.shape[1] != 2*self.n
                or self.basis.shape[0] == 0 or not np.isfinite(self.basis).all()
                or not np.allclose(self.basis@self.basis.T, np.eye(self.basis.shape[0]),
                                   rtol=0, atol=1e-12)):
            raise ValueError("Invalid full-state projection basis")

    def correction(self, psi, omega):
        state = np.concatenate((psi[1:-1, 1:-1].ravel(), omega[1:-1, 1:-1].ravel()))
        coefficients = self.basis@(state-self.target)
        correction = self.basis.T@coefficients
        return (correction[:self.n].reshape(self.shape),
                correction[self.n:].reshape(self.shape), coefficients)
