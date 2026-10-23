"""C++ pipeline: solvePsi -> ApplyThomBoundary -> solveOmega -> computeResiduals.

This is pseudo-time relaxation, not a physical unsteady Navier-Stokes integrator.
The Arakawa term is explicit; only diffusion is in the omega sweeps, as in the
current C++ implementation. Factorization corrections are retained in both RHSs.
"""
from __future__ import annotations

import math

import numba
import numpy as np

from . import kernels as k
from .config import Config, grid
from .forcing import build_forcing


class Solver:
    def __init__(self, cfg: Config):
        if cfg.threads > numba.config.NUMBA_NUM_THREADS:
            raise ValueError(f"execution.threads exceeds available Numba threads ({numba.config.NUMBA_NUM_THREADS})")
        self.cfg = cfg
        self.xs, self.ys = grid(cfg)
        self.dx, self.dy = cfg.lx/(cfg.nx-1), cfg.ly/(cfg.ny-1)
        self.nu = 1.0/cfg.re
        self.idx2, self.idy2 = 1/self.dx**2, 1/self.dy**2
        self.inv_4dxdy = 1/(4*self.dx*self.dy)
        self.psi = np.zeros((cfg.nx, cfg.ny), dtype=np.float64)
        self.omega = np.zeros_like(self.psi)
        self.rhs, self.work = np.zeros_like(self.psi), np.zeros_like(self.psi)
        self.zero_x, self.zero_y = np.zeros(cfg.nx), np.zeros(cfg.ny)
        self.left_g, self.right_g = np.zeros(cfg.ny), np.zeros(cfg.ny)
        self.source, self.bz = build_forcing(cfg, self.xs, self.ys)
        self.coefficients = tuple(cfg.dt*diffusion*spacing for diffusion in (1., self.nu)
                                  for spacing in (self.idx2, self.idy2))
        if not all(math.isfinite(v) and v > 0 for v in self.coefficients):
            raise ValueError("Non-finite/zero sweep coefficient; check grid, Re and dt")
        self.factors = [k.thomas_factors(size, coefficient)
                        for size, coefficient in zip((cfg.nx-2, cfg.ny-2)*2, self.coefficients)]
        self.sweep_x = k.sweep_x_parallel if cfg.threads > 1 else k.sweep_x_serial
        self.sweep_y = k.sweep_y_parallel if cfg.threads > 1 else k.sweep_y_serial
        self.completed_steps = 0
        self.last_residuals = (math.inf, math.inf)
        self.stabilized = cfg.stabilization is not None
        self.last_control = 0.0
        if self.stabilized:
            self._load_stabilization(cfg.stabilization)

    def _load_stabilization(self, settings):
        """-gain*P(omega - omega*) with P = B^T B, B the orthonormal rows written by stability.py."""
        with np.load(settings["file"]) as data:
            xs, ys, re = data["xs"], data["ys"], float(data["re"])
            omega_star, basis = data["omega_star"], data["basis"]
        if not (np.array_equal(xs, self.xs) and np.array_equal(ys, self.ys)):
            raise ValueError("Stabilization file was computed on a different grid")
        if re != self.cfg.re:
            raise ValueError(f"Stabilization file was computed for Re={re:g}, config has Re={self.cfg.re:g}")
        n = (self.cfg.nx-2)*(self.cfg.ny-2)
        if omega_star.shape != self.psi.shape or basis.ndim != 2 or basis.shape[1] != n:
            raise ValueError("Stabilization file has inconsistent array shapes")
        if not np.allclose(basis@basis.T, np.eye(basis.shape[0]), rtol=0, atol=1e-12):
            raise ValueError("Stabilization basis rows are not orthonormal")
        self.gain = settings["gain"]
        self.basis = np.ascontiguousarray(basis)
        self.omega_star = np.ascontiguousarray(omega_star[1:-1, 1:-1]).ravel()
        self.controlled_source = self.source.copy()

    def stabilized_source(self):
        """S - gain*P(omega - omega*) on the interior, at the same time level as J_A."""
        deviation = self.omega[1:-1, 1:-1].ravel() - self.omega_star
        control = self.gain*(self.basis.T@(self.basis@deviation))
        self.controlled_source[1:-1, 1:-1] = self.source[1:-1, 1:-1] - control.reshape(self.cfg.nx-2, self.cfg.ny-2)
        self.last_control = float(np.max(np.abs(control)))
        return self.controlled_source

    def solve_psi(self):
        k.psi_rhs(self.psi, self.omega, self.cfg.dt, self.idx2, self.idy2, self.rhs)
        self.sweep_x(self.rhs, self.work, self.zero_y, self.zero_y, self.coefficients[0], *self.factors[0])
        self.sweep_y(self.work, self.psi, self.zero_x, self.zero_x, self.coefficients[1], *self.factors[1])

    def apply_thom_boundary(self):
        bc = self.cfg.bc
        k.thom_boundary(self.psi, self.omega, self.dx, self.dy,
                        bc.bottom.u, bc.top.u, bc.left.v, bc.right.v)

    def solve_omega(self):
        # The control enters the right-hand side only; residuals keep the original S,
        # so convergence still means a steady state of the uncontrolled equations.
        source = self.stabilized_source() if self.stabilized else self.source
        k.omega_rhs(self.psi, self.omega, source, self.cfg.dt, self.nu,
                    self.idx2, self.idy2, self.inv_4dxdy, self.rhs)
        # g = (I - dt*nu*Dyy) omega_new. Its side values are NOT omega_wall.
        ay = self.coefficients[3]
        for edge, target in ((self.omega[0, :], self.left_g), (self.omega[-1, :], self.right_g)):
            target[1:-1] = edge[1:-1] - ay*(edge[:-2]-2*edge[1:-1]+edge[2:])
        self.sweep_x(self.rhs, self.work, self.left_g, self.right_g, self.coefficients[2], *self.factors[2])
        self.sweep_y(self.work, self.omega, self.omega[:, 0], self.omega[:, -1],
                     self.coefficients[3], *self.factors[3])

    def compute_residuals(self):
        values = k.residuals(self.psi, self.omega, self.source, self.nu,
                             self.idx2, self.idy2, self.inv_4dxdy)
        if not all(math.isfinite(v) for v in values):
            raise FloatingPointError("Non-finite fields/residuals; convergence is invalid. Check source and reduce dt.")
        self.last_residuals = values
        return values

    def step(self):
        if self.cfg.threads > 1 and numba.get_num_threads() != self.cfg.threads:
            numba.set_num_threads(self.cfg.threads)
        self.solve_psi()
        self.apply_thom_boundary()
        self.solve_omega()
        result = self.compute_residuals()
        self.completed_steps += 1
        return result

    def warm_up(self):
        """Compile kernels outside timing, then restore the zero initial state."""
        if self.completed_steps != 0 or np.any(self.psi) or np.any(self.omega):
            raise ValueError("warm_up is only valid before initial conditions are set")
        saved = self.source
        self.source = np.zeros_like(saved)
        try:
            self.step()
        finally:
            self.source = saved
            self.psi.fill(0)
            self.omega.fill(0)
            self.completed_steps = 0
            self.last_residuals = (math.inf, math.inf)
            self.last_control = 0.0
