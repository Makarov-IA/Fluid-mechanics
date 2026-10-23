#!/usr/bin/env python3
"""Exact discrete steady state and its linear stability, in the solver's own equations.

F(psi, omega) = (lap psi + omega,  (1/Re) lap omega + J_A(psi, omega) + S) on the interior,
with psi = 0 and Thom's omega on the walls: the very system whose max-norm the solver
reports. Nothing here is a second discretisation: the kernels are the solver's.

Newton uses the exact Jacobian, assembled sparse by colouring the linear tangent kernel
(9-point stencil, so 3x3 colours per field make every column exact), and direct LU solves.
There are no Krylov tolerances, preconditioners or line searches.

Linearised dynamics: omega_t = F_omega with psi slaved by F_psi = 0, i.e. the Schur
complement A = J_ww - J_wp J_pp^{-1} J_pw on the interior omega. The orthogonal projector
onto the real span of its unstable eigenvectors is an exact invariant-subspace projector,
so A - sigma*P shifts exactly those eigenvalues by -sigma and leaves the rest unchanged;
both facts are checked numerically before anything is saved.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigs, splu

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from solver_python.snapshot_io import read_snapshot, write_snapshot
from solver_python import kernels as k
from solver_python.config import DEFAULT_CONFIG, ConfigError, grid, load_config
from solver_python.forcing import build_forcing

ALEX = Path(__file__).resolve().parents[1]
EIGS_COUNT = 30
EIGS_TOL = 1e-10
EIGS_MAX_ITER = 20000
MAX_NEWTON = 20


class SteadyProblem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.xs, self.ys = grid(cfg)
        self.source, _ = build_forcing(cfg, self.xs, self.ys)
        self.nx, self.ny = cfg.nx, cfg.ny
        self.ni, self.nj = cfg.nx-2, cfg.ny-2
        self.n = self.ni*self.nj
        self.dx, self.dy = cfg.lx/(cfg.nx-1), cfg.ly/(cfg.ny-1)
        self.nu = 1.0/cfg.re
        self.idx2, self.idy2 = 1/self.dx**2, 1/self.dy**2
        self.inv_4dxdy = 1/(4*self.dx*self.dy)

    def pack(self, p, w):
        return np.concatenate((p[1:-1, 1:-1].ravel(), w[1:-1, 1:-1].ravel()))

    def fields(self, z):
        """Full arrays from interior unknowns: psi walls 0, omega walls from Thom."""
        p = np.zeros((self.nx, self.ny))
        w = np.zeros((self.nx, self.ny))
        p[1:-1, 1:-1] = z[:self.n].reshape(self.ni, self.nj)
        w[1:-1, 1:-1] = z[self.n:].reshape(self.ni, self.nj)
        bc = self.cfg.bc
        k.thom_boundary(p, w, self.dx, self.dy, bc.bottom.u, bc.top.u, bc.left.v, bc.right.v)
        return p, w

    def tangent_fields(self, dz):
        """Thom's formula is affine in psi: its derivative drops the wall velocities."""
        dp = np.zeros((self.nx, self.ny))
        dw = np.zeros((self.nx, self.ny))
        dp[1:-1, 1:-1] = dz[:self.n].reshape(self.ni, self.nj)
        dw[1:-1, 1:-1] = dz[self.n:].reshape(self.ni, self.nj)
        k.thom_boundary(dp, dw, self.dx, self.dy, 0.0, 0.0, 0.0, 0.0)
        return dp, dw

    def residual(self, z):
        p, w = self.fields(z)
        rp, rw = np.zeros_like(p), np.zeros_like(w)
        k.steady_residual(p, w, self.source, self.nu, self.idx2, self.idy2, self.inv_4dxdy, rp, rw)
        return self.pack(rp, rw)

    def solver_residuals(self, z):
        """The solver's own convergence measure, for an independent check."""
        p, w = self.fields(z)
        return k.residuals(p, w, self.source, self.nu, self.idx2, self.idy2, self.inv_4dxdy)

    def tangent(self, z, dz):
        p, w = self.fields(z)
        dp, dw = self.tangent_fields(dz)
        rp, rw = np.zeros_like(p), np.zeros_like(w)
        k.steady_tangent(p, w, dp, dw, self.nu, self.idx2, self.idy2, self.inv_4dxdy, rp, rw)
        return self.pack(rp, rw)

    def jacobian(self, z):
        """Exact sparse Jacobian. A column at (i, j) only touches rows within one cell
        (including the Thom coupling through the walls), so unknowns 3 apart never share
        a row: one tangent evaluation recovers a whole colour class exactly."""
        ii, jj = np.meshgrid(np.arange(self.ni), np.arange(self.nj), indexing="ij")
        rows, cols, vals = [], [], []
        for field in (0, 1):
            for ci in range(3):
                for cj in range(3):
                    mask = (ii % 3 == ci) & (jj % 3 == cj)
                    seed = np.zeros(2*self.n)
                    col_index = field*self.n + (ii[mask]*self.nj + jj[mask])
                    seed[col_index] = 1.0
                    response = self.tangent(z, seed)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ri, rj = ii[mask]+di, jj[mask]+dj
                            inside = (ri >= 0) & (ri < self.ni) & (rj >= 0) & (rj < self.nj)
                            for row_field in (0, 1):
                                row_index = row_field*self.n + ri[inside]*self.nj + rj[inside]
                                value = response[row_index]
                                keep = value != 0.0
                                rows.append(row_index[keep])
                                cols.append(col_index[inside][keep])
                                vals.append(value[keep])
        size = 2*self.n
        return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(size, size))


def check_jacobian(problem, z, jac, rng):
    """Assembled matrix against the tangent kernel (exact) and against finite differences."""
    v = rng.standard_normal(2*problem.n)
    exact = problem.tangent(z, v)
    assembled_error = float(np.max(np.abs(jac@v - exact))/np.max(np.abs(exact)))
    # F is quadratic (J_A is bilinear), so the central difference is exact for any h;
    # a moderate h only keeps round-off small.
    h = 1e-3*max(1.0, float(np.max(np.abs(z))))/float(np.max(np.abs(v)))
    fd = (problem.residual(z+h*v) - problem.residual(z-h*v))/(2*h)
    fd_error = float(np.max(np.abs(fd - exact))/np.max(np.abs(exact)))
    if assembled_error > 1e-12:
        raise AssertionError(f"Assembled Jacobian differs from the tangent kernel: {assembled_error:.3e}")
    if fd_error > 1e-6:
        raise AssertionError(f"Tangent kernel differs from finite differences: {fd_error:.3e}")
    return assembled_error, fd_error


def newton(problem, z, tol):
    """Pure Newton with exact Jacobian and exact linear solves."""
    history = []
    for iteration in range(MAX_NEWTON+1):
        f = problem.residual(z)
        norm = float(np.max(np.abs(f)))
        history.append({"iteration": iteration, "max_residual": norm})
        print(f"[newton] iter={iteration} max|F|={norm:.3e}", flush=True)
        if norm <= tol:
            return z, history
        if iteration == MAX_NEWTON:
            break
        jac = problem.jacobian(z)
        z = z + splu(jac).solve(-f)
    raise RuntimeError(f"Newton did not reach {tol:g} in {MAX_NEWTON} iterations; "
                       "the initial state is not in its convergence basin")


def schur_operator(problem, jac):
    """A = J_ww - J_wp J_pp^{-1} J_pw on interior omega; J_pw is the identity."""
    n = problem.n
    jpp = jac[:n, :n].tocsc()
    jpw = jac[:n, n:].tocsr()
    jwp = jac[n:, :n].tocsr()
    jww = jac[n:, n:].tocsr()
    if (jpw - sp.identity(n, format="csr")).count_nonzero():
        raise AssertionError("dF_psi/domega must be the identity")
    lu = splu(jpp)
    return jww, jwp, lu


def apply(parts, x, basis=None, gain=0.0):
    jww, jwp, lu = parts
    y = jww@x - jwp@lu.solve(np.ascontiguousarray(x))
    if basis is not None:
        y = y - gain*(basis.T@(basis@x))
    return y


def leading_eigenpairs(parts, n, seed, basis=None, gain=0.0):
    op = LinearOperator((n, n), matvec=lambda x: apply(parts, x, basis, gain), dtype=float)
    v0 = np.random.default_rng(seed).standard_normal(n)
    values, vectors = eigs(op, k=EIGS_COUNT, which="LR", tol=EIGS_TOL, maxiter=EIGS_MAX_ITER, v0=v0)
    order = np.argsort(-values.real)
    values, vectors = values[order], vectors[:, order]
    residuals = []
    for idx in range(values.size):
        v = vectors[:, idx]
        av = apply(parts, v.real, basis, gain) + 1j*apply(parts, v.imag, basis, gain)
        residuals.append(float(np.linalg.norm(av - values[idx]*v)/np.linalg.norm(v)))
    return values, vectors, np.array(residuals)


def write_eigenvalues(path, values, residuals):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "real", "imag", "residual", "unstable"])
        for idx, (value, res) in enumerate(zip(values, residuals)):
            writer.writerow([idx, f"{value.real:.16e}", f"{value.imag:.16e}", f"{res:.3e}", int(value.real > 0)])


def plot_spectrum(path, values, shifted, gain):
    figure, axes = plt.subplots(figsize=(7, 5))
    axes.scatter(values.real, values.imag, s=28, label="A")
    axes.scatter(shifted.real, shifted.imag, s=12, marker="x", label=f"A - {gain:g} P")
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("Re λ")
    axes.set_ylabel("Im λ")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyse(config_path: Path, gain: float) -> dict:
    cfg = load_config(config_path)
    binaries = cfg.output_dir
    stationary = cfg.result_dir/"stationary"
    newton_dir = cfg.result_dir/"newton"
    stability_dir = cfg.result_dir/"linear_stability"
    for directory in (stationary, newton_dir, stability_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Initial guess exactly as before: the snapshot with the smallest relative change.
    subprocess.run([sys.executable, str(ALEX/"solver_python/find_stationary.py"),
                    "--results-dir", str(binaries), "--config", str(cfg.path),
                    "--metrics-csv", str(stationary/"consecutive_difference_norm.csv"),
                    "--plot-png", str(stationary/"consecutive_difference_norm.png"),
                    "--stationary-snapshot", str(stationary/"stationary_state.bin"),
                    "--streamplot-png", str(stationary/"stationary_streamplot.png"),
                    "--field", "state", "--skip-newest", "1"], check=True)

    problem = SteadyProblem(cfg)
    initial = read_snapshot(stationary/"stationary_state.bin")
    z = problem.pack(initial.psi, initial.omega)
    rng = np.random.default_rng(0)
    assembled_error, fd_error = check_jacobian(problem, z, problem.jacobian(z), rng)
    print(f"[newton] Jacobian: assembled vs tangent {assembled_error:.1e}, "
          f"tangent vs finite differences {fd_error:.1e}", flush=True)

    z, history = newton(problem, z, cfg.steady_tolerance)
    rp, rw = problem.solver_residuals(z)
    if max(rp, rw) > cfg.steady_tolerance:
        raise AssertionError(f"Solver residuals of the Newton state: {rp:.3e}, {rw:.3e}")
    p_star, w_star = problem.fields(z)
    write_snapshot(newton_dir/"equilibrium.bin", problem.xs, problem.ys, p_star, w_star,
                   step=0, time=0.0, re=cfg.re, bc=cfg.bc)
    with (newton_dir/"newton_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["iteration", "max_residual"])
        writer.writeheader()
        writer.writerows(history)
    print(f"[newton] equilibrium: solver residuals Rpsi={rp:.3e} Romega={rw:.3e}", flush=True)

    jac = problem.jacobian(z)
    parts = schur_operator(problem, jac)
    print(f"[stability] ARPACK: {EIGS_COUNT} eigenvalues with largest real part", flush=True)
    values, vectors, residuals = leading_eigenpairs(parts, problem.n, seed=1)
    write_eigenvalues(stability_dir/"eigenvalues.csv", values, residuals)
    if values.real.min() >= 0:
        raise AssertionError("All computed eigenvalues are unstable: raise EIGS_COUNT")
    unstable = np.flatnonzero(values.real > 0)
    for idx in range(values.size):
        mark = "unstable" if idx in unstable else ""
        print(f"[stability] λ{idx:02d} = {values[idx].real:+.8f} {values[idx].imag:+.8f}i  "
              f"residual {residuals[idx]:.1e} {mark}", flush=True)
    if unstable.size == 0:
        raise AssertionError("No unstable eigenvalues: nothing to stabilise")

    # Real basis of the unstable invariant subspace; a conjugate pair gives Re and Im.
    columns = []
    for idx in unstable:
        if values[idx].imag == 0.0:
            columns.append(vectors[:, idx].real)
        elif values[idx].imag > 0.0:
            columns += [vectors[:, idx].real, vectors[:, idx].imag]
    q, r = np.linalg.qr(np.column_stack(columns))
    if np.min(np.abs(np.diag(r))) <= 1e-10*np.max(np.abs(np.diag(r))):
        raise AssertionError("Unstable eigenvectors are numerically dependent")
    basis = q.T  # rows orthonormal
    shift_error = 0.0
    for idx in unstable:
        v = vectors[:, idx]
        moved = (apply(parts, v.real, basis, gain) + 1j*apply(parts, v.imag, basis, gain)
                 - (values[idx]-gain)*v)
        shift_error = max(shift_error, float(np.linalg.norm(moved)/np.linalg.norm(v)))
    shifted, _, shifted_res = leading_eigenpairs(parts, problem.n, seed=2, basis=basis, gain=gain)
    write_eigenvalues(stability_dir/"eigenvalues_stabilized.csv", shifted, shifted_res)
    # Every stable eigenvalue of A above the lowest computed one of A - sigma*P is an
    # eigenvalue of A - sigma*P too and must therefore be in that computed window.
    stable = values[values.real < 0]
    required = stable[stable.real > shifted.real.min()]
    unchanged = max(float(np.min(np.abs(shifted - s))) for s in required)
    print(f"[stability] A - {gain:g}P: unstable pair moved by -{gain:g} (error {shift_error:.1e}); "
          f"max Re = {shifted.real.max():+.8f}; stable eigenvalues unchanged to {unchanged:.1e}", flush=True)
    if shifted.real.max() >= 0:
        raise AssertionError("Gain does not stabilise: some eigenvalue of A - sigma P is unstable")

    from solver_python.projection import full_state_basis
    state_basis = full_state_basis(basis, cfg.nx, cfg.ny, problem.dx, problem.dy)
    np.savez(stability_dir/"stabilization.npz", state_basis=state_basis, xs=problem.xs, ys=problem.ys,
             psi_star=p_star, omega_star=w_star, basis=basis,
             eigenvalues=values[unstable], re=cfg.re, source=problem.source,
             wall_velocities=np.array([(w.u, w.v) for w in
                 (cfg.bc.left, cfg.bc.right, cfg.bc.bottom, cfg.bc.top)]))
    plot_spectrum(stability_dir/"spectrum.png", values, shifted, gain)
    report = {
        "spectrum_scope": "physical operator with psi eliminated; controlled spectrum is A-gain*P_omega, not full-state feedback",
        "state_basis_metric": "unweighted Euclidean, concatenated interior psi and omega",
        "initial_snapshot": str(stationary/"stationary_state.bin"),
        "newton_iterations": len(history)-1,
        "newton_history": [h["max_residual"] for h in history],
        "solver_residuals": [rp, rw],
        "jacobian_check": {"assembled_vs_tangent": assembled_error, "tangent_vs_fd": fd_error},
        "unstable_eigenvalues": [[float(v.real), float(v.imag)] for v in values[unstable]],
        "eigen_residual_max": float(residuals.max()),
        "gain": gain, "shift_error": shift_error,
        "stabilized_max_real": float(shifted.real.max()),
        "stable_eigenvalues_unchanged_to": unchanged,
        "stabilization_file": str(stability_dir/"stabilization.npz"),
    }
    (stability_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"[stability] saved: {stability_dir/'stabilization.npz'}", flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gain", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        analyse(args.config, args.gain)
        return 0
    except (ConfigError, ValueError, OSError, RuntimeError, AssertionError,
            subprocess.CalledProcessError) as exc:
        print(f"[stability] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
