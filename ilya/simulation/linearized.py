"""Matrix-free steady Navier-Stokes linearization around a MAC state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from rich.console import Console
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from simulation.steady import _validate_time_independent
from solver.config import SimConfig
from solver.state_io import (
    full_mode_grids,
    join_full_state,
    load_mac_state_pickle,
    split_full_state,
)

console = Console()


@dataclass
class ProjectionSolver:
    """Sparse factorization for the MAC Helmholtz-Hodge pressure projection."""

    lu: spla.SuperLU
    velocity_size: int
    state_size: int


@dataclass
class LinearizedEigenResult:
    """Eigenpairs of the stationary Navier-Stokes operator linearized around U0."""

    state_path: Path
    state_metadata: dict[str, object]
    base_residual_inf: float
    matvec_count: int
    arpack_converged: bool
    arpack_message: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    u_modes: np.ndarray
    v_modes: np.ndarray
    p_modes: np.ndarray


def resolve_linear_state_path(project_dir: Path, cfg: SimConfig) -> Path:
    """Resolve the configured linearization state path relative to project root."""
    path = Path(cfg.linear_state_path)
    return path if path.is_absolute() else project_dir / path


def _u_unknown_idx(cfg: SimConfig, i: int, j: int) -> int:
    return j * (cfg.nx - 1) + (i - 1)


def _v_unknown_idx(cfg: SimConfig, i: int, j: int) -> int:
    return (cfg.nx - 1) * cfg.ny + (j - 1) * cfg.nx + i


def _p_idx(cfg: SimConfig, i: int, j: int) -> int:
    return j * cfg.nx + i


def _pack_full_state(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    p_grid: np.ndarray,
) -> np.ndarray:
    """Pack MAC grids into C++ unknown ordering [u_vec, v_vec, p]."""
    return np.concatenate(
        [
            u_grid.T.reshape(-1),
            v_grid.T.reshape(-1),
            p_grid.T.reshape(-1),
        ]
    )


def _force_grid_u(cfg: SimConfig, fu: np.ndarray | None) -> np.ndarray:
    if fu is None:
        return np.zeros((cfg.nx - 1, cfg.ny), dtype=np.float64)
    return np.asarray(fu, dtype=np.float64).reshape(cfg.ny, cfg.nx - 1).T


def _force_grid_v(cfg: SimConfig, fv: np.ndarray | None) -> np.ndarray:
    if fv is None:
        return np.zeros((cfg.nx, cfg.ny - 1), dtype=np.float64)
    return np.asarray(fv, dtype=np.float64).reshape(cfg.ny - 1, cfg.nx).T


def _full_fields_from_state(
    cfg: SimConfig,
    state: np.ndarray,
    bcs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert [u_vec, v_vec, p] to full MAC fields with velocity BCs applied."""
    nu_u = (cfg.nx - 1) * cfg.ny
    nv_u = cfg.nx * (cfg.ny - 1)
    u_vec, v_vec, p_vec = split_full_state(state, nu_u, nv_u)

    u = np.zeros((cfg.nx + 1, cfg.ny), dtype=np.float64)
    v = np.zeros((cfg.nx, cfg.ny + 1), dtype=np.float64)
    p = p_vec.reshape(cfg.ny, cfg.nx).T.copy()

    u[1 : cfg.nx, :] = u_vec.reshape(cfg.ny, cfg.nx - 1).T
    v[:, 1 : cfg.ny] = v_vec.reshape(cfg.ny - 1, cfg.nx).T
    u[0, :] = bcs["u_left"]
    u[cfg.nx, :] = bcs["u_right"]
    v[:, 0] = bcs["v_bot"]
    v[:, cfg.ny] = bcs["v_top"]
    return u, v, p


def _u_y_ghost_padded(
    cfg: SimConfig,
    u: np.ndarray,
    bcs: dict[str, np.ndarray],
) -> np.ndarray:
    """Pad u by one ghost layer in y, matching the C++ wall extension."""
    padded = np.zeros((cfg.nx + 1, cfg.ny + 2), dtype=np.float64)
    padded[:, 1 : cfg.ny + 1] = u
    padded[:, 0] = -u[:, 0]
    padded[1 : cfg.nx, 0] = 2.0 * bcs["u_bot"] - u[1 : cfg.nx, 0]
    padded[:, cfg.ny + 1] = 0.0
    padded[1 : cfg.nx, cfg.ny + 1] = 2.0 * bcs["u_top"] - u[1 : cfg.nx, cfg.ny - 1]
    return padded


def _v_x_ghost_padded(
    cfg: SimConfig,
    v: np.ndarray,
    bcs: dict[str, np.ndarray],
) -> np.ndarray:
    """Pad v by one ghost layer in x, matching the C++ wall extension."""
    padded = np.zeros((cfg.nx + 2, cfg.ny + 1), dtype=np.float64)
    padded[1 : cfg.nx + 1, :] = v
    padded[0, :] = -v[0, :]
    padded[0, 1 : cfg.ny] = 2.0 * bcs["v_left"] - v[0, 1 : cfg.ny]
    padded[cfg.nx + 1, :] = -v[cfg.nx - 1, :]
    padded[cfg.nx + 1, 1 : cfg.ny] = 2.0 * bcs["v_right"] - v[cfg.nx - 1, 1 : cfg.ny]
    return padded


def steady_residual(
    cfg: SimConfig,
    state: np.ndarray,
    fu: np.ndarray | None,
    fv: np.ndarray | None,
    bcs: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate stationary MAC Navier-Stokes residual R(u,p) with u_t=0."""
    dx = cfg.lx / cfg.nx
    dy = cfg.ly / cfg.ny
    dx2 = dx * dx
    dy2 = dy * dy

    u, v, p = _full_fields_from_state(cfg, state, bcs)
    u_pad = _u_y_ghost_padded(cfg, u, bcs)
    v_pad = _v_x_ghost_padded(cfg, v, bcs)

    u_int = u[1 : cfg.nx, :]
    du_dx = (u[2 : cfg.nx + 1, :] - u[0 : cfg.nx - 1, :]) / (2.0 * dx)
    du_dy = (u_pad[1 : cfg.nx, 2 : cfg.ny + 2] - u_pad[1 : cfg.nx, 0 : cfg.ny]) / (
        2.0 * dy
    )
    v_at_u = 0.25 * (
        v[0 : cfg.nx - 1, 0 : cfg.ny]
        + v[1 : cfg.nx, 0 : cfg.ny]
        + v[0 : cfg.nx - 1, 1 : cfg.ny + 1]
        + v[1 : cfg.nx, 1 : cfg.ny + 1]
    )
    adv_u = u_int * du_dx + v_at_u * du_dy
    lap_u = (
        (u[2 : cfg.nx + 1, :] - 2.0 * u_int + u[0 : cfg.nx - 1, :]) / dx2
        + (u_pad[1 : cfg.nx, 2 : cfg.ny + 2] - 2.0 * u_int + u_pad[1 : cfg.nx, 0 : cfg.ny])
        / dy2
    )
    dp_dx = (p[1 : cfg.nx, :] - p[0 : cfg.nx - 1, :]) / dx
    res_u = adv_u - cfg.nu * lap_u + dp_dx - _force_grid_u(cfg, fu)

    v_int = v[:, 1 : cfg.ny]
    dv_dx = (v_pad[2 : cfg.nx + 2, 1 : cfg.ny] - v_pad[0 : cfg.nx, 1 : cfg.ny]) / (
        2.0 * dx
    )
    dv_dy = (v[:, 2 : cfg.ny + 1] - v[:, 0 : cfg.ny - 1]) / (2.0 * dy)
    u_at_v = 0.25 * (
        u[0 : cfg.nx, 0 : cfg.ny - 1]
        + u[1 : cfg.nx + 1, 0 : cfg.ny - 1]
        + u[0 : cfg.nx, 1 : cfg.ny]
        + u[1 : cfg.nx + 1, 1 : cfg.ny]
    )
    adv_v = u_at_v * dv_dx + v_int * dv_dy
    lap_v = (
        (v_pad[2 : cfg.nx + 2, 1 : cfg.ny] - 2.0 * v_int + v_pad[0 : cfg.nx, 1 : cfg.ny])
        / dx2
        + (v[:, 2 : cfg.ny + 1] - 2.0 * v_int + v[:, 0 : cfg.ny - 1]) / dy2
    )
    dp_dy = (p[:, 1 : cfg.ny] - p[:, 0 : cfg.ny - 1]) / dy
    res_v = adv_v - cfg.nu * lap_v + dp_dy - _force_grid_v(cfg, fv)

    div = (u[1 : cfg.nx + 1, :] - u[0 : cfg.nx, :]) / dx + (
        v[:, 1 : cfg.ny + 1] - v[:, 0 : cfg.ny]
    ) / dy
    div = div.copy()
    div[0, 0] = p[0, 0]

    return _pack_full_state(res_u, res_v, div)


def _build_projection_solver(cfg: SimConfig) -> ProjectionSolver:
    """Build and factor [I G; D gauge] for pressure projection of velocity RHS."""
    dx = cfg.lx / cfg.nx
    dy = cfg.ly / cfg.ny
    nu_u = (cfg.nx - 1) * cfg.ny
    nv_u = cfg.nx * (cfg.ny - 1)
    np_u = cfg.nx * cfg.ny
    velocity_size = nu_u + nv_u
    state_size = velocity_size + np_u

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add(row: int, col: int, value: float) -> None:
        rows.append(row)
        cols.append(col)
        vals.append(value)

    for row in range(velocity_size):
        add(row, row, 1.0)

    for j in range(cfg.ny):
        for i in range(1, cfg.nx):
            row = _u_unknown_idx(cfg, i, j)
            add(row, velocity_size + _p_idx(cfg, i, j), +1.0 / dx)
            add(row, velocity_size + _p_idx(cfg, i - 1, j), -1.0 / dx)

    for j in range(1, cfg.ny):
        for i in range(cfg.nx):
            row = _v_unknown_idx(cfg, i, j)
            add(row, velocity_size + _p_idx(cfg, i, j), +1.0 / dy)
            add(row, velocity_size + _p_idx(cfg, i, j - 1), -1.0 / dy)

    for j in range(cfg.ny):
        for i in range(cfg.nx):
            row = velocity_size + _p_idx(cfg, i, j)
            if i == 0 and j == 0:
                add(row, velocity_size + _p_idx(cfg, 0, 0), 1.0)
                continue

            if i + 1 <= cfg.nx - 1:
                add(row, _u_unknown_idx(cfg, i + 1, j), +1.0 / dx)
            if i >= 1:
                add(row, _u_unknown_idx(cfg, i, j), -1.0 / dx)
            if j + 1 <= cfg.ny - 1:
                add(row, _v_unknown_idx(cfg, i, j + 1), +1.0 / dy)
            if j >= 1:
                add(row, _v_unknown_idx(cfg, i, j), -1.0 / dy)

    matrix = sp.csc_matrix((vals, (rows, cols)), shape=(state_size, state_size))
    lu = spla.splu(matrix)
    return ProjectionSolver(lu=lu, velocity_size=velocity_size, state_size=state_size)


def _sort_eigenpairs(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    which: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort ARPACK output in the same direction requested by `which`."""
    if which == "LM":
        order = np.argsort(np.abs(eigenvalues))[::-1]
    elif which == "SM":
        order = np.argsort(np.abs(eigenvalues))
    elif which == "LR":
        order = np.argsort(eigenvalues.real)[::-1]
    elif which == "SR":
        order = np.argsort(eigenvalues.real)
    elif which == "LI":
        order = np.argsort(eigenvalues.imag)[::-1]
    else:
        order = np.argsort(eigenvalues.imag)
    return eigenvalues[order], eigenvectors[:, order]


def _normalise_velocity_modes(eigenvectors: np.ndarray) -> np.ndarray:
    """Scale each velocity eigenvector column to max-norm one."""
    vectors = eigenvectors.copy()
    for idx in range(vectors.shape[1]):
        norm = float(np.max(np.abs(vectors[:, idx])))
        if norm > 0.0:
            vectors[:, idx] /= norm
    return vectors


def solve_linearized_eigenmodes(
    cfg: SimConfig,
    project_dir: Path,
) -> LinearizedEigenResult:
    """Find eigenpairs of L=-P D_momentum R(U0) on velocity perturbations."""
    _validate_time_independent(cfg)

    state_path = resolve_linear_state_path(project_dir, cfg)
    mac_state, metadata = load_mac_state_pickle(state_path, cfg, check_dt=False)
    base_state = join_full_state(mac_state)
    projection = _build_projection_solver(cfg)
    velocity_size = projection.velocity_size

    if cfg.linear_n_eigs >= velocity_size - 1:
        raise ValueError(
            f"linearization.n_eigs={cfg.linear_n_eigs} must be < velocity size - 1 "
            f"({velocity_size - 1})"
        )

    fu0, fv0 = cfg.make_force_arrays(t=0.0)
    bcs = cfg.make_bc_arrays()
    base_residual = steady_residual(cfg, base_state, fu0, fv0, bcs)
    base_residual_inf = float(np.max(np.abs(base_residual)))
    base_velocity_inf = max(float(np.max(np.abs(base_state[:velocity_size]))), 1.0)
    matvec_count = 0

    console.print(
        f"  base state: [dim]{state_path}[/dim]   "
        f"||R(U0)||∞ = {base_residual_inf:.2e}"
    )

    def raw_velocity_action(vec: np.ndarray, *, count: bool) -> np.ndarray:
        nonlocal matvec_count
        vec = np.asarray(vec, dtype=np.float64)
        vec_inf = float(np.max(np.abs(vec)))
        if vec_inf < 1e-14:
            return np.zeros_like(vec)

        perturb = np.zeros_like(base_state)
        perturb[:velocity_size] = vec
        eps = cfg.linear_jacobian_rdiff * base_velocity_inf / vec_inf
        trial_residual = steady_residual(cfg, base_state + eps * perturb, fu0, fv0, bcs)
        if count:
            matvec_count += 1
        return -(trial_residual[:velocity_size] - base_residual[:velocity_size]) / eps

    def project_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rhs = np.zeros(projection.state_size, dtype=np.float64)
        rhs[:velocity_size] = raw
        solved = projection.lu.solve(rhs)
        return solved[:velocity_size], solved[velocity_size:]

    def real_matvec(vec: np.ndarray, *, count: bool = True) -> np.ndarray:
        raw = raw_velocity_action(vec, count=count)
        projected_velocity, _ = project_raw(raw)
        return projected_velocity

    def matvec(vec: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(vec):
            return real_matvec(vec.real) + 1j * real_matvec(vec.imag)
        return real_matvec(vec)

    linop = spla.LinearOperator((velocity_size, velocity_size), matvec=matvec, dtype=np.float64)
    ncv = min(velocity_size - 1, max(4 * cfg.linear_n_eigs + 20, 40))

    try:
        eigenvalues, velocity_eigenvectors = spla.eigs(
            linop,
            k=cfg.linear_n_eigs,
            which=cfg.linear_which,
            tol=cfg.linear_tol,
            maxiter=cfg.linear_maxiter,
            ncv=ncv,
        )
        arpack_converged = True
        arpack_message = "converged"
    except spla.ArpackNoConvergence as exc:
        eigenvalues = exc.eigenvalues
        velocity_eigenvectors = exc.eigenvectors
        arpack_converged = False
        arpack_message = str(exc)
        if eigenvalues is None or velocity_eigenvectors is None or len(eigenvalues) == 0:
            raise
        console.print(
            f"  [yellow]ARPACK stopped early; saving {len(eigenvalues)} converged modes[/yellow]"
        )

    eigenvalues, velocity_eigenvectors = _sort_eigenpairs(
        eigenvalues,
        velocity_eigenvectors,
        cfg.linear_which,
    )
    velocity_eigenvectors = _normalise_velocity_modes(velocity_eigenvectors)

    pressure_modes: list[np.ndarray] = []
    for mode_idx in range(velocity_eigenvectors.shape[1]):
        mode = velocity_eigenvectors[:, mode_idx]
        raw_real = raw_velocity_action(mode.real, count=False)
        _, pressure_real = project_raw(raw_real)
        if np.max(np.abs(mode.imag)) > 0.0:
            raw_imag = raw_velocity_action(mode.imag, count=False)
            _, pressure_imag = project_raw(raw_imag)
            pressure_modes.append(pressure_real + 1j * pressure_imag)
        else:
            pressure_modes.append(pressure_real.astype(np.complex128))

    p_eigenvectors = np.column_stack(pressure_modes)
    eigenvectors = np.vstack([velocity_eigenvectors, p_eigenvectors])
    u_modes, v_modes, p_modes = full_mode_grids(cfg, eigenvectors)

    return LinearizedEigenResult(
        state_path=state_path,
        state_metadata=metadata,
        base_residual_inf=base_residual_inf,
        matvec_count=matvec_count,
        arpack_converged=arpack_converged,
        arpack_message=arpack_message,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        u_modes=u_modes,
        v_modes=v_modes,
        p_modes=p_modes,
    )


def save_linearized_eigenmodes(
    result: LinearizedEigenResult,
    cfg: SimConfig,
    out_dir: Path,
) -> Path:
    """Save linearized eigenpairs and metadata as one pickle dictionary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eigenpairs.pkl"
    payload = {
        "operator": "steady_ns_velocity_projection",
        "operator_definition": (
            "L=-P D_momentum R(U0), where R is the stationary MAC Navier-Stokes "
            "residual and P enforces divergence-free velocity through pressure"
        ),
        "state_path": str(result.state_path),
        "state_metadata": result.state_metadata,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "lx": cfg.lx,
        "ly": cfg.ly,
        "nu": cfg.nu,
        "n_eigs": len(result.eigenvalues),
        "which": cfg.linear_which,
        "tol": cfg.linear_tol,
        "maxiter": cfg.linear_maxiter,
        "jacobian_rdiff": cfg.linear_jacobian_rdiff,
        "base_residual_inf": result.base_residual_inf,
        "matvec_count": result.matvec_count,
        "arpack_converged": result.arpack_converged,
        "arpack_message": result.arpack_message,
        "eigenvalues": result.eigenvalues,
        "eigenvectors": result.eigenvectors,
        "u_modes": result.u_modes,
        "v_modes": result.v_modes,
        "p_modes": result.p_modes,
        "state_ordering": "[u_vec, v_vec, p]",
        "operator_state": "velocity perturbations; pressure modes recovered by projection",
        "u_mode_shape": "(mode, nx-1, ny)",
        "v_mode_shape": "(mode, nx, ny-1)",
        "p_mode_shape": "(mode, nx, ny)",
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path
