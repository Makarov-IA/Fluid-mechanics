"""Thin Python wrapper for C++ stationary Navier-Stokes linearization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import numpy as np
from rich.console import Console

from solver.config import SimConfig
from solver.lib import StokesMACLib, find_solver_lib
from solver.state_io import full_mode_grids, load_mac_state_pickle

console = Console()

_TIME_VAR_RE = re.compile(r"(?<![A-Za-z0-9_])t(?![A-Za-z0-9_])")


@dataclass
class LinearizedEigenResult:
    """Eigenpairs of the stationary Navier-Stokes operator linearized around U0."""

    state_path: Path
    state_metadata: dict[str, object]
    base_residual_inf: float
    matvec_count: int
    dense_operator_bytes: int
    eig_message: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    u_modes: np.ndarray
    v_modes: np.ndarray
    p_modes: np.ndarray


def resolve_linear_state_path(project_dir: Path, cfg: SimConfig) -> Path:
    """Resolve the configured linearization state path relative to project root."""
    path = Path(cfg.linear_state_path)
    return path if path.is_absolute() else project_dir / path


def _uses_time(expr: str | None) -> bool:
    """Return True when the symbolic expression depends on time."""
    return expr is not None and bool(_TIME_VAR_RE.search(expr))


def _validate_time_independent(cfg: SimConfig) -> None:
    """Linearization requires time-independent forcing and boundary conditions."""
    exprs = {
        "forcing.fu": cfg.forcing_u,
        "forcing.fv": cfg.forcing_v,
        "boundary.u_top": cfg.bc_u_top,
        "boundary.u_bot": cfg.bc_u_bot,
        "boundary.v_left": cfg.bc_v_left,
        "boundary.v_right": cfg.bc_v_right,
        "boundary.u_left": cfg.bc_u_left,
        "boundary.u_right": cfg.bc_u_right,
        "boundary.v_bot": cfg.bc_v_bot,
        "boundary.v_top": cfg.bc_v_top,
    }
    timed = [name for name, expr in exprs.items() if _uses_time(expr)]
    if timed:
        names = ", ".join(timed)
        raise ValueError(
            "linearize mode requires time-independent forcing/BCs; "
            f"'t' appears in: {names}"
        )


def solve_linearized_eigenmodes(
    cfg: SimConfig,
    project_dir: Path,
) -> LinearizedEigenResult:
    """Compute selected eigenpairs using the C++ analytic dense linearization."""
    _validate_time_independent(cfg)

    state_path = resolve_linear_state_path(project_dir, cfg)
    mac_state, metadata = load_mac_state_pickle(state_path, cfg, check_dt=False)

    velocity_size = (cfg.nx - 1) * cfg.ny + cfg.nx * (cfg.ny - 1)
    if cfg.linear_n_eigs > velocity_size:
        raise ValueError(
            f"linearization.n_eigs={cfg.linear_n_eigs} must be <= velocity size "
            f"({velocity_size})"
        )

    dense_equivalent_bytes = velocity_size * velocity_size * np.dtype(np.float64).itemsize
    console.print(
        f"  operator size: {velocity_size} velocity unknowns   "
        f"dense equivalent ≈ {dense_equivalent_bytes / 1024**3:.2f} GiB"
    )

    lib_path = find_solver_lib(project_dir)
    fu0, fv0 = cfg.make_force_arrays(t=0.0)
    with StokesMACLib(lib_path, cfg.nx, cfg.ny, cfg.lx, cfg.ly, cfg.nu, cfg.dt) as solver:
        solver.set_bc_arrays(cfg.make_bc_arrays())
        solver.set_state(mac_state.u_vec, mac_state.v_vec, mac_state.p)
        (
            eigenvalues,
            eigenvectors,
            base_residual_inf,
            matvec_count,
            dense_operator_bytes,
        ) = solver.solve_linearized_eig(cfg.linear_n_eigs, cfg.linear_which, fu0, fv0)

    eig_message = (
        "C++ dense Eigen::EigenSolver completed"
        if matvec_count == velocity_size
        else f"C++ Arnoldi completed; Eigen::EigenSolver used on {matvec_count}×{matvec_count} Hessenberg"
    )
    console.print(
        f"  base state: [dim]{state_path}[/dim]   "
        f"||R(U0)||∞ = {base_residual_inf:.2e}"
    )
    console.print(
        f"  eig backend: {eig_message}   "
        f"operator storage ≈ {dense_operator_bytes / 1024**3:.2f} GiB"
    )

    u_modes, v_modes, p_modes = full_mode_grids(cfg, eigenvectors)
    return LinearizedEigenResult(
        state_path=state_path,
        state_metadata=metadata,
        base_residual_inf=base_residual_inf,
        matvec_count=matvec_count,
        dense_operator_bytes=dense_operator_bytes,
        eig_message=eig_message,
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
        "eigen_solver": "C++ Eigen::EigenSolver",
        "linearization": "analytic_cpp",
        "base_residual_inf": result.base_residual_inf,
        "matvec_count": result.matvec_count,
        "dense_operator_bytes": result.dense_operator_bytes,
        "operator_storage_bytes": result.dense_operator_bytes,
        "eig_message": result.eig_message,
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
