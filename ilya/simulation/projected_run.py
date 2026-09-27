"""Simulation from a steady state with the unstable modes suppressed.

Two methods (``projected_run.method``):

* ``forcing``  — remove the component of [fu, fv] along the unstable modes
  once (open loop).
* ``feedback_field`` / ``feedback_deviation`` — every time step the time
  derivative uses u*_n = u^n - delta_n instead of u^n, i.e. -delta_n / dt is
  moved to the right-hand side; convection keeps u^n.
  ``feedback_field``:     delta_n = alpha * Pi u^n (the field without its
  unstable modes); ``feedback_deviation``: delta_n = alpha * Pi (u^n - u_s).  Pi = Q W^T is the
  oblique projector onto the unstable eigenspace along the stable one
  (W from adjoint modes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from solver.config import MacState, SimConfig
from solver.state_io import load_mac_state_pickle


@dataclass
class ProjectionInfo:
    """Metadata for the forcing projection used by projected-run."""

    state_path: Path
    eigenpairs_path: Path
    real_threshold: float
    selected_indices: np.ndarray
    selected_eigenvalues: np.ndarray
    basis_rank: int
    force_norm: float
    removed_norm: float
    remaining_norm: float
    method: str = "forcing"
    feedback_alpha: float | None = None


@dataclass
class FeedbackStabilization:
    """Data for the per-step feedback correction in the C++ solver."""

    basis: np.ndarray    # Q, (velocity_size, m)
    adjoint: np.ndarray  # W, (velocity_size, m), W^T Q = I
    u_ref: np.ndarray    # projected is u^n - u_ref (zeros: the field itself)
    u_steady: np.ndarray # steady velocity [u_vec, v_vec], for diagnostics
    alpha: float
    projection: str      # "field" | "deviation"


class ForceProjection:
    """Remove the component of [fu, fv] lying in the selected eigenmode span."""

    def __init__(self, cfg: SimConfig, basis: np.ndarray) -> None:
        self.cfg = cfg
        self.basis = basis
        self.nu_u = (cfg.nx - 1) * cfg.ny
        self.nv_u = cfg.nx * (cfg.ny - 1)

    def vector_from_arrays(
        self,
        fu: np.ndarray | None,
        fv: np.ndarray | None,
    ) -> np.ndarray:
        """Pack force arrays into the velocity-state ordering [u_vec, v_vec]."""
        fu_vec = (
            np.zeros(self.nu_u, dtype=np.float64)
            if fu is None
            else np.asarray(fu, dtype=np.float64)
        )
        fv_vec = (
            np.zeros(self.nv_u, dtype=np.float64)
            if fv is None
            else np.asarray(fv, dtype=np.float64)
        )
        if fu_vec.shape != (self.nu_u,):
            raise ValueError(f"fu must have shape {(self.nu_u,)}, got {fu_vec.shape}")
        if fv_vec.shape != (self.nv_u,):
            raise ValueError(f"fv must have shape {(self.nv_u,)}, got {fv_vec.shape}")
        return np.concatenate([fu_vec, fv_vec])

    def arrays_from_vector(self, force: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split one force vector back into u/v force arrays."""
        return force[: self.nu_u].copy(), force[self.nu_u :].copy()

    def project(self, force: np.ndarray) -> np.ndarray:
        """Return the orthogonal projection of force onto selected real modes."""
        if self.basis.size == 0:
            return np.zeros_like(force)
        return self.basis @ (self.basis.T @ force)

    def __call__(
        self,
        fu: np.ndarray | None,
        fv: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        force = self.vector_from_arrays(fu, fv)
        filtered = force - self.project(force)
        return self.arrays_from_vector(filtered)


def resolve_projected_state_path(project_dir: Path, cfg: SimConfig) -> Path:
    """Resolve projected-run steady state path relative to project root."""
    path = Path(cfg.projected_state_path)
    return path if path.is_absolute() else project_dir / path


def resolve_projected_eigenpairs_path(project_dir: Path, cfg: SimConfig) -> Path:
    """Resolve projected-run eigenpairs path relative to project root."""
    path = Path(cfg.projected_eigenpairs_path)
    return path if path.is_absolute() else project_dir / path


def _load_eigenpairs(path: Path, cfg: SimConfig) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Eigenpairs pickle not found: {path}. Run `make linearize` first.")
    with path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a pickle dictionary")

    required = ("eigenvectors", "eigenvalues", "nx", "ny")
    missing = [key for key in required if key not in data]
    if missing:
        keys = ", ".join(missing)
        raise ValueError(f"{path} is missing keys: {keys}")
    if int(data["nx"]) != cfg.nx or int(data["ny"]) != cfg.ny:
        raise ValueError(f"{path} grid does not match config grid {cfg.nx}x{cfg.ny}")
    return data


def _real_basis_from_modes(modes: np.ndarray, rcond: float) -> np.ndarray:
    """Build an orthonormal real basis from complex eigenvector columns."""
    if modes.size == 0:
        return np.empty((modes.shape[0], 0), dtype=np.float64)

    columns: list[np.ndarray] = []
    for idx in range(modes.shape[1]):
        mode = modes[:, idx]
        real = np.asarray(mode.real, dtype=np.float64)
        imag = np.asarray(mode.imag, dtype=np.float64)
        if np.linalg.norm(real) > 0.0:
            columns.append(real)
        if np.linalg.norm(imag) > 0.0:
            columns.append(imag)

    if not columns:
        return np.empty((modes.shape[0], 0), dtype=np.float64)

    raw_basis = np.column_stack(columns)
    u, s, _ = np.linalg.svd(raw_basis, full_matrices=False)
    if len(s) == 0 or s[0] == 0.0:
        return np.empty((modes.shape[0], 0), dtype=np.float64)

    cutoff = rcond * s[0]
    rank = int(np.count_nonzero(s > cutoff))
    return u[:, :rank]


def build_projected_run(
    cfg: SimConfig,
    project_dir: Path,
) -> tuple[MacState, ForceProjection, ProjectionInfo]:
    """Load steady state/eigenpairs and build the forcing projector."""
    state_path = resolve_projected_state_path(project_dir, cfg)
    eigenpairs_path = resolve_projected_eigenpairs_path(project_dir, cfg)
    mac_state, _ = load_mac_state_pickle(state_path, cfg, check_dt=False)
    eigen_data = _load_eigenpairs(eigenpairs_path, cfg)

    eigenvalues = np.asarray(eigen_data["eigenvalues"], dtype=np.complex128)
    eigenvectors = np.asarray(eigen_data["eigenvectors"], dtype=np.complex128)
    nu_u = (cfg.nx - 1) * cfg.ny
    nv_u = cfg.nx * (cfg.ny - 1)
    velocity_size = nu_u + nv_u
    expected_size = velocity_size + cfg.nx * cfg.ny
    if eigenvectors.shape[0] != expected_size:
        raise ValueError(
            f"{eigenpairs_path} eigenvectors must have {expected_size} rows "
            f"([u_vec, v_vec, p]), got {eigenvectors.shape[0]}"
        )

    selected = np.flatnonzero(eigenvalues.real > cfg.projected_real_threshold)
    selected_vectors = eigenvectors[:velocity_size, selected]
    basis = _real_basis_from_modes(selected_vectors, cfg.projected_projection_rcond)
    projector = ForceProjection(cfg, basis)

    fu0, fv0 = cfg.make_force_arrays(t=0.0)
    force0 = projector.vector_from_arrays(fu0, fv0)
    removed0 = projector.project(force0)
    remaining0 = force0 - removed0
    info = ProjectionInfo(
        state_path=state_path,
        eigenpairs_path=eigenpairs_path,
        real_threshold=cfg.projected_real_threshold,
        selected_indices=selected,
        selected_eigenvalues=eigenvalues[selected],
        basis_rank=basis.shape[1],
        force_norm=float(np.linalg.norm(force0)),
        removed_norm=float(np.linalg.norm(removed0)),
        remaining_norm=float(np.linalg.norm(remaining0)),
    )
    return mac_state, projector, info


def _oblique_projector(
    eigenvalues: np.ndarray,
    right: np.ndarray,
    left: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Real Q, W with W^T Q = I spanning the given (complex) modes.

    A complex pair enters through one member (Im lambda > 0) as
    [Re q, Im q]; Pi = Q W^T is then the real oblique projector onto the
    invariant subspace along the complementary one (left modes).
    """
    q_cols: list[np.ndarray] = []
    a_cols: list[np.ndarray] = []
    for k, lam in enumerate(eigenvalues):
        if lam.imag < 0.0 and np.any(np.isclose(eigenvalues, np.conj(lam))):
            continue  # conjugate partner already represented
        q_cols.append(right[:, k].real)
        a_cols.append(left[:, k].real)
        if lam.imag != 0.0:
            q_cols.append(right[:, k].imag)
            a_cols.append(left[:, k].imag)
    if not q_cols:
        empty = np.empty((right.shape[0], 0))
        return empty, empty
    q_mat = np.column_stack(q_cols)
    a_mat = np.column_stack(a_cols)
    w_mat = a_mat @ np.linalg.inv(q_mat.T @ a_mat)
    return q_mat, w_mat


def build_feedback_stabilization(
    cfg: SimConfig,
    project_dir: Path,
) -> tuple[MacState, FeedbackStabilization, ProjectionInfo]:
    """Load steady state/eigenpairs/adjoints and build the feedback projector."""
    state_path = resolve_projected_state_path(project_dir, cfg)
    eigenpairs_path = resolve_projected_eigenpairs_path(project_dir, cfg)
    mac_state, _ = load_mac_state_pickle(state_path, cfg, check_dt=False)
    eigen_data = _load_eigenpairs(eigenpairs_path, cfg)
    if "adjoint_vectors" not in eigen_data:
        raise ValueError(
            f"{eigenpairs_path} has no adjoint modes; rerun `make linearize` "
            "(needed for projected_run.method: feedback_*)"
        )

    eigenvalues = np.asarray(eigen_data["eigenvalues"], dtype=np.complex128)
    eigenvectors = np.asarray(eigen_data["eigenvectors"], dtype=np.complex128)
    adjoints = np.asarray(eigen_data["adjoint_vectors"], dtype=np.complex128)
    velocity_size = (cfg.nx - 1) * cfg.ny + cfg.nx * (cfg.ny - 1)

    selected = np.flatnonzero(eigenvalues.real > cfg.projected_real_threshold)
    basis, adjoint = _oblique_projector(
        eigenvalues[selected],
        eigenvectors[:velocity_size, selected],
        adjoints[:, selected],
    )
    if basis.shape[1] == 0:
        raise ValueError(
            f"No eigenvalue with Re(lambda) > {cfg.projected_real_threshold} in "
            f"{eigenpairs_path}; nothing to stabilise"
        )
    u_steady = np.concatenate([mac_state.u_vec, mac_state.v_vec])
    stabilization = FeedbackStabilization(
        basis=basis,
        adjoint=adjoint,
        u_ref=u_steady if cfg.projected_feedback_projection == "deviation" else np.zeros_like(u_steady),
        u_steady=u_steady,
        alpha=cfg.projected_feedback_alpha,
        projection=cfg.projected_feedback_projection,
    )
    info = ProjectionInfo(
        state_path=state_path,
        eigenpairs_path=eigenpairs_path,
        real_threshold=cfg.projected_real_threshold,
        selected_indices=selected,
        selected_eigenvalues=eigenvalues[selected],
        basis_rank=basis.shape[1],
        force_norm=0.0,
        removed_norm=0.0,
        remaining_norm=0.0,
        method=cfg.projected_method,
        feedback_alpha=cfg.projected_feedback_alpha,
    )
    return mac_state, stabilization, info


def save_projection_info(info: ProjectionInfo, out_dir: Path) -> Path:
    """Save projected-run projection metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "projection.pkl"
    payload = {
        "state_path": str(info.state_path),
        "eigenpairs_path": str(info.eigenpairs_path),
        "real_threshold": info.real_threshold,
        "selected_indices": info.selected_indices,
        "selected_eigenvalues": info.selected_eigenvalues,
        "basis_rank": info.basis_rank,
        "force_norm": info.force_norm,
        "removed_norm": info.removed_norm,
        "remaining_norm": info.remaining_norm,
        "method": info.method,
        "feedback_alpha": info.feedback_alpha,
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path
