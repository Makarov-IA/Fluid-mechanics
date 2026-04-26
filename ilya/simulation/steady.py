"""Steady Navier-Stokes solve via fixed-point Newton-GMRES."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from rich.console import Console
import scipy.sparse.linalg as spla

from solver.config import SimConfig, Snapshot
from solver.lib import StokesMACLib, find_solver_lib

console = Console()

_TIME_VAR_RE = re.compile(r"(?<![A-Za-z0-9_])t(?![A-Za-z0-9_])")
_PROJECT_DIR = Path(__file__).parent.parent
_STEADY_GUESS_CSV = _PROJECT_DIR / "plots" / "fixed_time_state" / "state.csv"


@dataclass
class ResidualEval:
    """One residual evaluation G(U) = Phi(U) - U."""

    residual: np.ndarray
    residual_inf: float
    next_state: np.ndarray
    p_cells: np.ndarray
    p_mac: np.ndarray
    u_mac: np.ndarray
    v_mac: np.ndarray
    max_div: float


@dataclass
class SteadySolveResult:
    """Outputs of the fixed-point steady solve."""

    snapshot: Snapshot
    max_div: float
    newton_iters: int
    residual_inf: float
    converged: bool
    stop_reason: str
    iterate_change_inf: list[float]


def _join_state(u_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Concatenate interior u- and v-unknowns into one nonlinear state vector."""
    return np.concatenate([u_vec, v_vec])


def _split_state(state: np.ndarray, nu_u: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the nonlinear state into interior u- and v-unknown blocks."""
    return state[:nu_u], state[nu_u:]


def _uses_time(expr: str | None) -> bool:
    """Return True when the symbolic expression depends on time."""
    return expr is not None and bool(_TIME_VAR_RE.search(expr))


def _validate_time_independent(cfg: SimConfig) -> None:
    """Steady mode requires time-independent forcing and boundary conditions."""
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
            "steady mode requires time-independent forcing/BCs; "
            f"'t' appears in: {names}"
        )


def _build_reconstruction_matrix(n_cells: int) -> np.ndarray:
    """Map interior face values to cell-centred averages for one line."""
    mat = np.zeros((n_cells, n_cells - 1), dtype=np.float64)
    if n_cells <= 1:
        return mat
    mat[0, 0] = 1.0
    for i in range(1, n_cells - 1):
        mat[i, i - 1] = 1.0
        mat[i, i] = 1.0
    mat[n_cells - 1, n_cells - 2] = 1.0
    return mat


def _reconstruct_u_interior(
    uc: np.ndarray,
    bc_u_left: np.ndarray,
    bc_u_right: np.ndarray,
) -> np.ndarray:
    """Recover interior u-face values from cell-centred u and wall BCs."""
    nx, ny = uc.shape
    mat = _build_reconstruction_matrix(nx)
    u_face = np.zeros((nx + 1, ny), dtype=np.float64)
    u_face[0, :] = bc_u_left
    u_face[nx, :] = bc_u_right

    for j in range(ny):
        rhs = 2.0 * uc[:, j].astype(np.float64, copy=True)
        rhs[0] -= u_face[0, j]
        rhs[-1] -= u_face[nx, j]
        u_face[1:nx, j] = np.linalg.lstsq(mat, rhs, rcond=None)[0]

    return u_face[1:nx, :].reshape(-1, order="F")


def _reconstruct_v_interior(
    vc: np.ndarray,
    bc_v_bot: np.ndarray,
    bc_v_top: np.ndarray,
) -> np.ndarray:
    """Recover interior v-face values from cell-centred v and wall BCs."""
    nx, ny = vc.shape
    mat = _build_reconstruction_matrix(ny)
    v_face = np.zeros((nx, ny + 1), dtype=np.float64)
    v_face[:, 0] = bc_v_bot
    v_face[:, ny] = bc_v_top

    for i in range(nx):
        rhs = 2.0 * vc[i, :].astype(np.float64, copy=True)
        rhs[0] -= v_face[i, 0]
        rhs[-1] -= v_face[i, ny]
        v_face[i, 1:ny] = np.linalg.lstsq(mat, rhs, rcond=None)[0]

    return v_face[:, 1:ny].reshape(-1, order="F")


def _load_state_csv(path: Path, cfg: SimConfig) -> np.ndarray:
    """Load a cell-centred CSV initial guess and reconstruct interior faces."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape != (cfg.nx * cfg.ny, 5):
        raise ValueError(
            f"{path} does not match expected state.csv shape {(cfg.nx * cfg.ny, 5)}"
        )

    uc = data[:, 2].reshape(cfg.nx, cfg.ny)
    vc = data[:, 3].reshape(cfg.nx, cfg.ny)
    bcs = cfg.make_bc_arrays()
    u_vec = _reconstruct_u_interior(uc, bcs["u_left"], bcs["u_right"])
    v_vec = _reconstruct_v_interior(vc, bcs["v_bot"], bcs["v_top"])
    return _join_state(u_vec, v_vec)


def _load_initial_guess(cfg: SimConfig) -> tuple[np.ndarray, str]:
    """Load the only supported steady initial guess: nearest-time simulation state."""
    if not _STEADY_GUESS_CSV.exists():
        raise FileNotFoundError(
            f"Initial guess CSV not found: {_STEADY_GUESS_CSV}. "
            "Run simulation mode first so it writes plots/fixed_time_state/state.csv."
        )
    return _load_state_csv(_STEADY_GUESS_CSV, cfg), str(_STEADY_GUESS_CSV)


def _snapshot_from_fields(
    cfg: SimConfig,
    p_mac: np.ndarray,
    u_mac: np.ndarray,
    v_mac: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    newton_iters: int,
) -> Snapshot:
    """Convert MAC-grid fields to the plotting snapshot format."""
    uc = 0.5 * (u_mac[:-1, :] + u_mac[1:, :])
    vc = 0.5 * (v_mac[:, :-1] + v_mac[:, 1:])
    omega = np.gradient(vc, xc, axis=0) - np.gradient(uc, yc, axis=1)
    return Snapshot(
        step=newton_iters,
        t=0.0,
        p=p_mac.astype(np.float32),
        uc=uc.astype(np.float32),
        vc=vc.astype(np.float32),
        omega=omega.astype(np.float32),
    )


class FixedPointMap:
    """Wrap the one-step Navier-Stokes map Φ(U)."""

    def __init__(
        self,
        solver: StokesMACLib,
        fu: np.ndarray | None,
        fv: np.ndarray | None,
    ) -> None:
        self.solver = solver
        self.fu = fu
        self.fv = fv
        self.nu_u = solver.nu_u

    def evaluate(self, state: np.ndarray) -> ResidualEval:
        """Evaluate G(U) = Φ(U) - U for one interior-velocity state vector."""
        u_vec, v_vec = _split_state(state, self.nu_u)
        self.solver.set_state(u_vec, v_vec)
        divs = self.solver.run_steps_with_force(0.0, 1, self.fu, self.fv)
        max_div = float(divs[-1])
        p_mac, u_mac, v_mac = self.solver.get_fields()
        u_next, v_next, p_cells = self.solver.get_state()
        next_state = _join_state(u_next, v_next)

        if not np.isfinite(next_state).all() or not np.isfinite(max_div):
            raise RuntimeError("Solver produced NaN/Inf during fixed-point evaluation")

        residual = next_state - state
        return ResidualEval(
            residual=residual,
            residual_inf=float(np.max(np.abs(residual))),
            next_state=next_state,
            p_cells=p_cells,
            p_mac=p_mac,
            u_mac=u_mac,
            v_mac=v_mac,
            max_div=max_div,
        )


def _jacobian_vec(
    fp_map: FixedPointMap,
    state: np.ndarray,
    base_residual: np.ndarray,
    vec: np.ndarray,
    rdiff: float,
) -> np.ndarray:
    """Finite-difference product [DG(U)] v."""
    vec_inf = float(np.max(np.abs(vec)))
    if vec_inf < 1e-14:
        return np.zeros_like(vec)

    state_inf = max(float(np.max(np.abs(state))), 1.0)
    eps = rdiff * state_inf / vec_inf
    trial_eval = fp_map.evaluate(state + eps * vec)
    return (trial_eval.residual - base_residual) / eps


def solve_steady(
    cfg: SimConfig,
    xc: np.ndarray,
    yc: np.ndarray,
) -> SteadySolveResult:
    """Solve G(U)=0 with damped Newton-GMRES on the discrete step map."""
    _validate_time_independent(cfg)

    lib_path = find_solver_lib(_PROJECT_DIR)
    fu0, fv0 = cfg.make_force_arrays(t=0.0)
    state, guess_desc = _load_initial_guess(cfg)

    with StokesMACLib(lib_path, cfg.nx, cfg.ny, cfg.lx, cfg.ly, cfg.nu, cfg.dt) as solver:
        solver.set_bc_arrays(cfg.make_bc_arrays())
        fp_map = FixedPointMap(solver, fu0, fv0)
        iterate_change_inf: list[float] = []
        converged = False
        stop_reason = "unknown"

        current = fp_map.evaluate(state)
        console.print(
            f"  initial guess: [dim]{guess_desc}[/dim]   "
            f"||G(U)||∞ = {current.residual_inf:.2e}"
        )

        newton_iters = 0
        for k in range(1, cfg.steady_max_newton_iters + 1):
            if current.residual_inf < cfg.steady_residual_tol:
                converged = True
                stop_reason = "residual tolerance reached"
                break

            base_state = state.copy()
            base_residual = current.residual.copy()

            linop = spla.LinearOperator(
                (base_state.size, base_state.size),
                matvec=lambda vec: _jacobian_vec(
                    fp_map,
                    base_state,
                    base_residual,
                    vec,
                    cfg.steady_jacobian_rdiff,
                ),
                dtype=np.float64,
            )

            gmres_hist: list[float] = []
            delta, gmres_info = spla.gmres(
                linop,
                -base_residual,
                atol=0.0,
                rtol=cfg.steady_krylov_tol,
                restart=min(cfg.steady_krylov_restart, base_state.size),
                maxiter=cfg.steady_krylov_maxiter,
                callback=gmres_hist.append,
                callback_type="pr_norm",
            )

            if gmres_info < 0 or not np.isfinite(delta).all():
                stop_reason = "GMRES failed while solving the Newton correction"
                console.print(f"[yellow]⚠ {stop_reason}[/yellow]")
                break

            alpha = 1.0
            accepted_state: np.ndarray | None = None
            accepted_eval: ResidualEval | None = None

            if cfg.steady_line_search == "none":
                trial_state = base_state + delta
                accepted_state = trial_state
                accepted_eval = fp_map.evaluate(trial_state)

            while alpha >= cfg.steady_min_step:
                if accepted_state is not None and accepted_eval is not None:
                    break

                trial_state = base_state + alpha * delta
                trial_eval = fp_map.evaluate(trial_state)
                target = (1.0 - 1e-4 * alpha) * current.residual_inf

                if trial_eval.residual_inf < target:
                    accepted_state = trial_state
                    accepted_eval = trial_eval
                    break

                alpha *= 0.5

            if accepted_state is None or accepted_eval is None:
                stop_reason = (
                    "Newton line search could not reduce ||G(U)||; "
                    "saving the last accepted state"
                )
                console.print(f"[yellow]⚠ {stop_reason}[/yellow]")
                break

            state = accepted_state
            current = accepted_eval
            newton_iters = k
            change_inf = float(np.max(np.abs(accepted_state - base_state)))
            iterate_change_inf.append(change_inf)

            gmres_tail = gmres_hist[-1] if gmres_hist else float("nan")
            gmres_msg = "ok" if gmres_info == 0 else f"info={gmres_info}"
            console.print(
                f"  Newton {k:02d}: ||G||∞ {np.max(np.abs(base_residual)):.2e}"
                f" → {current.residual_inf:.2e}   "
                f"||ΔU||∞={change_inf:.2e}   α={alpha:.3f}   "
                f"GMRES {gmres_msg} ({gmres_tail:.1e})"
            )

        if not converged and stop_reason == "unknown":
            stop_reason = (
                "Fixed-point solver reached the Newton iteration limit; "
                "saving the last accepted state"
            )
            console.print(f"[yellow]⚠ {stop_reason}[/yellow]")

        snap = _snapshot_from_fields(
            cfg,
            current.p_mac,
            current.u_mac,
            current.v_mac,
            xc,
            yc,
            newton_iters,
        )
        return SteadySolveResult(
            snapshot=snap,
            max_div=current.max_div,
            newton_iters=newton_iters,
            residual_inf=current.residual_inf,
            converged=converged,
            stop_reason=stop_reason,
            iterate_change_inf=iterate_change_inf,
        )
