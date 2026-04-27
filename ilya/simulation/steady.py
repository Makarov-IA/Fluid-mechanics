"""Steady Navier-Stokes solve through the C++ Newton-GMRES backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from rich.console import Console

from solver.config import MacState, SimConfig, Snapshot
from solver.lib import StokesMACLib, find_solver_lib
from solver.state_io import load_mac_state_pickle

console = Console()

_TIME_VAR_RE = re.compile(r"(?<![A-Za-z0-9_])t(?![A-Za-z0-9_])")
_PROJECT_DIR = Path(__file__).parent.parent
_STEADY_GUESS_PATH = _PROJECT_DIR / "plots" / "run" / "fixed_time_state" / "state_internal.pkl"

_STOP_REASONS = {
    1: "residual tolerance reached",
    2: "GMRES failed while solving the Newton correction",
    3: "Newton line search could not reduce ||G(U)||; saving the last accepted state",
    4: "Fixed-point solver reached the Newton iteration limit; saving the last accepted state",
}


@dataclass
class SteadySolveResult:
    """Outputs of the fixed-point steady solve."""

    snapshot: Snapshot
    mac_state: MacState
    max_div: float
    newton_iters: int
    residual_inf: float
    converged: bool
    stop_reason: str
    iterate_change_inf: list[float]


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


def _load_initial_guess(cfg: SimConfig) -> tuple[MacState, str]:
    """Load exact internal MAC state saved by simulation mode."""
    try:
        mac_state, _ = load_mac_state_pickle(_STEADY_GUESS_PATH, cfg)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Initial guess pickle not found: {_STEADY_GUESS_PATH}. "
            "Run simulation mode first so it writes "
            "plots/run/fixed_time_state/state_internal.pkl."
        ) from exc
    return mac_state, str(_STEADY_GUESS_PATH)


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


def solve_steady(
    cfg: SimConfig,
    xc: np.ndarray,
    yc: np.ndarray,
) -> SteadySolveResult:
    """Solve G(U)=0 with C++ damped Newton-GMRES on the discrete step map."""
    _validate_time_independent(cfg)

    lib_path = find_solver_lib(_PROJECT_DIR)
    fu0, fv0 = cfg.make_force_arrays(t=0.0)
    initial_state, guess_desc = _load_initial_guess(cfg)
    console.print(f"  initial guess: [dim]{guess_desc}[/dim]")

    with StokesMACLib(lib_path, cfg.nx, cfg.ny, cfg.lx, cfg.ly, cfg.nu, cfg.dt) as solver:
        solver.set_bc_arrays(cfg.make_bc_arrays())
        solver.set_state(initial_state.u_vec, initial_state.v_vec, initial_state.p)
        (
            newton_iters,
            residual_inf,
            max_div,
            converged,
            stop_code,
            iterate_change,
        ) = solver.solve_steady_newton(
            fu0,
            fv0,
            cfg.steady_max_newton_iters,
            cfg.steady_residual_tol,
            cfg.steady_krylov_tol,
            cfg.steady_krylov_maxiter,
            cfg.steady_krylov_restart,
            cfg.steady_jacobian_rdiff,
            cfg.steady_line_search,
            cfg.steady_min_step,
        )

        p_mac, u_mac, v_mac = solver.get_fields()
        u_final, v_final, p_cells = solver.get_state()

    snap = _snapshot_from_fields(cfg, p_mac, u_mac, v_mac, xc, yc, newton_iters)
    return SteadySolveResult(
        snapshot=snap,
        mac_state=MacState(u_vec=u_final, v_vec=v_final, p=p_cells),
        max_div=max_div,
        newton_iters=newton_iters,
        residual_inf=residual_inf,
        converged=converged,
        stop_reason=_STOP_REASONS.get(
            stop_code,
            f"C++ steady backend stopped with code {stop_code}",
        ),
        iterate_change_inf=iterate_change.tolist(),
    )
