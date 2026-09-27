"""Time-integration runner for the Stokes MAC solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from solver.config import MacState, SimConfig, Snapshot
from solver.lib import StokesMACLib

if TYPE_CHECKING:
    from simulation.projected_run import FeedbackStabilization

console = Console()


@dataclass
class SimulationResult:
    """Full set of outputs collected during one simulation run."""

    snapshots: list[Snapshot]
    mac_states: list[MacState]
    t_history: list[float]
    div_history: list[float]
    velocity_change_history: list[float]
    # Feedback stabilisation only: per step ||delta_n||_inf and ||u^n - u*||_inf
    correction_history: list[float] = field(default_factory=list)
    deviation_history: list[float] = field(default_factory=list)
    # ||f_c||_2 / ||F + f_c||_2 per step, f_c = -delta_n / dt
    force_ratio_history: list[float] = field(default_factory=list)


def _cell_centred_velocity(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average face-centred MAC velocities to cell centres."""
    return 0.5 * (u[:-1, :] + u[1:, :]), 0.5 * (v[:, :-1] + v[:, 1:])


def _vorticity(
    uc: np.ndarray,
    vc: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
) -> np.ndarray:
    """Scalar vorticity  ω = ∂v/∂x − ∂u/∂y  on cell-centred coordinates."""
    return np.gradient(vc, xc, axis=0) - np.gradient(uc, yc, axis=1)


def _raise_if_solver_diverged(
    divs: np.ndarray,
    changes: np.ndarray,
    batch_start: int,
    dt: float,
) -> None:
    """Raise an informative error when the solver produces NaN/Inf diagnostics."""
    bad_divs = ~np.isfinite(divs)
    bad_changes = ~np.isfinite(changes)
    if not bad_divs.any() and not bad_changes.any():
        return

    first_bad = len(divs)
    if bad_divs.any():
        first_bad = min(first_bad, int(np.argmax(bad_divs)))
    if bad_changes.any():
        first_bad = min(first_bad, int(np.argmax(bad_changes)))

    nan_step = batch_start + first_bad + 1
    raise RuntimeError(
        f"Solver diverged at step ~{nan_step} (t={nan_step * dt:.4f}). "
        f"CFL too large or Re too high for current grid/dt."
    )


def run_simulation(
    cfg: SimConfig,
    lib_path: Path,
    xc: np.ndarray,
    yc: np.ndarray,
    initial_state: MacState | None = None,
    initial_step: int = 0,
    initial_t: float = 0.0,
    force_modifier: Callable[
        [np.ndarray | None, np.ndarray | None],
        tuple[np.ndarray | None, np.ndarray | None],
    ]
    | None = None,
    description: str = "Simulation",
    stabilization: FeedbackStabilization | None = None,
) -> SimulationResult:
    """Run the time integration using batch C++ steps."""
    snapshots: list[Snapshot] = []
    mac_states: list[MacState] = []
    t_history: list[float] = []
    div_history: list[float] = []
    velocity_change_history: list[float] = []
    correction_history: list[float] = []
    deviation_history: list[float] = []
    force_ratio_history: list[float] = []
    converged = False
    n_batches = -(-cfg.n_steps // cfg.frame_every)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=38),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[info]}"),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task(description, total=n_batches, info="starting…")
        with StokesMACLib(
            lib_path,
            cfg.nx,
            cfg.ny,
            cfg.lx,
            cfg.ly,
            cfg.nu,
            cfg.dt,
        ) as solver:
            solver.set_bc_arrays(cfg.make_bc_arrays())
            solver.set_linear_solver(
                cfg.linear_solver_method,
                cfg.linear_solver_fast_tol,
                cfg.linear_solver_fast_extrapolation,
                cfg.linear_solver_fast_parallel,
            )
            if initial_state is not None:
                solver.set_state(initial_state.u_vec, initial_state.v_vec, initial_state.p)
            if stabilization is not None:
                solver.set_stabilization(
                    stabilization.basis,
                    stabilization.adjoint,
                    stabilization.u_ref,
                    stabilization.alpha,
                    u_diag=stabilization.u_steady,
                )

            step_done = 0
            for batch_start in range(0, cfg.n_steps, cfg.frame_every):
                batch_n = min(cfg.frame_every, cfg.n_steps - batch_start)
                t_start = initial_t + batch_start * cfg.dt

                fu = fv = None
                if cfg.has_forcing:
                    fu, fv = cfg.make_force_arrays(t=t_start)
                if force_modifier is not None:
                    fu, fv = force_modifier(fu, fv)

                if fu is not None or fv is not None:
                    divs, changes = solver.run_steps_with_force_diagnostics(
                        t_start,
                        batch_n,
                        fu,
                        fv,
                    )
                else:
                    divs, changes = solver.run_steps_diagnostics(t_start, batch_n)

                _raise_if_solver_diverged(divs, changes, batch_start, cfg.dt)

                step_done += batch_n
                t_now = initial_t + step_done * cfg.dt

                t_history.extend(
                    initial_t + (batch_start + k + 1) * cfg.dt
                    for k in range(batch_n)
                )
                div_history.extend(divs.tolist())
                velocity_change_history.extend(changes.tolist())
                if stabilization is not None:
                    corr, dev, ratio = solver.take_control_history(batch_n)
                    correction_history.extend(corr.tolist())
                    deviation_history.extend(dev.tolist())
                    force_ratio_history.extend(ratio.tolist())

                p, u, v = solver.get_fields()
                u_vec, v_vec, p_vec = solver.get_state()
                uc, vc = _cell_centred_velocity(u, v)
                omega = _vorticity(uc, vc, xc, yc)
                snapshots.append(
                    Snapshot(
                        step=initial_step + step_done,
                        t=t_now,
                        p=p.astype(np.float32),
                        uc=uc.astype(np.float32),
                        vc=vc.astype(np.float32),
                        omega=omega.astype(np.float32),
                    )
                )
                mac_states.append(MacState(u_vec=u_vec, v_vec=v_vec, p=p_vec))

                vel_change = float(changes[-1]) if len(changes) else None
                if vel_change is not None and cfg.conv_tol > 0 and vel_change < cfg.conv_tol:
                    progress.update(task, info=f"converged Δu={vel_change:.1e}")
                    progress.stop()
                    console.print(
                        f"[green]✓ Converged[/green] at step "
                        f"[bold]{initial_step + step_done}[/bold]  "
                        f"t={t_now:.3f}  Δu={vel_change:.2e}"
                    )
                    converged = True
                    break

                du_str = f"  Δu={vel_change:.2e}" if vel_change is not None else ""
                progress.update(
                    task,
                    advance=1,
                    info=f"t={t_now:.2f}  |div|={divs[-1]:.2e}{du_str}",
                )

            fast_steps, fast_iters, fast_max_iters = solver.linear_solver_stats()

    if not converged:
        tol_str = "disabled" if cfg.conv_tol == 0 else "not reached"
        console.print(
            f"[dim]Reached t={initial_t + cfg.t_end:.3f}  "
            f"(advanced by {cfg.t_end:.3f}; tol {tol_str})[/dim]"
        )
    if fast_steps:
        console.print(
            f"[dim]Fast linear solver: {fast_iters / fast_steps:.2f} CG iterations/step "
            f"(max {fast_max_iters})[/dim]"
        )

    return SimulationResult(
        snapshots=snapshots,
        mac_states=mac_states,
        t_history=t_history,
        div_history=div_history,
        velocity_change_history=velocity_change_history,
        correction_history=correction_history,
        deviation_history=deviation_history,
        force_ratio_history=force_ratio_history,
    )
