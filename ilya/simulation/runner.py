"""Time-integration runner for the Stokes MAC solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

from solver.config import SimConfig, Snapshot
from solver.lib import StokesMACLib

console = Console()


@dataclass
class SimulationResult:
    """Full set of outputs collected during one simulation run."""

    snapshots: list[Snapshot]
    t_history: list[float]
    div_history: list[float]
    velocity_change_history: list[float]


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
) -> SimulationResult:
    """Run the time integration using batch C++ steps."""
    snapshots: list[Snapshot] = []
    t_history: list[float] = []
    div_history: list[float] = []
    velocity_change_history: list[float] = []
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
        task = progress.add_task("Simulation", total=n_batches, info="starting…")
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

            step_done = 0
            for batch_start in range(0, cfg.n_steps, cfg.frame_every):
                batch_n = min(cfg.frame_every, cfg.n_steps - batch_start)
                t_start = batch_start * cfg.dt

                if cfg.has_forcing:
                    fu, fv = cfg.make_force_arrays(t=t_start)
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
                t_now = step_done * cfg.dt

                t_history.extend((batch_start + k + 1) * cfg.dt for k in range(batch_n))
                div_history.extend(divs.tolist())
                velocity_change_history.extend(changes.tolist())

                p, u, v = solver.get_fields()
                uc, vc = _cell_centred_velocity(u, v)
                omega = _vorticity(uc, vc, xc, yc)
                snapshots.append(
                    Snapshot(
                        step=step_done,
                        t=t_now,
                        p=p.astype(np.float32),
                        uc=uc.astype(np.float32),
                        vc=vc.astype(np.float32),
                        omega=omega.astype(np.float32),
                    )
                )

                vel_change = float(changes[-1]) if len(changes) else None
                if vel_change is not None and cfg.conv_tol > 0 and vel_change < cfg.conv_tol:
                    progress.update(task, info=f"converged Δu={vel_change:.1e}")
                    progress.stop()
                    console.print(
                        f"[green]✓ Converged[/green] at step [bold]{step_done}[/bold]  "
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

    if not converged:
        tol_str = "disabled" if cfg.conv_tol == 0 else "not reached"
        console.print(f"[dim]Reached t_end={cfg.t_end:.3f}  (tol {tol_str})[/dim]")

    return SimulationResult(
        snapshots=snapshots,
        t_history=t_history,
        div_history=div_history,
        velocity_change_history=velocity_change_history,
    )
