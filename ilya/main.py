"""Entry point for time-dependent and steady Navier-Stokes solves."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from simulation.runner import run_simulation
from simulation.steady import solve_steady
from solver.config import SimConfig, Snapshot
from solver.lib import find_solver_lib
from viz.levels import compute_colour_levels
from viz.plots import (
    save_divergence_plot,
    save_final_figure,
    save_iterate_change_plot,
    save_mac_state_pickle,
    save_state_pickle,
    save_velocity_change_plot,
)
from viz.video import render_videos

console = Console()
PROJECT_DIR = Path(__file__).parent


def _cell_centres(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-centred x/y coordinates for the MAC pressure grid."""
    xc = (np.arange(cfg.nx) + 0.5) * (cfg.lx / cfg.nx)
    yc = (np.arange(cfg.ny) + 0.5) * (cfg.ly / cfg.ny)
    return xc, yc


def _nearest_snapshot(
    snapshots: list[Snapshot],
    t_target: float,
) -> tuple[int, Snapshot, float]:
    """Return the snapshot whose time is closest to t_target."""
    idx, snap = min(enumerate(snapshots), key=lambda item: abs(item[1].t - t_target))
    return idx, snap, abs(float(snap.t) - t_target)


def _print_simulation_config(cfg: SimConfig) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("Domain", f"{cfg.lx} × {cfg.ly}")
    table.add_row("Grid", f"{cfg.nx} × {cfg.ny}")
    table.add_row("ν", f"{cfg.nu}")
    table.add_row("t_end", f"{cfg.t_end}")
    table.add_row("dt", f"{cfg.dt:.2e}")
    table.add_row("n_steps", f"{cfg.n_steps:,}")
    table.add_row(
        "video",
        f"{cfg.video_fps} fps  (speed {cfg.video_speed}×, {cfg.frame_every:,} steps/frame)",
    )
    conv = f"{cfg.conv_tol:.1e}" if cfg.conv_tol > 0 else "disabled"
    table.add_row("conv_tol", conv)
    table.add_row("fixed state t", f"{cfg.fixed_time_state_t}")
    table.add_row(
        "ΔU(t) plot",
        "enabled" if cfg.save_velocity_change_plot else "disabled",
    )
    console.print(Panel(table, title="[bold]Simulation config[/bold]", expand=False))


def _print_steady_config(cfg: SimConfig) -> None:
    guess_path = PROJECT_DIR / "plots" / "fixed_time_state" / "state_internal.pkl"
    nu_u = (cfg.nx - 1) * cfg.ny
    nv_u = cfg.nx * (cfg.ny - 1)
    np_u = cfg.nx * cfg.ny

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("Domain", f"{cfg.lx} × {cfg.ly}")
    table.add_row("Grid", f"{cfg.nx} × {cfg.ny}")
    table.add_row("ν", f"{cfg.nu}")
    table.add_row("Unknowns", f"{nu_u + nv_u + np_u:,}  (u:{nu_u}, v:{nv_u}, p:{np_u})")
    table.add_row("dt", f"{cfg.dt:.2e}")
    table.add_row("guess file", str(guess_path))
    table.add_row("guess target", f"nearest snapshot to t = {cfg.fixed_time_state_t}")
    table.add_row(
        "Newton",
        f"{cfg.steady_max_newton_iters} iters, tol {cfg.steady_residual_tol:.1e}",
    )
    table.add_row(
        "GMRES",
        (
            f"rtol {cfg.steady_krylov_tol:.1e}, "
            f"maxiter {cfg.steady_krylov_maxiter}, "
            f"restart {cfg.steady_krylov_restart}"
        ),
    )
    console.print(
        Panel(
            table,
            title="[bold]Steady Navier-Stokes (fixed-point Newton-GMRES)[/bold]",
            expand=False,
        )
    )


def _run_simulation(cfg: SimConfig) -> None:
    _print_simulation_config(cfg)

    lib_path = find_solver_lib(PROJECT_DIR)
    console.print(f"  Library: [dim]{lib_path.name}[/dim]")

    out_dir = PROJECT_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    xc, yc = _cell_centres(cfg)

    result = run_simulation(cfg, lib_path, xc, yc)
    snapshots = result.snapshots
    console.print(f"  [green]✓[/green] {len(snapshots)} snapshots collected")

    final_snapshot = snapshots[-1]
    guess_idx, guess_snapshot, guess_dt = _nearest_snapshot(snapshots, cfg.fixed_time_state_t)
    speed_levels, p_levels, omega_levels = compute_colour_levels(snapshots)

    with console.status("[cyan]Saving plots…[/cyan]"):
        divergence_path = save_divergence_plot(result.t_history, result.div_history, out_dir)
        velocity_change_path: Path | None = None
        if cfg.save_velocity_change_plot:
            velocity_change_path = save_velocity_change_plot(
                result.t_history,
                result.velocity_change_history,
                out_dir,
            )

        final_paths = save_final_figure(
            final_snapshot,
            cfg,
            xc,
            yc,
            out_dir,
            speed_levels,
            p_levels,
            omega_levels,
            panel_subdir="final_state",
        )
        final_state_path = out_dir / "final_state" / "state.pkl"
        save_state_pickle(final_snapshot, xc, yc, final_state_path)

        guess_paths = save_final_figure(
            guess_snapshot,
            cfg,
            xc,
            yc,
            out_dir,
            speed_levels,
            p_levels,
            omega_levels,
            panel_subdir="fixed_time_state",
        )
        guess_state_path = out_dir / "fixed_time_state" / "state.pkl"
        guess_internal_path = out_dir / "fixed_time_state" / "state_internal.pkl"
        save_state_pickle(guess_snapshot, xc, yc, guess_state_path)
        save_mac_state_pickle(
            result.mac_states[guess_idx],
            cfg,
            guess_snapshot,
            guess_internal_path,
        )

    video_paths = render_videos(
        snapshots,
        cfg,
        out_dir,
        xc,
        yc,
        speed_levels,
        p_levels,
        omega_levels,
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold green")
    table.add_column(style="dim")
    table.add_row("✓ divergence", str(divergence_path))
    if velocity_change_path is not None:
        table.add_row("✓ velocity-change", str(velocity_change_path))
    for path in final_paths:
        table.add_row(f"✓ final/{path.name}", str(path))
    table.add_row("✓ final/state.pkl", str(final_state_path))
    for path in guess_paths:
        table.add_row(f"✓ fixed-time/{path.name}", str(path))
    table.add_row("✓ fixed-time/state.pkl", str(guess_state_path))
    table.add_row("✓ fixed-time/state_internal.pkl", str(guess_internal_path))
    table.add_row(
        "✓ steady guess target",
        (
            f"target t={cfg.fixed_time_state_t:.3f}  →  "
            f"snapshot t={guess_snapshot.t:.3f}  (|Δt|={guess_dt:.3f})"
        ),
    )
    for kind, path in video_paths.items():
        table.add_row(f"✓ {kind}", str(path))
    console.print(Panel(table, title="[bold]Saved files[/bold]", expand=False))


def _run_steady(cfg: SimConfig) -> None:
    _print_steady_config(cfg)

    out_dir = PROJECT_DIR / "plots" / "steady"
    out_dir.mkdir(parents=True, exist_ok=True)
    xc, yc = _cell_centres(cfg)

    result = solve_steady(cfg, xc, yc)
    snap = result.snapshot

    status_style = "green" if result.converged else "yellow"
    status_text = "converged" if result.converged else "stopped early"
    console.print(
        f"  [{status_style}]{status_text}[/{status_style}]  "
        f"max|div u| = {result.max_div:.2e}  "
        f"Newton iters = {result.newton_iters}  "
        f"||G(U)||∞ = {result.residual_inf:.2e}"
    )
    if not result.converged:
        console.print(f"  [yellow]reason:[/yellow] {result.stop_reason}")

    speed_levels, p_levels, omega_levels = compute_colour_levels([snap])

    with console.status("[cyan]Saving plots…[/cyan]"):
        final_paths = save_final_figure(
            snap,
            cfg,
            xc,
            yc,
            out_dir,
            speed_levels,
            p_levels,
            omega_levels,
            panel_subdir="",
        )
        state_path = out_dir / "state.pkl"
        internal_path = out_dir / "state_internal.pkl"
        change_plot_path = save_iterate_change_plot(result.iterate_change_inf, out_dir)
        save_state_pickle(snap, xc, yc, state_path)
        save_mac_state_pickle(result.mac_state, cfg, snap, internal_path)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold green")
    table.add_column(style="dim")
    for path in final_paths:
        table.add_row(f"✓ steady/{path.name}", str(path))
    table.add_row("✓ steady/state.pkl", str(state_path))
    table.add_row("✓ steady/state_internal.pkl", str(internal_path))
    table.add_row("✓ steady/steady_iterate_change.png", str(change_plot_path))
    console.print(Panel(table, title="[bold]Saved files[/bold]", expand=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="2-D Navier-Stokes solver")
    parser.add_argument(
        "--mode",
        choices=["simulation", "steady"],
        default="simulation",
        help="Select which solver workflow to run",
    )
    args = parser.parse_args()

    config_path = PROJECT_DIR / "config.yaml"
    cfg = SimConfig.from_yaml(config_path)

    if args.mode == "steady":
        _run_steady(cfg)
    else:
        _run_simulation(cfg)


if __name__ == "__main__":
    main()
