"""2-D Navier-Stokes solver — entry point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from simulation.runner import run_simulation
from solver.config import SimConfig
from solver.lib import find_solver_lib
from viz.levels import compute_colour_levels
from viz.plots import save_divergence_plot, save_final_figure, save_state_csv
from viz.video import render_videos

console = Console()


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = SimConfig.from_yaml(config_path)

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="bold cyan")
    tbl.add_column(style="white")
    tbl.add_row("Domain",     f"{cfg.lx} × {cfg.ly}")
    tbl.add_row("Grid",       f"{cfg.nx} × {cfg.ny}")
    tbl.add_row("ν",          f"{cfg.nu}")
    tbl.add_row("t_end",      f"{cfg.t_end}")
    tbl.add_row("dt",         f"{cfg.dt:.2e}")
    tbl.add_row("n_steps",    f"{cfg.n_steps:,}")
    tbl.add_row("video_fps",  f"{cfg.video_fps} fps  (speed {cfg.video_speed}×,  {cfg.frame_every:,} steps/frame)")
    tbl.add_row("video_speed", f"{cfg.video_speed}× real time")
    conv = f"{cfg.conv_tol:.1e}" if cfg.conv_tol > 0 else "disabled"
    tbl.add_row("conv_tol",   conv)
    console.print(Panel(tbl, title="[bold]Simulation config[/bold]", expand=False))

    lib_path = find_solver_lib(Path(__file__).parent)
    console.print(f"  Library: [dim]{lib_path.name}[/dim]")

    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    xc = (np.arange(cfg.nx) + 0.5) * (cfg.lx / cfg.nx)
    yc = (np.arange(cfg.ny) + 0.5) * (cfg.ly / cfg.ny)

    snapshots, t_hist, div_hist = run_simulation(cfg, lib_path, xc, yc)
    console.print(f"  [green]✓[/green] {len(snapshots)} snapshots collected")

    sl, pl, ol, speed_max = compute_colour_levels(snapshots)

    with console.status("[cyan]Saving plots…[/cyan]"):
        div_path = save_divergence_plot(t_hist, div_hist, out_dir)
        final_paths = save_final_figure(snapshots[-1], cfg, xc, yc, out_dir, sl, pl, ol)
        csv_path = out_dir / "final_state" / "state.csv"
        save_state_csv(snapshots[-1], xc, yc, csv_path)

    video_paths = render_videos(snapshots, cfg, out_dir, xc, yc, sl, pl, ol, speed_max)

    out_tbl = Table(show_header=False, box=None, padding=(0, 2))
    out_tbl.add_column(style="bold green")
    out_tbl.add_column(style="dim")
    out_tbl.add_row("✓ divergence", str(div_path))
    for p in final_paths:
        out_tbl.add_row(f"✓ final/{p.name}", str(p))
    for kind, p in video_paths.items():
        out_tbl.add_row(f"✓ {kind}", str(p))
    console.print(Panel(out_tbl, title="[bold]Saved files[/bold]", expand=False))


if __name__ == "__main__":
    main()
