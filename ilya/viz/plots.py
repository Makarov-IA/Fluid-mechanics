"""Static plot generation helpers for simulation and steady-state outputs."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from solver.config import MacState, SimConfig, Snapshot

matplotlib.use("Agg")


def style_axes(ax, title: str, lx: float, ly: float) -> None:
    """Apply a consistent style to one field plot."""
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)


def fig_to_pil(fig, dpi: int = 110) -> Image.Image:
    """Render a matplotlib figure to a PIL image without touching disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).copy()
    buf.close()
    return image


def draw_streamlines(ax, fig, snap, xc, yc, x_grid, y_grid, speed_levels) -> None:
    """Draw velocity magnitude background and streamline overlay."""
    speed = np.hypot(snap.uc, snap.vc)
    bg = ax.contourf(x_grid, y_grid, speed, levels=speed_levels, cmap="viridis")
    fig.colorbar(bg, ax=ax, label="|u|")
    ax.streamplot(
        xc,
        yc,
        snap.uc.T,
        snap.vc.T,
        color="white",
        linewidth=0.8,
        density=1.5,
        arrowsize=0.9,
    )


def draw_pressure(ax, fig, snap, x_grid, y_grid, p_levels) -> None:
    """Draw pressure contours and filled levels."""
    contours = ax.contourf(x_grid, y_grid, snap.p, levels=p_levels, cmap="coolwarm")
    ax.contour(
        x_grid,
        y_grid,
        snap.p,
        levels=p_levels,
        colors="black",
        linewidths=0.25,
        alpha=0.7,
    )
    fig.colorbar(contours, ax=ax, label="p")


def draw_vorticity(ax, fig, snap, x_grid, y_grid, omega_levels) -> None:
    """Draw vorticity contours and filled levels."""
    contours = ax.contourf(
        x_grid,
        y_grid,
        snap.omega,
        levels=omega_levels,
        cmap="coolwarm",
    )
    ax.contour(
        x_grid,
        y_grid,
        snap.omega,
        levels=omega_levels,
        colors="black",
        linewidths=0.25,
        alpha=0.7,
    )
    fig.colorbar(contours, ax=ax, label="ω")


def find_vortex_centers(
    snap: Snapshot,
    xc: np.ndarray,
    yc: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Detect vortex centres via local extrema of the stream function ψ."""
    uc = snap.uc.astype(np.float64)
    vc = snap.vc.astype(np.float64)

    dy = float(yc[1] - yc[0])
    dx = float(xc[1] - xc[0])
    psi = 0.5 * (np.cumsum(uc * dy, axis=1) + np.cumsum(-vc * dx, axis=0))

    window = max(3, min(psi.shape) // 7)
    local_max = psi == ndi.maximum_filter(psi, size=window, mode="nearest")
    local_min = psi == ndi.minimum_filter(psi, size=window, mode="nearest")

    margin = 1
    for arr in (local_max, local_min):
        arr[:margin, :] = False
        arr[-margin:, :] = False
        arr[:, :margin] = False
        arr[:, -margin:] = False

    ccw = [(float(xc[i]), float(yc[j])) for i, j in zip(*np.where(local_max))]
    cw = [(float(xc[i]), float(yc[j])) for i, j in zip(*np.where(local_min))]
    return ccw, cw


def overlay_vortex_markers(
    ax,
    snap: Snapshot,
    xc: np.ndarray,
    yc: np.ndarray,
) -> tuple[bool, bool]:
    """Draw vortex-centre markers onto ax and report which classes were found."""
    ccw, cw = find_vortex_centers(snap, xc, yc)

    for x, y in ccw:
        ax.scatter(x, y, marker="+", c="red", s=150, linewidths=2.5, zorder=6)
        ax.annotate(
            f"({x:.2f}, {y:.2f})",
            xy=(x, y),
            xytext=(5, 4),
            textcoords="offset points",
            color="red",
            fontsize=7,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65, lw=0),
        )

    for x, y in cw:
        ax.scatter(x, y, marker="x", c="cyan", s=150, linewidths=2.5, zorder=6)
        ax.annotate(
            f"({x:.2f}, {y:.2f})",
            xy=(x, y),
            xytext=(5, 4),
            textcoords="offset points",
            color="cyan",
            fontsize=7,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0),
        )

    return bool(ccw), bool(cw)


def _make_legend_handles(has_ccw: bool, has_cw: bool) -> list[Line2D]:
    """Build legend handles for vortex markers."""
    handles: list[Line2D] = []
    if has_ccw:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="+",
                color="red",
                linestyle="none",
                markersize=10,
                markeredgewidth=2.0,
                label="Vortex ↺ — counterclockwise (CCW)",
            )
        )
    if has_cw:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="cyan",
                linestyle="none",
                markersize=10,
                markeredgewidth=2.0,
                label="Vortex ↻ — clockwise (CW)",
            )
        )
    return handles


def _downsample_history(
    x: list[float],
    y: list[float],
    max_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Downsample long history arrays for plotting without changing trends."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    stride = max(1, len(x_arr) // max_points)
    return x_arr[::stride], y_arr[::stride], stride


def save_divergence_plot(
    t_history: list[float],
    div_history: list[float],
    out_dir: Path,
) -> Path:
    """Save a time-series plot of max |div u|."""
    t_plot, div_plot, stride = _downsample_history(t_history, div_history)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(t_plot, div_plot, color="#0d47a1", linewidth=1.0)
    ax.set_title("Max divergence vs time")
    ax.set_xlabel("t")
    ax.set_ylabel("max |div u|")
    ax.grid(True, alpha=0.35)
    if len(t_history) > stride:
        ax.text(
            0.98,
            0.97,
            f"(every {stride}th point shown)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="gray",
        )
    fig.tight_layout()
    path = out_dir / "stokes_max_divergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_velocity_change_plot(
    t_history: list[float],
    change_history: list[float],
    out_dir: Path,
) -> Path:
    """Save the time-scaled velocity change ||U_n - U_{n-1}||_inf / Δt."""
    path = out_dir / "stokes_velocity_change.png"
    t_plot = np.asarray(t_history, dtype=np.float64)
    change_plot = np.asarray(change_history, dtype=np.float64)
    cm_per_second = 3.5
    seconds_span = float(t_plot[-1] - t_plot[0]) if len(t_plot) > 1 else 1.0
    width_in = max(8.0, seconds_span * cm_per_second / 2.54)

    fig, ax = plt.subplots(figsize=(width_in, 4.5))
    if len(change_plot) > 0:
        step_dt = np.diff(np.concatenate(([0.0], t_plot)))
        if np.any(step_dt <= 0.0):
            positive_dt = step_dt[step_dt > 0.0]
            fallback_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0
            step_dt = np.where(step_dt > 0.0, step_dt, fallback_dt)
        change_rate = change_plot / step_dt
        ax.semilogy(
            t_plot,
            np.maximum(change_rate, 1e-30),
            color="#ad1457",
            linewidth=1.2,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No solver-step history available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    ax.set_title(r"Velocity Change Rate vs Time  $\|U_n-U_{n-1}\|_\infty / \Delta t$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|U_n-U_{n-1}\|_\infty / \Delta t$")
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_iterate_change_plot(
    change_history: list[float],
    out_dir: Path,
) -> Path:
    """Save ||U^{n+1} - U^n||_inf versus steady-iteration index n."""
    path = out_dir / "steady_iterate_change.png"
    indices = np.arange(len(change_history), dtype=int)
    values = np.asarray(change_history, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if len(values) == 0:
        ax.text(
            0.5,
            0.5,
            "No accepted Newton updates",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    else:
        ax.semilogy(
            indices,
            np.maximum(values, 1e-30),
            marker="o",
            color="#c62828",
            linewidth=1.2,
        )
        x_max = float(indices[-1]) if len(indices) > 1 else float(indices[0] + 1)
        ax.set_xlim(float(indices[0]), x_max)

    ax.set_title(r"Steady Iteration Change  $\|U^{n+1}-U^n\|_\infty$")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\|U^{n+1}-U^n\|_\infty$")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_final_figure(
    snap: Snapshot,
    cfg: SimConfig,
    xc: np.ndarray,
    yc: np.ndarray,
    out_dir: Path,
    speed_levels: np.ndarray,
    p_levels: np.ndarray,
    omega_levels: np.ndarray,
    panel_subdir: str = "final_state",
) -> list[Path]:
    """Save each state panel as a separate PNG file."""
    x_grid, y_grid = np.meshgrid(xc, yc, indexing="ij")
    panel_dir = out_dir / panel_subdir if panel_subdir else out_dir
    panel_dir.mkdir(parents=True, exist_ok=True)

    suptitle = f"State   ν = {cfg.nu},  grid {cfg.nx}×{cfg.ny},  t = {snap.t:.2f}"
    panels = [
        ("streamlines", "Streamlines", speed_levels),
        ("pressure", "Pressure", p_levels),
        ("vorticity", "Vorticity", omega_levels),
    ]

    saved: list[Path] = []
    for kind, title, levels in panels:
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.suptitle(suptitle, fontsize=12)

        if kind == "streamlines":
            draw_streamlines(ax, fig, snap, xc, yc, x_grid, y_grid, levels)
        elif kind == "pressure":
            draw_pressure(ax, fig, snap, x_grid, y_grid, levels)
        else:
            draw_vorticity(ax, fig, snap, x_grid, y_grid, levels)

        has_ccw, has_cw = overlay_vortex_markers(ax, snap, xc, yc)
        style_axes(ax, title, cfg.lx, cfg.ly)

        handles = _make_legend_handles(has_ccw, has_cw)
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=len(handles),
                fontsize=9,
                framealpha=0.9,
                facecolor="white",
                bbox_to_anchor=(0.5, 0.0),
            )

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        path = panel_dir / f"{kind}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def save_state_pickle(
    snap: Snapshot,
    xc: np.ndarray,
    yc: np.ndarray,
    path: Path,
) -> None:
    """Save one cell-centred state as a pickle dict with x/y/u/v/p arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x_grid, y_grid = np.meshgrid(xc, yc, indexing="ij")
    state = {
        "x": x_grid,
        "y": y_grid,
        "u": snap.uc,
        "v": snap.vc,
        "p": snap.p,
    }
    with path.open("wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)


def save_mac_state_pickle(
    mac_state: MacState,
    cfg: SimConfig,
    snap: Snapshot,
    path: Path,
) -> None:
    """Save exact internal MAC unknowns used by steady mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "nx": cfg.nx,
        "ny": cfg.ny,
        "lx": cfg.lx,
        "ly": cfg.ly,
        "nu": cfg.nu,
        "dt": cfg.dt,
        "step": snap.step,
        "t": snap.t,
        "u_vec": np.asarray(mac_state.u_vec, dtype=np.float64),
        "v_vec": np.asarray(mac_state.v_vec, dtype=np.float64),
        "p": np.asarray(mac_state.p, dtype=np.float64),
    }
    with path.open("wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
