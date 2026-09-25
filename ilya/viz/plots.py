"""Static plot generation helpers for simulation and steady-state outputs."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
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


def _field_colorbar(fig, ax, mappable, x_grid, y_grid, label: str) -> None:
    """Add a colorbar scaled to the visible field aspect ratio."""
    x_span = float(np.max(x_grid) - np.min(x_grid))
    y_span = float(np.max(y_grid) - np.min(y_grid))
    shrink = 1.0
    if x_span > 0.0 and y_span > 0.0:
        shrink = min(1.0, max(0.35, y_span / x_span))
    fig.colorbar(mappable, ax=ax, label=label, shrink=shrink, pad=0.04)


def fig_to_rgb(fig, dpi: int = 110) -> np.ndarray:
    """Rasterise a figure straight from the Agg canvas (same pixels as savefig)."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def draw_streamlines(ax, fig, snap, xc, yc, x_grid, y_grid, speed_levels) -> None:
    """Draw velocity magnitude background and streamline overlay."""
    speed = np.hypot(snap.uc, snap.vc)
    bg = ax.contourf(x_grid, y_grid, speed, levels=speed_levels, cmap="viridis")
    _field_colorbar(fig, ax, bg, x_grid, y_grid, "|u|")
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
    _field_colorbar(fig, ax, contours, x_grid, y_grid, "p")


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
    _field_colorbar(fig, ax, contours, x_grid, y_grid, "ω")


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
        if len(t_plot) > 1:
            dt_between_points = np.diff(t_plot)
            positive_dt = dt_between_points[dt_between_points > 0.0]
            fallback_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0
            step_dt = np.concatenate(([fallback_dt], dt_between_points))
        else:
            step_dt = np.ones_like(change_plot)
        step_dt = np.where(step_dt > 0.0, step_dt, fallback_dt if len(t_plot) > 1 else 1.0)
        change_rate = change_plot / step_dt
        finite_positive = change_rate[np.isfinite(change_rate) & (change_rate > 0.0)]
        if len(finite_positive):
            floor = max(float(np.min(finite_positive)) * 0.1, 1e-300)
            y_plot = np.where(np.isfinite(change_rate) & (change_rate > 0.0), change_rate, floor)
            y_max = max(float(np.max(y_plot)), floor * 10.0)
            ax.set_ylim(floor, y_max * 10.0)
        else:
            floor = 1e-16
            y_plot = np.full_like(change_rate, floor)
            ax.set_ylim(floor * 0.1, floor * 10.0)
        ax.semilogy(
            t_plot,
            y_plot,
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


def save_center_velocity_plot(
    snapshots: list[Snapshot],
    cfg: SimConfig,
    xc: np.ndarray,
    yc: np.ndarray,
    out_dir: Path,
) -> Path:
    """Save u/v velocity at a shifted control point versus time."""
    path = out_dir / "center_velocity.png"
    probe_x = 0.35 * cfg.lx
    probe_y = 0.40 * cfg.ly
    ix = int(np.argmin(np.abs(xc - probe_x)))
    iy = int(np.argmin(np.abs(yc - probe_y)))
    actual_x = float(xc[ix])
    actual_y = float(yc[iy])

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if snapshots:
        times = np.asarray([snap.t for snap in snapshots], dtype=np.float64)
        u_probe = np.asarray([snap.uc[ix, iy] for snap in snapshots], dtype=np.float64)
        v_probe = np.asarray([snap.vc[ix, iy] for snap in snapshots], dtype=np.float64)
        ax.plot(times, u_probe, label="u", color="#1565c0", linewidth=1.4)
        ax.plot(times, v_probe, label="v", color="#c62828", linewidth=1.4)
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "No snapshots available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
    ax.set_title(f"Velocity at selected point x={actual_x:.3f}, y={actual_y:.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel("velocity")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _select_time_panels(
    snapshots: list[Snapshot],
    max_panels: int = 4,
) -> list[Snapshot]:
    """Pick snapshots spread evenly across the available time interval."""
    if not snapshots:
        return []
    count = min(max_panels, len(snapshots))
    indices = np.linspace(0, len(snapshots) - 1, count, dtype=int)
    unique_indices = list(dict.fromkeys(int(idx) for idx in indices))
    return [snapshots[idx] for idx in unique_indices]


def _display_stride(nx: int, ny: int, max_points: int = 170) -> tuple[int, int]:
    """Return field strides that keep streamplot generation reasonably cheap."""
    return max(1, nx // max_points), max(1, ny // max_points)


def save_velocity_field_sequence(
    snapshots: list[Snapshot],
    cfg: SimConfig,
    xc: np.ndarray,
    yc: np.ndarray,
    out_dir: Path,
    *,
    max_panels: int = 16,
) -> Path:
    """Save velocity fields at several time moments in one figure."""
    path = out_dir / "velocity_fields_times.png"
    selected = _select_time_panels(snapshots, max_panels=max_panels)

    if not selected:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.text(
            0.5,
            0.5,
            "No snapshots available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    sx, sy = _display_stride(len(xc), len(yc))
    x_plot = xc[::sx]
    y_plot = yc[::sy]
    x_grid, y_grid = np.meshgrid(x_plot, y_plot, indexing="ij")

    speed_max = max(
        float(np.max(np.hypot(snap.uc[::sx, ::sy], snap.vc[::sx, ::sy])))
        for snap in selected
    )
    if speed_max <= 0.0 or not np.isfinite(speed_max):
        speed_max = 1.0
    levels = np.linspace(0.0, speed_max, 36)

    ncols = min(4, len(selected)) if len(selected) > 1 else 1
    nrows = int(np.ceil(len(selected) / ncols))
    panel_width = 2.45
    panel_height = panel_width * cfg.ly / cfg.lx
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False,
        gridspec_kw={"wspace": 0.01, "hspace": 0.01},
    )

    for ax, snap in zip(axes.ravel(), selected):
        uc = snap.uc[::sx, ::sy]
        vc = snap.vc[::sx, ::sy]
        speed = np.hypot(uc, vc)
        ax.contourf(x_grid, y_grid, speed, levels=levels, cmap="viridis")
        ax.streamplot(
            x_plot,
            y_plot,
            uc.T,
            vc.T,
            color="white",
            linewidth=0.75,
            density=1.25,
            arrowsize=0.8,
        )
        ax.text(
            0.5,
            0.985,
            f"t = {snap.t:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
            zorder=8,
        )
        ax.set_aspect("equal")
        ax.set_xlim(0.0, cfg.lx)
        ax.set_ylim(0.0, cfg.ly)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.ravel()[len(selected):]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.998)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def _is_re10000_cavity_case(cfg: SimConfig) -> bool:
    return (
        np.isclose(cfg.lx, 1.0)
        and np.isclose(cfg.ly, 1.0)
        and np.isclose(cfg.nu, 1.0e-4)
        and not cfg.has_forcing
    )


def _streamfunction_from_velocity(
    snap: Snapshot,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate cell-centred streamfunction with zero value on the walls."""
    dx = cfg.lx / cfg.nx
    dy = cfg.ly / cfg.ny
    xc = (np.arange(cfg.nx) + 0.5) * dx
    yc = (np.arange(cfg.ny) + 0.5) * dy
    psi = np.cumsum(snap.uc.astype(np.float64), axis=1) * dy - 0.5 * dy * snap.uc

    top_drift = np.sum(snap.uc.astype(np.float64), axis=1) * dy
    if cfg.ly > 0.0:
        psi = psi - (yc[None, :] / cfg.ly) * top_drift[:, None]

    x = np.concatenate(([0.0], xc, [cfg.lx]))
    y = np.concatenate(([0.0], yc, [cfg.ly]))
    psi_ext = np.zeros((cfg.nx + 2, cfg.ny + 2), dtype=np.float64)
    psi_ext[1:-1, 1:-1] = psi
    return x, y, psi_ext


def _format_contour_value(value: float) -> str:
    if abs(value) < 1e-4:
        return f"{value:.0e}"
    if abs(value) < 1e-3:
        return f"{value:.5f}".rstrip("0")
    if abs(value) < 1e-2:
        return f"{value:.4f}".rstrip("0")
    return f"{value:.3f}".rstrip("0")


def _streamfunction_contour_levels(psi: np.ndarray) -> np.ndarray:
    candidates = np.asarray(
        [
            -0.1175,
            -0.115,
            -0.11,
            -0.10,
            -0.09,
            -0.07,
            -0.05,
            -0.03,
            -0.01,
            -0.005,
            -0.001,
            -0.0001,
            0.00001,
            0.000025,
            0.00005,
            0.0001,
            0.00025,
            0.0005,
            0.001,
            0.0015,
            0.002,
            0.003,
            0.005,
            0.01,
        ],
        dtype=np.float64,
    )
    psi_min = float(np.min(psi))
    psi_max = float(np.max(psi))
    levels = candidates[(candidates > psi_min) & (candidates < psi_max)]
    if len(levels) >= 4:
        return levels
    return np.linspace(psi_min, psi_max, 12)[1:-1]


def save_cavity_streamfunction_contours(
    snap: Snapshot,
    cfg: SimConfig,
    out_dir: Path,
) -> Path | None:
    """Save Re=10000 cavity streamfunction contours with secondary-vortex zooms."""
    if not _is_re10000_cavity_case(cfg):
        return None

    x, y, psi = _streamfunction_from_velocity(snap, cfg)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    levels = _streamfunction_contour_levels(psi)

    path = out_dir / "cavity_streamfunction_contours.png"
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 10.0))

    panels = [
        (axes[0, 0], (0.0, 0.27), (0.55, 1.0), "Eddy TL1"),
        (axes[0, 1], (0.0, 1.0), (0.0, 1.0), ""),
        (axes[1, 0], (0.0, 0.45), (0.0, 0.40), "Eddies BL1, BL2, BL3"),
        (axes[1, 1], (0.55, 1.0), (0.0, 0.50), "Eddies BR1, BR2, BR3"),
    ]

    for ax, xlim, ylim, label in panels:
        contours = ax.contour(
            x_grid,
            y_grid,
            psi,
            levels=levels,
            colors="#202020",
            linewidths=0.8,
            linestyles="solid",
        )
        ax.clabel(contours, inline=True, fontsize=7, fmt=_format_contour_value)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        if label:
            ax.text(
                0.5,
                0.83,
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                alpha=0.9,
            )

    axes[0, 1].set_xlabel("X", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("Y", fontsize=13, fontweight="bold")
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Streamfunction contours of primary and secondary vortices, Re=10000", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_cavity_centerline_profiles(
    snap: Snapshot,
    cfg: SimConfig,
    xc: np.ndarray,
    yc: np.ndarray,
    out_dir: Path,
) -> Path | None:
    """Save Re=10000 cavity velocity profiles on the central cross-lines."""
    if not _is_re10000_cavity_case(cfg):
        return None

    y_ref = np.asarray(
        [
            1.000,
            0.990,
            0.980,
            0.970,
            0.960,
            0.950,
            0.940,
            0.930,
            0.920,
            0.910,
            0.900,
            0.500,
            0.200,
            0.180,
            0.160,
            0.140,
            0.120,
            0.100,
            0.080,
            0.060,
            0.040,
            0.020,
            0.000,
        ],
        dtype=np.float64,
    )
    u_ref = np.asarray(
        [
            1.0000,
            0.5891,
            0.4837,
            0.4891,
            0.4917,
            0.4843,
            0.4711,
            0.4556,
            0.4398,
            0.4243,
            0.4095,
            -0.0268,
            -0.2998,
            -0.3179,
            -0.3361,
            -0.3543,
            -0.3721,
            -0.3899,
            -0.4142,
            -0.4469,
            -0.4259,
            -0.2907,
            0.0000,
        ],
        dtype=np.float64,
    )

    x_ref = np.asarray(
        [
            1.000,
            0.985,
            0.970,
            0.955,
            0.940,
            0.925,
            0.910,
            0.895,
            0.880,
            0.865,
            0.850,
            0.500,
            0.150,
            0.135,
            0.120,
            0.105,
            0.090,
            0.075,
            0.060,
            0.045,
            0.030,
            0.015,
            0.000,
        ],
        dtype=np.float64,
    )
    v_ref = np.asarray(
        [
            0.0000,
            -0.3419,
            -0.5712,
            -0.5124,
            -0.4592,
            -0.4411,
            -0.4256,
            -0.4078,
            -0.3895,
            -0.3715,
            -0.3538,
            0.0088,
            0.3562,
            0.3722,
            0.3885,
            0.4056,
            0.4247,
            0.4449,
            0.4566,
            0.4409,
            0.3844,
            0.2756,
            0.0000,
        ],
        dtype=np.float64,
    )

    ix = int(np.argmin(np.abs(xc - 0.5 * cfg.lx)))
    iy = int(np.argmin(np.abs(yc - 0.5 * cfg.ly)))
    y_line = np.concatenate(([0.0], yc, [cfg.ly]))
    u_line = np.concatenate(([0.0], snap.uc[ix, :], [1.0]))
    x_line = np.concatenate(([0.0], xc, [cfg.lx]))
    v_line = np.concatenate(([0.0], snap.vc[:, iy], [0.0]))

    path = out_dir / "cavity_centerline_profiles.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))

    ax = axes[0]
    ax.plot(u_line, y_line, color="#1565c0", linewidth=1.6, label="steady solution")
    ax.scatter(u_ref, y_ref, color="#c62828", s=22, label="table data")
    ax.axvline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.set_title(r"$u(0.5,y)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(x_line, v_line, color="#1565c0", linewidth=1.6, label="steady solution")
    ax.scatter(x_ref, v_ref, color="#c62828", s=22, label="table data")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_title(r"$v(x,0.5)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle("Lid-driven cavity centreline velocity profiles, Re=10000", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
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
    show_state_title: bool = True,
    show_vortex_markers: bool = True,
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
        if show_state_title:
            fig.suptitle(suptitle, fontsize=12)

        if kind == "streamlines":
            draw_streamlines(ax, fig, snap, xc, yc, x_grid, y_grid, levels)
        elif kind == "pressure":
            draw_pressure(ax, fig, snap, x_grid, y_grid, levels)
        else:
            draw_vorticity(ax, fig, snap, x_grid, y_grid, levels)

        has_ccw = has_cw = False
        if show_vortex_markers:
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

        bottom = 0.06 if handles else 0.0
        top = 0.96 if show_state_title else 1.0
        fig.tight_layout(rect=[0, bottom, 1, top])
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
