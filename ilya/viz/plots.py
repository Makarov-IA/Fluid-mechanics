"""Static plot generation: final-state panels, divergence plot, CSV export."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from solver.config import SimConfig, Snapshot

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------


def style_axes(ax, title: str, lx: float, ly: float) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)


def fig_to_pil(fig, dpi: int = 110) -> Image.Image:
    """Render a matplotlib figure to a PIL Image without touching disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def draw_streamlines(ax, fig, snap, xc, yc, Xc, Yc, speed_levels) -> None:
    speed = np.hypot(snap.uc, snap.vc)
    bg = ax.contourf(Xc, Yc, speed, levels=speed_levels, cmap="viridis")
    fig.colorbar(bg, ax=ax, label="|u|")
    ax.streamplot(
        xc, yc, snap.uc.T, snap.vc.T,
        color="white", linewidth=0.8, density=1.5, arrowsize=0.9,
    )


def draw_pressure(ax, fig, snap, Xc, Yc, p_levels) -> None:
    cp = ax.contourf(Xc, Yc, snap.p, levels=p_levels, cmap="coolwarm")
    ax.contour(Xc, Yc, snap.p, levels=p_levels, colors="black", linewidths=0.25, alpha=0.7)
    fig.colorbar(cp, ax=ax, label="p")


def draw_vorticity(ax, fig, snap, Xc, Yc, omega_levels) -> None:
    cw = ax.contourf(Xc, Yc, snap.omega, levels=omega_levels, cmap="coolwarm")
    ax.contour(
        Xc, Yc, snap.omega, levels=omega_levels,
        colors="black", linewidths=0.25, alpha=0.7,
    )
    fig.colorbar(cw, ax=ax, label="ω")


# ---------------------------------------------------------------------------
# Vortex detection
# ---------------------------------------------------------------------------


def find_vortex_centers(
    snap: Snapshot,
    xc: np.ndarray,
    yc: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Detect vortex centers via local extrema of the stream function ψ."""
    uc = snap.uc.astype(np.float64)
    vc = snap.vc.astype(np.float64)

    dy = float(yc[1] - yc[0])
    dx = float(xc[1] - xc[0])

    psi = 0.5 * (np.cumsum(uc * dy, axis=1) + np.cumsum(-vc * dx, axis=0))

    nbr = max(3, min(psi.shape) // 7)
    local_max = psi == ndi.maximum_filter(psi, size=nbr, mode="nearest")
    local_min = psi == ndi.minimum_filter(psi, size=nbr, mode="nearest")

    m = 1
    for arr in (local_max, local_min):
        arr[:m, :] = False
        arr[-m:, :] = False
        arr[:, :m] = False
        arr[:, -m:] = False

    ccw = [(float(xc[i]), float(yc[j])) for i, j in zip(*np.where(local_max))]
    cw  = [(float(xc[i]), float(yc[j])) for i, j in zip(*np.where(local_min))]
    return ccw, cw


def overlay_vortex_markers(
    ax, snap: Snapshot, xc: np.ndarray, yc: np.ndarray
) -> tuple[bool, bool]:
    """Draw vortex-core markers onto *ax*. Returns (has_ccw, has_cw)."""
    ccw, cw = find_vortex_centers(snap, xc, yc)

    for x, y in ccw:
        ax.scatter(x, y, marker="+", c="red", s=150, linewidths=2.5, zorder=6)
        ax.annotate(
            f"({x:.2f}, {y:.2f})", xy=(x, y), xytext=(5, 4),
            textcoords="offset points", color="red", fontsize=7, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65, lw=0),
        )

    for x, y in cw:
        ax.scatter(x, y, marker="x", c="cyan", s=150, linewidths=2.5, zorder=6)
        ax.annotate(
            f"({x:.2f}, {y:.2f})", xy=(x, y), xytext=(5, 4),
            textcoords="offset points", color="cyan", fontsize=7, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0),
        )

    return bool(ccw), bool(cw)


def _make_legend_handles(has_ccw: bool, has_cw: bool) -> list:
    handles = []
    if has_ccw:
        handles.append(Line2D(
            [0], [0], marker="+", color="red", linestyle="none",
            markersize=10, markeredgewidth=2.0, label="Vortex ↺ — counterclockwise (CCW)",
        ))
    if has_cw:
        handles.append(Line2D(
            [0], [0], marker="x", color="cyan", linestyle="none",
            markersize=10, markeredgewidth=2.0, label="Vortex ↻ — clockwise (CW)",
        ))
    return handles


# ---------------------------------------------------------------------------
# Saved outputs
# ---------------------------------------------------------------------------


def save_divergence_plot(
    t_history: list[float], div_history: list[float], out_dir: Path
) -> Path:
    """Save a time-series plot of max |div u|."""
    t = np.asarray(t_history)
    d = np.asarray(div_history)

    stride = max(1, len(t) // 5000)
    t_plot, d_plot = t[::stride], d[::stride]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(t_plot, d_plot, color="#0d47a1", linewidth=1.0)
    ax.set_title("Max divergence vs time")
    ax.set_xlabel("t")
    ax.set_ylabel("max |div u|")
    ax.grid(True, alpha=0.35)
    if len(t) > stride:
        ax.text(
            0.98, 0.97, f"(every {stride}th point shown)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="gray",
        )
    fig.tight_layout()
    path = out_dir / "stokes_max_divergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_final_figure(
    snap: Snapshot,
    cfg: SimConfig,
    xc: np.ndarray,
    yc: np.ndarray,
    out_dir: Path,
    sl: np.ndarray,
    pl: np.ndarray,
    ol: np.ndarray,
) -> list[Path]:
    """Save each final-state panel as a separate PNG in plots/final_state/."""
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    panel_dir = out_dir / "final_state"
    panel_dir.mkdir(parents=True, exist_ok=True)

    suptitle = f"Final state   ν = {cfg.nu},  grid {cfg.nx}×{cfg.ny},  t = {snap.t:.2f}"

    panels = [
        ("streamlines", "Streamlines", sl),
        ("pressure",    "Pressure",    pl),
        ("vorticity",   "Vorticity",   ol),
    ]

    saved: list[Path] = []
    for kind, title, levels in panels:
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.suptitle(suptitle, fontsize=12)

        if kind == "streamlines":
            draw_streamlines(ax, fig, snap, xc, yc, Xc, Yc, levels)
        elif kind == "pressure":
            draw_pressure(ax, fig, snap, Xc, Yc, levels)
        else:
            draw_vorticity(ax, fig, snap, Xc, Yc, levels)

        has_ccw, has_cw = overlay_vortex_markers(ax, snap, xc, yc)
        style_axes(ax, title, cfg.lx, cfg.ly)

        handles = _make_legend_handles(has_ccw, has_cw)
        if handles:
            fig.legend(
                handles=handles, loc="lower center", ncol=len(handles),
                fontsize=9, framealpha=0.9, facecolor="white",
                bbox_to_anchor=(0.5, 0.0),
            )

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        path = panel_dir / f"{kind}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def save_state_csv(snap: Snapshot, xc: np.ndarray, yc: np.ndarray, path: Path) -> None:
    """Save final snapshot (cell-centred x, y, u, v, p) as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    data = np.column_stack(
        [X.ravel(), Y.ravel(), snap.uc.ravel(), snap.vc.ravel(), snap.p.ravel()]
    )
    np.savetxt(path, data, delimiter=",", header="x,y,u,v,p", comments="", fmt="%.8f")
