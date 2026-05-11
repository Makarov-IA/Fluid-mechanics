#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import snapshot_io

matplotlib.use("Agg")


def load_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    xs, ys, fields_xy = snapshot_io.load_snapshot(path)
    fields = {name: values.T for name, values in fields_xy.items()}
    return xs, ys, fields


def velocities_from_psi(psi: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dpsi_dy, dpsi_dx = np.gradient(psi, ys, xs, edge_order=2)
    return dpsi_dy, -dpsi_dx


def mark_vortex_centers(ax, psi: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
    if psi.shape[0] < 3 or psi.shape[1] < 3:
        return
    inner = psi[1:-1, 1:-1]
    min_j, min_i = np.unravel_index(int(np.argmin(inner)), inner.shape)
    max_j, max_i = np.unravel_index(int(np.argmax(inner)), inner.shape)
    seen = set()
    for i, j in ((min_i + 1, min_j + 1), (max_i + 1, max_j + 1)):
        if (i, j) in seen:
            continue
        seen.add((i, j))
        ax.scatter(
            [xs[i]],
            [ys[j]],
            marker="x",
            s=72,
            linewidths=1.9,
            color="#ffeb3b",
            zorder=8,
        )


def default_output_path(snapshot_path: Path) -> Path:
    return snapshot_path.with_name(f"{snapshot_path.stem}_streamplot.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a streamplot from one Alex binary snapshot.")
    parser.add_argument("snapshot", type=Path, help="Input binary snapshot")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--density", type=float, default=1.6, help="Matplotlib streamplot density")
    parser.add_argument("--dpi", type=int, default=180, help="Output PNG DPI")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--no-contours", action="store_true", help="Do not draw psi contours")
    args = parser.parse_args()

    snapshot_path = args.snapshot.expanduser().resolve()
    out_path = (args.out.expanduser().resolve() if args.out is not None else default_output_path(snapshot_path))

    xs, ys, fields = load_snapshot(snapshot_path)
    if "u" in fields and "v" in fields:
        u = fields["u"]
        v = fields["v"]
    elif "psi" in fields:
        u, v = velocities_from_psi(fields["psi"], xs, ys)
    else:
        raise RuntimeError("Need either u,v columns or a psi column to build a streamplot")

    speed = np.hypot(u, v)
    x_grid, y_grid = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bg = ax.contourf(x_grid, y_grid, speed, levels=36, cmap="viridis")

    stream = ax.streamplot(
        xs,
        ys,
        u,
        v,
        color=speed,
        cmap="plasma",
        density=args.density,
        linewidth=0.7 + 1.5 * speed / max(float(speed.max()), 1e-12),
        arrowsize=1.0,
    )
    stream.arrows.set_color("white")

    if "psi" in fields:
        psi = fields["psi"]
        if not args.no_contours:
            levels = np.linspace(float(psi.min()), float(psi.max()), 21)
            ax.contour(x_grid, y_grid, psi, levels=levels, colors="white", linewidths=0.35, alpha=0.55)
        mark_vortex_centers(ax, psi, xs, ys)

    ax.set_title(args.title or snapshot_path.stem)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(float(ys[0]), float(ys[-1]))
    fig.colorbar(bg, ax=ax, label="|u|")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[streamplot] saved: {out_path}")


if __name__ == "__main__":
    main()
