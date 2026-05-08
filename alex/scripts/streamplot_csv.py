#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"CSV is empty: {path}")

    xs = np.unique(data["x"])
    ys = np.unique(data["y"])
    nx, ny = xs.size, ys.size

    fields: dict[str, np.ndarray] = {}
    for name in data.dtype.names:
        if name in ("x", "y"):
            continue
        # Alex writes x in the outer loop and y in the inner loop.
        fields[name] = np.asarray(data[name], dtype=float).reshape(nx, ny).T

    return xs, ys, fields


def velocities_from_psi(psi: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dpsi_dy, dpsi_dx = np.gradient(psi, ys, xs, edge_order=2)
    return dpsi_dy, -dpsi_dx


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_streamplot.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a streamplot from one Alex CSV file.")
    parser.add_argument("csv", type=Path, help="Input CSV with columns x,y,psi,omega,u,v")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--density", type=float, default=1.6, help="Matplotlib streamplot density")
    parser.add_argument("--dpi", type=int, default=180, help="Output PNG DPI")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--no-contours", action="store_true", help="Do not draw psi contours")
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    out_path = (args.out.expanduser().resolve() if args.out is not None else default_output_path(csv_path))

    xs, ys, fields = load_csv(csv_path)
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

    if not args.no_contours and "psi" in fields:
        psi = fields["psi"]
        levels = np.linspace(float(psi.min()), float(psi.max()), 21)
        ax.contour(x_grid, y_grid, psi, levels=levels, colors="white", linewidths=0.35, alpha=0.55)

    ax.set_title(args.title or csv_path.stem)
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
