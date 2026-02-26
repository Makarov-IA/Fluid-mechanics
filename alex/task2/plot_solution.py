#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIELDS = ("psi", "omega", "u", "v", "g", "speed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fields for task2 omega-psi solver")
    parser.add_argument("--input-dir", default="results", help="Directory with psi.txt, omega.txt, u.txt, v.txt, g.txt")
    parser.add_argument("--output-dir", default="plots", help="Directory for PNG output")
    parser.add_argument("--field", choices=("all",) + FIELDS, default="all", help="Field to plot")
    parser.add_argument("--cmap", default="turbo", help="Colormap")
    parser.add_argument("--dpi", type=int, default=180, help="Saved image dpi")
    parser.set_defaults(show=True)
    parser.add_argument("--show", dest="show", action="store_true", help="Show matplotlib windows")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not show windows")
    parser.add_argument("--save", action="store_true", help="Save PNG files")
    return parser.parse_args()


def load_field(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing field file: {path}")
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def plot_scalar(name: str, arr: np.ndarray, out_dir: Path, cmap: str, dpi: int, save: bool, show: bool) -> None:
    ny, nx = arr.shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(arr, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, aspect="auto")
    ax1.set_title(f"{name} heatmap [{nx}x{ny}]")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(xx, yy, arr, cmap=cmap, linewidth=0, antialiased=True)
    ax2.set_title(f"{name} surface")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel(name)
    fig.colorbar(surf, ax=ax2, shrink=0.7, pad=0.08)

    fig.tight_layout()
    out = out_dir / f"{name}.png"
    if save:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_velocity_quiver(u: np.ndarray, v: np.ndarray, out_dir: Path, dpi: int, save: bool, show: bool) -> None:
    ny, nx = u.shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)
    speed = np.sqrt(u * u + v * v)

    step = max(1, nx // 25)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(speed, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", alpha=0.9, aspect="auto")
    ax.quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        color="white",
        scale=25,
        width=0.0025,
    )
    ax.set_title("velocity quiver + speed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="|V|")
    fig.tight_layout()

    out = out_dir / "velocity_quiver.png"
    if save:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    psi = load_field(input_dir / "psi.txt")
    omega = load_field(input_dir / "omega.txt")
    u = load_field(input_dir / "u.txt")
    v = load_field(input_dir / "v.txt")
    g = load_field(input_dir / "g.txt")

    speed = np.sqrt(u * u + v * v)
    fields = {
        "psi": psi,
        "omega": omega,
        "u": u,
        "v": v,
        "g": g,
        "speed": speed,
    }

    selected = FIELDS if args.field == "all" else (args.field,)
    for name in selected:
        plot_scalar(name, fields[name], out_dir, args.cmap, args.dpi, args.save, args.show)

    if args.field in ("all", "speed"):
        plot_velocity_quiver(u, v, out_dir, args.dpi, args.save, args.show)

    print(f"Input dir: {input_dir}")
    print(f"Grid: {psi.shape[1]}x{psi.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
