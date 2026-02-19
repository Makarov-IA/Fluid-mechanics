#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 2D Poisson solution from text grid.")
    parser.add_argument("--input", default="solution.txt", help="Path to solution txt file")
    parser.add_argument("--output-dir", default="plots", help="Directory for output images")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap")
    parser.add_argument("--dpi", type=int, default=180, help="Output dpi")
    parser.set_defaults(show=True)
    parser.add_argument("--show", dest="show", action="store_true", help="Show plot windows")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not show plot windows")
    parser.add_argument("--save", action="store_true", help="Save PNG files")
    parser.add_argument("--title", default="Solution u(x, y)", help="Figure title")
    return parser.parse_args()


def load_grid(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        raise ValueError("Input file is empty")
    return data


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    u = load_grid(input_path)
    ny, nx = u.shape

    x = np.linspace(1.0 / (nx + 1), nx / (nx + 1), nx)
    y = np.linspace(1.0 / (ny + 1), ny / (ny + 1), ny)
    xx, yy = np.meshgrid(x, y)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(1, 1, 1, projection="3d")
    surf = ax3.plot_surface(xx, yy, u, cmap=args.cmap, linewidth=0, antialiased=True)
    ax3.set_title(f"{args.title}  [{nx}x{ny}]")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("u")
    fig.colorbar(surf, ax=ax3, shrink=0.7, pad=0.08)
    fig.tight_layout()

    out_main = output_dir / "solution_3d.png"
    if args.save:
        fig.savefig(out_main, dpi=args.dpi, bbox_inches="tight")

    print(f"Input: {input_path}")
    print(f"Grid: {nx}x{ny}")
    if args.save:
        print(f"Saved: {out_main}")

    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
