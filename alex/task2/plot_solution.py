#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot psi/omega from task2 results")
    parser.add_argument("--input-dir", default="results", help="Directory with output files")
    parser.add_argument("--output-dir", default="plots", help="Directory for PNG output")
    parser.add_argument("--field", choices=("psi", "omega"), default="omega", help="Field to plot")
    parser.add_argument("--graph", choices=("heatmap", "surface", "contour", "all"), default="all", help="Graph type")
    parser.add_argument("--snapshot-index", type=int, default=-1, help="Snapshot index from snapshots/index.csv. -1 means final field")
    parser.add_argument("--list-snapshots", action="store_true", help="Print snapshot list and exit")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap")
    parser.add_argument("--levels", type=int, default=25, help="Contour levels")
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


def load_snapshots(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    rows = []
    with index_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def resolve_field_path(input_dir: Path, field: str, snapshot_index: int, snapshots: list[dict]) -> tuple[Path, str]:
    if snapshot_index < 0:
        return input_dir / f"{field}.txt", "final"

    if not snapshots:
        raise ValueError("No snapshots found (missing snapshots/index.csv)")
    if snapshot_index >= len(snapshots):
        raise IndexError(f"snapshot-index {snapshot_index} is out of range [0, {len(snapshots)-1}]")

    row = snapshots[snapshot_index]
    rel = row[f"{field}_file"]
    tag = f"snap{snapshot_index:03d}_step{int(row['step']):05d}_t{float(row['time']):.3f}"
    return input_dir / rel, tag


def plot_heatmap(ax, arr: np.ndarray, cmap: str, title: str):
    im = ax.imshow(arr, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im


def plot_surface(ax, arr: np.ndarray, cmap: str, title: str):
    ny, nx = arr.shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)
    surf = ax.plot_surface(xx, yy, arr, cmap=cmap, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("value")
    return surf


def plot_contour(ax, arr: np.ndarray, cmap: str, levels: int, title: str):
    ny, nx = arr.shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    cs = ax.contourf(x, y, arr, levels=levels, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return cs


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = load_snapshots(input_dir / "snapshots" / "index.csv")
    if args.list_snapshots:
        if not snapshots:
            print("No snapshots found")
            return 0
        for row in snapshots:
            print(f"index={row['index']} step={row['step']} t={row['time']}")
        return 0

    field_path, tag = resolve_field_path(input_dir, args.field, args.snapshot_index, snapshots)
    arr = load_field(field_path)
    ny, nx = arr.shape

    title_prefix = f"{args.field} [{nx}x{ny}] ({tag})"

    if args.graph == "heatmap":
        fig, ax = plt.subplots(figsize=(7, 6))
        m = plot_heatmap(ax, arr, args.cmap, f"{title_prefix} heatmap")
        fig.colorbar(m, ax=ax)
    elif args.graph == "surface":
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        s = plot_surface(ax, arr, args.cmap, f"{title_prefix} surface")
        fig.colorbar(s, ax=ax, shrink=0.7, pad=0.08)
    elif args.graph == "contour":
        fig, ax = plt.subplots(figsize=(7, 6))
        c = plot_contour(ax, arr, args.cmap, args.levels, f"{title_prefix} contour")
        fig.colorbar(c, ax=ax)
    else:
        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        m = plot_heatmap(ax1, arr, args.cmap, "heatmap")
        fig.colorbar(m, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        s = plot_surface(ax2, arr, args.cmap, "surface")
        fig.colorbar(s, ax=ax2, shrink=0.7, pad=0.08)

        ax3 = fig.add_subplot(1, 3, 3)
        c = plot_contour(ax3, arr, args.cmap, args.levels, "contour")
        fig.colorbar(c, ax=ax3, fraction=0.046, pad=0.04)

        fig.suptitle(title_prefix)

    fig.tight_layout()

    out_name = f"{args.field}_{tag}_{args.graph}.png"
    out_path = output_dir / out_name
    if args.save:
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Input: {field_path}")
    print(f"Grid: {nx}x{ny}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
