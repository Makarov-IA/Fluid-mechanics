#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ALEX_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ALEX_DIR / "common"))

import snapshot_io  # noqa: E402

matplotlib.use("Agg")


RESULT_RE = re.compile(r"result_(\d+)\.bin$")


def latest_snapshot(results_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in results_dir.glob("result_*.bin"):
        match = RESULT_RE.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"No result_*.bin files found in {results_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def vortex_summary(snapshot: snapshot_io.Snapshot) -> dict[str, float]:
    psi = snapshot.psi
    xs = snapshot.xs
    ys = snapshot.ys
    inner = psi[1:-1, 1:-1]
    min_flat = int(np.argmin(inner))
    max_flat = int(np.argmax(inner))
    min_i, min_j = np.unravel_index(min_flat, inner.shape)
    max_i, max_j = np.unravel_index(max_flat, inner.shape)
    min_i += 1
    min_j += 1
    max_i += 1
    max_j += 1
    return {
        "psi_min": float(psi[min_i, min_j]),
        "psi_min_x": float(xs[min_i]),
        "psi_min_y": float(ys[min_j]),
        "psi_max": float(psi[max_i, max_j]),
        "psi_max_x": float(xs[max_i]),
        "psi_max_y": float(ys[max_j]),
    }


def mark_vortex_centers(ax, snap: snapshot_io.Snapshot) -> None:
    summary = vortex_summary(snap)
    points = [
        (summary["psi_min_x"], summary["psi_min_y"]),
        (summary["psi_max_x"], summary["psi_max_y"]),
    ]
    seen = set()
    for x, y in points:
        key = (round(x, 14), round(y, 14))
        if key in seen:
            continue
        seen.add(key)
        ax.scatter(
            [x],
            [y],
            marker="x",
            s=76,
            linewidths=2.0,
            color="#ffeb3b",
            zorder=8,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot centerline velocity profiles from an Alex binary snapshot.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve() if args.tables_dir is not None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.snapshot.expanduser().resolve() if args.snapshot is not None else latest_snapshot(results_dir)
    snap = snapshot_io.read_snapshot(snapshot_path)

    mid_i = nearest_index(snap.xs, 0.5 * snap.xs[-1])
    mid_j = nearest_index(snap.ys, 0.5 * snap.ys[-1])
    u_center = snap.u[mid_i, :]
    v_center = snap.v[:, mid_j]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].plot(u_center, snap.ys, color="#1565c0", linewidth=1.8)
    axes[0].set_xlabel("u(0.5, y)")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_title("Vertical centerline")

    axes[1].plot(snap.xs, v_center, color="#c62828", linewidth=1.8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v(x, 0.5)")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_title("Horizontal centerline")

    fig.suptitle(f"{snapshot_path.parent.name}: {snapshot_path.name}")
    fig.tight_layout()
    out_png = out_dir / "final_profiles.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    speed = np.hypot(snap.u.T, snap.v.T)
    x_grid, y_grid = np.meshgrid(snap.xs, snap.ys)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    bg = ax.contourf(x_grid, y_grid, speed, levels=40, cmap="viridis")
    stream = ax.streamplot(
        snap.xs,
        snap.ys,
        snap.u.T,
        snap.v.T,
        color=speed,
        cmap="plasma",
        density=1.6,
        linewidth=0.7 + 1.5 * speed / max(float(speed.max()), 1e-12),
        arrowsize=1.0,
    )
    stream.arrows.set_color("white")
    psi = snap.psi.T
    if float(psi.max()) > float(psi.min()):
        ax.contour(x_grid, y_grid, psi, levels=25, colors="white", linewidths=0.35, alpha=0.6)
    mark_vortex_centers(ax, snap)
    ax.set_title(f"Final streamplot, {snapshot_path.parent.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(float(snap.xs[0]), float(snap.xs[-1]))
    ax.set_ylim(float(snap.ys[0]), float(snap.ys[-1]))
    fig.colorbar(bg, ax=ax, label="|u|")
    fig.tight_layout()
    stream_png = out_dir / "final_streamplot.png"
    fig.savefig(stream_png, dpi=180)
    plt.close(fig)

    summary = vortex_summary(snap)
    summary_path = tables_dir / "vortex_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot", *summary.keys()])
        writer.writerow([str(snapshot_path), *(f"{value:.16e}" for value in summary.values())])

    print(f"[profiles] snapshot: {snapshot_path}")
    print(f"[profiles] plot: {out_png}")
    print(f"[profiles] streamplot: {stream_png}")
    print(f"[profiles] summary: {summary_path}")


if __name__ == "__main__":
    main()
