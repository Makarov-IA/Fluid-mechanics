import argparse
import csv
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ALEX_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ALEX_DIR / "common"))

import snapshot_io

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

plt.switch_backend("Agg")
plt.ioff()
plt.rcParams["path.simplify"] = True
plt.rcParams["agg.path.chunksize"] = 10000

FRAME_KINDS = ("psi", "omega", "streamplot")


def step_sort_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)(?!.*\d)", base)
    if match is None:
        return (-1, base)
    return (int(match.group(1)), base)


def collect_snapshot_files(results_dir, snapshot_index=None):
    if snapshot_index is None:
        return sorted(
            glob.glob(os.path.join(results_dir, "result_*.bin")),
            key=step_sort_key,
        )

    files = []
    with open(snapshot_index, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = (
                row.get("filtered_snapshot")
                or row.get("source_snapshot")
                or row.get("filtered_csv")
                or row.get("source_csv")
                or ""
            )
            if not raw_path:
                continue
            snapshot_path = os.path.abspath(os.path.normpath(os.path.expanduser(raw_path)))
            if os.path.exists(snapshot_path):
                files.append(snapshot_path)

    return sorted(files, key=step_sort_key)


def load_grids(path):
    xs, ys, fields_xy = snapshot_io.load_snapshot(path)
    grids = {name: values.T for name, values in fields_xy.items()}
    return grids, xs, ys


def regularize_axis(axis):
    if axis.size <= 2:
        return axis.copy()
    return np.linspace(float(axis[0]), float(axis[-1]), axis.size)


def increasing_levels(vmin, vmax, n_levels=31):
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    if abs(vmax - vmin) < 1e-14 * max(1.0, abs(vmin), abs(vmax)):
        center = 0.5 * (vmin + vmax)
        half_width = max(1e-12, 1e-6 * max(1.0, abs(center)))
        vmin = center - half_width
        vmax = center + half_width
    return np.linspace(vmin, vmax, n_levels)


def scan_file_ranges(snapshot_path):
    grids, _, _ = load_grids(snapshot_path)
    speed = np.hypot(grids["u"], grids["v"])
    return {
        "psi_min": float(np.min(grids["psi"])),
        "psi_max": float(np.max(grids["psi"])),
        "omega_min": float(np.min(grids["omega"])),
        "omega_max": float(np.max(grids["omega"])),
        "speed_max": float(np.max(speed)),
    }


def merge_plot_scale(ranges):
    psi_min = min(item["psi_min"] for item in ranges)
    psi_max = max(item["psi_max"] for item in ranges)
    omega_min = min(item["omega_min"] for item in ranges)
    omega_max = max(item["omega_max"] for item in ranges)
    speed_max = max(max(item["speed_max"] for item in ranges), 1e-12)
    return {
        "psi_limits": increasing_levels(psi_min, psi_max),
        "omega_limits": increasing_levels(omega_min, omega_max),
        "speed_limits": (0.0, speed_max),
    }


def build_frame_stats(grids, xs, ys, plot_scale=None):
    x_span = xs[-1] - xs[0] if len(xs) > 1 else 1.0
    y_span = ys[-1] - ys[0] if len(ys) > 1 else 1.0
    domain_span = max(min(x_span, y_span), 1e-12)
    speed = np.hypot(grids["u"], grids["v"])
    if plot_scale is None:
        plot_scale = {
            "psi_limits": increasing_levels(float(np.min(grids["psi"])), float(np.max(grids["psi"]))),
            "omega_limits": increasing_levels(float(np.min(grids["omega"])), float(np.max(grids["omega"]))),
            "speed_limits": (0.0, max(float(np.max(speed)), 1e-12)),
        }
    speed_max = max(float(plot_scale["speed_limits"][1]), 1e-12)

    return {
        "psi_limits": plot_scale["psi_limits"],
        "omega_limits": plot_scale["omega_limits"],
        "speed_limits": plot_scale["speed_limits"],
        "speed": speed,
        "arrow_factor": 0.075 * domain_span / speed_max,
        "quiver_scale": 1.0,
    }


def mark_vortex_centers(ax, grids, xs, ys):
    psi = grids.get("psi")
    if psi is None or psi.shape[0] < 3 or psi.shape[1] < 3:
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
            s=70,
            linewidths=1.8,
            color="#ffeb3b",
            zorder=8,
        )


def style_axes(ax, title, xs, ys):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])


def save_figure(fig, out_png):
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.10, top=0.92)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def build_scalar_frame_with_quiver(
    grids, xs, ys, out_png, field, cmap, colorbar_label, title, stats, base
):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    try:
        xs_plot = regularize_axis(xs)
        ys_plot = regularize_axis(ys)
        x_grid, y_grid = np.meshgrid(xs_plot, ys_plot)
        skip = max(1, min(len(xs_plot), len(ys_plot)) // 24)
        scalar = grids[field]
        levels = stats[f"{field}_limits"]

        contourf = ax.contourf(
            x_grid, y_grid, scalar, levels=levels, cmap=cmap, extend="both"
        )
        if float(np.max(scalar)) > float(np.min(scalar)):
            ax.contour(
                x_grid,
                y_grid,
                scalar,
                levels=levels,
                colors="black",
                linewidths=0.25,
                alpha=0.45,
            )
        ax.quiver(
            x_grid[::skip, ::skip],
            y_grid[::skip, ::skip],
            stats["arrow_factor"] * grids["u"][::skip, ::skip],
            stats["arrow_factor"] * grids["v"][::skip, ::skip],
            color="white",
            pivot="mid",
            width=0.0032,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.5,
            angles="xy",
            scale_units="xy",
            scale=stats["quiver_scale"],
            alpha=0.9,
        )
        mark_vortex_centers(ax, grids, xs_plot, ys_plot)
        fig.colorbar(contourf, ax=ax, label=colorbar_label)
        style_axes(ax, f"{title}, {base}", xs_plot, ys_plot)
        save_figure(fig, out_png)
    finally:
        plt.close(fig)


def build_streamplot_frame(grids, xs, ys, out_png, stats, base):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    try:
        xs_plot = regularize_axis(xs)
        ys_plot = regularize_axis(ys)
        speed = stats["speed"]
        _, speed_max = stats["speed_limits"]
        speed_max = max(float(speed_max), 1e-12)
        speed_levels = np.linspace(0.0, speed_max, 31)
        speed_norm = Normalize(vmin=0.0, vmax=speed_max)
        x_grid, y_grid = np.meshgrid(xs_plot, ys_plot)

        bg = ax.contourf(
            x_grid,
            y_grid,
            speed,
            levels=speed_levels,
            cmap="viridis",
            norm=speed_norm,
            extend="max",
        )
        stream = ax.streamplot(
            xs_plot,
            ys_plot,
            grids["u"],
            grids["v"],
            color=speed,
            cmap="plasma",
            norm=speed_norm,
            linewidth=0.8 + 1.6 * speed / speed_max,
            density=1.45,
            arrowsize=1.0,
        )
        stream.arrows.set_color("white")
        mark_vortex_centers(ax, grids, xs_plot, ys_plot)
        fig.colorbar(bg, ax=ax, label="|u|")
        style_axes(ax, f"Streamplot, {base}", xs_plot, ys_plot)
        save_figure(fig, out_png)
    finally:
        plt.close(fig)


def make_frame_set_task(task):
    snapshot_path, frames_dir, plot_scale = task
    grids, xs, ys = load_grids(snapshot_path)
    stats = build_frame_stats(grids, xs, ys, plot_scale)
    base = os.path.splitext(os.path.basename(snapshot_path))[0]
    outputs = {}

    psi_png = os.path.join(frames_dir, f"{base}_psi.png")
    build_scalar_frame_with_quiver(
        grids,
        xs,
        ys,
        psi_png,
        "psi",
        "coolwarm",
        "psi",
        "Psi field",
        stats,
        base,
    )
    outputs["psi"] = psi_png

    omega_png = os.path.join(frames_dir, f"{base}_omega.png")
    build_scalar_frame_with_quiver(
        grids,
        xs,
        ys,
        omega_png,
        "omega",
        "RdBu_r",
        "omega",
        "Omega field",
        stats,
        base,
    )
    outputs["omega"] = omega_png

    stream_png = os.path.join(frames_dir, f"{base}_streamplot.png")
    build_streamplot_frame(grids, xs, ys, stream_png, stats, base)
    outputs["streamplot"] = stream_png

    return outputs


def read_residual_history(path):
    times = []
    psi_residuals = []
    omega_residuals = []
    max_residuals = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            psi_residuals.append(float(row.get("psi_res", row["max_residual"])))
            omega_residuals.append(float(row.get("omega_res", row["max_residual"])))
            max_residuals.append(float(row["max_residual"]))

    return (
        np.array(times, dtype=float),
        np.array(psi_residuals, dtype=float),
        np.array(omega_residuals, dtype=float),
        np.array(max_residuals, dtype=float),
    )


def build_residual_plot(results_dir, plot_root):
    residual_path = os.path.join(results_dir, "residual_history.csv")
    if not os.path.exists(residual_path):
        return

    times, psi_residuals, omega_residuals, max_residuals = read_residual_history(residual_path)
    if max_residuals.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogy(
        times,
        np.maximum(psi_residuals, 1e-30),
        color="#1565c0",
        linewidth=1.6,
        label="psi_res",
    )
    ax.semilogy(
        times,
        np.maximum(omega_residuals, 1e-30),
        color="#c62828",
        linewidth=1.6,
        label="omega_res",
    )
    ax.semilogy(
        times,
        np.maximum(max_residuals, 1e-30),
        color="#2e7d32",
        linewidth=1.2,
        linestyle="--",
        label="max_residual",
    )
    ax.set_title("Residual history")
    ax.set_xlabel("t")
    ax.set_ylabel("residual")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    out_png = os.path.join(plot_root, "residual_history.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[plot] png: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Render Alex binary snapshot fields to PNG frames.")
    parser.add_argument("results_dir")
    parser.add_argument("frames_dir")
    parser.add_argument(
        "--snapshot-index",
        default=None,
        help="Optional CSV index with filtered_snapshot/source_snapshot columns; prevents stale files from old runs.",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(os.path.normpath(args.results_dir))
    frames_dir = os.path.abspath(os.path.normpath(args.frames_dir))
    snapshot_index = (
        os.path.abspath(os.path.normpath(args.snapshot_index))
        if args.snapshot_index is not None
        else None
    )
    plot_root = os.path.dirname(frames_dir)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(plot_root, exist_ok=True)

    snapshot_files = collect_snapshot_files(results_dir, snapshot_index)
    if snapshot_index is not None:
        print(f"[plot] Using snapshot index: {snapshot_index}")
    if not snapshot_files:
        raise RuntimeError(f"No result_*.bin found in {results_dir}")

    workers = max(1, min(os.cpu_count() or 1, len(snapshot_files)))

    print(f"[plot] Scanning shared scale from {len(snapshot_files)} snapshots")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_file_ranges, snapshot_path) for snapshot_path in snapshot_files]
        ranges = []
        progress_total = len(futures)
        if tqdm is not None:
            with tqdm(total=progress_total, desc="scan scale", unit="file") as bar:
                for fut in as_completed(futures):
                    ranges.append(fut.result())
                    bar.update(1)
        else:
            for fut in as_completed(futures):
                ranges.append(fut.result())

    plot_scale = merge_plot_scale(ranges)
    print(
        "[plot] Shared scale: "
        f"psi=[{plot_scale['psi_limits'][0]:.6e}, {plot_scale['psi_limits'][-1]:.6e}], "
        f"omega=[{plot_scale['omega_limits'][0]:.6e}, {plot_scale['omega_limits'][-1]:.6e}], "
        f"speed=[0, {plot_scale['speed_limits'][1]:.6e}]"
    )

    tasks = [(snapshot_path, frames_dir, plot_scale) for snapshot_path in snapshot_files]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(make_frame_set_task, task) for task in tasks]

        progress_total = len(futures)
        if tqdm is not None:
            with tqdm(total=progress_total, desc="render frames", unit="frame") as bar:
                for fut in as_completed(futures):
                    fut.result()
                    bar.update(1)
        else:
            for fut in as_completed(futures):
                fut.result()

    build_residual_plot(results_dir, plot_root)


if __name__ == "__main__":
    main()
