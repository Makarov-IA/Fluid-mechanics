import csv
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

plt.switch_backend("Agg")
plt.ioff()

COL_IDX = {"psi": 2, "omega": 3, "u": 4, "v": 5}
FRAME_KINDS = ("psi", "omega", "streamplot")
GIF_NAMES = {
    "psi": "psi.gif",
    "omega": "omega.gif",
    "streamplot": "streamplot.gif",
}


def step_sort_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)(?!.*\d)", base)
    if match is None:
        return (-1, base)
    return (int(match.group(1)), base)


def read_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                (
                    float(r["x"]),
                    float(r["y"]),
                    float(r["psi"]),
                    float(r["omega"]),
                    float(r["u"]),
                    float(r["v"]),
                )
            )
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def build_grids(rows):
    xs = np.array(sorted({r[0] for r in rows}), dtype=float)
    ys = np.array(sorted({r[1] for r in rows}), dtype=float)
    nx = len(xs)
    ny = len(ys)

    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_j = {y: j for j, y in enumerate(ys)}

    grids = {
        "psi": np.zeros((ny, nx), dtype=float),
        "omega": np.zeros((ny, nx), dtype=float),
        "u": np.zeros((ny, nx), dtype=float),
        "v": np.zeros((ny, nx), dtype=float),
    }

    for r in rows:
        i = x_to_i[r[0]]
        j = y_to_j[r[1]]
        grids["psi"][j, i] = r[COL_IDX["psi"]]
        grids["omega"][j, i] = r[COL_IDX["omega"]]
        grids["u"][j, i] = r[COL_IDX["u"]]
        grids["v"][j, i] = r[COL_IDX["v"]]

    return grids, xs, ys


def regularize_axis(axis):
    if axis.size <= 2:
        return axis.copy()
    return np.linspace(float(axis[0]), float(axis[-1]), axis.size)


def symmetric_levels(values, n_levels=61, percentile=98.0):
    bound = float(np.percentile(np.abs(values), percentile))
    bound = max(bound, 1e-12)
    return np.linspace(-bound, bound, n_levels)


def collect_plot_stats(csv_files):
    psi_values = []
    omega_values = []
    speed_values = []
    domain_span = 1.0

    iterator = csv_files
    if tqdm is not None:
        iterator = tqdm(csv_files, desc="scan limits", unit="file")

    for idx, path in enumerate(iterator):
        rows = read_rows(path)
        grids, xs, ys = build_grids(rows)

        if idx == 0:
            x_span = xs[-1] - xs[0] if len(xs) > 1 else 1.0
            y_span = ys[-1] - ys[0] if len(ys) > 1 else 1.0
            domain_span = max(min(x_span, y_span), 1e-12)

        psi_values.append(grids["psi"].ravel())
        omega_values.append(grids["omega"].ravel())
        speed_values.append(np.hypot(grids["u"], grids["v"]).ravel())

    psi_all = np.concatenate(psi_values)
    omega_all = np.concatenate(omega_values)
    speed_all = np.concatenate(speed_values)
    speed_max = max(float(np.max(speed_all)), 1e-12)

    return {
        "psi_levels": symmetric_levels(psi_all),
        "omega_levels": symmetric_levels(omega_all),
        "speed_levels": np.linspace(0.0, speed_max, 25),
        "arrow_factor": 0.075 * domain_span / speed_max,
        "quiver_scale": 1.0,
    }


def style_axes(ax, title, xs, ys):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])


def save_figure(fig, out_png):
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def build_scalar_frame_with_quiver(
    grids, xs, ys, out_png, field, levels, cmap, colorbar_label, title, stats, base
):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    try:
        xs_plot = regularize_axis(xs)
        ys_plot = regularize_axis(ys)
        x_grid, y_grid = np.meshgrid(xs_plot, ys_plot)
        skip = max(1, min(len(xs_plot), len(ys_plot)) // 24)
        scalar = grids[field]

        contourf = ax.contourf(
            x_grid, y_grid, scalar, levels=levels, cmap=cmap
        )
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
        speed = np.hypot(grids["u"], grids["v"])
        x_grid, y_grid = np.meshgrid(xs_plot, ys_plot)

        bg = ax.contourf(
            x_grid, y_grid, speed, levels=stats["speed_levels"], cmap="viridis"
        )
        stream = ax.streamplot(
            xs_plot,
            ys_plot,
            grids["u"],
            grids["v"],
            color=speed,
            cmap="plasma",
            linewidth=0.8 + 1.6 * speed / max(float(np.max(speed)), 1e-12),
            density=1.45,
            arrowsize=1.0,
        )
        stream.arrows.set_color("white")
        fig.colorbar(bg, ax=ax, label="|u|")
        style_axes(ax, f"Streamplot, {base}", xs_plot, ys_plot)
        save_figure(fig, out_png)
    finally:
        plt.close(fig)


def make_frame_task(task):
    kind, csv_path, out_png, stats = task
    rows = read_rows(csv_path)
    grids, xs, ys = build_grids(rows)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    if kind == "psi":
        build_scalar_frame_with_quiver(
            grids,
            xs,
            ys,
            out_png,
            "psi",
            stats["psi_levels"],
            "coolwarm",
            "psi",
            "Psi field",
            stats,
            base,
        )
    elif kind == "omega":
        build_scalar_frame_with_quiver(
            grids,
            xs,
            ys,
            out_png,
            "omega",
            stats["omega_levels"],
            "RdBu_r",
            "omega",
            "Omega field",
            stats,
            base,
        )
    elif kind == "streamplot":
        build_streamplot_frame(grids, xs, ys, out_png, stats, base)
    else:
        raise RuntimeError(f"Unknown frame kind: {kind}")

    return kind, out_png


def make_gif(frame_paths, gif_path, duration_ms=120):
    if len(frame_paths) < 2:
        return False

    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
    )
    return True


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
    if len(sys.argv) < 4:
        print("Usage: plot_fields.py <results_dir> <frames_dir> <gifs_dir>")
        sys.exit(1)

    results_dir = os.path.abspath(os.path.normpath(sys.argv[1]))
    frames_dir = os.path.abspath(os.path.normpath(sys.argv[2]))
    gifs_dir = os.path.abspath(os.path.normpath(sys.argv[3]))
    plot_root = os.path.dirname(frames_dir)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(gifs_dir, exist_ok=True)
    os.makedirs(plot_root, exist_ok=True)

    csv_files = sorted(
        glob.glob(os.path.join(results_dir, "result_*.csv")),
        key=step_sort_key,
    )
    if not csv_files:
        raise RuntimeError(f"No result_*.csv found in {results_dir}")

    stats = collect_plot_stats(csv_files)

    tasks = []
    for csv_path in csv_files:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        for kind in FRAME_KINDS:
            out_png = os.path.join(frames_dir, f"{base}_{kind}.png")
            tasks.append((kind, csv_path, out_png, stats))

    frame_paths = {kind: [] for kind in FRAME_KINDS}
    workers = max(1, min(os.cpu_count() or 1, len(tasks)))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(make_frame_task, task) for task in tasks]

        progress_total = len(futures)
        if tqdm is not None:
            with tqdm(total=progress_total, desc="render frames", unit="frame") as bar:
                for fut in as_completed(futures):
                    kind, out_png = fut.result()
                    frame_paths[kind].append(out_png)
                    bar.update(1)
        else:
            for fut in as_completed(futures):
                kind, out_png = fut.result()
                frame_paths[kind].append(out_png)

    for kind in FRAME_KINDS:
        frame_paths[kind].sort(key=step_sort_key)

    for kind, gif_name in GIF_NAMES.items():
        gif_path = os.path.join(gifs_dir, gif_name)
        if make_gif(frame_paths[kind], gif_path):
            print(f"[plot] gif: {gif_path}")
        else:
            print(f"[plot] Only one frame for {kind}: GIF skipped")

    build_residual_plot(results_dir, plot_root)


if __name__ == "__main__":
    main()
