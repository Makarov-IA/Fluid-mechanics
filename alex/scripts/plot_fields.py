import csv
import glob
import os
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

# Оставляем только psi и omega для основных графиков
FIELDS = ["psi", "omega"]
COL_IDX = {"psi": 2, "omega": 3, "u": 4, "v": 5}


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


def build_grid(rows, col_idx):
    xs = sorted({r[0] for r in rows})
    ys = sorted({r[1] for r in rows})
    nx = len(xs)
    ny = len(ys)

    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_j = {y: j for j, y in enumerate(ys)}

    arr = np.zeros((ny, nx), dtype=float)
    for r in rows:
        i = x_to_i[r[0]]
        j = y_to_j[r[1]]
        arr[j, i] = r[col_idx]

    return arr, xs, ys


def collect_limits(csv_files):
    values = {name: [] for name in FIELDS}

    iterator = csv_files
    if tqdm is not None:
        iterator = tqdm(csv_files, desc="scan limits", unit="file")

    for path in iterator:
        rows = read_rows(path)
        for name in FIELDS:
            vals = np.array([r[COL_IDX[name]] for r in rows], dtype=float)
            values[name].append(vals)

    out = {}
    for name in FIELDS:
        arr = np.concatenate(values[name])

        # Robust limits: ignore extreme outliers that flatten dynamics in colormap.
        q_low = float(np.percentile(arr, 5.0))
        q_high = float(np.percentile(arr, 95.0))

        if name in ("omega", "psi"):
            # For signed fields, keep symmetric scale around 0 for visual consistency.
            bound = max(abs(q_low), abs(q_high))
            vmin = -bound
            vmax = bound
        else:
            vmin = q_low
            vmax = q_high

        if abs(vmax - vmin) < 1e-14:
            eps = 1e-12
            vmin -= eps
            vmax += eps

        out[name] = (vmin, vmax)

    return out


def make_frame(csv_path, out_png, limits):
    rows = read_rows(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    base = os.path.splitext(os.path.basename(csv_path))[0]
    fig.suptitle(base)

    # Строим сетки для всех полей
    grid_psi, xs, ys = build_grid(rows, COL_IDX["psi"])
    grid_omega, _, _ = build_grid(rows, COL_IDX["omega"])
    grid_u, _, _ = build_grid(rows, COL_IDX["u"])
    grid_v, _, _ = build_grid(rows, COL_IDX["v"])

    X, Y = np.meshgrid(xs, ys)
    skip = 12

    # Автоматический подбор factor на основе масштаба скоростей
    speed = np.sqrt(grid_u**2 + grid_v**2)
    max_speed = np.max(speed)
    
    # Размер ячейки сетки
    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    grid_spacing = min(dx, dy)
    
    # Желаемая длина самой большой стрелки (доля от размера ячейки)
    # Например, 0.4 означает 40% от размера ячейки
    desired_arrow_fraction = 0.4
    
    # scale в quiver: чем больше, тем мельче стрелки
    quiver_scale = 20
    
    factor = 1.1

    for ax, field in zip(axes, FIELDS):
        grid = grid_psi if field == "psi" else grid_omega
        vmin, vmax = limits[field]

        # Тепловая карта
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            aspect="equal",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{field} field")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Векторы скорости с адаптивным factor
        ax.quiver(
            X[::skip, ::skip],
            Y[::skip, ::skip],
            factor * grid_u[::skip, ::skip],
            factor * grid_v[::skip, ::skip],
            color="black",
            pivot="mid",
            width=0.005,
            headwidth=5.0,
            headlength=6.0,
            headaxislength=5.0,
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            alpha=0.9
        )
        
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_aspect("equal")

    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def make_frame_task(task):
    csv_path, out_png, limits = task
    make_frame(csv_path, out_png, limits)
    return out_png


def make_gif(frame_paths, gif_path, duration_ms=120):
    if len(frame_paths) < 2:
        return False

    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
    )
    return True


def main():
    if len(sys.argv) < 4:
        print("Usage: plot_fields.py <results_dir> <frames_dir> <gifs_dir>")
        sys.exit(1)

    results_dir = os.path.abspath(os.path.normpath(sys.argv[1]))
    frames_dir = os.path.abspath(os.path.normpath(sys.argv[2]))
    gifs_dir = os.path.abspath(os.path.normpath(sys.argv[3]))
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(gifs_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(results_dir, "result_*.csv")))
    if not csv_files:
        raise RuntimeError(f"No result_*.csv found in {results_dir}")

    limits = collect_limits(csv_files)

    tasks = []
    for csv_path in csv_files:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_png = os.path.join(frames_dir, f"{base}_fields.png")
        tasks.append((csv_path, out_png, limits))

    frame_paths = []
    workers = max(1, min(os.cpu_count() or 1, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(make_frame_task, task) for task in tasks]

        progress_total = len(futures)
        if tqdm is not None:
            with tqdm(total=progress_total, desc="render frames", unit="frame") as bar:
                for fut in as_completed(futures):
                    frame_paths.append(fut.result())
                    bar.update(1)
        else:
            for fut in as_completed(futures):
                frame_paths.append(fut.result())

    frame_paths.sort()

    gif_path = os.path.join(gifs_dir, "evolution.gif")
    if make_gif(frame_paths, gif_path):
        print(f"[plot] gif: {gif_path}")
    else:
        print("[plot] Only one frame found: GIF skipped")


if __name__ == "__main__":
    main()