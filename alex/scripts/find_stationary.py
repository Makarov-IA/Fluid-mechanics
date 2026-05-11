#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import snapshot_io

matplotlib.use("Agg")


RESULT_RE = re.compile(r"result_(\d+)\.bin$")


def parse_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def collect_result_files(results_dir: Path, skip_newest: int) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in results_dir.glob("result_*.bin"):
        match = RESULT_RE.match(path.name)
        if match is None:
            continue
        files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    if skip_newest > 0:
        files = files[:-skip_newest]
    return files


def collect_result_files_from_index(index_csv: Path, skip_newest: int) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    if not index_csv.exists():
        raise RuntimeError(f"Snapshot index not found: {index_csv}")

    with index_csv.open("r", newline="") as f:
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
            path = Path(raw_path).expanduser().resolve()
            match = RESULT_RE.match(path.name)
            if match is None or not path.exists():
                continue
            files.append((int(match.group(1)), path))

    files.sort(key=lambda item: item[0])
    if skip_newest > 0:
        files = files[:-skip_newest]
    return files


def read_residual_times(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}

    times: dict[int, float] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                times[int(float(row["step"]))] = float(row["time"])
            except (KeyError, TypeError, ValueError):
                continue
    return times


def estimate_time(step: int, cfg: dict[str, str], residual_times: dict[int, float]) -> float:
    if step in residual_times:
        return residual_times[step]

    dt = float(cfg.get("dt", "-1"))
    if dt > 0.0:
        return (step + 1) * dt

    t_max = float(cfg.get("t_max", "1"))
    n_time_steps = float(cfg.get("n_time_steps", "1"))
    return (step + 1) * t_max / n_time_steps


def load_state(path: Path, field: str) -> np.ndarray:
    snapshot = snapshot_io.read_snapshot(path)
    if field == "psi":
        return snapshot.psi.reshape(-1)
    if field == "omega":
        return snapshot.omega.reshape(-1)
    if field == "state":
        return np.concatenate((snapshot.psi.reshape(-1), snapshot.omega.reshape(-1)))
    raise ValueError(f"Unknown field: {field}")


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    xs, ys, fields_xy = snapshot_io.load_snapshot(path)
    fields = {name: values.T for name, values in fields_xy.items()}
    return xs, ys, fields


def compute_differences(
    files: list[tuple[int, Path]],
    cfg: dict[str, str],
    residual_times: dict[int, float],
    field: str,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    previous_step: int | None = None
    previous_state: np.ndarray | None = None
    processed = 0

    print(f"[stationary] scanning files from scratch: {len(files)}")

    for step, path in files:
        try:
            state = load_state(path, field)
        except Exception as exc:
            print(f"[stationary] skip unreadable file {path}: {exc}")
            continue

        if previous_step is not None and previous_state is not None:
            diff = state - previous_state
            abs_l2 = float(np.linalg.norm(diff))
            rel_l2 = abs_l2 / max(float(np.linalg.norm(state)), 1e-30)
            linf = float(np.max(np.abs(diff)))
            rows.append(
                {
                    "previous_step": previous_step,
                    "step": step,
                    "time": estimate_time(step, cfg, residual_times),
                    "abs_l2": abs_l2,
                    "rel_l2": rel_l2,
                    "linf": linf,
                }
            )

        previous_step = step
        previous_state = state
        processed += 1
        if processed % 100 == 0:
            print(f"[stationary] scanned {processed}/{len(files)} files, current step {step}", flush=True)

    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["previous_step", "step", "time", "abs_l2", "rel_l2", "linf"])
        for row in rows:
            writer.writerow(
                [
                    row["previous_step"],
                    row["step"],
                    f"{row['time']:.15g}",
                    f"{row['abs_l2']:.15g}",
                    f"{row['rel_l2']:.15g}",
                    f"{row['linf']:.15g}",
                ]
            )


def plot_norm(path: Path, rows: list[dict[str, float | int]], field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([float(row["time"]) for row in rows], dtype=float)
    rel_l2 = np.array([float(row["rel_l2"]) for row in rows], dtype=float)

    best_idx = int(np.argmin(rel_l2))
    best_time = times[best_idx]
    best_value = rel_l2[best_idx]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.semilogy(times, np.maximum(rel_l2, 1e-30), color="#1565c0", linewidth=1.5)
    ax.scatter([best_time], [best_value], color="#c62828", s=34, zorder=3)
    ax.axvline(best_time, color="#c62828", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title(f"Difference between consecutive snapshots ({field})")
    ax.set_xlabel("time")
    ax.set_ylabel("||q_i - q_{i-1}||_2 / ||q_i||_2")
    ax.grid(True, which="both", alpha=0.35)
    ax.text(
        0.02,
        0.04,
        f"min at t={best_time:.6g}, value={best_value:.3e}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_streamplot(snapshot_path: Path, output_path: Path, title: str, density: float, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs, ys, fields = load_grid(snapshot_path)

    if "u" not in fields or "v" not in fields:
        raise RuntimeError(f"Need u and v fields for streamplot: {snapshot_path}")

    u = fields["u"]
    v = fields["v"]
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
        density=density,
        linewidth=0.7 + 1.5 * speed / max(float(speed.max()), 1e-12),
        arrowsize=1.0,
    )
    stream.arrows.set_color("white")

    if "psi" in fields:
        psi = fields["psi"]
        psi_min = float(psi.min())
        psi_max = float(psi.max())
        if psi_max > psi_min + 1e-14 * max(1.0, abs(psi_min), abs(psi_max)):
            levels = np.linspace(psi_min, psi_max, 21)
            ax.contour(x_grid, y_grid, psi, levels=levels, colors="white", linewidths=0.35, alpha=0.55)
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

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(float(ys[0]), float(ys[-1]))
    fig.colorbar(bg, ax=ax, label="|u|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot norm of difference between consecutive Alex binary snapshots.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--plot-png", type=Path, required=True)
    parser.add_argument("--stationary-snapshot", type=Path, required=True)
    parser.add_argument("--streamplot-png", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=Path, default=None)
    parser.add_argument("--field", choices=("state", "psi", "omega"), default="state")
    parser.add_argument("--skip-newest", type=int, default=1)
    parser.add_argument("--stream-density", type=float, default=1.6)
    parser.add_argument("--stream-dpi", type=int, default=180)
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    metrics_csv = args.metrics_csv.expanduser().resolve()
    plot_png = args.plot_png.expanduser().resolve()
    stationary_snapshot = args.stationary_snapshot.expanduser().resolve()
    streamplot_png = args.streamplot_png.expanduser().resolve()
    snapshot_index = args.snapshot_index.expanduser().resolve() if args.snapshot_index is not None else None

    cfg = parse_config(config_path)
    residual_times = read_residual_times(results_dir / "residual_history.csv")
    if snapshot_index is not None:
        files = collect_result_files_from_index(snapshot_index, args.skip_newest)
        print(f"[stationary] using snapshot index: {snapshot_index}")
    else:
        files = collect_result_files(results_dir, args.skip_newest)
    if len(files) < 2:
        raise RuntimeError(f"Need at least two result_*.bin files in {results_dir}")

    rows = compute_differences(files, cfg, residual_times, args.field)
    if not rows:
        raise RuntimeError("No readable consecutive snapshot pairs found")

    write_csv(metrics_csv, rows)
    plot_norm(plot_png, rows, args.field)

    best = min(rows, key=lambda row: float(row["rel_l2"]))
    best_step = int(best["step"])
    file_by_step = {step: path for step, path in files}
    best_source = file_by_step[best_step]

    stationary_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_source, stationary_snapshot)
    plot_streamplot(
        stationary_snapshot,
        streamplot_png,
        title=f"Stationary candidate, step={best_step}, t={float(best['time']):.6g}",
        density=args.stream_density,
        dpi=args.stream_dpi,
    )

    print(f"[stationary] snapshots used: {len(files)}")
    print(f"[stationary] metrics: {metrics_csv}")
    print(f"[stationary] plot: {plot_png}")
    print(f"[stationary] saved snapshot: {stationary_snapshot}")
    print(f"[stationary] streamplot: {streamplot_png}")
    print(
        "[stationary] min rel_l2: "
        f"step {best['previous_step']}..{best['step']}, "
        f"time={float(best['time']):.6g}, "
        f"value={float(best['rel_l2']):.6e}"
    )


if __name__ == "__main__":
    main()
