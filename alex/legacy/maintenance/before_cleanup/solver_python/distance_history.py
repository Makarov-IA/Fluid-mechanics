#!/usr/bin/env python3
"""Distance to the Newton equilibrium for every SAVED snapshot of both runs.

Nothing is recomputed: the snapshots of the run without control and of the run
with control are read as they are. Norms are relative L2 over the whole grid,
psi and omega separately (omega is ~300 times larger than psi, a joint norm
would hide psi). One CSV row per snapshot, so every saved step can be inspected.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from common.snapshot_io import read_snapshot
from solver_python.config import load_config

LABELS = ("без поправки", "с поправкой")


def snapshots(directory: Path):
    paths = sorted(directory.glob("result_*.bin"), key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise FileNotFoundError(f"No result_*.bin in {directory}")
    return paths


def distance_history(directory: Path, psi_star: np.ndarray, omega_star: np.ndarray) -> np.ndarray:
    """Rows (step, time, |psi-psi*|/|psi*|, |omega-omega*|/|omega*|), one per snapshot."""
    psi_scale, omega_scale = np.linalg.norm(psi_star), np.linalg.norm(omega_star)
    rows = []
    for path in snapshots(directory):
        s = read_snapshot(path)
        rows.append((s.step, s.time,
                     np.linalg.norm(s.psi - psi_star)/psi_scale,
                     np.linalg.norm(s.omega - omega_star)/omega_scale))
    return np.array(rows)


def write_table(path: Path, history: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time", "psi_distance", "omega_distance"])
        for step, t, dp, dw in history:
            writer.writerow([int(step), f"{t:.8e}", f"{dp:.12e}", f"{dw:.12e}"])


def plot(path: Path, histories: dict[str, np.ndarray]):
    figure, panels = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for name, h in histories.items():
        for axes, column in zip(panels, (2, 3)):
            axes.semilogy(h[:, 1], h[:, column], label=name)
    for axes, title in zip(panels, ("‖ψ − ψ*‖ / ‖ψ*‖", "‖ω − ω*‖ / ‖ω*‖")):
        axes.set_title(title)
        axes.set_xlabel("τ")
    panels[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--uncontrolled", required=True, help="experiment name of the run without control")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.stabilization is None:
        raise ValueError("The working YAML has no stabilization block")
    root = Path(cfg.resolved["output"]["root"])
    with np.load(cfg.stabilization["file"]) as data:
        psi_star, omega_star = data["psi_star"], data["omega_star"]

    runs = dict(zip(LABELS, ((args.uncontrolled, root/"binaries"/args.uncontrolled),
                             (cfg.experiment, cfg.output_dir))))
    histories = {}
    for name, (experiment, directory) in runs.items():
        histories[name] = h = distance_history(directory, psi_star, omega_star)
        table = root/"tables"/experiment/"distance_to_equilibrium.csv"
        write_table(table, h)
        best = int(np.argmin(h[:, 3]))
        print(f"[distance] {name}: {len(h)} snapshots -> {table}")
        print(f"[distance]   min ω-distance {h[best, 3]:.3e} at step {int(h[best, 0])} (τ={h[best, 1]:g}); "
              f"last ψ {h[-1, 2]:.3e}, ω {h[-1, 3]:.3e}")
    figure = root/"figures"/cfg.experiment/"distance_history.png"
    plot(figure, histories)
    print(f"[distance] figure: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
