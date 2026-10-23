#!/usr/bin/env python3
"""Uncontrolled vs controlled run: distance to the Newton equilibrium, residual, control."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from solver_python.config import load_config
from solver_python.distance_history import distance_history, plot as distance_plot


def column(path: Path, name: str):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return np.array([float(r["time"]) for r in rows]), np.array([float(r[name]) for r in rows])


def log_plot(path: Path, title: str, curves):
    figure, axes = plt.subplots(figsize=(8, 4))
    for t, y, label in curves:
        axes.semilogy(t, np.maximum(y, 1e-300), label=label)
    axes.set_title(title)
    axes.set_xlabel("τ")
    if len(curves) > 1:
        axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def compare(config_path: Path, uncontrolled: str) -> dict:
    cfg = load_config(config_path)
    if cfg.stabilization is None:
        raise ValueError("The working YAML has no stabilization block")
    root = Path(cfg.resolved["output"]["root"])
    runs = {"без поправки": root/"binaries"/uncontrolled, "с поправкой": cfg.output_dir}
    with np.load(cfg.stabilization["file"]) as data:
        psi_star, omega_star = data["psi_star"], data["omega_star"]
    out = root/"figures"/cfg.experiment
    out.mkdir(parents=True, exist_ok=True)

    distances = {name: distance_history(d, psi_star, omega_star) for name, d in runs.items()}
    distance_plot(out/"distance_to_equilibrium.png", distances)
    residuals = {name: column(d/"residual_history.csv", "omega_res") for name, d in runs.items()}
    log_plot(out/"omega_residual.png", "Невязка R_ω",
             [(t, y, name) for name, (t, y) in residuals.items()])
    t, control = column(cfg.output_dir/"control_history.csv", "max_control")
    log_plot(out/"control.png", "max|σ P(ω − ω*)|", [(t, control, "")])

    report = {name: {"final_distance_psi": float(d[-1, 2]), "final_distance_omega": float(d[-1, 3]),
                     "min_distance_omega": float(d[:, 3].min()),
                     "time_of_min_omega": float(d[np.argmin(d[:, 3]), 1]),
                     "final_omega_residual": float(residuals[name][1][-1])}
              for name, d in distances.items()}
    report["с поправкой"]["final_control"] = float(control[-1])
    (out/"comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[compare] figures: {out}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--uncontrolled", required=True, help="experiment name of the run without control")
    args = parser.parse_args(argv)
    compare(args.config, args.uncontrolled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
