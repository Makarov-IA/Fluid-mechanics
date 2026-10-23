#!/usr/bin/env python3
"""Compare base, full post-step projection, and full-state RHS feedback."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from solver_python.config import load_config
from solver_python.distance_history import snapshots
from solver_python.snapshot_io import read_snapshot


def compare(base_path: Path, projection_path: Path, rhs_path: Path, time_offset: float = 0.) -> dict:
    if not np.isfinite(time_offset) or time_offset < 0:
        raise ValueError("time_offset must be finite and nonnegative")
    configs = [load_config(p) for p in (base_path, projection_path, rhs_path)]
    if configs[0].stabilization is not None:
        raise ValueError("Base run must be uncontrolled")
    for cfg, method in zip(configs[1:], ("projection", "rhs")):
        if not cfg.stabilization or (cfg.stabilization["components"], cfg.stabilization["method"]) != ("state", method):
            raise ValueError(f"Expected full-state {method}: {cfg.path}")
    bundle = Path(configs[1].stabilization["file"])
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    with np.load(bundle) as data:
        pstar, wstar, basis = data["psi_star"], data["omega_star"], data["state_basis"]
        xs, ys, re = data["xs"], data["ys"], float(data["re"])
    target = np.concatenate((pstar[1:-1, 1:-1].ravel(), wstar[1:-1, 1:-1].ravel()))
    scales = (np.linalg.norm(pstar), np.linalg.norm(wstar))
    histories, residuals, report = {}, {}, {}
    reference_config, reference_source, reference_times = None, None, None
    for cfg in configs:
        directory = cfg.output_dir
        summary = json.loads((directory/"summary.json").read_text())
        if summary["status"] != "completed" or summary["completed_steps"] != cfg.n_time_steps:
            raise ValueError(f"Incomplete run: {directory}")
        recorded = yaml.safe_load((directory/"config.resolved.yaml").read_text())
        common = {key: recorded[key] for key in ("grid", "physics", "time", "boundaries")}
        with np.load(directory/"forcing.npz") as data:
            source = data["source"]
        if reference_config is None:
            reference_config, reference_source = common, source
        elif common != reference_config or not np.array_equal(source, reference_source):
            raise ValueError("Runs differ in numerical/physical parameters or source")
        if cfg.stabilization:
            if summary.get("stabilization_sha256") != digest:
                raise ValueError("Runs used different stabilization bundles")
            if summary["stabilization"] != cfg.stabilization:
                raise ValueError("Control settings changed since the recorded run")
        rows = []
        for path in snapshots(directory):
            s = read_snapshot(path)
            if s.re != re or not np.array_equal(s.xs, xs) or not np.array_equal(s.ys, ys):
                raise ValueError(f"Snapshot does not match equilibrium: {path}")
            z = np.concatenate((s.psi[1:-1, 1:-1].ravel(), s.omega[1:-1, 1:-1].ravel()))
            rows.append((s.step, s.time+time_offset, np.linalg.norm(s.psi-pstar)/scales[0],
                         np.linalg.norm(s.omega-wstar)/scales[1], np.linalg.norm(basis@(z-target))))
        h = histories[cfg.experiment] = np.array(rows)
        if reference_times is None:
            reference_times = h[:, :2]
        elif not np.array_equal(reference_times, h[:, :2]):
            raise ValueError("Snapshots have different steps/times")
        tables = cfg.result_dir/"tables"
        tables.mkdir(parents=True, exist_ok=True)
        np.savetxt(tables/"state_projection_history.csv", h, delimiter=",", comments="",
                   header="step,time,psi_distance,omega_distance,modal_norm",
                   fmt=["%d"]+["%.16e"]*4)
        with (directory/"residual_history.csv").open() as stream:
            r = list(csv.DictReader(stream))
        residuals[cfg.experiment] = np.array([(float(row["time"])+time_offset, float(row["psi_res"]),
                                              float(row["omega_res"])) for row in r])
        report[cfg.experiment] = dict(
            snapshots=len(h), completed_steps=summary["completed_steps"], pseudo_time=summary["pseudo_time"]+time_offset,
            run_pseudo_time=summary["pseudo_time"],
            converged=summary["converged"], final_psi_distance=float(h[-1, 2]),
            final_omega_distance=float(h[-1, 3]), final_modal_norm=float(h[-1, 4]),
            max_saved_modal_norm=float(h[:, 4].max()),
            min_psi_distance=float(h[:, 2].min()), time_of_min_psi=float(h[h[:, 2].argmin(), 1]),
            min_omega_distance=float(h[:, 3].min()), time_of_min_omega=float(h[h[:, 3].argmin(), 1]),
            final_psi_residual=summary["psi_res"], final_omega_residual=summary["omega_res"],
            elapsed_seconds=summary["elapsed_seconds"], baseline_reused_from=summary.get("reused_from"))
    out = configs[0].result_dir/"figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    labels = ("Base", "Post-step projection (psi, omega)", "RHS (psi, omega), gain=1")
    for cfg, label in zip(configs, labels):
        h, r = histories[cfg.experiment], residuals[cfg.experiment]
        for ax, col in zip(axes[0], (2, 3)):
            ax.semilogy(h[:, 1], np.maximum(h[:, col], 1e-18), label=label)
        for ax, col in zip(axes[1], (1, 2)):
            ax.semilogy(r[:, 0], np.maximum(r[:, col], 1e-18), label=label)
    titles = (r"Relative distance $\|\psi-\psi_*\|/\|\psi_*\|$",
              r"Relative distance $\|\omega-\omega_*\|/\|\omega_*\|$",
              r"Original residual $R_\psi$", r"Original residual $R_\omega$")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Pseudo-time")
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"projection_comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for cfg, label in zip(configs, labels):
        h = histories[cfg.experiment]
        ax.semilogy(h[:, 1], np.maximum(h[:, 4], 1e-18), label=label)
    ax.set(xlabel="Pseudo-time", ylabel="Unstable modal coefficient norm",
           title="Projection onto the fixed paired basis (interior Euclidean metric)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"projection_modes.png", dpi=150)
    plt.close(fig)
    output = dict(metric="relative full-grid Euclidean distances, separately psi and omega",
                  time_offset=time_offset,
                  modal_metric="unweighted Euclidean, concatenated interior psi and omega",
                  sampling="saved snapshots; residual CSV rows",
                  intervention="full subtraction versus dt*gain RHS term; different strengths",
                  runs=report)
    (out/"projection_comparison.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--rhs", required=True, type=Path)
    parser.add_argument("--time-offset", type=float, default=0., help="explicit offset for plot/table times; does not change snapshots")
    args = parser.parse_args()
    compare(args.base, args.projection, args.rhs, args.time_offset)
