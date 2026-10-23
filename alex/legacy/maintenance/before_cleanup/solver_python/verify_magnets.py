#!/usr/bin/env python3
"""Verify the magnets from the ONE working YAML, then run and read its snapshots."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.snapshot_io import read_snapshot
from solver_python.config import grid, load_config, parse_config
from solver_python.forcing import build_forcing, dipole_field
from solver_python.run import run
from solver_python.solver import Solver


def verify(config_path: Path, steps: int, output_dir: Path) -> dict:
    cfg = load_config(config_path)
    if cfg.forcing["type"] != "magnets":
        raise ValueError("Set forcing.type: magnets in the working YAML before running this check")
    if steps < 2:
        raise ValueError("Verification requires at least two steps")
    data = copy.deepcopy(cfg.resolved)
    data["time"] = {"mode": "fixed_dt_steps", "dt": cfg.dt, "n_time_steps": steps,
                    "steady_tolerance": cfg.steady_tolerance}
    cfg = parse_config(data, cfg.path)
    f = cfg.forcing
    xs, ys = grid(cfg)
    length, speed = f["scales"]["length_m"], f["scales"]["velocity_m_s"]
    x, y = xs[:, None]*length, ys[None, :]*length
    magnets, height = f["magnets"], f["plane_height_m"]
    bz, dbx, dby = dipole_field(x, y, magnets, height)

    # Test each magnet in isolation: other magnets contribute at its axis too.
    axis_errors = []
    for magnet in magnets:
        xk, yk, _, b = magnet
        value = float(dipole_field(xk, yk, [magnet], height)[0])
        np.testing.assert_allclose(value, b, rtol=1e-12, atol=1e-15)
        axis_errors.append(abs(value-b))
    individual_sum = sum(dipole_field(x, y, [m], height)[0] for m in magnets)
    np.testing.assert_allclose(bz, individual_sum, rtol=1e-13, atol=1e-15)

    # Differentiate field VALUES independently; this checks the analytic derivatives.
    eps = min(m[2]+height for m in magnets)*1e-5
    numerical_dx = (dipole_field(x+eps, y, magnets, height)[0]-dipole_field(x-eps, y, magnets, height)[0])/(2*eps)
    numerical_dy = (dipole_field(x, y+eps, magnets, height)[0]-dipole_field(x, y-eps, magnets, height)[0])/(2*eps)
    derivative_scale = max(float(np.max(np.abs(dbx))), float(np.max(np.abs(dby))), 1e-30)
    derivative_error = max(float(np.max(np.abs(dbx-numerical_dx))), float(np.max(np.abs(dby-numerical_dy))))/derivative_scale
    if derivative_error > 1e-7:
        raise AssertionError(f"Magnetic derivative relative max error: {derivative_error}")
    source, saved_bz = build_forcing(cfg, xs, ys)
    jx, jy = f["current_density_A_m2"]
    numerical_source = -(jx*numerical_dx+jy*numerical_dy)/f["density_kg_m3"]*(length/speed)**2
    source_scale = max(float(np.max(np.abs(source))), 1e-30)
    source_error = float(np.max(np.abs(source-numerical_source)))/source_scale
    if source_error > 1e-7:
        raise AssertionError(f"Vorticity source relative max error: {source_error}")
    np.testing.assert_array_equal(saved_bz, bz)

    reversed_data = copy.deepcopy(data)
    reversed_data["forcing"]["magnets"] = [m[:3]+[-m[3]] for m in magnets]
    reversed_cfg = parse_config(reversed_data, cfg.path)
    reverse_source, reverse_bz = build_forcing(reversed_cfg, xs, ys)
    np.testing.assert_allclose(reverse_source, -source, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(reverse_bz, -bz, rtol=1e-13, atol=1e-15)

    # With zero initial flow and stationary walls, a nonzero source must drive omega.
    stationary_walls = all(w.u == 0 and w.v == 0 for w in (cfg.bc.left, cfg.bc.right, cfg.bc.bottom, cfg.bc.top))
    source_active = bool(np.any(source[1:-1, 1:-1]))
    first_step = Solver(cfg)
    first_step.step()
    first_response = float(np.sum(first_step.omega[1:-1, 1:-1]*source[1:-1, 1:-1]))
    if stationary_walls and source_active and first_response <= 0:
        raise AssertionError("First vorticity step does not respond with the expected source sign")

    print(f"[magnets] field/source checks passed; derivative error={derivative_error:.3e}, "
          f"source error={source_error:.3e}", flush=True)
    summary = run(cfg, output_dir=output_dir)
    output_dir = Path(summary["output_dir"])
    snapshot_paths = sorted(output_dir.glob("result_*.bin"), key=lambda p: int(p.stem.split("_")[-1]))
    for path in snapshot_paths:
        snapshot = read_snapshot(path)
        for name in ("psi", "omega", "u", "v"):
            if not np.isfinite(getattr(snapshot, name)).all():
                raise AssertionError(f"Non-finite {name} in {path.name}")
        np.testing.assert_allclose(snapshot.time, (snapshot.step+1)*cfg.dt, rtol=1e-14)
    last = read_snapshot(snapshot_paths[-1])
    assert last.step == steps-1
    if stationary_walls and source_active and (not np.any(last.psi) or not np.any(last.omega)):
        raise AssertionError("Nonzero source produced a zero final flow")
    with np.load(output_dir/"forcing.npz") as stored:
        np.testing.assert_array_equal(stored["Bz_T"], bz)
        np.testing.assert_array_equal(stored["source"], source)
        np.testing.assert_array_equal(stored["xs"], xs)
        np.testing.assert_array_equal(stored["ys"], ys)
    report = {
        "passed": True, "magnet_count": len(magnets), "grid": [cfg.nx, cfg.ny],
        "completed_steps": summary["completed_steps"], "snapshots_checked": len(snapshot_paths),
        "max_axis_normalization_error_T": max(axis_errors),
        "derivative_relative_max_error": derivative_error, "source_relative_max_error": source_error,
        "Bz_range_T": [float(bz.min()), float(bz.max())],
        "source_range_nondimensional": [float(source.min()), float(source.max())],
        "first_step_source_response": first_response,
        "final_psi_range": [float(last.psi.min()), float(last.psi.max())],
        "final_omega_range": [float(last.omega.min()), float(last.omega.max())],
        "final_max_speed": float(np.hypot(last.u, last.v).max()),
        "final_psi_res": summary["psi_res"], "final_omega_res": summary["omega_res"],
        "converged": summary["converged"], "output_dir": str(output_dir),
    }
    (output_dir/"verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.config, args.steps, args.output)


if __name__ == "__main__":
    main()
