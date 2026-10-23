#!/usr/bin/env python3
"""One YAML, one simulation. All physical and threading settings live in YAML."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from common.snapshot_io import write_snapshot
from solver_python.config import Config, ConfigError, load_config
from solver_python.solver import Solver


def benchmark(cfg: Config, steps: int) -> dict:
    solver = Solver(cfg)
    start = time.perf_counter()
    solver.warm_up()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter()
    for _ in range(steps):
        solver.step()
    elapsed = time.perf_counter()-start
    return {"steps": steps, "threads": cfg.threads, "grid": [cfg.nx, cfg.ny],
            "warmup_seconds": compile_seconds, "compute_seconds": elapsed,
            "milliseconds_per_step": 1000*elapsed/steps,
            "psi_res": solver.last_residuals[0], "omega_res": solver.last_residuals[1]}


# Files this solver writes itself; their presence marks a directory as ours to replace.
OWN_OUTPUT = ("summary.json", "config.resolved.yaml", "forcing.npz")


def replaceable(out: Path) -> bool:
    """True for a missing, empty, or previously-written output directory."""
    if not out.exists():
        return True
    if not out.is_dir():
        return False
    return not any(out.iterdir()) or any((out/name).exists() for name in OWN_OUTPUT)


def run(cfg: Config, output_dir: Path | None = None) -> dict:
    out = (output_dir or cfg.output_dir).resolve()
    # Checked before the solver is built, so a bad config cannot cost old results.
    if not replaceable(out):
        raise FileExistsError(f"{out} is not empty and holds no output of this solver; "
                              "refusing to erase it. Choose another experiment name/root.")
    solver = Solver(cfg)
    print(f"[python] {cfg.experiment}: {cfg.nx}x{cfg.ny}, Re={cfg.re:g}, "
          f"dt={cfg.dt:g}, threads={cfg.threads}, forcing={cfg.forcing['type']}", flush=True)
    print("[python] Compiling/loading numerical kernels...", flush=True)
    solver.warm_up()
    # Re-running an experiment REPLACES it: snapshots of the previous run must not
    # survive, or the plotting and analysis tools would read two runs as one.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with (out / "config.resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg.resolved, stream, sort_keys=False, allow_unicode=True)
    if cfg.path.exists():
        (out / "config.input.yaml").write_text(cfg.path.read_text(encoding="utf-8"), encoding="utf-8")
    if cfg.forcing["type"] == "custom":
        (out / "source.input.py").write_bytes(Path(cfg.forcing["file"]).read_bytes())
    np.savez(out / "forcing.npz", xs=solver.xs, ys=solver.ys, source=solver.source, Bz_T=solver.bz)
    summary = {"experiment": cfg.experiment, "scheme": "factorized", "jacobian": "arakawa",
               "time_kind": "pseudo_time", "status": "running", "threads": cfg.threads,
               "config": str(cfg.path), "output_dir": str(out), "completed_steps": 0}
    if solver.stabilized:
        summary["stabilization"] = cfg.stabilization
    started = time.perf_counter()
    control_stream = (out / "control_history.csv").open("w", newline="", encoding="utf-8") if solver.stabilized else None
    control_writer = csv.writer(control_stream) if control_stream else None
    if control_writer:
        control_writer.writerow(["step", "time", "max_control"])
    try:
        with (out / "residual_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "time", "psi_res", "omega_res", "max_residual"])
            for step in range(cfg.n_time_steps):
                rp, rw = solver.step()
                pseudo_time = solver.completed_steps*cfg.dt
                converged = max(rp, rw) <= cfg.steady_tolerance
                stop = cfg.mode == "till_converges" and converged
                last = step == cfg.n_time_steps-1 or stop
                save = step % cfg.save_every_step == 0 or last
                log = step % cfg.log_every_step == 0 or last
                if save or log:
                    writer.writerow([step, f"{pseudo_time:.16e}", f"{rp:.16e}",
                                     f"{rw:.16e}", f"{max(rp, rw):.16e}"])
                    stream.flush()
                    if control_writer:
                        control_writer.writerow([step, f"{pseudo_time:.16e}", f"{solver.last_control:.16e}"])
                        control_stream.flush()
                if save:
                    write_snapshot(out / f"result_{step}.bin", solver.xs, solver.ys,
                                   solver.psi, solver.omega, step=step, time=pseudo_time,
                                   re=cfg.re, bc=cfg.bc)
                if log:
                    control = f" control={solver.last_control:.3e}" if solver.stabilized else ""
                    print(f"[python] step={step} tau={pseudo_time:.7g} "
                          f"Rpsi={rp:.3e} Romega={rw:.3e}{control}", flush=True)
                if stop:
                    break
        status = "completed" if cfg.mode != "till_converges" else ("converged" if converged else "step_limit")
        summary.update(status=status, converged=converged, psi_res=rp, omega_res=rw)
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if control_stream:
            control_stream.close()
        summary.update(completed_steps=solver.completed_steps,
                       pseudo_time=solver.completed_steps*cfg.dt,
                       elapsed_seconds=time.perf_counter()-started)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"[python] {summary['status']}: {out}", flush=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate configuration and source without running/saving")
    mode.add_argument("--benchmark", type=int, metavar="STEPS", help="time steps without output files (after JIT warm-up)")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.check:
            solver = Solver(cfg)
            print(f"Valid: {cfg.nx}x{cfg.ny}, Re={cfg.re:g}, dt={cfg.dt:g}, threads={cfg.threads}")
            print(f"Arakawa, factorized; forcing={cfg.forcing['type']}; "
                  f"source range=[{np.min(solver.source):.6g}, {np.max(solver.source):.6g}]")
            if cfg.forcing["type"] == "magnets":
                print(f"Magnets: {len(cfg.forcing['magnets'])}; "
                      f"Bz [T]=[{np.min(solver.bz):.6g}, {np.max(solver.bz):.6g}]")
            print(f"Output: {cfg.output_dir}")
            return 0
        if args.benchmark is not None:
            if args.benchmark < 1:
                raise ConfigError("--benchmark must be a positive integer")
            print(json.dumps(benchmark(cfg, args.benchmark), indent=2, allow_nan=False))
            return 0
        result = run(cfg)
        return 2 if result["status"] == "step_limit" else 0
    except (ValueError, OSError, FloatingPointError, OverflowError) as exc:
        print(f"[python] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
