#!/usr/bin/env python3
"""Draw the forcing of the ONE working YAML before spending time on a run.

Field and source come from the same grid() and build_forcing() calls that
Solver.__init__ makes, so the picture is exactly what the run would use.
Nothing is integrated here and no snapshot is written.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from solver_python.config import ConfigError, grid, load_config
from solver_python.forcing import build_forcing

MODES_X = 8
MODES_Y = 6
SHOWN_MODES = 8


def sine_spectrum(source, xs, ys, lx, ly):
    """Project S onto sin(m*pi*x/lx)*sin(n*pi*y/ly), the basis with S=0 on the walls.

    The x integral does not depend on n, so it is computed once per m.
    """
    coefficients = np.zeros((MODES_X, MODES_Y))
    for a in range(MODES_X):
        column = np.trapezoid(source*np.sin((a+1)*np.pi*xs/lx)[:, None], xs, axis=0)
        for b in range(MODES_Y):
            coefficients[a, b] = np.trapezoid(column*np.sin((b+1)*np.pi*ys/ly), ys)
    return 4.0/(lx*ly)*coefficients


def ranked_modes(coefficients):
    """Modes ordered by magnitude, normalised to the strongest one."""
    flat = np.argsort(np.abs(coefficients), axis=None)[::-1][:SHOWN_MODES]
    peak = coefficients.flat[flat[0]] if coefficients.any() else 1.0
    modes = []
    for index in flat:
        a, b = np.unravel_index(index, coefficients.shape)
        modes.append({"m": int(a+1), "n": int(b+1),
                      "coefficient": float(coefficients[a, b]),
                      "relative": float(coefficients[a, b]/peak)})
    return modes


def field_panel(figure, axes, xs, ys, values, title):
    span = float(np.max(np.abs(values)))
    if span == 0.0:
        span = 1.0
    image = axes.contourf(xs, ys, values.T, levels=np.linspace(-span, span, 31),
                          cmap="RdBu_r", extend="neither")
    axes.set_aspect("equal")
    axes.set_title(title)
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    return axes


def mark_magnets(axes, magnets, length):
    """Positions only: the colour map already shows each magnet's polarity."""
    for xk, yk, _, _ in magnets:
        axes.plot(xk/length, yk/length, marker="o", markersize=7, markerfacecolor="none",
                  markeredgecolor="black", markeredgewidth=1.2, linestyle="none")


def preview(config_path: Path) -> dict:
    cfg = load_config(config_path)
    # The same two calls Solver.__init__ makes; the physics is not recomputed here.
    xs, ys = grid(cfg)
    source, bz = build_forcing(cfg, xs, ys)
    magnetic = cfg.forcing["type"] == "magnets"
    coefficients = sine_spectrum(source, xs, ys, cfg.lx, cfg.ly)
    modes = ranked_modes(coefficients)

    # Only the two fields are drawn; the spectrum stays in the printout and the report.
    figure, panels = plt.subplots(1, 2 if magnetic else 1,
                                  figsize=(12 if magnetic else 6, 4.2), squeeze=False)
    panels = panels[0]
    if magnetic:
        field_panel(figure, panels[0], xs, ys, 1e3*bz, "Bz [mT]")
        mark_magnets(panels[0], cfg.forcing["magnets"], cfg.forcing["scales"]["length_m"])
    field_panel(figure, panels[-1], xs, ys, source, "S")
    figure.tight_layout()

    out = Path(cfg.resolved["output"]["root"])/"figures"/cfg.experiment
    out.mkdir(parents=True, exist_ok=True)
    # A preview is re-drawn while tuning the magnets, so it overwrites itself.
    image_path = out/"forcing_preview.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)

    report = {
        "experiment": cfg.experiment, "config": str(cfg.path), "forcing": cfg.forcing["type"],
        "grid": [cfg.nx, cfg.ny], "lx": cfg.lx, "ly": cfg.ly, "re": cfg.re,
        "source_range_nondimensional": [float(source.min()), float(source.max())],
        "source_spectrum": modes, "figure": str(image_path),
    }
    if magnetic:
        report["magnets"] = cfg.forcing["magnets"]
        report["length_scale_m"] = cfg.forcing["scales"]["length_m"]
        report["Bz_range_T"] = [float(bz.min()), float(bz.max())]
    (out/"forcing_preview.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                            encoding="utf-8")

    print(f"[preview] {cfg.experiment}: {cfg.forcing['type']} forcing, "
          f"{cfg.nx}x{cfg.ny}, Re={cfg.re:g}", flush=True)
    if magnetic:
        print(f"[preview] {len(cfg.forcing['magnets'])} magnets, "
              f"Bz in [{1e3*bz.min():.4g}, {1e3*bz.max():.4g}] mT", flush=True)
    print(f"[preview] S in [{source.min():.6g}, {source.max():.6g}]", flush=True)
    for mode in modes:
        print(f"[preview]   ({mode['m']},{mode['n']}) {mode['coefficient']:+.6g}"
              f"   relative {mode['relative']:+.4f}", flush=True)
    print(f"[preview] {image_path}", flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--open", action="store_true", help="show the figure in the desktop viewer")
    args = parser.parse_args(argv)
    try:
        report = preview(args.config)
        if args.open:
            subprocess.run(["open", report["figure"]], check=True)
        return 0
    except (ConfigError, ValueError, OSError, FloatingPointError, OverflowError) as exc:
        print(f"[preview] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
