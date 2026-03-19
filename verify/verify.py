"""
Verification: compare Ilya's MAC/IMEX solver vs Alex's ADI psi-omega solver.

Grid alignment
--------------
Alex  : node-centred,  u[i,j] at x = i/(nx-1),       y = j/(ny-1)
Ilya  : cell-centred,  u[i,j] at x = (i+0.5)/nx,     y = (j+0.5)/ny

To bring Alex onto a cell-centred grid we average the four surrounding nodes:

    u_cell[i,j] = (u[i,j] + u[i+1,j] + u[i,j+1] + u[i+1,j+1]) / 4

This gives (nx-1)×(ny-1) cell-centre values at x = (i+0.5)/(nx-1).
Ilya's cell-centres sit at x = (i+0.5)/nx — the difference per cell is
(i+0.5)·1/(nx(nx-1)) ≈ O(1/nx²), negligible for nx=105.

We then compare the first (nx-1) rows/cols of both arrays index-by-index.

Usage
-----
    python verify.py
    python verify.py --alex data_alex --ilya ../ilya/plots/final_state/state.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load a solver CSV.  Returns (x_1d, y_1d, fields_dict).
    Data must be written x-outer, y-inner (both solvers do this)."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    x1d = np.unique(data["x"])
    y1d = np.unique(data["y"])
    nx, ny = len(x1d), len(y1d)
    fields: dict[str, np.ndarray] = {
        name: data[name].reshape(nx, ny)
        for name in data.dtype.names
        if name not in ("x", "y")
    }
    return x1d, y1d, fields


def load_alex(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    csvs = sorted(
        data_dir.glob("result_*.csv"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not csvs:
        sys.exit(f"[error] No result_*.csv files found in {data_dir}")
    latest = csvs[-1]
    print(f"  Alex  : {latest}  ({len(csvs)} snapshots total)")
    return _load_csv(latest)


def load_ilya(csv_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not csv_path.exists():
        sys.exit(
            f"[error] Ilya's CSV not found: {csv_path}\n"
            "        Run `python main.py` in ilya/ first."
        )
    print(f"  Ilya  : {csv_path}")
    return _load_csv(csv_path)


# ---------------------------------------------------------------------------
# Grid alignment: node-centred → cell-centred via 4-point half-sum
# ---------------------------------------------------------------------------


def to_cell_centres(f: np.ndarray) -> np.ndarray:
    """
    Average the four corners of each cell:
        f_cell[i,j] = (f[i,j] + f[i+1,j] + f[i,j+1] + f[i+1,j+1]) / 4

    Input : (nx, ny)   — node-centred
    Output: (nx-1, ny-1) — cell-centred
    """
    return 0.25 * (f[:-1, :-1] + f[1:, :-1] + f[:-1, 1:] + f[1:, 1:])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def print_metrics(
    name: str, diff: np.ndarray, ref: np.ndarray,
    dx: float, dy: float,
) -> None:
    l2_abs  = np.sqrt(np.mean(diff ** 2))
    l2_rel  = l2_abs / (np.sqrt(np.mean(ref ** 2)) + 1e-30)
    linf    = np.max(np.abs(diff))
    # Integral metrics weighted by cell area dx·dy
    i_abs   = np.sum(np.abs(diff)) * dx * dy   # ∫|δu| dA
    i_signed = np.sum(diff)        * dx * dy   # ∫δu  dA  (systematic bias)
    print(f"    {name:6s}  L2_abs={l2_abs:.3e}  L2_rel={l2_rel:.3e}  "
          f"Linf={linf:.3e}  I_abs={i_abs:.3e}  I_signed={i_signed:+.3e}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _field_row(
    axes: list,
    f_a: np.ndarray, f_i: np.ndarray,
    xc: np.ndarray, yc: np.ndarray,
    label: str,
    cmap: str = "RdBu_r",
) -> None:
    vmax = max(np.abs(f_a).max(), np.abs(f_i).max())
    diff = f_i - f_a
    dmax = max(np.abs(diff).max(), 1e-12)

    kw = dict(origin="lower", extent=[xc[0], xc[-1], yc[0], yc[-1]],
              aspect="equal", cmap=cmap, interpolation="bilinear")

    im0 = axes[0].imshow(f_a.T, vmin=-vmax, vmax=vmax, **kw)
    axes[0].set_title(f"Alex — {label}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(f_i.T, vmin=-vmax, vmax=vmax, **kw)
    axes[1].set_title(f"Ilya — {label}")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff.T, vmin=-dmax, vmax=dmax,
                         cmap="PiYG", **{k: v for k, v in kw.items() if k != "cmap"})
    axes[2].set_title(f"Ilya − Alex  ({label})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")


def plot_fields(
    fields_a: dict, fields_i: dict,
    xc: np.ndarray, yc: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle("Comparison: Alex (ADI ψ-ω)  vs  Ilya (MAC IMEX)", fontsize=13)
    _field_row(axes[0], fields_a["u"],     fields_i["u"],     xc, yc, "u")
    _field_row(axes[1], fields_a["v"],     fields_i["v"],     xc, yc, "v")
    _field_row(axes[2], fields_a["omega"], fields_i["omega"], xc, yc, "ω")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {out_path}")


def plot_profiles(
    fields_a: dict, fields_i: dict,
    xc: np.ndarray, yc: np.ndarray,
    out_path: Path,
) -> None:
    ix = np.argmin(np.abs(xc - 0.5))
    iy = np.argmin(np.abs(yc - 0.5))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Centreline profiles — Alex vs Ilya", fontsize=12)

    ax = axes[0]
    ax.plot(fields_a["u"][ix, :], yc, "b-",  lw=2,   label="Alex (ADI ψ-ω)")
    ax.plot(fields_i["u"][ix, :], yc, "r--", lw=1.5, label="Ilya (MAC IMEX)")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("u"); ax.set_ylabel("y")
    ax.set_title("u(x=0.5, y)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xc, fields_a["v"][:, iy], "b-",  lw=2,   label="Alex (ADI ψ-ω)")
    ax.plot(xc, fields_i["v"][:, iy], "r--", lw=1.5, label="Ilya (MAC IMEX)")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("x"); ax.set_ylabel("v")
    ax.set_title("v(x, y=0.5)")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Compare Alex vs Ilya solver outputs.")
    parser.add_argument("--alex", type=Path, default=here / "data_alex")
    parser.add_argument("--ilya", type=Path,
                        default=here / "../ilya/plots/final_state/state.csv")
    parser.add_argument("--out",  type=Path, default=here / "plots")
    args = parser.parse_args()

    print("\n=== Loading data ===")
    x_a, y_a, f_a = load_alex(args.alex)
    x_i, y_i, f_i = load_ilya(args.ilya.resolve())

    nx_a, ny_a = len(x_a), len(y_a)
    nx_i, ny_i = len(x_i), len(y_i)
    print(f"\n  Alex  grid : {nx_a} × {ny_a}  (node-centred)")
    print(f"  Ilya  grid : {nx_i} × {ny_i}  (cell-centred)")

    # ── Bring Alex to cell centres via 4-point average ─────────────────────
    # Alex cell centres: (i+0.5)/(nx-1),  i = 0..nx-2  →  nx-1 cells
    # Ilya cell centres: (i+0.5)/nx,      i = 0..nx-1  →  nx   cells
    # We compare the first (nx-1) indices of both (difference < 1/nx² per cell)
    nc = nx_a - 1  # number of comparison cells per axis

    shared_a: dict[str, np.ndarray] = {}
    shared_i: dict[str, np.ndarray] = {}

    for key in ("u", "v", "omega"):
        shared_a[key] = to_cell_centres(f_a[key])          # (nc, nc)
        shared_i[key] = f_i[key][:nc, :nc]                 # (nc, nc)

    # Common coordinates: Alex's cell-midpoints
    xc = (np.arange(nc) + 0.5) / (nx_a - 1)
    yc = (np.arange(nc) + 0.5) / (ny_a - 1)

    print(f"\n  Common grid : {nc} × {nc}  (Alex cell-midpoints)")
    print(f"  x ∈ [{xc[0]:.4f}, {xc[-1]:.4f}]   "
          f"Δx_mismatch ≈ {abs(xc[0] - x_i[0]):.2e} per cell")

    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\n=== Error metrics (Ilya − Alex) ===")
    for key in ("u", "v", "omega"):
        print_metrics(key, shared_i[key] - shared_a[key], shared_a[key], dx, dy)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n=== Saving plots ===")
    plot_fields(shared_a, shared_i, xc, yc, args.out / "fields_comparison.png")
    plot_profiles(shared_a, shared_i, xc, yc, args.out / "profiles_comparison.png")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
