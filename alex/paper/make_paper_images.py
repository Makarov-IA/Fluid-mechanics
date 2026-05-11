#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "alex" / "paper"
IMAGES = PAPER / "images"


def copy_image(src: str, dst_name: str) -> None:
    source = ROOT / src
    if not source.exists():
        print(f"[paper-images] missing: {source}")
        return
    IMAGES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, IMAGES / dst_name)
    print(f"[paper-images] copied: {dst_name}")


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_spectrum() -> None:
    path = ROOT / "alex" / "linear_stability" / "eigenvalues.csv"
    rows = read_csv_dict(path)
    real = np.array([float(r["real"]) for r in rows])
    imag = np.array([float(r["imag"]) for r in rows])
    unstable = np.array([int(r["unstable"]) for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axvline(0.0, color="#777777", linewidth=1.0)
    ax.axhline(0.0, color="#cccccc", linewidth=0.8)
    ax.scatter(real[~unstable], imag[~unstable], s=34, color="#1f77b4", label="stable modes")
    ax.scatter(real[unstable], imag[unstable], s=58, color="#c62828", label="unstable modes")
    for idx, (x, y) in enumerate(zip(real, imag)):
        if unstable[idx]:
            ax.annotate(str(idx), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel(r"$\operatorname{Re}\lambda$")
    ax.set_ylabel(r"$\operatorname{Im}\lambda$")
    ax.set_title("Spectrum near the four-vortex equilibrium")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(IMAGES / "stability_spectrum.png", dpi=220)
    plt.close(fig)
    print("[paper-images] generated: stability_spectrum.png")


def plot_unstable_mode() -> None:
    path = ROOT / "alex" / "linear_stability" / "unstable_modes" / "eig_000.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    xs = np.unique(data["x"])
    ys = np.unique(data["y"])
    nx, ny = xs.size, ys.size

    fields = {
        r"$\operatorname{Re}\varphi$": data["psi_real"].reshape(nx, ny).T,
        r"$\operatorname{Im}\varphi$": data["psi_imag"].reshape(nx, ny).T,
        r"$\operatorname{Re}\eta$": data["omega_real"].reshape(nx, ny).T,
        r"$\operatorname{Im}\eta$": data["omega_imag"].reshape(nx, ny).T,
    }
    xg, yg = np.meshgrid(xs, ys)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2), constrained_layout=True)
    for ax, (title, values) in zip(axes.flat, fields.items()):
        vmax = max(float(np.max(np.abs(values))), 1e-14)
        levels = np.linspace(-vmax, vmax, 33)
        cf = ax.contourf(xg, yg, values, levels=levels, cmap="RdBu_r", extend="both")
        ax.contour(xg, yg, values, levels=11, colors="black", linewidths=0.25, alpha=0.35)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        fig.colorbar(cf, ax=ax, shrink=0.84)
    fig.suptitle(r"Unstable eigenmode $\lambda=0.2170+1.7868 i$", y=1.02)
    fig.savefig(IMAGES / "unstable_mode_000.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[paper-images] generated: unstable_mode_000.png")


def plot_projection_norm() -> None:
    path = ROOT / "alex" / "linear_stability" / "filtered_snapshots.csv"
    rows = read_csv_dict(path)
    time = np.array([float(r["time"]) for r in rows])
    projection = np.array([float(r["projection_norm"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(time, projection, color="#6a1b9a", linewidth=1.4)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|P_{\mathrm{unst}}(U(t)-U_*)\|_2$")
    ax.set_title("Norm of the removed unstable component")
    ax.grid(True, alpha=0.32)
    fig.tight_layout()
    fig.savefig(IMAGES / "projection_norm.png", dpi=220)
    plt.close(fig)
    print("[paper-images] generated: projection_norm.png")


def plot_raw_filtered_differences() -> None:
    raw = read_csv_dict(ROOT / "alex" / "stationary_detection" / "consecutive_difference_norm.csv")
    filtered = read_csv_dict(
        ROOT / "alex" / "linear_stability" / "filtered_detection" / "consecutive_difference_norm.csv"
    )
    tr = np.array([float(r["time"]) for r in raw])
    yr = np.array([float(r["rel_l2"]) for r in raw])
    tf = np.array([float(r["time"]) for r in filtered])
    yf = np.array([float(r["rel_l2"]) for r in filtered])

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.semilogy(tr, np.maximum(yr, 1e-30), color="#1565c0", linewidth=1.25, label="original")
    ax.semilogy(tf, np.maximum(yf, 1e-30), color="#c62828", linewidth=1.25, label="after projection")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|U_k-U_{k-1}\|_2/\|U_k\|_2$")
    ax.set_title("Consecutive snapshot differences")
    ax.grid(True, which="both", alpha=0.32)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "raw_filtered_difference.png", dpi=220)
    plt.close(fig)
    print("[paper-images] generated: raw_filtered_difference.png")


def plot_residual_history() -> None:
    path = ROOT / "alex" / "data" / "results" / "residual_history.csv"
    rows = read_csv_dict(path)
    time = []
    psi = []
    omega = []
    for r in rows:
        try:
            time.append(float(r["time"]))
            psi.append(float(r["psi_res"]))
            omega.append(float(r["omega_res"]))
        except (KeyError, ValueError):
            continue
    if not time:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.semilogy(time, np.maximum(psi, 1e-30), label=r"$R_\psi$", linewidth=1.25)
    ax.semilogy(time, np.maximum(omega, 1e-30), label=r"$R_\omega$", linewidth=1.25)
    ax.set_xlabel("$t$")
    ax.set_ylabel("max residual")
    ax.set_title("Pseudo-time residuals")
    ax.grid(True, which="both", alpha=0.32)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "residual_history.png", dpi=220)
    plt.close(fig)
    print("[paper-images] generated: residual_history.png")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    copies = {
        "alex/stationary_detection/consecutive_difference_norm.png": "stationary_consecutive_difference.png",
        "alex/stationary_detection/stationary_streamplot.png": "stationary_candidate_streamplot.png",
        "alex/stationary_detection/newton_equilibrium_streamplot.png": "newton_equilibrium_streamplot.png",
        "alex/linear_stability/filtered_detection/consecutive_difference_norm.png": "filtered_consecutive_difference.png",
        "alex/plots/frames/result_1000000_streamplot.png": "raw_result_1000000_streamplot.png",
        "alex/plots/frames/result_1000000_psi.png": "raw_result_1000000_psi.png",
        "alex/plots/frames/result_1000000_omega.png": "raw_result_1000000_omega.png",
        "alex/linear_stability/plots/frames/result_1000000_streamplot.png": "filtered_result_1000000_streamplot.png",
        "alex/linear_stability/plots/frames/result_1000000_psi.png": "filtered_result_1000000_psi.png",
        "alex/linear_stability/plots/frames/result_1000000_omega.png": "filtered_result_1000000_omega.png",
    }
    for src, dst in copies.items():
        copy_image(src, dst)

    plot_spectrum()
    plot_unstable_mode()
    plot_projection_norm()
    plot_raw_filtered_differences()
    plot_residual_history()


if __name__ == "__main__":
    main()
