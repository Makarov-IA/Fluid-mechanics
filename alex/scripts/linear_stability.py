#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import newton_from_csv as steady

matplotlib.use("Agg")

RESULT_RE = re.compile(r"result_(\d+)\.csv$")


@dataclass(frozen=True)
class EigenResult:
    index: int
    eigenvalue: complex
    residual: float
    growth_rate: float
    frequency: float
    is_unstable: bool


@dataclass(frozen=True)
class SavedMode:
    label: str
    eigenvalue: complex
    psi: np.ndarray
    omega: np.ndarray


class LinearizedOmegaOperator:
    def __init__(self, problem: steady.Problem, psi0: np.ndarray, omega0: np.ndarray):
        self.problem = problem
        self.psi0 = np.asarray(psi0)
        self.omega0 = np.asarray(omega0)
        self.ni = problem.nx - 2
        self.nj = problem.ny - 2
        self.n = self.ni * self.nj
        self.laplace_inverse = steady.DirichletLaplaceInverse(problem)

    def full_fields_from_eta(self, eta_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eta_inner = eta_vec.reshape(self.ni, self.nj)
        phi = np.zeros((self.problem.nx, self.problem.ny), dtype=eta_vec.dtype)
        eta = np.zeros((self.problem.nx, self.problem.ny), dtype=eta_vec.dtype)

        phi[1:-1, 1:-1] = self.laplace_inverse.solve(-eta_inner)
        eta[1:-1, 1:-1] = eta_inner
        steady.apply_thom_boundary_tangent(self.problem, phi, eta)
        return phi, eta

    def apply(self, eta_vec: np.ndarray) -> np.ndarray:
        phi, eta = self.full_fields_from_eta(eta_vec)
        rhs = (
            (1.0 / self.problem.re) * steady.laplacian_center(self.problem, eta)
            - steady.arakawa_jacobian(self.problem, phi, self.omega0)
            - steady.arakawa_jacobian(self.problem, self.psi0, eta)
        )
        return rhs.reshape(-1)


def load_problem(csv_path: Path, config_path: Path, re_override: float | None):
    cfg = steady.parse_config(config_path)
    xs, ys, fields = steady.load_csv(csv_path)
    if "psi" not in fields or "omega" not in fields:
        raise RuntimeError("Input CSV must contain psi and omega columns")

    bc = steady.BoundaryConditions(
        left=steady.WallVelocity(steady.cfg_float(cfg, "bc.left.u"), steady.cfg_float(cfg, "bc.left.v")),
        right=steady.WallVelocity(steady.cfg_float(cfg, "bc.right.u"), steady.cfg_float(cfg, "bc.right.v")),
        bottom=steady.WallVelocity(steady.cfg_float(cfg, "bc.bottom.u"), steady.cfg_float(cfg, "bc.bottom.v")),
        top=steady.WallVelocity(steady.cfg_float(cfg, "bc.top.u"), steady.cfg_float(cfg, "bc.top.v")),
    )
    re_value = float(re_override if re_override is not None else cfg.get("Re", 100.0))
    problem = steady.Problem(xs=xs, ys=ys, re=re_value, bc=bc)
    steady.validate_uniform_grid(problem)

    z0 = steady.pack(fields["psi"], fields["omega"])
    psi0, omega0 = steady.unpack(z0, problem)
    return problem, psi0, omega0


def vector_norm(v: np.ndarray) -> float:
    return float(np.sqrt(np.real(np.vdot(v, v))))


def compute_ritz_pairs(
    operator: LinearizedOmegaOperator,
    eigs_count: int,
    seed: int,
    eigs_tol: float,
    eigs_max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for the eigenvalue solver. Install it with: "
            "python3 -m pip install scipy"
        ) from exc

    k = min(int(eigs_count), operator.n - 2)
    if k < 1:
        raise RuntimeError(f"Need at least one eigenpair, got k={k}")

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(operator.n) + 1j * rng.standard_normal(operator.n)
    v0 /= vector_norm(v0)

    def matvec(v: np.ndarray) -> np.ndarray:
        return operator.apply(np.asarray(v, dtype=complex))

    linear_operator = LinearOperator(
        shape=(operator.n, operator.n),
        matvec=matvec,
        dtype=np.complex128,
    )

    print(
        "[stability] SciPy ARPACK eigs: "
        f"k={k}, which=LR, tol={eigs_tol:g}, maxiter={eigs_max_iter}",
        flush=True,
    )
    try:
        eigenvalues, eigenvectors = eigs(
            linear_operator,
            k=k,
            which="LR",
            tol=eigs_tol,
            maxiter=eigs_max_iter,
            v0=v0,
            return_eigenvectors=True,
        )
    except ArpackNoConvergence as exc:
        eigenvalues = exc.eigenvalues
        eigenvectors = exc.eigenvectors
        if eigenvalues is None or eigenvectors is None or eigenvalues.size == 0:
            raise RuntimeError(
                "ARPACK did not converge and returned no eigenpairs. "
                "Try increasing EIGS_MAX_ITER or EIGS_COUNT in linear_stability.sh."
            ) from exc
        print(
            "[stability] warning: ARPACK did not fully converge; "
            f"using {eigenvalues.size} converged eigenpairs",
            flush=True,
        )

    for k in range(eigenvectors.shape[1]):
        norm = vector_norm(eigenvectors[:, k])
        if norm > 0.0:
            eigenvectors[:, k] /= norm

    residuals = np.zeros(eigenvalues.size, dtype=float)
    for k, value in enumerate(eigenvalues):
        residuals[k] = vector_norm(operator.apply(eigenvectors[:, k]) - value * eigenvectors[:, k])

    order = np.argsort(-eigenvalues.real)
    return eigenvalues[order], eigenvectors[:, order], residuals[order]


def write_eigenvalues(path: Path, eigenvalues: np.ndarray, residuals: np.ndarray, unstable_tol: float) -> list[EigenResult]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[EigenResult] = []
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "real", "imag", "abs", "residual", "growth_rate", "frequency", "unstable"])
        for idx, (value, residual) in enumerate(zip(eigenvalues, residuals)):
            result = EigenResult(
                index=idx,
                eigenvalue=value,
                residual=float(residual),
                growth_rate=float(value.real),
                frequency=float(value.imag / (2.0 * math.pi)),
                is_unstable=bool(value.real > unstable_tol),
            )
            rows.append(result)
            writer.writerow(
                [
                    result.index,
                    f"{value.real:.16e}",
                    f"{value.imag:.16e}",
                    f"{abs(value):.16e}",
                    f"{result.residual:.16e}",
                    f"{result.growth_rate:.16e}",
                    f"{result.frequency:.16e}",
                    int(result.is_unstable),
                ]
            )
    return rows


def write_unstable(path: Path, rows: list[EigenResult]) -> list[EigenResult]:
    unstable = [row for row in rows if row.is_unstable]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "real", "imag", "abs", "residual", "growth_rate", "frequency"])
        for row in unstable:
            writer.writerow(
                [
                    row.index,
                    f"{row.eigenvalue.real:.16e}",
                    f"{row.eigenvalue.imag:.16e}",
                    f"{abs(row.eigenvalue):.16e}",
                    f"{row.residual:.16e}",
                    f"{row.growth_rate:.16e}",
                    f"{row.frequency:.16e}",
                ]
            )
    return unstable


def write_unstable_modes(
    out_dir: Path,
    operator: LinearizedOmegaOperator,
    unstable: list[EigenResult],
    eigenvectors: np.ndarray,
) -> None:
    modes_dir = out_dir / "unstable_modes"
    modes_dir.mkdir(parents=True, exist_ok=True)
    remove_matching_files(modes_dir, "eig_*.csv")

    for result in unstable:
        mode_vec = normalized_complex_mode(operator, eigenvectors[:, result.index])
        phi, eta = operator.full_fields_from_eta(mode_vec)
        mode_path = modes_dir / f"eig_{result.index:03d}.csv"
        with mode_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "x",
                    "y",
                    "psi_real",
                    "psi_imag",
                    "omega_real",
                    "omega_imag",
                    "lambda_real",
                    "lambda_imag",
                ]
            )
            for i, x in enumerate(operator.problem.xs):
                for j, y in enumerate(operator.problem.ys):
                    writer.writerow(
                        [
                            f"{x:.16e}",
                            f"{y:.16e}",
                            f"{np.real(phi[i, j]):.16e}",
                            f"{np.imag(phi[i, j]):.16e}",
                            f"{np.real(eta[i, j]):.16e}",
                            f"{np.imag(eta[i, j]):.16e}",
                            f"{result.eigenvalue.real:.16e}",
                            f"{result.eigenvalue.imag:.16e}",
                        ]
                    )
        print(f"[stability] unstable mode: {mode_path}", flush=True)


def load_saved_unstable_modes(out_dir: Path, problem: steady.Problem, max_modes: int) -> list[SavedMode]:
    modes_dir = out_dir / "unstable_modes"
    mode_paths = sorted(modes_dir.glob("eig_*.csv"))
    if max_modes > 0:
        mode_paths = mode_paths[:max_modes]

    modes: list[SavedMode] = []
    for mode_path in mode_paths:
        data = np.genfromtxt(mode_path, delimiter=",", names=True)
        if data.size == 0:
            continue

        xs = np.unique(data["x"])
        ys = np.unique(data["y"])
        if xs.size != problem.nx or ys.size != problem.ny:
            raise RuntimeError(f"Saved mode grid differs from equilibrium grid: {mode_path}")

        psi = (
            np.asarray(data["psi_real"], dtype=float)
            + 1j * np.asarray(data["psi_imag"], dtype=float)
        ).reshape(problem.nx, problem.ny)
        omega = (
            np.asarray(data["omega_real"], dtype=float)
            + 1j * np.asarray(data["omega_imag"], dtype=float)
        ).reshape(problem.nx, problem.ny)
        eigenvalue = complex(float(data["lambda_real"][0]), float(data["lambda_imag"][0]))
        modes.append(SavedMode(label=mode_path.stem, eigenvalue=eigenvalue, psi=psi, omega=omega))

    print(f"[stability] loaded saved unstable modes: {len(modes)} from {modes_dir}", flush=True)
    return modes


def save_csv(path: Path, problem: steady.Problem, psi: np.ndarray, omega: np.ndarray) -> None:
    steady.save_csv(path, problem, np.asarray(psi, dtype=float), np.asarray(omega, dtype=float))


def collect_result_files(results_dir: Path, limit: int | None) -> list[Path]:
    files: list[tuple[int, Path]] = []
    for path in results_dir.glob("result_*.csv"):
        match = RESULT_RE.match(path.name)
        if match is None:
            continue
        files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    paths = [path for _, path in files]
    if limit is not None:
        paths = paths[:limit]
    return paths


def remove_matching_files(directory: Path, pattern: str) -> None:
    if not directory.exists():
        return
    removed = 0
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        print(f"[stability] removed {removed} old files from {directory}", flush=True)


def read_snapshot_times(results_dir: Path) -> dict[int, float]:
    path = results_dir / "residual_history.csv"
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


def snapshot_step(path: Path) -> int:
    match = RESULT_RE.match(path.name)
    if match is None:
        raise RuntimeError(f"Cannot read step from snapshot name: {path}")
    return int(match.group(1))


def estimate_snapshot_time(path: Path, cfg: dict[str, str], snapshot_times: dict[int, float]) -> float:
    step = snapshot_step(path)
    if step in snapshot_times:
        return snapshot_times[step]

    dt = float(cfg.get("dt", "-1"))
    if dt > 0.0:
        return (step + 1) * dt

    t_max = float(cfg.get("t_max", "1"))
    n_time_steps = float(cfg.get("n_time_steps", "1"))
    return (step + 1) * t_max / n_time_steps


def normalized_complex_mode(operator: LinearizedOmegaOperator, eta_vec: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(eta_vec)))
    phase = np.exp(-1j * np.angle(eta_vec[idx])) if abs(eta_vec[idx]) > 0.0 else 1.0 + 0.0j
    mode = phase * eta_vec
    _, eta = operator.full_fields_from_eta(mode)
    scale = max(float(np.max(np.abs(eta))), 1e-30)
    return mode / scale


def evolved_real_mode(
    operator: LinearizedOmegaOperator,
    eta_vec: np.ndarray,
    eigenvalue: complex,
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    evolved_eta_vec = np.exp(eigenvalue * time) * eta_vec
    phi, eta = operator.full_fields_from_eta(evolved_eta_vec)
    return np.real(phi), np.real(eta)


def build_unstable_modes(
    operator: LinearizedOmegaOperator,
    eigenvectors: np.ndarray,
    unstable: list[EigenResult],
    max_modes: int,
) -> list[tuple[EigenResult, np.ndarray]]:
    modes: list[tuple[EigenResult, np.ndarray]] = []
    for result in unstable[:max_modes]:
        modes.append((result, normalized_complex_mode(operator, eigenvectors[:, result.index])))
    return modes


def collect_plot_limits(csv_paths: list[Path]) -> tuple[float, float, float]:
    psi_min = math.inf
    psi_max = -math.inf
    speed_max = 0.0

    for idx, csv_path in enumerate(csv_paths):
        _, _, fields = steady.load_csv(csv_path)
        psi = fields["psi"]
        u = fields["u"]
        v = fields["v"]
        speed = np.hypot(u, v)
        psi_min = min(psi_min, float(np.min(psi)))
        psi_max = max(psi_max, float(np.max(psi)))
        speed_max = max(speed_max, float(np.max(speed)))
        if (idx + 1) % 100 == 0 or idx + 1 == len(csv_paths):
            print(f"[stability] scanned plot limits {idx + 1}/{len(csv_paths)}", flush=True)

    if not csv_paths:
        psi_min = 0.0
        psi_max = 0.0
    return psi_min, psi_max, max(speed_max, 1e-12)


def plot_streamplot(
    csv_path: Path,
    output_path: Path,
    title: str,
    dpi: int,
    plot_limits: tuple[float, float, float] | None = None,
) -> None:
    xs, ys, fields = steady.load_csv(csv_path)
    psi = fields["psi"].T
    if "u" in fields and "v" in fields:
        u = fields["u"].T
        v = fields["v"].T
    else:
        dpsi_dy, dpsi_dx = np.gradient(psi, ys, xs, edge_order=2)
        u = dpsi_dy
        v = -dpsi_dx

    speed = np.hypot(u, v)
    x_grid, y_grid = np.meshgrid(xs, ys)
    if plot_limits is None:
        psi_min = float(np.min(psi))
        psi_max = float(np.max(psi))
        speed_max = max(float(np.max(speed)), 1e-12)
    else:
        psi_min, psi_max, speed_max = plot_limits

    speed_levels = np.linspace(0.0, speed_max, 36)
    speed_norm = Normalize(vmin=0.0, vmax=speed_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bg = ax.contourf(x_grid, y_grid, speed, levels=speed_levels, cmap="viridis", norm=speed_norm)
    stream = ax.streamplot(
        xs,
        ys,
        u,
        v,
        color=speed,
        cmap="plasma",
        norm=speed_norm,
        density=1.45,
        linewidth=0.7 + 1.5 * speed / speed_max,
        arrowsize=1.0,
    )
    stream.arrows.set_color("white")
    if psi_max > psi_min + 1e-14 * max(1.0, abs(psi_min), abs(psi_max)):
        levels = np.linspace(psi_min, psi_max, 21)
        ax.contour(x_grid, y_grid, psi, levels=levels, colors="white", linewidths=0.35, alpha=0.55)
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


def subtract_unstable_modes_from_snapshots(
    out_dir: Path,
    problem: steady.Problem,
    operator: LinearizedOmegaOperator,
    unstable: list[EigenResult],
    eigenvectors: np.ndarray,
    snapshots_dir: Path,
    filtered_dir: Path,
    max_modes: int,
    mode_amplitude: float,
    config: dict[str, str],
    limit: int | None,
    plot: bool,
    plot_dpi: int,
) -> None:
    modes = build_unstable_modes(operator, eigenvectors, unstable, max_modes)
    labels = [f"eig_{result.index:03d}" for result, _ in modes]
    filtered_dir.mkdir(parents=True, exist_ok=True)
    remove_matching_files(filtered_dir, "result_*.csv")
    plots_dir = out_dir / "filtered_streamplots"
    if plot:
        plots_dir.mkdir(parents=True, exist_ok=True)
        remove_matching_files(plots_dir, "*_filtered_streamplot.png")
    snapshot_times = read_snapshot_times(snapshots_dir)

    metadata_path = out_dir / "filtered_snapshots.csv"
    filtered_csvs: list[Path] = []
    plot_jobs: list[tuple[Path, Path, str]] = []
    with metadata_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_csv",
                "filtered_csv",
                "time",
                "removed_norm",
                "mode_amplitude",
                "modes",
                "plot_png",
            ]
        )
        f.flush()

        files = collect_result_files(snapshots_dir, limit)
        print(f"[stability] filtering snapshots: {len(files)} files from {snapshots_dir}")

        for idx, source_csv in enumerate(files):
            xs, ys, fields = steady.load_csv(source_csv)
            if xs.size != problem.nx or ys.size != problem.ny:
                raise RuntimeError(f"Snapshot grid differs from equilibrium grid: {source_csv}")

            time = estimate_snapshot_time(source_csv, config, snapshot_times)
            remove_psi = np.zeros((problem.nx, problem.ny), dtype=float)
            remove_omega = np.zeros((problem.nx, problem.ny), dtype=float)

            for result, mode_vec in modes:
                mode_psi, mode_omega = evolved_real_mode(operator, mode_vec, result.eigenvalue, time)
                remove_psi += mode_amplitude * mode_psi
                remove_omega += mode_amplitude * mode_omega

            filtered_psi = fields["psi"] - remove_psi
            filtered_omega = fields["omega"] - remove_omega
            steady.apply_thom_boundary(problem, filtered_psi, filtered_omega)
            filtered_csv = filtered_dir / source_csv.name
            save_csv(filtered_csv, problem, filtered_psi, filtered_omega)
            filtered_csvs.append(filtered_csv)

            plot_png = ""
            if plot:
                plot_path = plots_dir / f"{source_csv.stem}_filtered_streamplot.png"
                plot_png = str(plot_path)
                plot_jobs.append((filtered_csv, plot_path, f"Filtered {source_csv.stem}"))

            removed_norm = float(np.sqrt(np.linalg.norm(remove_psi[1:-1, 1:-1]) ** 2 +
                                         np.linalg.norm(remove_omega[1:-1, 1:-1]) ** 2))

            writer.writerow(
                [
                    str(source_csv),
                    str(filtered_csv),
                    f"{time:.16e}",
                    f"{removed_norm:.16e}",
                    f"{mode_amplitude:.16e}",
                    ";".join(labels),
                    plot_png,
                ]
            )
            f.flush()

            if (idx + 1) % 10 == 0 or idx + 1 == len(files):
                print(f"[stability] filtered {idx + 1}/{len(files)} snapshots", flush=True)

    print(f"[stability] filtered snapshot index: {metadata_path}")
    if plot and plot_jobs:
        print("[stability] computing shared plot scale for filtered streamplots", flush=True)
        plot_limits = collect_plot_limits(filtered_csvs)
        print(
            "[stability] shared plot scale: "
            f"psi=[{plot_limits[0]:.6e}, {plot_limits[1]:.6e}], "
            f"speed=[0, {plot_limits[2]:.6e}]",
            flush=True,
        )
        for idx, (filtered_csv, plot_path, title) in enumerate(plot_jobs):
            plot_streamplot(filtered_csv, plot_path, title=title, dpi=plot_dpi, plot_limits=plot_limits)
            if (idx + 1) % 10 == 0 or idx + 1 == len(plot_jobs):
                print(f"[stability] plotted {idx + 1}/{len(plot_jobs)} filtered streamplots", flush=True)


def subtract_saved_modes_from_snapshots(
    out_dir: Path,
    problem: steady.Problem,
    modes: list[SavedMode],
    snapshots_dir: Path,
    filtered_dir: Path,
    mode_amplitude: float,
    config: dict[str, str],
    limit: int | None,
    plot: bool,
    plot_dpi: int,
) -> None:
    labels = [mode.label for mode in modes]
    filtered_dir.mkdir(parents=True, exist_ok=True)
    remove_matching_files(filtered_dir, "result_*.csv")
    plots_dir = out_dir / "filtered_streamplots"
    if plot:
        plots_dir.mkdir(parents=True, exist_ok=True)
        remove_matching_files(plots_dir, "*_filtered_streamplot.png")
    snapshot_times = read_snapshot_times(snapshots_dir)

    metadata_path = out_dir / "filtered_snapshots.csv"
    filtered_csvs: list[Path] = []
    plot_jobs: list[tuple[Path, Path, str]] = []
    with metadata_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_csv",
                "filtered_csv",
                "time",
                "removed_norm",
                "mode_amplitude",
                "modes",
                "plot_png",
            ]
        )
        f.flush()

        files = collect_result_files(snapshots_dir, limit)
        print(f"[stability] filtering snapshots from saved modes: {len(files)} files from {snapshots_dir}")

        for idx, source_csv in enumerate(files):
            xs, ys, fields = steady.load_csv(source_csv)
            if xs.size != problem.nx or ys.size != problem.ny:
                raise RuntimeError(f"Snapshot grid differs from equilibrium grid: {source_csv}")

            time = estimate_snapshot_time(source_csv, config, snapshot_times)
            remove_psi = np.zeros((problem.nx, problem.ny), dtype=float)
            remove_omega = np.zeros((problem.nx, problem.ny), dtype=float)

            for mode in modes:
                factor = np.exp(mode.eigenvalue * time)
                remove_psi += mode_amplitude * np.real(factor * mode.psi)
                remove_omega += mode_amplitude * np.real(factor * mode.omega)

            filtered_psi = fields["psi"] - remove_psi
            filtered_omega = fields["omega"] - remove_omega
            steady.apply_thom_boundary(problem, filtered_psi, filtered_omega)
            filtered_csv = filtered_dir / source_csv.name
            save_csv(filtered_csv, problem, filtered_psi, filtered_omega)
            filtered_csvs.append(filtered_csv)

            plot_png = ""
            if plot:
                plot_path = plots_dir / f"{source_csv.stem}_filtered_streamplot.png"
                plot_png = str(plot_path)
                plot_jobs.append((filtered_csv, plot_path, f"Filtered {source_csv.stem}"))

            removed_norm = float(np.sqrt(np.linalg.norm(remove_psi[1:-1, 1:-1]) ** 2 +
                                         np.linalg.norm(remove_omega[1:-1, 1:-1]) ** 2))

            writer.writerow(
                [
                    str(source_csv),
                    str(filtered_csv),
                    f"{time:.16e}",
                    f"{removed_norm:.16e}",
                    f"{mode_amplitude:.16e}",
                    ";".join(labels),
                    plot_png,
                ]
            )
            f.flush()

            if (idx + 1) % 10 == 0 or idx + 1 == len(files):
                print(f"[stability] filtered {idx + 1}/{len(files)} snapshots", flush=True)

    print(f"[stability] filtered snapshot index: {metadata_path}")
    if plot and plot_jobs:
        print("[stability] computing shared plot scale for filtered streamplots", flush=True)
        plot_limits = collect_plot_limits(filtered_csvs)
        print(
            "[stability] shared plot scale: "
            f"psi=[{plot_limits[0]:.6e}, {plot_limits[1]:.6e}], "
            f"speed=[0, {plot_limits[2]:.6e}]",
            flush=True,
        )
        for idx, (filtered_csv, plot_path, title) in enumerate(plot_jobs):
            plot_streamplot(filtered_csv, plot_path, title=title, dpi=plot_dpi, plot_limits=plot_limits)
            if (idx + 1) % 10 == 0 or idx + 1 == len(plot_jobs):
                print(f"[stability] plotted {idx + 1}/{len(plot_jobs)} filtered streamplots", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix-free linear stability analysis around a Newton equilibrium.")
    parser.add_argument("csv", type=Path, help="Equilibrium CSV with x,y,psi,omega,u,v")
    parser.add_argument("--config", type=Path, default=Path("alex/configs/config.cfg"))
    parser.add_argument("--out-dir", type=Path, default=Path("alex/linear_stability"))
    parser.add_argument("--re", type=float, default=None)
    parser.add_argument("--eigs-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eigs-tol", type=float, default=1e-8)
    parser.add_argument("--eigs-max-iter", type=int, default=3000)
    parser.add_argument("--unstable-tol", type=float, default=1e-9)
    parser.add_argument("--snapshots-dir", type=Path, default=Path("alex/data/results"))
    parser.add_argument("--filtered-dir", type=Path, default=None)
    parser.add_argument("--snapshot-limit", type=int, default=None)
    parser.add_argument("--max-unstable-modes", type=int, default=8)
    parser.add_argument("--mode-amplitude", type=float, default=1e-3)
    parser.add_argument("--filter-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-dpi", type=int, default=160)
    args = parser.parse_args()

    input_csv = args.csv.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    snapshots_dir = args.snapshots_dir.expanduser().resolve()
    filtered_dir = (
        args.filtered_dir.expanduser().resolve()
        if args.filtered_dir is not None
        else out_dir / "filtered_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    problem, psi0, omega0 = load_problem(input_csv, config_path, args.re)
    operator = LinearizedOmegaOperator(problem, psi0, omega0)

    steady_residual = steady.make_residual(problem)(steady.pack(psi0, omega0))
    linf, l2 = steady.norm_info(steady_residual)
    print(
        "[stability] base residual: "
        f"Linf={linf:.6e} L2={l2:.6e} grid={problem.nx}x{problem.ny} "
        f"unknowns={operator.n} Re={problem.re:g}"
    )
    config = steady.parse_config(config_path)

    if args.filter_only:
        modes = load_saved_unstable_modes(out_dir, problem, args.max_unstable_modes)
        subtract_saved_modes_from_snapshots(
            out_dir,
            problem,
            modes,
            snapshots_dir=snapshots_dir,
            filtered_dir=filtered_dir,
            mode_amplitude=args.mode_amplitude,
            config=config,
            limit=args.snapshot_limit,
            plot=not args.no_plots,
            plot_dpi=args.plot_dpi,
        )
        return

    print(
        "[stability] eigensolver: "
        f"SciPy sparse.linalg.eigs, k={args.eigs_count}, "
        f"tol={args.eigs_tol:g}, maxiter={args.eigs_max_iter}"
    )

    eigenvalues, eigenvectors, residuals = compute_ritz_pairs(
        operator,
        eigs_count=args.eigs_count,
        seed=args.seed,
        eigs_tol=args.eigs_tol,
        eigs_max_iter=args.eigs_max_iter,
    )

    eigenvalues_csv = out_dir / "eigenvalues.csv"
    unstable_csv = out_dir / "unstable_eigenvalues.csv"
    rows = write_eigenvalues(eigenvalues_csv, eigenvalues, residuals, args.unstable_tol)
    unstable = write_unstable(unstable_csv, rows)
    write_unstable_modes(out_dir, operator, unstable, eigenvectors)

    print(f"[stability] eigenvalues: {eigenvalues_csv}")
    print(f"[stability] unstable eigenvalues: {unstable_csv}")
    print(f"[stability] unstable count: {len(unstable)}")
    for row in unstable:
        print(
            "[stability] unstable "
            f"#{row.index}: lambda={row.eigenvalue.real:.8e}"
            f"{row.eigenvalue.imag:+.8e}j residual={row.residual:.3e}"
        )

    subtract_unstable_modes_from_snapshots(
        out_dir,
        problem,
        operator,
        unstable,
        eigenvectors,
        snapshots_dir=snapshots_dir,
        filtered_dir=filtered_dir,
        max_modes=args.max_unstable_modes,
        mode_amplitude=args.mode_amplitude,
        config=config,
        limit=args.snapshot_limit,
        plot=not args.no_plots,
        plot_dpi=args.plot_dpi,
    )


if __name__ == "__main__":
    main()
