#!/usr/bin/env python3
"""Place finite magnets to approximate the omega part of saved state modes.

Offline geometry prototype, not a feedback/stability test or a flow solver.
Independent signed amplitudes are an actuator assumption, not a property of
permanent magnets driven by one common current.
"""
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
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import differential_evolution
import yaml

from solver_python.config import ALEX, UniqueSafeLoader, integer, number, section
from solver_python.finite_magnets import cuboid_field, vorticity_source


def load_design(path):
    path = Path(path).resolve()
    raw = yaml.load(path.read_text(), Loader=UniqueSafeLoader)
    cfg = section(raw, "design", {"mode_file", "physics", "magnets", "search", "control"})
    if not {"mode_file", "physics", "magnets", "search"} <= set(cfg):
        raise ValueError("Design needs mode_file, physics, magnets and search")
    p = section(cfg["physics"], "physics", {"length_m", "velocity_m_s",
        "density_kg_m3", "current_density_A_m2", "bottom_thickness_m", "plane_height_m"})
    m = section(cfg["magnets"], "magnets", {"count", "size_m", "gap_m",
        "magnetization_A_m", "edge_clearance_m", "spacing_m", "frame_extension_m"})
    s = section(cfg["search"], "search", {"seed", "maxiter", "popsize", "tolerance",
        "method", "candidate_pitch_m", "comparison_counts"})
    for key in ("length_m", "velocity_m_s", "density_kg_m3"):
        p[key] = number(p[key], key, positive=True)
    for key in ("bottom_thickness_m", "plane_height_m"):
        p[key] = number(p[key], key, nonnegative=True)
    p["current_density_A_m2"] = [number(v, "current") for v in p["current_density_A_m2"]]
    if len(p["current_density_A_m2"]) != 2 or np.linalg.norm(p["current_density_A_m2"]) == 0:
        raise ValueError("Supply a nonzero horizontal current with two components")
    m["count"] = integer(m["count"], "count")
    m["size_m"] = [number(v, "size", positive=True) for v in m["size_m"]]
    if len(m["size_m"]) != 3:
        raise ValueError("size_m must have three dimensions")
    m["magnetization_A_m"] = number(m["magnetization_A_m"], "magnetization", positive=True)
    for key in ("gap_m", "edge_clearance_m", "spacing_m"):
        m[key] = number(m[key], key, nonnegative=True)
    m["frame_extension_m"] = number(m.get("frame_extension_m", 0), "frame_extension_m", nonnegative=True)
    if m["gap_m"] + p["bottom_thickness_m"] + p["plane_height_m"] <= 0:
        raise ValueError("Observation plane must be above the top magnet face")
    s["method"] = s.get("method", "differential_evolution")
    if s["method"] == "differential_evolution":
        for key in ("seed", "maxiter", "popsize"):
            s[key] = integer(s[key], key, minimum=0 if key == "seed" else 1)
        s["tolerance"] = number(s["tolerance"], "tolerance", positive=True)
    elif s["method"] == "greedy_dictionary":
        s["candidate_pitch_m"] = [number(v, "pitch", positive=True) for v in s["candidate_pitch_m"]]
        if (len(s["candidate_pitch_m"]) != 2 or
                np.any(np.asarray(s["candidate_pitch_m"]) < np.asarray(m["size_m"][:2])+m["spacing_m"])):
            raise ValueError("Candidate pitch must allow magnet width plus spacing in each direction")
        s["comparison_counts"] = sorted(set(integer(v, "comparison count") for v in s["comparison_counts"]))
        if not s["comparison_counts"] or s["comparison_counts"][-1] != m["count"]:
            raise ValueError("Last comparison count must equal magnets.count")
    else:
        raise ValueError("Unknown search method")
    if "control" in cfg:
        c = section(cfg["control"], "control", {"modal_history", "gain", "max_amplitude"})
        c["modal_history"] = str((path.parent/c["modal_history"]).resolve())
        c["gain"] = number(c["gain"], "gain", positive=True)
        c["max_amplitude"] = number(c["max_amplitude"], "max_amplitude", positive=True)
    cfg["mode_file"] = str((path.parent / cfg["mode_file"]).resolve())
    return path, cfg


def load_modes(path):
    with np.load(path, allow_pickle=False) as data:
        xs, ys = data["xs"].copy(), data["ys"].copy()
        for axis in (xs, ys):
            if (axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all()
                    or not np.all(np.diff(axis) > 0)):
                raise ValueError("Invalid mode grid")
        n = (len(xs) - 2) * (len(ys) - 2)
        basis = data["state_basis"]
        if (basis.ndim != 2 or basis.shape[1] != 2 * n or not np.isfinite(basis).all()
                or not np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-12, rtol=0)):
            raise ValueError("Expected an orthonormal full-state basis")
        target = basis[:, n:].T.copy()
        if np.linalg.matrix_rank(target) != len(basis):
            raise ValueError("Dependent omega components in state modes")
    return xs, ys, target


def valid_positions(positions, size, spacing, bounds):
    positions = np.asarray(positions)
    if not np.isfinite(positions).all():
        return False
    if np.any(positions < bounds[:, 0]) or np.any(positions > bounds[:, 1]):
        return False
    for i in range(len(positions)):
        for j in range(i):
            # Axis-aligned footprints, minimum Euclidean edge-to-edge gap.
            separation = np.abs(positions[i] - positions[j]) - np.asarray(size[:2])
            if np.all(separation < 0) or np.linalg.norm(np.maximum(separation, 0)) < spacing:
                return False
    return True


def fit_sources(sources, target):
    """Full-grid Euclidean least squares, with column scaling for conditioning.

    Return physical amplitude matrix C (G C approximates target) and the
    worst relative error over every combination of target modes. No adjoint
    modes or dynamical controllability claims enter this spatial criterion.
    """
    norms = np.linalg.norm(sources, axis=0)
    if np.any(norms <= np.finfo(float).tiny):
        raise ValueError("Zero source column")
    normalized = sources / norms
    coefficients = np.linalg.lstsq(normalized, target, rcond=1e-10)[0] / norms[:, None]
    fitted = sources @ coefficients
    q, r = np.linalg.qr(target, mode="reduced")
    error_on_orthonormal_modes = (target - fitted) @ np.linalg.inv(r)
    worst_error = float(np.linalg.norm(error_on_orthonormal_modes, ord=2))
    return coefficients, fitted, worst_error


def fit_with_amplitude_limit(sources, target, radius, gain, limit):
    """Tikhonov fit with a certified row-norm bound over ||a||_2 <= radius.

    For u=-gain*C*a, sup |u_i| = gain*radius*||C_i||_2. Bisection finds
    a feasible common regularization. This is not the optimal SOCP solution;
    only the resulting bound, not global constrained optimality, is claimed.
    """
    coefficients, fitted, worst = fit_sources(sources, target)
    amplitude = gain*radius*np.linalg.norm(coefficients, axis=1)
    regularization = 0.0
    if amplitude.max() > limit:
        u, values, vh = np.linalg.svd(sources, full_matrices=False)
        projected = u.T@target

        def solve(alpha):
            return vh.T@((values/(values**2+alpha))[:, None]*projected)

        hi = values[0]**2
        while gain*radius*np.linalg.norm(solve(hi), axis=1).max() > limit:
            hi *= 10
        lo = 0.0
        for _ in range(60):
            mid = (lo+hi)/2
            if gain*radius*np.linalg.norm(solve(mid), axis=1).max() <= limit:
                hi = mid
            else:
                lo = mid
        regularization = hi
        coefficients = solve(hi)
        fitted = sources@coefficients
        r = np.linalg.qr(target, mode="reduced")[1]
        worst = float(np.linalg.norm((target-fitted)@np.linalg.inv(r), ord=2))
        amplitude = gain*radius*np.linalg.norm(coefficients, axis=1)
    return coefficients, fitted, worst, dict(regularization=regularization,
        max_amplitude=float(amplitude.max()), amplitude_by_magnet=amplitude.tolist(),
        modal_radius=radius, gain=gain, amplitude_limit=limit)


def candidate_positions(problem):
    pitch = np.asarray(problem.cfg["search"]["candidate_pitch_m"])
    counts = np.floor(np.diff(problem.bounds, axis=1).ravel()/pitch + 1e-10).astype(int)+1
    return np.array([(x, y) for x in np.linspace(*problem.bounds[0], counts[0])
                     for y in np.linspace(*problem.bounds[1], counts[1])])


def greedy_selection(sources, target, count):
    """Nested minimax spatial selection using full-grid Gram products.

    Refit every selected source jointly, never subtract a single local peak.
    Schur updates avoid repeating full-grid least squares for every candidate.
    Full SVD least squares independently verifies each reported checkpoint.
    """
    if count > sources.shape[1]:
        raise ValueError("More magnets requested than nonoverlapping candidate sites")
    g = sources/np.linalg.norm(sources, axis=0)
    q = np.linalg.qr(target, mode="reduced")[0]
    gram, overlap = g.T@g, g.T@q
    captured = np.zeros((q.shape[1], q.shape[1]))
    selected, history = [], []
    available = np.ones(g.shape[1], dtype=bool)
    for _ in range(count):
        variance = np.diag(gram).copy()
        eligible = available & (variance > 1e-12)
        if not eligible.any():
            raise ValueError("Candidate dictionary lost independent directions")
        ids = np.flatnonzero(eligible)
        modes = overlap[ids]/np.sqrt(variance[ids, None])
        trial = captured[None, :, :] + np.einsum("ki,kj->kij", modes, modes)
        eigenvalues = np.linalg.eigvalsh(trial)
        # The first rank-one source always has zero minimum eigenvalue for two
        # targets. Resolve equal minima by total captured energy.
        minimum = eigenvalues[:, 0]
        tied = np.flatnonzero(minimum >= minimum.max()-1e-13)
        j = ids[tied[np.argmax(np.trace(trial[tied], axis1=1, axis2=2))]]
        pivot = np.sqrt(variance[j])
        direction, response = gram[:, j].copy()/pivot, overlap[j].copy()/pivot
        gram -= np.outer(direction, direction)
        overlap -= np.outer(direction, response)
        captured += np.outer(response, response)
        selected.append(int(j))
        available[j] = False
        history.append(float(np.sqrt(max(0., 1-np.linalg.eigvalsh(captured)[0]))))
    return selected, history


class PlacementProblem:
    def __init__(self, cfg, xs, ys, target):
        self.cfg, self.xs, self.ys, self.target = cfg, xs, ys, target
        self.p, self.m = cfg["physics"], cfg["magnets"]
        length = self.p["length_m"]
        self.x, self.y = xs[1:-1, None] * length, ys[None, 1:-1] * length
        clearance = np.asarray(self.m["size_m"][:2]) / 2 + self.m["edge_clearance_m"]
        self.bounds = np.array([[xs[0]*length+clearance[0], xs[-1]*length-clearance[0]],
                                [ys[0]*length+clearance[1], ys[-1]*length-clearance[1]]])
        self.bounds += np.array([-self.m.get("frame_extension_m", 0), self.m.get("frame_extension_m", 0)])
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Magnets do not fit inside the domain")
        self.separation = (self.m["gap_m"] + self.p["bottom_thickness_m"]
                           + self.p["plane_height_m"])
        self.best = np.inf
        self.best_positions = None
        self.history = []

    def fields(self, center, full=False):
        length = self.p["length_m"]
        x = self.xs[:, None]*length if full else self.x
        y = self.ys[None, :]*length if full else self.y
        b, bx, by = cuboid_field(x, y, center, self.m["size_m"], self.separation,
                                self.m["magnetization_A_m"])
        source = vorticity_source(bx, by, self.p["current_density_A_m2"],
                                  self.p["density_kg_m3"], length, self.p["velocity_m_s"])
        return b, source

    def matrix(self, positions):
        return np.column_stack([self.fields(center)[1].ravel() for center in positions])

    def objective(self, flat):
        positions = np.asarray(flat).reshape(-1, 2)
        if not valid_positions(positions, self.m["size_m"], self.m["spacing_m"], self.bounds):
            return 2.0
        error = fit_sources(self.matrix(positions), self.target)[2]
        if error < self.best:
            self.best, self.best_positions = error, positions.copy()
        return error

    def callback(self, x, convergence):
        self.history.append(self.best)
        if len(self.history) % 10 == 0:
            print(f"[design] generation {len(self.history)}: worst spatial error {self.best:.3%}",
                  flush=True)


def draw_results(out, problem, positions, fitted, errors):
    xs, ys = problem.xs*problem.p["length_m"]*100, problem.ys*problem.p["length_m"]*100
    shape = (len(xs)-2, len(ys)-2)
    width, height = np.asarray(problem.m["size_m"][:2])*100

    def footprints(ax, show_magnets=True):
        for i, (x, y) in enumerate(positions*100 if show_magnets else []):
            ax.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                                   facecolor="none", edgecolor="black",
                                   linewidth=1.8 if len(positions) <= 32 else .6))
            if len(positions) <= 32:
                ax.annotate(str(i+1), (x, y), ha="center", va="center", fontsize=8,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
        extension = problem.m.get("frame_extension_m", 0)*100
        ax.add_patch(Rectangle((xs[0], ys[0]), xs[-1]-xs[0], ys[-1]-ys[0],
                               facecolor="none", edgecolor="#168043", linewidth=2))
        ax.set(xlim=(xs[0]-extension, xs[-1]+extension),
               ylim=(ys[0]-extension, ys[-1]+extension), xlabel="x [см]", ylabel="y [см]")
        ax.set_aspect("equal")

    def field(ax, values, title, span=None):
        span = max(float(np.max(np.abs(values))), 1e-20) if span is None else span
        im = ax.pcolormesh(xs[1:-1], ys[1:-1], values.reshape(shape).T,
                           cmap="RdBu_r", vmin=-span, vmax=span, shading="auto")
        footprints(ax, show_magnets=len(positions) <= 32)
        ax.set_title(title)
        return im

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    footprints(axes[0])
    axes[0].set_title("Вид сверху: размеры магнитов в масштабе")
    axes[0].grid(alpha=.2)
    current = np.asarray(problem.p["current_density_A_m2"])
    direction = current / np.linalg.norm(current)
    tail = np.array([xs[-1]-.13*(xs[-1]-xs[0]), ys[0]+.3*(ys[-1]-ys[0])])
    axes[0].annotate("", xy=tail+direction*.25*(ys[-1]-ys[0]), xytext=tail,
                     arrowprops=dict(arrowstyle="->", color="#336699", lw=2))
    axes[0].text(tail[0], tail[1]-.7, "ток j", ha="center", color="#336699")
    energy = np.sqrt(np.sum(problem.target**2, axis=1)).reshape(shape)
    im = axes[1].pcolormesh(xs[1:-1], ys[1:-1], energy.T, cmap="magma", shading="auto")
    footprints(axes[1])
    axes[1].set_title("Расположение на фоне двух целевых мод ω")
    fig.colorbar(im, ax=axes[1], shrink=.7, label="√(η₁² + η₂²), нормировка базиса")
    sx, sy, sz = np.asarray(problem.m["size_m"])*1000
    fig.suptitle(f"Магнитов: {len(positions)} | размер: {sx:g} × {sy:g} × {sz:g} мм | "
                 f"от верхней грани до плоскости расчёта: {problem.separation*1000:g} мм\n"
                 "Подбор формы источника; предполагается независимое управление амплитудами", fontsize=12)
    fig.savefig(out/"magnet_layout.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    count = problem.target.shape[1]
    fig, axes = plt.subplots(count, 3, figsize=(14, 3.3*count), squeeze=False, layout="constrained")
    for k in range(count):
        span = max(float(np.max(np.abs(problem.target[:, k]))), 1e-20)
        for ax, values, title in zip(axes[k],
                (problem.target[:, k], fitted[:, k], problem.target[:, k]-fitted[:, k]),
                (f"Целевой профиль ω: {k+1}", f"Магнитное приближение: {k+1}",
                 f"Остаток | ошибка L₂: {errors[k]:.1%}")):
            im = field(ax, values, title, span)
            fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle("Общая цветовая шкала в каждой строке | внутренние узлы | без расчёта течения")
    fig.savefig(out/"mode_fit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    shown = min(4, len(positions))
    fig, axes = plt.subplots(shown, 2, figsize=(11, 3.3*shown),
                             squeeze=False, layout="constrained")
    for i, center in enumerate(positions[:shown]):
        b, source = problem.fields(center)
        for ax, values, title in zip(axes[i], (1000*b, source),
                (f"Магнит {i+1}: Bz [мТл]", f"Магнит {i+1}: безразмерный источник ω")):
            im = field(ax, values, title)
            fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle(f"Индивидуальные отклики первых {shown} магнитов при номинальной намагниченности")
    fig.savefig(out/"magnet_responses.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def control_envelope(cfg):
    if "control" not in cfg:
        return None
    c = cfg["control"]
    history = np.genfromtxt(c["modal_history"], names=True, delimiter=",")
    values = np.atleast_1d(history["modal_norm"])
    if not np.isfinite(values).all() or np.any(values < 0) or values.max() <= 0:
        raise ValueError("Invalid modal-norm history")
    return dict(radius=float(values.max()), gain=c["gain"], limit=c["max_amplitude"])


def summarize_fit(sources, target, envelope):
    c, f, error = fit_sources(sources, target)
    row = dict(count=sources.shape[1], unconstrained_worst_error=error)
    if envelope is not None:
        row["unconstrained_max_amplitude"] = float(
            envelope["gain"]*envelope["radius"]*np.linalg.norm(c, axis=1).max())
        c, f, error, capacity = fit_with_amplitude_limit(sources, target, **envelope)
        row.update(limited_worst_error=error, max_amplitude=capacity["max_amplitude"],
                   regularization=capacity["regularization"])
    else:
        capacity = None
        row["limited_worst_error"] = error
    errors = np.linalg.norm(target-f, axis=0)/np.linalg.norm(target, axis=0)
    row.update({f"mode_{i+1}_error": float(value) for i, value in enumerate(errors)})
    return row, c, f, error, capacity


def dictionary_search(problem, envelope):
    sites = candidate_positions(problem)
    if not valid_positions(sites, problem.m["size_m"], problem.m["spacing_m"]-1e-12, problem.bounds):
        raise ValueError("Invalid candidate frame geometry")
    print(f"[design] evaluating {len(sites)} nonoverlapping candidate sites on the full grid", flush=True)
    sources = problem.matrix(sites)
    selected, history = greedy_selection(sources, problem.target, problem.m["count"])
    comparisons = []
    for count in problem.cfg["search"]["comparison_counts"]:
        row = summarize_fit(sources[:, selected[:count]], problem.target, envelope)[0]
        # Independently verify the small-Gram search against a full-grid SVD fit.
        if abs(row["unconstrained_worst_error"]-history[count-1]) > 1e-7:
            raise ArithmeticError("Dictionary selection disagrees with full-grid SVD")
        comparisons.append(row)
        print(f"[design] N={count}: spatial error {row['unconstrained_worst_error']:.2%}, "
              f"amplitude-limited error {row['limited_worst_error']:.2%}", flush=True)
    all_sites = summarize_fit(sources, problem.target, envelope)[0]
    return sites[selected], dict(method="greedy_dictionary", success=True,
        message="Completed nested candidate selection; no global-optimality claim",
        candidate_count=len(sites), best_error_by_addition=history,
        selection_indices=selected, all_candidates_fit=all_sites), comparisons, sites


def draw_comparison(out, comparisons, optimizer):
    n = [row["count"] for row in comparisons]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    axes[0].plot(n, [100*r["unconstrained_worst_error"] for r in comparisons], "o-", label="Без ограничения амплитуд")
    axes[0].plot(n, [100*r["limited_worst_error"] for r in comparisons], "s-", label="С ограничением амплитуд")
    lower = optimizer["all_candidates_fit"]["unconstrained_worst_error"]*100
    axes[0].axhline(lower, color="gray", linestyle=":", label="Все места: без ограничений")
    axes[0].set(xlabel="Количество магнитов", ylabel="Худшая относительная ошибка, %", ylim=(0, 105))
    axes[0].legend(fontsize=8)
    if "max_amplitude" in comparisons[0]:
        axes[1].plot(n, [r["unconstrained_max_amplitude"] for r in comparisons], "o-", label="Без ограничения")
        axes[1].plot(n, [r["max_amplitude"] for r in comparisons], "s-", label="После ограничения")
        axes[1].set(xlabel="Количество магнитов", ylabel="Максимальная амплитуда / номинальная")
        axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Одна сетка, размеры, зазор и ток; последовательно добавляем места")
    fig.savefig(out/"count_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_amplitudes(out, problem, positions, coefficients):
    fig, axes = plt.subplots(1, coefficients.shape[1], figsize=(12, 4.3),
                             squeeze=False, layout="constrained")
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    width, height = np.asarray(problem.m["size_m"][:2])*100
    for k, ax in enumerate(axes[0]):
        span = max(np.abs(coefficients[:, k]).max(), 1e-20)
        norm = Normalize(-span, span)
        for (x, y), amplitude in zip(positions*100, coefficients[:, k]):
            ax.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                                   facecolor=plt.cm.RdBu_r(norm(amplitude)), edgecolor="black", linewidth=.4))
        lx, ly = problem.xs[-1]*problem.p["length_m"]*100, problem.ys[-1]*problem.p["length_m"]*100
        ax.add_patch(Rectangle((0, 0), lx, ly, facecolor="none", edgecolor="#168043", linewidth=2))
        ax.set(xlim=tuple(problem.bounds[0]*100+[-width/2, width/2]),
               ylim=tuple(problem.bounds[1]*100+[-height/2, height/2]),
               xlabel="x [см]", ylabel="y [см]", title=f"Коэффициенты C для моды {k+1}")
        ax.set_aspect("equal")
        fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, shrink=.7)
    fig.suptitle("Знаки и относительные веса магнитов: u = −gain · C · a; зелёная рамка — кювета")
    fig.savefig(out/"magnet_amplitudes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_design(path):
    path, cfg = load_design(path)
    xs, ys, target = load_modes(cfg["mode_file"])
    problem = PlacementProblem(cfg, xs, ys, target)
    search = cfg["search"]
    envelope = control_envelope(cfg)
    comparisons, sites = None, None
    print(f"[design] {cfg['magnets']['count']} finite magnets, {len(xs)}x{len(ys)} grid, "
          f"{target.shape[1]} saved state modes", flush=True)
    if search["method"] == "greedy_dictionary":
        positions, optimizer, comparisons, sites = dictionary_search(problem, envelope)
    else:
        result = differential_evolution(problem.objective,
            bounds=list(map(tuple, problem.bounds))*cfg["magnets"]["count"],
            seed=search["seed"], maxiter=search["maxiter"], popsize=search["popsize"],
            tol=search["tolerance"], atol=0, polish=True, workers=1,
            callback=problem.callback)
        if problem.best_positions is None:
            raise ValueError("Search found no feasible placement; check sizes and spacing")
        positions = problem.best_positions
        optimizer = dict(method=search["method"], success=bool(result.success),
            message=str(result.message), evaluations=int(result.nfev), iterations=int(result.nit),
            seed=search["seed"], best_error_by_generation=problem.history)
    positions = positions[np.lexsort((positions[:, 1], positions[:, 0]))]
    sources = problem.matrix(positions)
    metrics, coefficients, fitted, worst, capacity = summarize_fit(sources, target, envelope)
    errors = np.linalg.norm(target-fitted, axis=0)/np.linalg.norm(target, axis=0)
    q = np.linalg.qr(target, mode="reduced")[0]
    coupling = q.T @ (sources / np.linalg.norm(sources, axis=0))
    out = ALEX/"results"/path.stem
    figures = out/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = [dict(id=i+1, center_m=center.tolist(), size_m=cfg["magnets"]["size_m"],
                    gap_m=cfg["magnets"]["gap_m"],
                    nominal_magnetization_A_m=cfg["magnets"]["magnetization_A_m"])
               for i, center in enumerate(positions)]
    report = dict(task="finite_magnet_spatial_fit", config=str(path),
        mode_file=cfg["mode_file"],
        mode_file_sha256=hashlib.sha256(Path(cfg["mode_file"]).read_bytes()).hexdigest(),
        grid=[len(xs), len(ys)], magnets=records, physics=cfg["physics"],
        top_face_to_observation_plane_m=problem.separation,
        relative_errors=errors.tolist(), worst_combination_relative_error=worst,
        coefficients_for_positive_target=coefficients.tolist(),
        control_law="u = -gain * coefficients_for_positive_target @ a; S_control = G @ u",
        normalized_spatial_overlap_singular_values=np.linalg.svd(coupling, compute_uv=False).tolist(),
        optimizer=optimizer, comparison=comparisons, fit_metrics=metrics,
        amplitude_constraint=capacity,
        assumptions=["Uniform vertical prescribed magnetization; nonmagnetic surroundings",
            "Independent signed amplitudes per magnet; not permanent magnets with one shared current",
            "Constant horizontal current; a single observation plane, no layer averaging",
            "Amplitude envelope, if configured, bounds an idealized actuator; no hardware feasibility claim",
            "Geometry selection is unconstrained spatial minimax; subsequent Tikhonov fitting enforces amplitude bounds",
            "Tikhonov fitting is feasible but not certified optimal among all bounded fits",
            "Only omega component of state correction is fitted; psi feedback is not reproduced",
            "No flow run, adjoint-mode controllability test, or closed-loop stability proof",
            "Model dimensions and strengths, not experimental measurements or article parameters"])
    if envelope is not None:
        report["control_history_file"] = cfg["control"]["modal_history"]
        report["control_history_sha256"] = hashlib.sha256(Path(cfg["control"]["modal_history"]).read_bytes()).hexdigest()
    (out/"design.input.yaml").write_bytes(path.read_bytes())
    (out/"design.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out/"placement.yaml").write_text(yaml.safe_dump(dict(magnets=records), sort_keys=False))
    (out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(out/"design.npz", xs=xs, ys=ys, target_omega=target,
        source_columns=sources, fitted_omega=fitted, coefficients=coefficients,
        centers_m=positions)
    draw_results(figures, problem, positions, fitted, errors)
    draw_amplitudes(figures, problem, positions, coefficients)
    if comparisons is not None:
        with (out/"count_comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        np.savez_compressed(out/"candidate_sites.npz", centers_m=sites,
                            selection_indices=optimizer["selection_indices"])
        draw_comparison(figures, comparisons, optimizer)
    print(json.dumps(dict(output=str(out), centers_m=positions.tolist(),
                          relative_errors=errors.tolist(), worst_error=worst,
                          optimizer_success=optimizer["success"]), indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run_design(parser.parse_args().config)
