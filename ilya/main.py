"""2-D Navier-Stokes lid-driven cavity solver — Python driver.

Physical setup
--------------
Domain     : [0, Lx] × [0, Ly]  (unit square by default)
Top wall   : u = u_lid,  v = 0
Other walls: u = 0,      v = 0
Reynolds   : Re = u_lid * min(Lx, Ly) / nu

Numerical method
----------------
MAC (staggered) grid, IMEX time integration:
  - viscosity + pressure : implicit (backward Euler)
  - convection           : explicit (forward Euler from previous step)
The constant linear system is factorised once at solver creation (SparseLU).
"""

from __future__ import annotations

import ctypes as ct
import io
import multiprocessing as mp
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy.ndimage as ndi
import yaml
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

matplotlib.use("Agg")

console = Console()

_Force3D = ct.CFUNCTYPE(ct.c_double, ct.c_double, ct.c_double, ct.c_double)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """All physical and numerical parameters for the lid-driven cavity run."""

    lx: float = 1.0
    ly: float = 1.0
    nx: int = 25
    ny: int = 25
    nu: float = 1e-3
    u_lid: float = 1.0
    t_end: float = 30.0
    n_steps: int = 300_000
    capture_fps: float = 4.0         # snapshots per simulation-second
    gif_playback_speed: float = 2.0  # simulation-s shown per real-s
    print_every: int = 10_000
    conv_tol: float = 1e-6           # 0 = disabled

    def __post_init__(self) -> None:
        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("nx and ny must be > 1")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.t_end <= 0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if self.capture_fps <= 0:
            raise ValueError(f"capture_fps must be positive, got {self.capture_fps}")
        if self.conv_tol < 0:
            raise ValueError(f"conv_tol must be >= 0, got {self.conv_tol}")

    @property
    def frame_every(self) -> int:
        """Solver steps between snapshots, derived from capture_fps."""
        return max(1, round(1.0 / (self.capture_fps * self.dt)))

    @classmethod
    def from_yaml(cls, path: Path) -> "SimConfig":
        with open(path) as fh:
            d = yaml.safe_load(fh)
        return cls(
            lx=d["domain"]["lx"],
            ly=d["domain"]["ly"],
            nx=d["grid"]["nx"],
            ny=d["grid"]["ny"],
            nu=d["physics"]["nu"],
            u_lid=d["physics"]["u_lid"],
            t_end=d["time"]["t_end"],
            n_steps=d["time"]["n_steps"],
            capture_fps=d["output"]["capture_fps"],
            gif_playback_speed=d["output"]["gif_playback_speed"],
            print_every=d["output"]["print_every"],
            conv_tol=d.get("convergence", {}).get("tol", 1e-6),
        )

    @property
    def dt(self) -> float:
        return self.t_end / self.n_steps

    @property
    def re(self) -> float:
        return self.u_lid * min(self.lx, self.ly) / self.nu

    def gif_fps(self, n_frames: int) -> float:
        """Compute fps so that the GIF plays at gif_playback_speed × real time."""
        video_duration_s = self.t_end / self.gif_playback_speed
        return max(1.0, n_frames / video_duration_s)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """Solution at one point in time, stored as float32 to halve memory."""

    step: int
    t: float
    p: np.ndarray  # (Nx, Ny) float32 — pressure
    uc: np.ndarray  # (Nx, Ny) float32 — cell-centred x-velocity
    vc: np.ndarray  # (Nx, Ny) float32 — cell-centred y-velocity
    omega: np.ndarray  # (Nx, Ny) float32 — vorticity


# ---------------------------------------------------------------------------
# Solver library wrapper
# ---------------------------------------------------------------------------


def _find_solver_lib(directory: Path) -> Path:
    """Return the path to the compiled solver shared library for the current OS."""
    ext_map = {"Darwin": ".dylib", "Windows": ".dll", "Linux": ".so"}
    ext = ext_map.get(platform.system())
    if ext is None:
        raise RuntimeError(f"Unsupported OS: {platform.system()}")

    exact = directory / f"solver{ext}"
    if exact.exists():
        return exact

    candidates = [
        f
        for f in directory.iterdir()
        if f.name.startswith("solver") and f.suffix == ext
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Solver library not found in {directory}. "
            f"Expected '{exact.name}'. Run compile.sh first."
        )
    lib = candidates[0]
    print(f"[warning] Using alternative library name: {lib.name}")
    return lib


def _check_stability(cfg: SimConfig) -> None:
    """
    Check CFL stability for the explicit convection term.

    The only stability constraint for IMEX is on explicit convection:
        CFL = u_lid * dt / min(dx, dy)
    Viscosity is implicit so there is no viscous CFL constraint.
    """
    dx = cfg.lx / cfg.nx
    dy = cfg.ly / cfg.ny
    cfl = cfg.u_lid * cfg.dt / min(dx, dy)

    # Minimum n_steps needed for CFL <= 0.5
    n_steps_safe = int(cfg.t_end * cfg.u_lid / (0.5 * min(dx, dy))) + 1

    if cfl > 1.0:
        raise RuntimeError(
            f"CFL = {cfl:.3f} > 1.0 — решение гарантированно расходится.\n"
            f"  Увеличь n_steps до >= {n_steps_safe} (текущее: {cfg.n_steps})."
        )
    if cfl > 0.5:
        console.print(
            f"[yellow]⚠  CFL = {cfl:.3f} > 0.5 — возможна нестабильность "
            f"при Re={cfg.re:.0f}.  Рекомендуется n_steps >= {n_steps_safe}.[/yellow]"
        )
    else:
        console.print(f"  CFL = [green]{cfl:.4f}[/green]  (stable)")


class StokesMACLib:
    """
    Python wrapper around the compiled C++ Stokes MAC solver.

    Grid conventions (see stokes_mac.h):
        p[i, j]  shape (Nx,   Ny  )  — pressure at cell centres
        u[i, j]  shape (Nx+1, Ny  )  — x-velocity at vertical faces
        v[i, j]  shape (Nx,   Ny+1)  — y-velocity at horizontal faces
    """

    def __init__(
        self,
        lib_path: Path,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        nu: float,
        dt: float,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self._handle: ct.c_void_p | None = None

        try:
            self._dll = ct.CDLL(str(lib_path), mode=ct.RTLD_GLOBAL)
        except AttributeError:
            self._dll = ct.CDLL(str(lib_path))

        self._bind_c_api()

        self._handle = self._dll.stokes_mac_create_c(nx, ny, lx, ly, nu, dt)
        if not self._handle:
            raise RuntimeError("stokes_mac_create_c returned NULL")

    def __enter__(self) -> "StokesMACLib":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def run_steps(self, t_start: float, n_steps: int) -> np.ndarray:
        """Run n_steps with zero body force entirely inside C++.

        Returns float64 array of length n_steps with max|div u| per step.
        """
        div_out = np.empty(n_steps, dtype=np.float64)
        self._dll.stokes_mac_run_steps_c(
            self._handle,
            ct.c_double(t_start),
            ct.c_int(n_steps),
            div_out.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        return div_out

    def get_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p, u, v) arrays on the MAC grid (fresh float64 copies)."""
        p = self._copy_field(
            self._dll.stokes_mac_get_p_c(self._handle), (self.nx, self.ny)
        )
        u = self._copy_field(
            self._dll.stokes_mac_get_u_c(self._handle), (self.nx + 1, self.ny)
        )
        v = self._copy_field(
            self._dll.stokes_mac_get_v_c(self._handle), (self.nx, self.ny + 1)
        )
        return p, u, v

    def close(self) -> None:
        if self._handle is not None:
            self._dll.stokes_mac_free_c(self._handle)
            self._handle = None

    def _bind_c_api(self) -> None:
        dll = self._dll
        dll.stokes_mac_create_c.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
        ]
        dll.stokes_mac_create_c.restype = ct.c_void_p
        dll.stokes_mac_free_c.argtypes = [ct.c_void_p]
        dll.stokes_mac_free_c.restype = None
        dll.stokes_mac_run_steps_c.argtypes = [
            ct.c_void_p,
            ct.c_double,
            ct.c_int,
            ct.POINTER(ct.c_double),
        ]
        dll.stokes_mac_run_steps_c.restype = None
        for name in ("stokes_mac_get_p_c", "stokes_mac_get_u_c", "stokes_mac_get_v_c"):
            fn = getattr(dll, name)
            fn.argtypes = [ct.c_void_p]
            fn.restype = ct.POINTER(ct.c_double)

    def _copy_field(
        self, ptr: ct.POINTER(ct.c_double), shape_xy: tuple[int, int]
    ) -> np.ndarray:
        """Copy a C row-major array into a (Nx, Ny) numpy array (x-first)."""
        nx, ny = shape_xy
        return np.ctypeslib.as_array(ptr, shape=(nx * ny,)).copy().reshape(ny, nx).T


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _cell_centred_velocity(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average face-centred MAC velocities to cell centres."""
    return 0.5 * (u[:-1, :] + u[1:, :]), 0.5 * (v[:, :-1] + v[:, 1:])


def _vorticity(
    uc: np.ndarray, vc: np.ndarray, xc: np.ndarray, yc: np.ndarray
) -> np.ndarray:
    """Scalar vorticity  ω = ∂v/∂x − ∂u/∂y  on cell-centred coordinates."""
    return np.gradient(vc, xc, axis=0) - np.gradient(uc, yc, axis=1)


def run_simulation(
    cfg: SimConfig,
    lib_path: Path,
    xc: np.ndarray,
    yc: np.ndarray,
) -> tuple[list[Snapshot], list[float], list[float]]:
    """
    Run the time integration using batch C++ steps.

    The outer Python loop advances `frame_every` steps per iteration,
    eliminating per-step Python↔C overhead.

    Returns
    -------
    snapshots   : one Snapshot per batch (float32 arrays)
    t_history   : time at every solver step
    div_history : max|div u| at every solver step
    """
    snapshots: list[Snapshot] = []
    t_history: list[float] = []
    div_history: list[float] = []

    prev_uc: np.ndarray | None = None
    prev_vc: np.ndarray | None = None
    converged = False
    n_batches = -(-cfg.n_steps // cfg.frame_every)  # ceil division

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=38),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[info]}"),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task("Simulation", total=n_batches, info="starting…")
        with StokesMACLib(
            lib_path, cfg.nx, cfg.ny, cfg.lx, cfg.ly, cfg.nu, cfg.dt
        ) as solver:
            step_done = 0
            for batch_start in range(0, cfg.n_steps, cfg.frame_every):
                batch_n = min(cfg.frame_every, cfg.n_steps - batch_start)
                t_start = batch_start * cfg.dt

                divs = solver.run_steps(t_start, batch_n)

                if not np.isfinite(divs).all():
                    nan_step = batch_start + int(np.argmax(~np.isfinite(divs))) + 1
                    raise RuntimeError(
                        f"Solver diverged at step ~{nan_step} (t={nan_step*cfg.dt:.4f}). "
                        f"CFL too large or Re too high for current grid/dt."
                    )

                step_done += batch_n
                t_now = step_done * cfg.dt

                t_history.extend((batch_start + k + 1) * cfg.dt for k in range(batch_n))
                div_history.extend(divs.tolist())

                p, u, v = solver.get_fields()
                uc, vc = _cell_centred_velocity(u, v)
                omega = _vorticity(uc, vc, xc, yc)
                snapshots.append(
                    Snapshot(
                        step=step_done,
                        t=t_now,
                        p=p.astype(np.float32),
                        uc=uc.astype(np.float32),
                        vc=vc.astype(np.float32),
                        omega=omega.astype(np.float32),
                    )
                )

                if cfg.conv_tol > 0 and prev_uc is not None:
                    vel_change = float(
                        max(
                            np.max(np.abs(uc - prev_uc)),
                            np.max(np.abs(vc - prev_vc)),
                        )
                    )
                    if vel_change < cfg.conv_tol:
                        progress.update(task, info=f"converged Δu={vel_change:.1e}")
                        progress.stop()
                        console.print(
                            f"[green]✓ Converged[/green] at step [bold]{step_done}[/bold]  "
                            f"t={t_now:.3f}  Δu={vel_change:.2e}"
                        )
                        converged = True
                        break

                prev_uc = uc
                prev_vc = vc

                progress.update(
                    task,
                    advance=1,
                    info=f"t={t_now:.2f}  |div|={divs[-1]:.2e}",
                )

    if not converged:
        tol_str = "disabled" if cfg.conv_tol == 0 else "not reached"
        console.print(f"[dim]Reached t_end={cfg.t_end:.1f}  (tol {tol_str})[/dim]")

    return snapshots, t_history, div_history


# ---------------------------------------------------------------------------
# Vortex detection
# ---------------------------------------------------------------------------


def find_vortex_centers(
    snap: Snapshot,
    xc: np.ndarray,
    yc: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Detect vortex centers via local extrema of the stream function ψ.

    For 2-D incompressible flow:  ∂ψ/∂y = u,  ∂ψ/∂x = −v.
    ψ is computed by averaging two cumulative integrations (along x and y)
    to cancel integration drift.

    Vortex centers are topological extrema of ψ:
        local maximum → CCW rotation (↺)
        local minimum → CW  rotation (↻)

    This is threshold-free and unaffected by the lid-corner singularities
    that break Q-criterion and speed-minimum approaches.
    """
    uc = snap.uc.astype(np.float64)
    vc = snap.vc.astype(np.float64)

    dy = float(yc[1] - yc[0])
    dx = float(xc[1] - xc[0])

    # Stream function ψ: integrate u along y (∂ψ/∂y = u)
    # and -v along x (∂ψ/∂x = -v), then average to reduce integration drift.
    psi = 0.5 * (np.cumsum(uc * dy, axis=1) + np.cumsum(-vc * dx, axis=0))

    # Window ≈ 1/7 of domain: small enough to catch compact corner vortices,
    # large enough that the primary vortex still gives exactly one extremum.
    nbr = max(3, min(psi.shape) // 7)  # ≈9 cells for 65×65

    local_max = psi == ndi.maximum_filter(psi, size=nbr, mode="nearest")
    local_min = psi == ndi.minimum_filter(psi, size=nbr, mode="nearest")

    # 1-cell border only — corner vortices sit very close to walls.
    m = 1
    for arr in (local_max, local_min):
        arr[:m, :] = False
        arr[-m:, :] = False
        arr[:, :m] = False
        arr[:, -m:] = False

    # local max of ψ → CCW (↺),  local min of ψ → CW (↻)
    ccw = [(float(xc[i]), float(yc[j])) for i, j in zip(*np.where(local_max))]
    cw = [(float(xc[i]), float(yc[j])) for i, j in zip(*np.where(local_min))]

    return ccw, cw


def _overlay_vortex_markers(
    ax, snap: Snapshot, xc: np.ndarray, yc: np.ndarray
) -> tuple[bool, bool]:
    """Draw vortex-core markers with coordinate labels onto *ax*.

    Returns (has_ccw, has_cw) so the caller can build a shared legend.
    """
    ccw, cw = find_vortex_centers(snap, xc, yc)

    for x, y in ccw:
        ax.scatter(x, y, marker="+", c="red", s=150, linewidths=2.5, zorder=6)
        ax.annotate(
            f"({x:.2f}, {y:.2f})",
            xy=(x, y),
            xytext=(5, 4),
            textcoords="offset points",
            color="red",
            fontsize=7,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65, lw=0),
        )

    for x, y in cw:
        ax.scatter(x, y, marker="x", c="cyan", s=150, linewidths=2.5, zorder=6)
        ax.annotate(
            f"({x:.2f}, {y:.2f})",
            xy=(x, y),
            xytext=(5, 4),
            textcoords="offset points",
            color="cyan",
            fontsize=7,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0),
        )

    return bool(ccw), bool(cw)


# ---------------------------------------------------------------------------
# Colour-level helpers  (single pass over all snapshots)
# ---------------------------------------------------------------------------


def _compute_colour_levels(
    snapshots: list[Snapshot],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (speed_levels, p_levels, omega_levels, speed_max)."""
    speed_max = 0.0
    p_parts: list[np.ndarray] = []
    omega_abs_parts: list[np.ndarray] = []

    for s in snapshots:
        speed_max = max(speed_max, float(np.max(np.hypot(s.uc, s.vc))))
        p_parts.append(s.p.ravel())
        omega_abs_parts.append(np.abs(s.omega.ravel()))

    speed_max = max(speed_max, 1e-12)
    sl = np.linspace(0.0, speed_max, 25)

    all_p = np.concatenate(p_parts)
    p_lo = float(np.percentile(all_p, 2.0))
    p_hi = float(np.percentile(all_p, 98.0))
    if p_hi <= p_lo:
        p_lo, p_hi = float(all_p.min()), float(all_p.max())
    pl = np.linspace(p_lo, p_hi, 61)

    omega_max = max(float(np.percentile(np.concatenate(omega_abs_parts), 98.0)), 1e-12)
    ol = np.linspace(-omega_max, omega_max, 61)

    return sl, pl, ol, speed_max


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _style_axes(ax, title: str, lx: float, ly: float) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)


def _fig_to_pil(fig, dpi: int = 110) -> Image.Image:
    """Render a matplotlib figure to a PIL Image without touching disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def _draw_streamlines(ax, fig, snap, xc, yc, Xc, Yc, speed_levels) -> None:
    speed = np.hypot(snap.uc, snap.vc)
    bg = ax.contourf(Xc, Yc, speed, levels=speed_levels, cmap="viridis")
    fig.colorbar(bg, ax=ax, label="|u|")
    ax.streamplot(
        xc,
        yc,
        snap.uc.T,
        snap.vc.T,
        color="white",
        linewidth=0.8,
        density=1.5,
        arrowsize=0.9,
    )


def _draw_pressure(ax, fig, snap, Xc, Yc, p_levels) -> None:
    cp = ax.contourf(Xc, Yc, snap.p, levels=p_levels, cmap="coolwarm")
    ax.contour(
        Xc, Yc, snap.p, levels=p_levels, colors="black", linewidths=0.25, alpha=0.7
    )
    fig.colorbar(cp, ax=ax, label="p")


def _draw_vorticity(ax, fig, snap, Xc, Yc, omega_levels) -> None:
    cw = ax.contourf(Xc, Yc, snap.omega, levels=omega_levels, cmap="coolwarm")
    ax.contour(
        Xc,
        Yc,
        snap.omega,
        levels=omega_levels,
        colors="black",
        linewidths=0.25,
        alpha=0.7,
    )
    fig.colorbar(cw, ax=ax, label="ω")


# ---------------------------------------------------------------------------
# GIF rendering (worker process)
# ---------------------------------------------------------------------------


def _gif_worker(task: dict) -> str:
    """
    Render every frame for one GIF type, assemble, and write the file.

    Runs in a spawned worker process. No temp files — frames go to BytesIO.
    """
    kind: str = task["kind"]
    lx, ly = task["lx"], task["ly"]
    xc: np.ndarray = task["xc"]
    yc: np.ndarray = task["yc"]
    Xc: np.ndarray = task["Xc"]
    Yc: np.ndarray = task["Yc"]
    snapshots: list[Snapshot] = task["snapshots"]
    frame_duration_ms: int = task["frame_duration_ms"]

    gif_frames: list[Image.Image] = []

    for snap in snapshots:
        fig, ax = plt.subplots(figsize=(6.2, 6.0))

        if kind == "streamlines":
            _draw_streamlines(ax, fig, snap, xc, yc, Xc, Yc, task["speed_levels"])
            title = f"Streamlines, t={snap.t:.3f}"
        elif kind == "pressure":
            _draw_pressure(ax, fig, snap, Xc, Yc, task["p_levels"])
            title = f"Pressure, t={snap.t:.3f}"
        elif kind == "vorticity":
            _draw_vorticity(ax, fig, snap, Xc, Yc, task["omega_levels"])
            title = f"Vorticity, t={snap.t:.3f}"
        else:
            raise ValueError(f"Unknown GIF kind: {kind!r}")

        _style_axes(ax, title, lx, ly)
        fig.tight_layout()
        gif_frames.append(_fig_to_pil(fig, dpi=110))

    gif_path = Path(task["gif_path"])
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    if gif_frames:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
    return str(gif_path)


def render_gifs(
    snapshots: list[Snapshot],
    cfg: SimConfig,
    out_dir: Path,
    xc: np.ndarray,
    yc: np.ndarray,
    sl: np.ndarray,
    pl: np.ndarray,
    ol: np.ndarray,
    speed_max: float,
) -> dict[str, Path]:
    """Render all four GIFs in parallel. Returns mapping kind → Path."""
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    fps = cfg.gif_fps(len(snapshots))
    frame_duration_ms = int(1000.0 / fps)

    common = {
        "lx": cfg.lx,
        "ly": cfg.ly,
        "xc": xc,
        "yc": yc,
        "Xc": Xc,
        "Yc": Yc,
        "frame_duration_ms": frame_duration_ms,
        "snapshots": snapshots,
    }
    tasks = [
        {
            **common,
            "kind": "streamlines",
            "gif_path": str(out_dir / "stokes_streamlines.gif"),
            "speed_levels": sl,
        },
        {
            **common,
            "kind": "pressure",
            "gif_path": str(out_dir / "stokes_pressure.gif"),
            "p_levels": pl,
        },
        {
            **common,
            "kind": "vorticity",
            "gif_path": str(out_dir / "stokes_vorticity.gif"),
            "omega_levels": ol,
        },
    ]

    n_workers = min(len(tasks), os.cpu_count() or 1)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        gif_paths = list(pool.map(_gif_worker, tasks))

    return {t["kind"]: Path(p) for t, p in zip(tasks, gif_paths)}


# ---------------------------------------------------------------------------
# Static output
# ---------------------------------------------------------------------------


def save_divergence_plot(
    t_history: list[float], div_history: list[float], out_dir: Path
) -> Path:
    """Save a time-series plot of max |div u|, downsampled for fast rendering."""
    t = np.asarray(t_history)
    d = np.asarray(div_history)

    # Downsample to ~5000 points so matplotlib doesn't labour over 300K markers
    stride = max(1, len(t) // 5000)
    t_plot, d_plot = t[::stride], d[::stride]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(t_plot, d_plot, color="#0d47a1", linewidth=1.0)
    ax.set_title("Max divergence vs time")
    ax.set_xlabel("t")
    ax.set_ylabel("max |div u|")
    ax.grid(True, alpha=0.35)
    if len(t) > stride:
        ax.text(
            0.98,
            0.97,
            f"(every {stride}th point shown)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="gray",
        )
    fig.tight_layout()
    path = out_dir / "stokes_max_divergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _make_legend_handles(has_ccw: bool, has_cw: bool) -> list:
    handles = []
    if has_ccw:
        handles.append(Line2D(
            [0], [0], marker="+", color="red", linestyle="none",
            markersize=10, markeredgewidth=2.0,
            label="Vortex ↺ — counterclockwise (CCW)",
        ))
    if has_cw:
        handles.append(Line2D(
            [0], [0], marker="x", color="cyan", linestyle="none",
            markersize=10, markeredgewidth=2.0,
            label="Vortex ↻ — clockwise (CW)",
        ))
    return handles


def save_final_figure(
    snap: Snapshot,
    cfg: SimConfig,
    xc: np.ndarray,
    yc: np.ndarray,
    out_dir: Path,
    sl: np.ndarray,
    pl: np.ndarray,
    ol: np.ndarray,
) -> list[Path]:
    """Save each final-state panel as a separate PNG in plots/final_state/."""
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    panel_dir = out_dir / "final_state"
    panel_dir.mkdir(parents=True, exist_ok=True)

    suptitle = (
        f"Lid-driven cavity — final state   "
        f"Re = {cfg.re:.0f},  grid {cfg.nx}×{cfg.ny},  t = {snap.t:.2f}"
    )

    panels = [
        ("streamlines", "Streamlines", sl),
        ("pressure",    "Pressure",    pl),
        ("vorticity",   "Vorticity",   ol),
    ]

    saved: list[Path] = []
    for kind, title, levels in panels:
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.suptitle(suptitle, fontsize=12)

        if kind == "streamlines":
            _draw_streamlines(ax, fig, snap, xc, yc, Xc, Yc, levels)
        elif kind == "pressure":
            _draw_pressure(ax, fig, snap, Xc, Yc, levels)
        else:
            _draw_vorticity(ax, fig, snap, Xc, Yc, levels)

        has_ccw, has_cw = _overlay_vortex_markers(ax, snap, xc, yc)
        _style_axes(ax, title, cfg.lx, cfg.ly)

        handles = _make_legend_handles(has_ccw, has_cw)
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=len(handles),
                fontsize=9,
                framealpha=0.9,
                facecolor="white",
                bbox_to_anchor=(0.5, 0.0),
            )

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        path = panel_dir / f"{kind}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = SimConfig.from_yaml(config_path)

    # ── Config table ──────────────────────────────────────────────────────
    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="bold cyan")
    tbl.add_column(style="white")
    tbl.add_row("Domain", f"{cfg.lx} × {cfg.ly}")
    tbl.add_row("Grid", f"{cfg.nx} × {cfg.ny}")
    tbl.add_row("Re", f"{cfg.re:.0f}")
    tbl.add_row("ν", f"{cfg.nu}")
    tbl.add_row("u_lid", f"{cfg.u_lid}")
    tbl.add_row("t_end", f"{cfg.t_end}")
    tbl.add_row("dt", f"{cfg.dt:.2e}")
    tbl.add_row("n_steps", f"{cfg.n_steps:,}")
    tbl.add_row("capture_fps", f"{cfg.capture_fps} /sim-s  ({cfg.frame_every:,} steps/frame)")
    tbl.add_row("GIF speed", f"{cfg.gif_playback_speed}× real time")
    conv = f"{cfg.conv_tol:.1e}" if cfg.conv_tol > 0 else "disabled"
    tbl.add_row("conv_tol", conv)
    console.print(Panel(tbl, title="[bold]Lid-driven cavity[/bold]", expand=False))

    _check_stability(cfg)

    lib_path = _find_solver_lib(Path(__file__).parent)
    console.print(f"  Library: [dim]{lib_path.name}[/dim]")

    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    xc = (np.arange(cfg.nx) + 0.5) * (cfg.lx / cfg.nx)
    yc = (np.arange(cfg.ny) + 0.5) * (cfg.ly / cfg.ny)

    snapshots, t_hist, div_hist = run_simulation(cfg, lib_path, xc, yc)
    console.print(f"  [green]✓[/green] {len(snapshots)} snapshots collected")

    sl, pl, ol, speed_max = _compute_colour_levels(snapshots)

    with console.status("[cyan]Saving plots…[/cyan]"):
        div_path = save_divergence_plot(t_hist, div_hist, out_dir)
        final_paths = save_final_figure(snapshots[-1], cfg, xc, yc, out_dir, sl, pl, ol)

    with console.status("[cyan]Rendering GIFs…[/cyan]"):
        gif_paths = render_gifs(snapshots, cfg, out_dir, xc, yc, sl, pl, ol, speed_max)

    # ── Saved files table ─────────────────────────────────────────────────
    out_tbl = Table(show_header=False, box=None, padding=(0, 2))
    out_tbl.add_column(style="bold green")
    out_tbl.add_column(style="dim")
    out_tbl.add_row("✓ divergence", str(div_path))
    for p in final_paths:
        out_tbl.add_row(f"✓ final/{p.name}", str(p))
    for kind, p in gif_paths.items():
        out_tbl.add_row(f"✓ {kind}", str(p))
    console.print(Panel(out_tbl, title="[bold]Saved files[/bold]", expand=False))


if __name__ == "__main__":
    main()
