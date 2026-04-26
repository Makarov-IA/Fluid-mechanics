"""Simulation configuration and expression-evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


_EVAL_NS: dict[str, object] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "tanh": np.tanh,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "pi": np.pi,
    "e": np.e,
    "np": np,
}


def eval_expr(expr: str, **kwargs) -> np.ndarray:
    """Evaluate a math expression string in a NumPy-aware namespace."""
    result = eval(expr, {"__builtins__": {}}, {**_EVAL_NS, **kwargs})  # noqa: S307
    return np.asarray(result, dtype=np.float64)


def is_zero_expr(expr: str | None) -> bool:
    """Return True if the expression is trivially zero."""
    return expr is None or expr.strip() in ("0", "0.0", "0.", "+0", "-0")


@dataclass
class Snapshot:
    """Solution at one point in time, stored as float32 to reduce memory use."""

    step: int
    t: float
    p: np.ndarray
    uc: np.ndarray
    vc: np.ndarray
    omega: np.ndarray


@dataclass
class MacState:
    """Exact internal MAC solver state in C++ unknown ordering."""

    u_vec: np.ndarray
    v_vec: np.ndarray
    p: np.ndarray


@dataclass
class SimConfig:
    """All physical, numerical, and output parameters for one run."""

    lx: float = 1.0
    ly: float = 1.0
    nx: int = 25
    ny: int = 25
    nu: float = 1e-3

    t_end: float = 30.0
    n_steps: int = 300_000
    video_fps: float = 30.0
    video_speed: float = 1.0
    save_velocity_change_plot: bool = False
    conv_tol: float = 1e-6

    fixed_time_state_t: float = 0.0
    steady_max_newton_iters: int = 12
    steady_residual_tol: float = 1e-8
    steady_krylov_tol: float = 1e-6
    steady_krylov_maxiter: int = 40
    steady_krylov_restart: int = 25
    steady_jacobian_rdiff: float = 1e-6
    steady_line_search: str = "armijo"
    steady_min_step: float = 1e-3

    forcing_u: str = "0.0"
    forcing_v: str = "0.0"

    bc_u_top: str | None = None
    bc_u_bot: str | None = None
    bc_v_left: str | None = None
    bc_v_right: str | None = None
    bc_u_left: str | None = None
    bc_u_right: str | None = None
    bc_v_bot: str | None = None
    bc_v_top: str | None = None

    def __post_init__(self) -> None:
        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("nx and ny must be > 1")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if self.t_end <= 0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.video_fps <= 0:
            raise ValueError(f"video_fps must be positive, got {self.video_fps}")
        if self.video_speed <= 0:
            raise ValueError(f"video_speed must be positive, got {self.video_speed}")
        if self.conv_tol < 0:
            raise ValueError(f"conv_tol must be >= 0, got {self.conv_tol}")
        if not (0.0 <= self.fixed_time_state_t <= self.t_end):
            raise ValueError("fixed_time_state_t must lie in [0, t_end]")
        if self.steady_max_newton_iters <= 0:
            raise ValueError("steady_max_newton_iters must be positive")
        if self.steady_residual_tol <= 0:
            raise ValueError("steady_residual_tol must be positive")
        if self.steady_krylov_tol <= 0:
            raise ValueError("steady_krylov_tol must be positive")
        if self.steady_krylov_maxiter <= 0:
            raise ValueError("steady_krylov_maxiter must be positive")
        if self.steady_krylov_restart <= 0:
            raise ValueError("steady_krylov_restart must be positive")
        if self.steady_jacobian_rdiff <= 0:
            raise ValueError("steady_jacobian_rdiff must be positive")
        if self.steady_line_search not in ("armijo", "none"):
            raise ValueError("steady_line_search must be 'armijo' or 'none'")
        if not (0 < self.steady_min_step <= 1):
            raise ValueError("steady_min_step must be in (0, 1]")

    @property
    def dt(self) -> float:
        return self.t_end / self.n_steps

    @property
    def capture_fps(self) -> float:
        """Frames to capture per simulation-second."""
        return self.video_fps / self.video_speed

    @property
    def frame_every(self) -> int:
        """Solver steps between snapshots."""
        return max(1, round(1.0 / (self.capture_fps * self.dt)))

    @property
    def has_forcing(self) -> bool:
        return not (is_zero_expr(self.forcing_u) and is_zero_expr(self.forcing_v))

    @classmethod
    def from_yaml(cls, path: Path) -> "SimConfig":
        with path.open() as fh:
            data = yaml.safe_load(fh)

        forcing = data.get("forcing", {}) or {}
        boundary = data.get("boundary", {}) or {}
        output = data.get("output", {}) or {}
        time_data = data.get("time", {}) or {}
        steady = data.get("steady_solver", {}) or {}

        return cls(
            lx=data["domain"]["lx"],
            ly=data["domain"]["ly"],
            nx=data["grid"]["nx"],
            ny=data["grid"]["ny"],
            nu=data["physics"]["nu"],
            t_end=time_data.get("t_end", 30.0),
            n_steps=time_data.get("n_steps", 1000),
            video_fps=output.get("video_fps", 30.0),
            video_speed=output.get("video_speed", 1.0),
            save_velocity_change_plot=bool(output.get("save_velocity_change_plot", False)),
            conv_tol=(data.get("convergence", {}) or {}).get("tol", 1e-6),
            fixed_time_state_t=output.get("fixed_time_state_t", 0.0),
            steady_max_newton_iters=steady.get("max_newton_iters", 12),
            steady_residual_tol=steady.get("residual_tol", 1e-8),
            steady_krylov_tol=steady.get("krylov_tol", 1e-6),
            steady_krylov_maxiter=steady.get("krylov_maxiter", 40),
            steady_krylov_restart=steady.get("krylov_restart", 25),
            steady_jacobian_rdiff=steady.get("jacobian_rdiff", 1e-6),
            steady_line_search=str(steady.get("line_search", "armijo")),
            steady_min_step=steady.get("min_step", 1e-3),
            forcing_u=str(forcing.get("fu", "0.0")),
            forcing_v=str(forcing.get("fv", "0.0")),
            bc_u_top=boundary.get("u_top"),
            bc_u_bot=boundary.get("u_bot"),
            bc_v_left=boundary.get("v_left"),
            bc_v_right=boundary.get("v_right"),
            bc_u_left=boundary.get("u_left"),
            bc_u_right=boundary.get("u_right"),
            bc_v_bot=boundary.get("v_bot"),
            bc_v_top=boundary.get("v_top"),
        )

    def make_bc_arrays(self) -> dict[str, np.ndarray]:
        """Evaluate boundary-condition expressions on the relevant grid segments."""
        dx = self.lx / self.nx
        dy = self.ly / self.ny

        x_u_int = np.arange(1, self.nx) * dx
        y_v_int = np.arange(1, self.ny) * dy
        y_u_face = (np.arange(self.ny) + 0.5) * dy
        x_v_face = (np.arange(self.nx) + 0.5) * dx

        def _ev(expr: str | None, default: np.ndarray, **kwargs) -> np.ndarray:
            if expr is None:
                return default
            values = eval_expr(expr, **kwargs)
            return np.broadcast_to(values, default.shape).copy()

        zeros_u_int = np.zeros(self.nx - 1)
        zeros_v_int = np.zeros(self.ny - 1)
        zeros_u_face = np.zeros(self.ny)
        zeros_v_face = np.zeros(self.nx)

        return {
            "u_top": _ev(self.bc_u_top, zeros_u_int, x=x_u_int, y=self.ly),
            "u_bot": _ev(self.bc_u_bot, zeros_u_int, x=x_u_int, y=0.0),
            "v_left": _ev(self.bc_v_left, zeros_v_int, x=0.0, y=y_v_int),
            "v_right": _ev(self.bc_v_right, zeros_v_int, x=self.lx, y=y_v_int),
            "u_left": _ev(self.bc_u_left, zeros_u_face, x=0.0, y=y_u_face),
            "u_right": _ev(self.bc_u_right, zeros_u_face, x=self.lx, y=y_u_face),
            "v_bot": _ev(self.bc_v_bot, zeros_v_face, x=x_v_face, y=0.0),
            "v_top": _ev(self.bc_v_top, zeros_v_face, x=x_v_face, y=self.ly),
        }

    def make_force_arrays(
        self,
        t: float = 0.0,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Evaluate body-force expressions on the u- and v-face grids."""
        dx = self.lx / self.nx
        dy = self.ly / self.ny

        fu = fv = None

        if not is_zero_expr(self.forcing_u):
            x_u = np.arange(1, self.nx) * dx
            y_u = (np.arange(self.ny) + 0.5) * dy
            x_grid_u, y_grid_u = np.meshgrid(x_u, y_u)
            values = eval_expr(self.forcing_u, x=x_grid_u, y=y_grid_u, t=t)
            fu = np.broadcast_to(values, x_grid_u.shape).astype(np.float64).ravel()

        if not is_zero_expr(self.forcing_v):
            x_v = (np.arange(self.nx) + 0.5) * dx
            y_v = np.arange(1, self.ny) * dy
            x_grid_v, y_grid_v = np.meshgrid(x_v, y_v)
            values = eval_expr(self.forcing_v, x=x_grid_v, y=y_grid_v, t=t)
            fv = np.broadcast_to(values, x_grid_v.shape).astype(np.float64).ravel()

        return fu, fv
