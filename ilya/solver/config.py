"""SimConfig, Snapshot, and expression-evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------

_EVAL_NS: dict = {
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
    """Evaluate a math expression string in a numpy-aware namespace."""
    result = eval(expr, {"__builtins__": {}}, {**_EVAL_NS, **kwargs})  # noqa: S307
    return np.asarray(result, dtype=np.float64)


def is_zero_expr(expr: str | None) -> bool:
    """Return True if the expression is trivially zero."""
    return expr is None or expr.strip() in ("0", "0.0", "0.", "+0", "-0")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """Solution at one point in time, stored as float32 to halve memory."""

    step: int
    t: float
    p: np.ndarray      # (Nx, Ny) float32 — pressure
    uc: np.ndarray     # (Nx, Ny) float32 — cell-centred x-velocity
    vc: np.ndarray     # (Nx, Ny) float32 — cell-centred y-velocity
    omega: np.ndarray  # (Nx, Ny) float32 — vorticity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """All physical and numerical parameters for a simulation run."""

    lx: float = 1.0
    ly: float = 1.0
    nx: int = 25
    ny: int = 25
    nu: float = 1e-3
    t_end: float = 30.0
    n_steps: int = 300_000
    video_fps: float = 30.0   # output video frame rate (fps of saved .mp4)
    video_speed: float = 1.0  # sim-seconds per real second (>1 = faster, <1 = slower)
    conv_tol: float = 1e-6    # 0 = disabled

    # Body-force expressions (functions of x, y, t).  "0.0" → no forcing.
    forcing_u: str = "0.0"
    forcing_v: str = "0.0"

    # Boundary-condition expressions.
    # Ghost-correction walls (u_top/u_bot: function of x; v_left/v_right: of y).
    # None → 0 everywhere.
    bc_u_top: str | None = None    # u at top wall.    None → 0
    bc_u_bot: str | None = None    # u at bottom wall. None → 0
    bc_v_left: str | None = None   # v at left wall.   None → 0
    bc_v_right: str | None = None  # v at right wall.  None → 0
    # Dirichlet face walls (u_left/u_right: function of y; v_bot/v_top: of x).
    bc_u_left: str | None = None   # u at left  wall. None → 0
    bc_u_right: str | None = None  # u at right wall. None → 0
    bc_v_bot: str | None = None    # v at bottom wall. None → 0
    bc_v_top: str | None = None    # v at top wall.    None → 0

    def __post_init__(self) -> None:
        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("nx and ny must be > 1")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.t_end <= 0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if self.video_fps <= 0:
            raise ValueError(f"video_fps must be positive, got {self.video_fps}")
        if self.video_speed <= 0:
            raise ValueError(f"video_speed must be positive, got {self.video_speed}")
        if self.conv_tol < 0:
            raise ValueError(f"conv_tol must be >= 0, got {self.conv_tol}")

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
        with open(path) as fh:
            d = yaml.safe_load(fh)
        forcing = d.get("forcing", {}) or {}
        bc = d.get("boundary", {}) or {}
        return cls(
            lx=d["domain"]["lx"],
            ly=d["domain"]["ly"],
            nx=d["grid"]["nx"],
            ny=d["grid"]["ny"],
            nu=d["physics"]["nu"],
            t_end=d["time"]["t_end"],
            n_steps=d["time"]["n_steps"],
            video_fps=d["output"]["video_fps"],
            video_speed=d["output"]["video_speed"],
            conv_tol=d.get("convergence", {}).get("tol", 1e-6),
            forcing_u=str(forcing.get("fu", "0.0")),
            forcing_v=str(forcing.get("fv", "0.0")),
            bc_u_top=bc.get("u_top"),
            bc_u_bot=bc.get("u_bot"),
            bc_v_left=bc.get("v_left"),
            bc_v_right=bc.get("v_right"),
            bc_u_left=bc.get("u_left"),
            bc_u_right=bc.get("u_right"),
            bc_v_bot=bc.get("v_bot"),
            bc_v_top=bc.get("v_top"),
        )

    def make_bc_arrays(self) -> dict[str, np.ndarray]:
        """Evaluate boundary-condition expressions on their respective grid segments."""
        dx = self.lx / self.nx
        dy = self.ly / self.ny

        x_u_int  = np.arange(1, self.nx) * dx
        y_v_int  = np.arange(1, self.ny) * dy
        y_u_face = (np.arange(self.ny) + 0.5) * dy
        x_v_face = (np.arange(self.nx) + 0.5) * dx

        def _ev(expr: str | None, default: np.ndarray, **kw) -> np.ndarray:
            if expr is None:
                return default
            r = eval_expr(expr, **kw)
            return np.broadcast_to(r, default.shape).copy()

        zeros_u_int  = np.zeros(self.nx - 1)
        zeros_v_int  = np.zeros(self.ny - 1)
        zeros_u_face = np.zeros(self.ny)
        zeros_v_face = np.zeros(self.nx)

        return {
            "u_top":   _ev(self.bc_u_top,   zeros_u_int,  x=x_u_int,  y=self.ly),
            "u_bot":   _ev(self.bc_u_bot,   zeros_u_int,  x=x_u_int,  y=0.0),
            "v_left":  _ev(self.bc_v_left,  zeros_v_int,  x=0.0,      y=y_v_int),
            "v_right": _ev(self.bc_v_right, zeros_v_int,  x=self.lx,  y=y_v_int),
            "u_left":  _ev(self.bc_u_left,  zeros_u_face, x=0.0,      y=y_u_face),
            "u_right": _ev(self.bc_u_right, zeros_u_face, x=self.lx,  y=y_u_face),
            "v_bot":   _ev(self.bc_v_bot,   zeros_v_face, x=x_v_face, y=0.0),
            "v_top":   _ev(self.bc_v_top,   zeros_v_face, x=x_v_face, y=self.ly),
        }

    def make_force_arrays(
        self, t: float = 0.0
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Evaluate body-force expressions on the u- and v-face grids."""
        dx = self.lx / self.nx
        dy = self.ly / self.ny

        fu = fv = None

        if not is_zero_expr(self.forcing_u):
            x_u = np.arange(1, self.nx) * dx
            y_u = (np.arange(self.ny) + 0.5) * dy
            X_u, Y_u = np.meshgrid(x_u, y_u)
            arr = eval_expr(self.forcing_u, x=X_u, y=Y_u, t=t)
            fu = np.broadcast_to(arr, X_u.shape).astype(np.float64).ravel()

        if not is_zero_expr(self.forcing_v):
            x_v = (np.arange(self.nx) + 0.5) * dx
            y_v = np.arange(1, self.ny) * dy
            X_v, Y_v = np.meshgrid(x_v, y_v)
            arr = eval_expr(self.forcing_v, x=X_v, y=Y_v, t=t)
            fv = np.broadcast_to(arr, X_v.shape).astype(np.float64).ravel()

        return fu, fv
