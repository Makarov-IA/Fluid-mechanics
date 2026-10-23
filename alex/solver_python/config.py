"""Strict, safe YAML input. Grid and wall velocities are nondimensional."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from solver_python.snapshot_io import BoundaryConditions, WallVelocity


ALEX = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ALEX / "results"
DEFAULT_CONFIG = ALEX / "configs/pulsation_re260/pulsation_re260.yaml"


class ConfigError(ValueError):
    pass


class UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently discarding a user's setting."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError(f"YAML keys must be strings (line {key_node.start_mark.line + 1})")
        if key in result:
            raise ConfigError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def section(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    extra = set(value) - allowed
    if extra:
        raise ConfigError(f"Unknown {name} keys: {', '.join(sorted(extra))}")
    return value


def number(value: Any, name: str, *, positive=False, nonnegative=False) -> float:
    # PyYAML may parse 1e-5 as a string; accept numeric strings, never bool/null.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{name} must be a finite number")
    try:
        result = float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ConfigError(f"Invalid {name}: {value}")
    return result


def integer(value: Any, name: str, minimum=1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Config:
    path: Path
    experiment: str
    nx: int
    ny: int
    lx: float
    ly: float
    re: float
    mode: str
    dt: float
    n_time_steps: int
    steady_tolerance: float
    bc: BoundaryConditions
    threads: int
    output_dir: Path
    save_every_step: int
    log_every_step: int
    forcing: dict
    resolved: dict
    stabilization: dict | None = None
    resume: dict | None = None
    magnetic_actuation: dict | None = None

    @property
    def result_dir(self) -> Path:
        return self.output_dir.parent


def parse_config(raw: Any, path: Path | str) -> Config:
    path = Path(path).expanduser().resolve()
    raw = section(raw, "root", {"version", "grid", "physics", "time",
                               "boundaries", "execution", "output", "forcing", "stabilization", "resume",
                               "magnetic_actuation"})
    if type(raw.get("version")) is not int or raw["version"] != 1:
        raise ConfigError("version must be 1")
    experiment = path.stem
    if not isinstance(experiment, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment):
        raise ConfigError("YAML filename must use letters/digits/_.-")
    grid = section(raw.get("grid"), "grid", {"nx", "ny", "lx", "ly"})
    nx, ny = (integer(grid.get(k), f"grid.{k}", 3) for k in ("nx", "ny"))
    lx, ly = (number(grid.get(k), f"grid.{k}", positive=True) for k in ("lx", "ly"))
    physics = section(raw.get("physics"), "physics", {"re"})
    re_value = number(physics.get("re"), "physics.re", positive=True)
    timing = section(raw.get("time"), "time", {"mode", "dt", "t_max", "n_time_steps", "steady_tolerance"})
    mode = timing.get("mode", "fixed_dt_steps")
    if mode not in ("fixed_steps", "fixed_dt_steps", "till_converges"):
        raise ConfigError("time.mode must be fixed_steps, fixed_dt_steps or till_converges")
    nsteps = integer(timing.get("n_time_steps"), "time.n_time_steps")
    if mode == "fixed_steps":
        if "dt" in timing:
            raise ConfigError("fixed_steps uses t_max/n_time_steps: remove time.dt")
        dt = number(timing.get("t_max"), "time.t_max", positive=True) / nsteps
    else:
        if "t_max" in timing:
            raise ConfigError(f"{mode} uses dt and n_time_steps: remove time.t_max")
        dt = number(timing.get("dt"), "time.dt", positive=True)
    tol = number(timing.get("steady_tolerance", 1e-10), "time.steady_tolerance", positive=True)
    boundaries = section(raw.get("boundaries", {}), "boundaries", {"left", "right", "bottom", "top"})
    walls = {}
    for wall in ("left", "right", "bottom", "top"):
        values = section(boundaries.get(wall, {}), f"boundaries.{wall}", {"u", "v"})
        walls[wall] = WallVelocity(*(number(values.get(k, 0), f"boundaries.{wall}.{k}") for k in ("u", "v")))
    bc = BoundaryConditions(**walls)
    if bc.left.u or bc.right.u or bc.bottom.v or bc.top.v:
        raise ConfigError("Normal wall velocities must be zero for psi=0 impermeable walls")
    execution = section(raw.get("execution", {}), "execution", {"threads"})
    threads = integer(execution.get("threads", 1), "execution.threads")
    output = section(raw.get("output", {}), "output", {"save_every_step", "log_every_step"})
    out_dir = RESULTS_ROOT / experiment / "binaries"
    save_every = integer(output.get("save_every_step", 1000), "output.save_every_step")
    log_every = integer(output.get("log_every_step", save_every), "output.log_every_step")
    forcing = parse_forcing(raw.get("forcing"), re_value, path.parent)
    stabilization = parse_stabilization(raw.get("stabilization"), path.parent)
    resume = parse_resume(raw.get("resume"), path.parent)
    magnetic_actuation = None
    if raw.get("magnetic_actuation") is not None:
        if not isinstance(raw["magnetic_actuation"], dict):
            raise ConfigError("magnetic_actuation must be a mapping")
        settings = dict(raw["magnetic_actuation"])
        reference = settings.pop("reference", None)
        if reference != "base_forcing":
            raise ConfigError("magnetic_actuation.reference must explicitly be base_forcing")
        if settings.get("type") != "magnets":
            raise ConfigError("magnetic_actuation.type must be magnets")
        magnetic_actuation = parse_forcing(settings, re_value, path.parent)
        magnetic_actuation["reference"] = reference
        if mode == "till_converges":
            raise ConfigError("magnetic_actuation requires a fixed-duration run; residuals refer to base_forcing")
    if not math.isfinite(dt*nsteps) or dt <= 0:
        raise ConfigError("Total pseudo-time must be finite and dt must be positive")
    resolved = {
        "version": 1,
        "grid": dict(nx=nx, ny=ny, lx=lx, ly=ly), "physics": {"re": re_value},
        "time": dict(mode=mode, n_time_steps=nsteps, steady_tolerance=tol,
                     **({"t_max": dt*nsteps} if mode == "fixed_steps" else {"dt": dt})),
        "boundaries": {key: {"u": value.u, "v": value.v} for key, value in walls.items()},
        "execution": {"threads": threads},
        "output": {"save_every_step": save_every, "log_every_step": log_every},
        "forcing": forcing,
        **({"stabilization": stabilization} if stabilization is not None else {}),
        **({"resume": resume} if resume is not None else {}),
        **({"magnetic_actuation": magnetic_actuation} if magnetic_actuation is not None else {}),
    }
    return Config(path, experiment, nx, ny, lx, ly, re_value, mode, dt, nsteps, tol,
                  bc, threads, out_dir, save_every, log_every, forcing, resolved, stabilization, resume,
                  magnetic_actuation)


def parse_resume(value: Any, directory: Path) -> dict | None:
    """Each component explicitly selects its own ALEXBIN1 snapshot."""
    if value is None:
        return None
    settings = section(value, "resume", {"psi", "omega"})
    result = {}
    for field in ("psi", "omega"):
        filename = settings.get(field)
        if not isinstance(filename, str) or not filename.strip():
            raise ConfigError(f"resume.{field} must name an ALEXBIN1 snapshot")
        path = Path(filename).expanduser()
        if not path.is_absolute():
            path = directory/path
        path = path.resolve()
        if not path.is_file():
            raise ConfigError(f"resume.{field} file does not exist: {path}")
        result[field] = str(path)
    return result


def parse_stabilization(value: Any, directory: Path) -> dict | None:
    """Optional RHS feedback or full post-step projection from saved modes."""
    if value is None:
        return None
    s = section(value, "stabilization", {"file", "gain", "method", "components"})
    filename = s.get("file")
    if not isinstance(filename, str) or not filename.strip():
        raise ConfigError("stabilization.file must be the .npz written by stability.py")
    file = Path(filename).expanduser()
    if not file.is_absolute():
        file = directory/file
    file = file.resolve()
    if file.suffix != ".npz" or not file.is_file():
        raise ConfigError(f"Stabilization file does not exist or is not .npz: {file}")
    # RHS uses -gain*P: a negative gain intentionally adds the projection.
    # Full post-step removal retains its separate gain=1 restriction below.
    gain = number(s.get("gain"), "stabilization.gain")
    method = s.get("method", "rhs")
    components = s.get("components", "omega")
    if method not in ("rhs", "projection") or components not in ("omega", "state"):
        raise ConfigError("stabilization method: rhs/projection; components: omega/state")
    if method == "projection" and (components != "state" or gain != 1.0):
        raise ConfigError("Full post-step projection requires components: state and gain: 1")
    return {"file": str(file), "gain": gain, "method": method, "components": components}


def parse_forcing(value: Any, re_value: float, directory: Path) -> dict:
    if not isinstance(value, dict):
        raise ConfigError("forcing must be a mapping with an explicit type")
    kind = value.get("type")
    if kind == "none":
        section(value, "forcing", {"type"})
        return {"type": kind}
    if kind == "custom":
        section(value, "forcing", {"type", "file"})
        filename = value.get("file")
        if not isinstance(filename, str) or not filename.strip():
            raise ConfigError("forcing.file must be a Python source file")
        file = Path(filename).expanduser()
        if not file.is_absolute():
            file = directory/file
        file = file.resolve()
        if file.suffix != ".py" or not file.is_file():
            raise ConfigError(f"Custom forcing file does not exist or is not .py: {file}")
        return {"type": kind, "file": str(file)}
    if kind != "magnets":
        raise ConfigError("forcing.type must be none, custom or magnets")
    f = section(value, "forcing", {"type", "magnets", "density_kg_m3", "current_density_A_m2",
                                  "plane_height_m", "scales"})
    density = number(f.get("density_kg_m3"), "forcing.density_kg_m3", positive=True)
    height = number(f.get("plane_height_m", 0), "forcing.plane_height_m", nonnegative=True)
    current = f.get("current_density_A_m2")
    if not isinstance(current, list) or len(current) != 2:
        raise ConfigError("forcing.current_density_A_m2 must be [jx, jy]")
    current = [number(v, "current_density_A_m2") for v in current]
    scales = section(f.get("scales"), "forcing.scales", {"length_m", "velocity_m_s", "kinematic_viscosity_m2_s"})
    scales = {k: number(scales.get(k), f"forcing.scales.{k}", positive=True)
              for k in ("length_m", "velocity_m_s", "kinematic_viscosity_m2_s")}
    physical_re = scales["length_m"] * scales["velocity_m_s"] / scales["kinematic_viscosity_m2_s"]
    if not math.isclose(re_value, physical_re, rel_tol=1e-8):
        raise ConfigError(f"physics.re={re_value} disagrees with U*L/nu={physical_re}")
    magnets = f.get("magnets")
    if not isinstance(magnets, list) or not magnets:
        raise ConfigError("forcing.magnets must be a nonempty list of [x_m, y_m, d_m, B_axis_T]")
    parsed = []
    for k, magnet in enumerate(magnets):
        if not isinstance(magnet, list) or len(magnet) != 4:
            raise ConfigError(f"magnet {k+1} must be [x_m, y_m, d_m, B_axis_T]")
        parsed.append([number(v, f"magnet {k+1}.{name}", positive=(name == "d"))
                       for name, v in zip(("x", "y", "d", "B"), magnet)])
    return {"type": kind, "magnets": parsed, "density_kg_m3": density,
            "current_density_A_m2": current, "plane_height_m": height, "scales": scales}


def grid(cfg: Config):
    """The single grid definition. Preview tools use it to match the solver exactly."""
    return np.linspace(0, cfg.lx, cfg.nx), np.linspace(0, cfg.ly, cfg.ny)


def load_config(path: Path | str) -> Config:
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=UniqueSafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    return parse_config(raw, path)
