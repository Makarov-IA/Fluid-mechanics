"""Physical curl of the body force, with the sign used in omega_t = nu*lap + J + S."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from types import ModuleType

from .config import Config


def dipole_field(x, y, magnets, plane_height=0.0):
    """SI Bz and its analytic x/y derivatives, summed over all vertical dipoles.

    d is from the dipole point to the cuvette bottom. B is this dipole's
    vertical field directly above it at the sampling plane, not remanence.
    """
    x, y = np.broadcast_arrays(x, y)
    bz = np.zeros(x.shape, dtype=np.float64)
    dbx, dby = np.zeros_like(bz), np.zeros_like(bz)
    for xk, yk, d, b in magnets:
        depth = d + plane_height
        rx, ry = x - xk, y - yk
        r2 = rx*rx + ry*ry
        q = r2 + depth*depth
        a = 0.5*b*depth**3
        bz += a*(2*depth*depth - r2)/q**2.5
        derivative = 3*a*(r2 - 4*depth*depth)/q**3.5
        dbx += derivative*rx
        dby += derivative*ry
    return bz, dbx, dby


def build_forcing(cfg: Config, xs, ys):
    x, y = xs[:, None], ys[None, :]
    source = np.zeros((len(xs), len(ys)), dtype=np.float64)
    bz = np.zeros_like(source)
    f = cfg.forcing
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        if f["type"] == "custom":
            file = Path(f["file"])
            # Explicit user-supplied Python function, analogous to the C++ forcing file.
            # Compile fresh: do not reuse a stale module/pyc after the file is edited.
            module = ModuleType("custom_vorticity_source")
            module.__file__ = str(file)
            exec(compile(file.read_bytes(), str(file), "exec"), module.__dict__)
            callback = getattr(module, "source", None)
            if not callable(callback):
                raise ValueError("Custom forcing file must define source(x, y, lx, ly)")
            values = np.asarray(callback(x, y, cfg.lx, cfg.ly))
            if np.iscomplexobj(values) or values.dtype.kind not in "fiu":
                raise ValueError("Custom source must return real numeric values")
            try:
                source[:] = np.broadcast_to(values, source.shape)
            except ValueError as exc:
                raise ValueError(f"Custom source shape {values.shape} cannot broadcast to {source.shape}") from exc
        elif f["type"] == "magnets":
            length, speed = f["scales"]["length_m"], f["scales"]["velocity_m_s"]
            bz, dbx, dby = dipole_field(x*length, y*length, f["magnets"], f["plane_height_m"])
            jx, jy = f["current_density_A_m2"]
            source = -(jx*dbx + jy*dby)/f["density_kg_m3"] * (length/speed)**2
    if not np.isfinite(source).all() or not np.isfinite(bz).all():
        raise FloatingPointError("Non-finite magnetic field or vorticity source")
    return np.ascontiguousarray(source), np.ascontiguousarray(bz)
