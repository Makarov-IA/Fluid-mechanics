from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"ALEXBIN1"
HEADER_FORMAT = "<8sIIQ12d"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass(frozen=True)
class WallVelocity:
    u: float = 0.0
    v: float = 0.0


@dataclass(frozen=True)
class BoundaryConditions:
    left: WallVelocity = WallVelocity()
    right: WallVelocity = WallVelocity()
    bottom: WallVelocity = WallVelocity()
    top: WallVelocity = WallVelocity()


@dataclass(frozen=True)
class Snapshot:
    xs: np.ndarray
    ys: np.ndarray
    psi: np.ndarray
    omega: np.ndarray
    u: np.ndarray
    v: np.ndarray
    step: int
    time: float
    re: float
    bc: BoundaryConditions


def velocities_from_psi(
    psi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    bc: BoundaryConditions,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)

    u[0, :] = bc.left.u
    v[0, :] = bc.left.v
    u[-1, :] = bc.right.u
    v[-1, :] = bc.right.v
    u[1:-1, 0] = bc.bottom.u
    v[1:-1, 0] = bc.bottom.v
    u[1:-1, -1] = bc.top.u
    v[1:-1, -1] = bc.top.v

    if xs.size > 2 and ys.size > 2:
        dx = float(xs[1] - xs[0])
        dy = float(ys[1] - ys[0])
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dy)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)

    return u, v


def fields_from_snapshot(snapshot: Snapshot, include_velocity: bool = True) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {
        "psi": snapshot.psi,
        "omega": snapshot.omega,
    }
    if include_velocity:
        fields["u"] = snapshot.u
        fields["v"] = snapshot.v
    return fields


def load_snapshot(path: Path | str, include_velocity: bool = True) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    snapshot = read_snapshot(path)
    return snapshot.xs, snapshot.ys, fields_from_snapshot(snapshot, include_velocity=include_velocity)


def read_snapshot(path: Path | str) -> Snapshot:
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise RuntimeError(f"Snapshot header is truncated: {path}")

        unpacked = struct.unpack(HEADER_FORMAT, header)
        magic = unpacked[0]
        if magic != MAGIC:
            raise RuntimeError(f"Invalid Alex binary snapshot magic in {path}: {magic!r}")

        nx = int(unpacked[1])
        ny = int(unpacked[2])
        step = int(unpacked[3])
        (
            time,
            lx,
            ly,
            re,
            left_u,
            left_v,
            right_u,
            right_v,
            bottom_u,
            bottom_v,
            top_u,
            top_v,
        ) = unpacked[4:]

        if nx < 2 or ny < 2:
            raise RuntimeError(f"Invalid snapshot grid {nx}x{ny}: {path}")

        count = nx * ny
        payload = np.fromfile(f, dtype="<f8", count=4 * count)
        if payload.size != 4 * count:
            raise RuntimeError(f"Snapshot payload is truncated: {path}")

    psi = payload[:count].reshape(nx, ny).copy()
    omega = payload[count:2 * count].reshape(nx, ny).copy()
    u = payload[2 * count:3 * count].reshape(nx, ny).copy()
    v = payload[3 * count:].reshape(nx, ny).copy()
    xs = np.linspace(0.0, float(lx), nx)
    ys = np.linspace(0.0, float(ly), ny)
    bc = BoundaryConditions(
        left=WallVelocity(float(left_u), float(left_v)),
        right=WallVelocity(float(right_u), float(right_v)),
        bottom=WallVelocity(float(bottom_u), float(bottom_v)),
        top=WallVelocity(float(top_u), float(top_v)),
    )
    return Snapshot(xs=xs, ys=ys, psi=psi, omega=omega, u=u, v=v, step=step, time=float(time), re=float(re), bc=bc)


def write_snapshot(
    path: Path | str,
    xs: np.ndarray,
    ys: np.ndarray,
    psi: np.ndarray,
    omega: np.ndarray,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    *,
    step: int = 0,
    time: float = 0.0,
    re: float = 0.0,
    bc: BoundaryConditions = BoundaryConditions(),
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    psi = np.asarray(psi, dtype="<f8")
    omega = np.asarray(omega, dtype="<f8")

    nx = int(xs.size)
    ny = int(ys.size)
    if psi.shape != (nx, ny) or omega.shape != (nx, ny):
        raise RuntimeError(
            f"Snapshot fields must have shape {(nx, ny)}, got psi={psi.shape}, omega={omega.shape}"
        )
    if u is None or v is None:
        u, v = velocities_from_psi(psi, xs, ys, bc)
    u = np.asarray(u, dtype="<f8")
    v = np.asarray(v, dtype="<f8")
    if u.shape != (nx, ny) or v.shape != (nx, ny):
        raise RuntimeError(
            f"Snapshot velocities must have shape {(nx, ny)}, got u={u.shape}, v={v.shape}"
        )

    lx = float(xs[-1] - xs[0])
    ly = float(ys[-1] - ys[0])
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        nx,
        ny,
        int(step),
        float(time),
        lx,
        ly,
        float(re),
        float(bc.left.u),
        float(bc.left.v),
        float(bc.right.u),
        float(bc.right.v),
        float(bc.bottom.u),
        float(bc.bottom.v),
        float(bc.top.u),
        float(bc.top.v),
    )

    with path.open("wb") as f:
        f.write(header)
        np.ascontiguousarray(psi, dtype="<f8").tofile(f)
        np.ascontiguousarray(omega, dtype="<f8").tofile(f)
        np.ascontiguousarray(u, dtype="<f8").tofile(f)
        np.ascontiguousarray(v, dtype="<f8").tofile(f)
