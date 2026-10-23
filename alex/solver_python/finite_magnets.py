"""Field above finite, uniformly vertically magnetized rectangular prisms.

External nonmagnetic medium; prescribed magnetization, no mutual demagnetization.
The surface-charge integral over the two horizontal faces is evaluated exactly.
This model does not change the runtime's existing point-dipole forcing.
"""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


def cuboid_field(x, y, center, size, separation, magnetization):
    """Return Bz [T], dBz/dx and dBz/dy [T/m] above one cuboid.

    All lengths are SI. separation > 0 is from the TOP FACE to the
    observation plane, not from the magnet center. Positive M points upward.
    size = (width_x, width_y, thickness); magnetization is in A/m.
    """
    size = np.asarray(size, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if (size.shape != (3,) or center.shape != (2,)
            or not np.isfinite(size).all() or np.any(size <= 0)
            or not np.isfinite(center).all() or not np.isfinite(separation)
            or separation <= 0 or not np.isfinite(magnetization)):
        raise ValueError("Invalid finite magnet dimensions, position or separation")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                               np.asarray(y, dtype=np.float64))
    bz, dbx, dby = (np.zeros_like(x) for _ in range(3))
    # sigma_m = +M on the top face and -M on the bottom face.
    for z, face_sign in ((separation, 1), (separation + size[2], -1)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                xx = center[0] + sx * size[0] / 2 - x
                yy = center[1] + sy * size[1] / 2 - y
                radius = np.sqrt(xx * xx + yy * yy + z * z)
                sign = face_sign * sx * sy
                bz += sign * np.arctan2(xx * yy, z * radius)
                dbx -= sign * yy * z / (radius * (xx * xx + z * z))
                dby -= sign * xx * z / (radius * (yy * yy + z * z))
    factor = MU0 * magnetization / (4 * np.pi)
    return bz * factor, dbx * factor, dby * factor


def vorticity_source(dbx, dby, current, density, length, speed):
    """Nondimensional curl(j x B / rho), for uniform horizontal current."""
    return -(current[0] * dbx + current[1] * dby) / density * (length / speed)**2
