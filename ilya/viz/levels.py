"""Compute shared colour levels over all snapshots."""

from __future__ import annotations

import numpy as np

from solver.config import Snapshot


def compute_colour_levels(
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
