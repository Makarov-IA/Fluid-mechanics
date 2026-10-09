"""Feedback force realised by magnets under the electrolyte layer.

Lorentz force (Tithof et al., arXiv:1601.00243): current J = J y_hat in the
layer, magnets below it, F = J x B ~ J B_z x_hat; the 2-D force is its average
over the layer thickness:

    f = (J / rho) <B_z(x, y, z)>_z  x_hat.

B_z of a cuboid magnet magnetised along z is evaluated in closed form.

The theoretical feedback force is f_c = -(alpha/dt) Q c(t), c = W^T (u - u_ref).
The magnet strengths are fitted to it by least squares:  B S ~ Q,  so the
magnet force is f_mag = -(alpha/dt) B S c(t) with strengths s(t) = -(alpha/dt) S c(t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver.config import SimConfig


def bz_cuboid(x, y, z, x1, x2, y1, y2, z1, z2) -> np.ndarray:
    """B_z of a cuboid magnetised along +z, in units of mu0 M / (4 pi)."""
    out = np.zeros(np.broadcast(x, y, z).shape)
    for i, xi in enumerate((x1, x2)):
        for j, yj in enumerate((y1, y2)):
            for k, zk in enumerate((z1, z2)):
                X, Y, Z = x - xi, y - yj, z - zk
                R = np.sqrt(X * X + Y * Y + Z * Z)
                out += (-1) ** (i + j + k + 1) * np.arctan2(X * Y, Z * R)
    return out


class MagnetModel:
    """Depth-averaged Lorentz force of one magnet on the u faces (f_v = 0)."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        nx, ny = cfg.nx, cfg.ny
        dx, dy = cfg.lx / nx, cfg.ly / ny
        self.nu = (nx - 1) * ny
        self.nvel = self.nu + nx * (ny - 1)
        self.xu = (np.arange(1, nx) * dx)[None, :] * np.ones((ny, 1))
        self.yu = ((np.arange(ny) + 0.5) * dy)[:, None] * np.ones((1, nx - 1))
        z, w = np.polynomial.legendre.leggauss(4)
        self.zq = 0.5 * cfg.magnets_layer * (z + 1.0)  # layer z in [0, h_e]
        self.wq = 0.5 * w

    def force(self, cx: float, cy: float, size: float) -> np.ndarray:
        """Unit force field (max |f| = 1) of a size x size magnet centred at (cx, cy)."""
        c = self.cfg
        z_top = -c.magnets_gap
        z_bot = z_top - c.magnets_thickness
        h = size / 2.0
        bz = sum(
            wk * bz_cuboid(self.xu, self.yu, zk, cx - h, cx + h, cy - h, cy + h, z_bot, z_top)
            for zk, wk in zip(self.zq, self.wq)
        )
        f = np.zeros(self.nvel)
        f[: self.nu] = bz.ravel()
        return f / np.abs(f).max()


@dataclass
class MagnetActuator:
    centres: list[tuple[float, float]]
    size: float
    fields: np.ndarray         # B: unit magnet forces, (velocity_size, N)
    strengths_map: np.ndarray  # S = lstsq(B, Q), (N, m): s = -(alpha/dt) S c
    basis: np.ndarray          # B S, the magnet force basis passed to the solver
    exact_basis: np.ndarray    # Q, the theoretical force basis
    fit_error: np.ndarray      # |B S e_k - Q e_k| / |Q e_k| per column

    def strengths(self, coefficients: np.ndarray, alpha: float, dt: float) -> np.ndarray:
        """Magnet strengths (peak force of each magnet) for c(t), shape (steps, N)."""
        return -(alpha / dt) * coefficients @ self.strengths_map.T

    def difference_norms(
        self, coefficients: np.ndarray, alpha: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per step, with f_mag = -(alpha/dt) B S c and f_theory = -(alpha/dt) Q c:

        |f_mag - f_theory|_2,  |f_mag - proj_{f_theory} f_mag|_2  and  |f_theory|_2.
        """
        A, Q = self.basis, self.exact_basis
        quad = lambda M: np.einsum("ti,ij,tj->t", coefficients, M, coefficients)  # noqa: E731
        aa, aq, qq = quad(A.T @ A), quad(A.T @ Q), quad(Q.T @ Q)
        scale = alpha / dt
        diff = scale * np.sqrt(np.maximum(aa - 2.0 * aq + qq, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            perp2 = np.where(qq > 0, aa - aq * aq / qq, aa)
        perp = scale * np.sqrt(np.maximum(perp2, 0.0))
        return diff, perp, scale * np.sqrt(np.maximum(qq, 0.0))


def _fit(B: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares strengths S (B S ~ Q) and the relative residual."""
    S, *_ = np.linalg.lstsq(B, Q, rcond=None)
    return S, float(np.linalg.norm(B @ S - Q) / np.linalg.norm(Q))


def _choose_optimal(Bc: np.ndarray, Q: np.ndarray, count: int) -> list[int]:
    """Greedily pick `count` candidate magnets with the smallest fit residual."""
    chosen: list[int] = []
    for _ in range(count):
        best, best_err = -1, np.inf
        for k in range(Bc.shape[1]):
            if k in chosen:
                continue
            _, err = _fit(Bc[:, chosen + [k]], Q)
            if err < best_err:
                best, best_err = k, err
        chosen.append(best)
    return chosen


def build_magnet_actuator(cfg: SimConfig, Q: np.ndarray) -> MagnetActuator:
    """Magnets whose Lorentz force best fits the theoretical feedback force Q."""
    model = MagnetModel(cfg)
    if cfg.magnets_layout == "array":
        mx, my = cfg.magnets_array
        size = min(cfg.lx / mx, cfg.ly / my)
        centres = [((i + 0.5) * cfg.lx / mx, (j + 0.5) * cfg.ly / my) for j in range(my) for i in range(mx)]
        B = np.column_stack([model.force(cx, cy, size) for cx, cy in centres])
    else:
        size = cfg.magnets_size
        gx = max(2, int(round(cfg.lx / size * 2)))
        gy = max(1, int(round(cfg.ly / size * 2)))
        cand = [((i + 0.5) * cfg.lx / gx, (j + 0.5) * cfg.ly / gy) for j in range(gy) for i in range(gx)]
        Bc = np.column_stack([model.force(cx, cy, size) for cx, cy in cand])
        chosen = _choose_optimal(Bc, Q, cfg.magnets_count)
        centres = [cand[k] for k in chosen]
        B = Bc[:, chosen]

    S, _ = _fit(B, Q)
    basis = B @ S
    fit_error = np.linalg.norm(basis - Q, axis=0) / np.linalg.norm(Q, axis=0)
    return MagnetActuator(
        centres=centres,
        size=size,
        fields=B,
        strengths_map=S,
        basis=basis,
        exact_basis=Q,
        fit_error=fit_error,
    )
