"""Helpers for exact MAC-state pickle files."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from solver.config import MacState, SimConfig

_REQUIRED_MAC_KEYS = ("nx", "ny", "lx", "ly", "nu", "dt", "u_vec", "v_vec", "p")


def split_full_state(
    state: np.ndarray,
    nu_u: int,
    nv_u: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split [u_vec, v_vec, p] into MAC unknown blocks."""
    return state[:nu_u], state[nu_u : nu_u + nv_u], state[nu_u + nv_u :]


def load_mac_state_pickle(
    path: Path,
    cfg: SimConfig,
    *,
    check_dt: bool = True,
) -> tuple[MacState, dict[str, object]]:
    """Load and validate an exact internal MAC-state pickle."""
    if not path.exists():
        raise FileNotFoundError(f"MAC-state pickle not found: {path}")

    with path.open("rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a pickle dictionary")

    missing = [key for key in _REQUIRED_MAC_KEYS if key not in data]
    if missing:
        keys = ", ".join(missing)
        raise ValueError(f"{path} is missing keys: {keys}")

    saved_nx = int(data["nx"])
    saved_ny = int(data["ny"])
    if saved_nx != cfg.nx or saved_ny != cfg.ny:
        raise ValueError(
            f"{path} grid {saved_nx}x{saved_ny} does not match "
            f"config grid {cfg.nx}x{cfg.ny}"
        )

    keys_to_check = [("lx", cfg.lx), ("ly", cfg.ly), ("nu", cfg.nu)]
    if check_dt:
        keys_to_check.append(("dt", cfg.dt))

    for key, current in keys_to_check:
        saved = float(data[key])
        if not np.isclose(saved, current, rtol=1e-12, atol=1e-14):
            raise ValueError(f"{path} {key}={saved} does not match config {key}={current}")

    expected_u = (cfg.nx - 1) * cfg.ny
    expected_v = cfg.nx * (cfg.ny - 1)
    expected_p = cfg.nx * cfg.ny
    u_vec = np.asarray(data["u_vec"], dtype=np.float64).reshape(-1)
    v_vec = np.asarray(data["v_vec"], dtype=np.float64).reshape(-1)
    p_vec = np.asarray(data["p"], dtype=np.float64).reshape(-1)

    if u_vec.shape != (expected_u,):
        raise ValueError(f"u_vec must have shape {(expected_u,)}, got {u_vec.shape}")
    if v_vec.shape != (expected_v,):
        raise ValueError(f"v_vec must have shape {(expected_v,)}, got {v_vec.shape}")
    if p_vec.shape != (expected_p,):
        raise ValueError(f"p must have shape {(expected_p,)}, got {p_vec.shape}")

    metadata = {
        key: data[key]
        for key in ("nx", "ny", "lx", "ly", "nu", "dt", "step", "t")
        if key in data
    }
    return MacState(u_vec=u_vec, v_vec=v_vec, p=p_vec), metadata


def full_mode_grids(
    cfg: SimConfig,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert full-state eigenvector columns into complex MAC u/v/p blocks."""
    nu_u = (cfg.nx - 1) * cfg.ny
    nv_u = cfg.nx * (cfg.ny - 1)
    modes_u = []
    modes_v = []
    modes_p = []
    for mode_idx in range(eigenvectors.shape[1]):
        u_vec, v_vec, p_vec = split_full_state(eigenvectors[:, mode_idx], nu_u, nv_u)
        modes_u.append(u_vec.reshape(cfg.ny, cfg.nx - 1).T)
        modes_v.append(v_vec.reshape(cfg.ny - 1, cfg.nx).T)
        modes_p.append(p_vec.reshape(cfg.ny, cfg.nx).T)
    return np.asarray(modes_u), np.asarray(modes_v), np.asarray(modes_p)
