#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ALEX_DIR = SCRIPT_DIR.parent
SCRIPTS_DIR = ALEX_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import snapshot_io  # noqa: E402


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
class Config:
    nx: int
    ny: int
    lx: float
    ly: float
    n_time_steps: int
    dt: float
    re: float
    save_every_step: int
    save_dir: Path
    use_arakawa: bool
    bc: BoundaryConditions


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "on", "yes"}:
        return True
    if value in {"false", "0", "off", "no"}:
        return False
    raise ValueError(f"invalid bool value: {value}")


def parse_config(path: Path) -> tuple[Config, dict[str, str]]:
    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw[key.strip()] = value.strip()

    def f(key: str, default: float = 0.0) -> float:
        return float(raw.get(key, default))

    def i(key: str, default: int = 0) -> int:
        return int(raw.get(key, default))

    cfg = Config(
        nx=i("nx", 601),
        ny=i("ny", 601),
        lx=f("lx", 1.0),
        ly=f("ly", 1.0),
        n_time_steps=i("n_time_steps", 100000),
        dt=f("dt", 1e-4),
        re=f("Re", 1000.0),
        save_every_step=i("save_every_step", 5000),
        save_dir=Path(raw.get("save_dir", "alex/torch_cavity_601/results/run")),
        use_arakawa=parse_bool(raw.get("use_arakawa", "false")),
        bc=BoundaryConditions(
            left=WallVelocity(f("bc.left.u"), f("bc.left.v")),
            right=WallVelocity(f("bc.right.u"), f("bc.right.v")),
            bottom=WallVelocity(f("bc.bottom.u"), f("bc.bottom.v")),
            top=WallVelocity(f("bc.top.u"), f("bc.top.v")),
        ),
    )
    return cfg, raw


def dst1(x: torch.Tensor, dim: int) -> torch.Tensor:
    n = x.shape[dim]
    ext_shape = list(x.shape)
    ext_shape[dim] = 2 * (n + 1)
    ext = torch.zeros(ext_shape, dtype=x.dtype, device=x.device)

    idx_mid = [slice(None)] * x.ndim
    idx_mid[dim] = slice(1, n + 1)
    ext[tuple(idx_mid)] = x

    idx_tail = [slice(None)] * x.ndim
    idx_tail[dim] = slice(n + 2, None)
    ext[tuple(idx_tail)] = -torch.flip(x, dims=(dim,))

    coeffs = -torch.fft.fft(ext, dim=dim).imag
    idx_out = [slice(None)] * x.ndim
    idx_out[dim] = slice(1, n + 1)
    return coeffs[tuple(idx_out)]


class PoissonSolver:
    def __init__(self, nx: int, ny: int, dx: float, dy: float, device: torch.device, dtype: torch.dtype):
        self.ni = nx - 2
        self.nj = ny - 2
        kx = torch.arange(1, self.ni + 1, device=device, dtype=dtype)
        ky = torch.arange(1, self.nj + 1, device=device, dtype=dtype)
        lam_x = -4.0 * torch.sin(0.5 * math.pi * kx / (self.ni + 1)) ** 2 / (dx * dx)
        lam_y = -4.0 * torch.sin(0.5 * math.pi * ky / (self.nj + 1)) ** 2 / (dy * dy)
        self.lam = lam_x[:, None] + lam_y[None, :]
        self.inverse_scale = 1.0 / (4.0 * (self.ni + 1) * (self.nj + 1))

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        rhs_hat = dst1(dst1(rhs, dim=0), dim=1)
        sol_hat = rhs_hat / self.lam
        return dst1(dst1(sol_hat, dim=0), dim=1) * self.inverse_scale


def apply_thom_boundary(psi: torch.Tensor, omega: torch.Tensor, cfg: Config, dx: float, dy: float) -> None:
    bc = cfg.bc
    omega[1:-1, 0] = -(2.0 * psi[1:-1, 1]) / (dy * dy) - 2.0 * bc.bottom.u / dy
    omega[1:-1, -1] = -(2.0 * psi[1:-1, -2]) / (dy * dy) - 2.0 * bc.top.u / dy
    omega[0, 1:-1] = -(2.0 * psi[1, 1:-1]) / (dx * dx) - 2.0 * bc.left.v / dx
    omega[-1, 1:-1] = -(2.0 * psi[-2, 1:-1]) / (dx * dx) - 2.0 * bc.right.v / dx
    omega[0, 0] = 0.0
    omega[0, -1] = 0.0
    omega[-1, 0] = 0.0
    omega[-1, -1] = 0.0


def solve_psi_from_omega(psi: torch.Tensor, omega: torch.Tensor, poisson: PoissonSolver) -> None:
    psi.zero_()
    psi[1:-1, 1:-1] = poisson.solve(-omega[1:-1, 1:-1])


def velocities(psi: torch.Tensor, cfg: Config, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.zeros_like(psi)
    v = torch.zeros_like(psi)
    bc = cfg.bc

    u[0, :] = bc.left.u
    v[0, :] = bc.left.v
    u[-1, :] = bc.right.u
    v[-1, :] = bc.right.v
    u[1:-1, 0] = bc.bottom.u
    v[1:-1, 0] = bc.bottom.v
    u[1:-1, -1] = bc.top.u
    v[1:-1, -1] = bc.top.v

    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)
    return u, v


def laplacian(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    return (
        (field[:-2, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[2:, 1:-1]) / (dx * dx)
        + (field[1:-1, :-2] - 2.0 * field[1:-1, 1:-1] + field[1:-1, 2:]) / (dy * dy)
    )


def central_jacobian(psi: torch.Tensor, omega: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    psi_x = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)
    psi_y = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dy)
    omega_x = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * dx)
    omega_y = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * dy)
    return psi_x * omega_y - psi_y * omega_x


def arakawa_jacobian(psi: torch.Tensor, omega: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    inv_4dxdy = 1.0 / (4.0 * dx * dy)
    psi_e = psi[2:, 1:-1]
    psi_w = psi[:-2, 1:-1]
    psi_n = psi[1:-1, 2:]
    psi_s = psi[1:-1, :-2]
    omega_e = omega[2:, 1:-1]
    omega_w = omega[:-2, 1:-1]
    omega_n = omega[1:-1, 2:]
    omega_s = omega[1:-1, :-2]

    j1 = ((psi_e - psi_w) * (omega_n - omega_s) - (psi_n - psi_s) * (omega_e - omega_w)) * inv_4dxdy
    j2 = (
        psi_e * (omega[2:, 2:] - omega[2:, :-2])
        - psi_w * (omega[:-2, 2:] - omega[:-2, :-2])
        - psi_n * (omega[2:, 2:] - omega[:-2, 2:])
        + psi_s * (omega[2:, :-2] - omega[:-2, :-2])
    ) * inv_4dxdy
    j3 = (
        omega_n * (psi[2:, 2:] - psi[:-2, 2:])
        - omega_s * (psi[2:, :-2] - psi[:-2, :-2])
        - omega_e * (psi[2:, 2:] - psi[2:, :-2])
        + omega_w * (psi[:-2, 2:] - psi[:-2, :-2])
    ) * inv_4dxdy
    return (j1 + j2 + j3) / 3.0


def steady_residuals(psi: torch.Tensor, omega: torch.Tensor, cfg: Config, dx: float, dy: float) -> tuple[float, float]:
    lap_psi = laplacian(psi, dx, dy)
    lap_omega = laplacian(omega, dx, dy)
    jac = arakawa_jacobian(psi, omega, dx, dy) if cfg.use_arakawa else central_jacobian(psi, omega, dx, dy)
    psi_res = torch.max(torch.abs(lap_psi + omega[1:-1, 1:-1]))
    omega_res = torch.max(torch.abs((1.0 / cfg.re) * lap_omega - jac))
    return float(psi_res.detach().cpu()), float(omega_res.detach().cpu())


def write_snapshot(
    output_dir: Path,
    step: int,
    time_value: float,
    xs: np.ndarray,
    ys: np.ndarray,
    cfg: Config,
    psi: torch.Tensor,
    omega: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> None:
    bc = snapshot_io.BoundaryConditions(
        left=snapshot_io.WallVelocity(cfg.bc.left.u, cfg.bc.left.v),
        right=snapshot_io.WallVelocity(cfg.bc.right.u, cfg.bc.right.v),
        bottom=snapshot_io.WallVelocity(cfg.bc.bottom.u, cfg.bc.bottom.v),
        top=snapshot_io.WallVelocity(cfg.bc.top.u, cfg.bc.top.v),
    )
    snapshot_io.write_snapshot(
        output_dir / f"result_{step}.bin",
        xs,
        ys,
        psi.detach().cpu().numpy(),
        omega.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
        v.detach().cpu().numpy(),
        step=step,
        time=time_value,
        re=cfg.re,
        bc=bc,
    )


def run(config_path: Path, output_dir: Path | None, device_name: str, dtype_name: str, clean: bool) -> None:
    cfg, _ = parse_config(config_path)
    if output_dir is not None:
        cfg = Config(**{**cfg.__dict__, "save_dir": output_dir})

    if clean and cfg.save_dir.exists():
        shutil.rmtree(cfg.save_dir)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")

    dx = cfg.lx / (cfg.nx - 1)
    dy = cfg.ly / (cfg.ny - 1)
    xs = np.linspace(0.0, cfg.lx, cfg.nx)
    ys = np.linspace(0.0, cfg.ly, cfg.ny)

    psi = torch.zeros((cfg.nx, cfg.ny), device=device, dtype=dtype)
    omega = torch.zeros_like(psi)
    poisson = PoissonSolver(cfg.nx, cfg.ny, dx, dy, device, dtype)

    history_path = cfg.save_dir / "residual_history.csv"
    started = time.time()
    print(
        f"[torch-cavity] config={config_path} grid={cfg.nx}x{cfg.ny} "
        f"Re={cfg.re:g} steps={cfg.n_time_steps} dt={cfg.dt:g} device={device} dtype={dtype_name}",
        flush=True,
    )

    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "psi_res", "omega_res", "max_residual"])

        for step in range(cfg.n_time_steps):
            solve_psi_from_omega(psi, omega, poisson)
            apply_thom_boundary(psi, omega, cfg, dx, dy)

            jac = arakawa_jacobian(psi, omega, dx, dy) if cfg.use_arakawa else central_jacobian(psi, omega, dx, dy)
            rhs = (1.0 / cfg.re) * laplacian(omega, dx, dy) - jac
            omega[1:-1, 1:-1] += cfg.dt * rhs
            apply_thom_boundary(psi, omega, cfg, dx, dy)

            if step % cfg.save_every_step == 0 or step == cfg.n_time_steps - 1:
                solve_psi_from_omega(psi, omega, poisson)
                apply_thom_boundary(psi, omega, cfg, dx, dy)
                u, v = velocities(psi, cfg, dx, dy)
                psi_res, omega_res = steady_residuals(psi, omega, cfg, dx, dy)
                max_res = max(psi_res, omega_res)
                time_value = (step + 1) * cfg.dt
                writer.writerow([step, f"{time_value:.16e}", f"{psi_res:.16e}", f"{omega_res:.16e}", f"{max_res:.16e}"])
                f.flush()
                write_snapshot(cfg.save_dir, step, time_value, xs, ys, cfg, psi, omega, u, v)
                elapsed = time.time() - started
                print(
                    f"[torch-cavity] step={step} t={time_value:.6g} "
                    f"psi_res={psi_res:.3e} omega_res={omega_res:.3e} elapsed={elapsed:.1f}s",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch CUDA lid-driven cavity solver for Alex binary snapshots.")
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    run(
        args.config.expanduser().resolve(),
        args.output_dir.expanduser().resolve() if args.output_dir is not None else None,
        args.device,
        args.dtype,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
