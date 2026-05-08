#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
class Problem:
    xs: np.ndarray
    ys: np.ndarray
    re: float
    bc: BoundaryConditions

    @property
    def nx(self) -> int:
        return int(self.xs.size)

    @property
    def ny(self) -> int:
        return int(self.ys.size)

    @property
    def lx(self) -> float:
        return float(self.xs[-1] - self.xs[0])

    @property
    def ly(self) -> float:
        return float(self.ys[-1] - self.ys[0])

    @property
    def dx(self) -> float:
        return float(self.xs[1] - self.xs[0])

    @property
    def dy(self) -> float:
        return float(self.ys[1] - self.ys[0])


class DirichletLaplaceInverse:
    def __init__(self, problem: Problem):
        self.ni = problem.nx - 2
        self.nj = problem.ny - 2

        ix = np.arange(1, self.ni + 1, dtype=float)
        jx = np.arange(1, self.ni + 1, dtype=float)
        iy = np.arange(1, self.nj + 1, dtype=float)
        jy = np.arange(1, self.nj + 1, dtype=float)

        self.sx = math.sqrt(2.0 / (self.ni + 1)) * np.sin(math.pi * np.outer(ix, jx) / (self.ni + 1))
        self.sy = math.sqrt(2.0 / (self.nj + 1)) * np.sin(math.pi * np.outer(iy, jy) / (self.nj + 1))

        lam_x = -4.0 * np.sin(0.5 * math.pi * jx / (self.ni + 1)) ** 2 / (problem.dx * problem.dx)
        lam_y = -4.0 * np.sin(0.5 * math.pi * jy / (self.nj + 1)) ** 2 / (problem.dy * problem.dy)
        self.lam = lam_x[:, None] + lam_y[None, :]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_hat = self.sx.T @ rhs @ self.sy
        sol_hat = rhs_hat / self.lam
        return self.sx @ sol_hat @ self.sy.T


def parse_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def cfg_float(cfg: dict[str, str], key: str, default: float = 0.0) -> float:
    return float(cfg.get(key, default))


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"CSV is empty: {path}")

    xs = np.unique(data["x"])
    ys = np.unique(data["y"])
    nx, ny = xs.size, ys.size

    fields: dict[str, np.ndarray] = {}
    for name in data.dtype.names:
        if name in ("x", "y"):
            continue
        # Keep the same orientation as Solver::psi_(i,j): shape is (nx, ny).
        fields[name] = np.asarray(data[name], dtype=float).reshape(nx, ny)

    return xs, ys, fields


def omega_forcing(problem: Problem) -> np.ndarray:
    x = problem.xs[1:-1, None]
    y = problem.ys[None, 1:-1]
    lx = problem.lx
    ly = problem.ly
    pi = math.pi

    a22 = -19.0 * ly / (2.0 * pi)
    a42 = -6.0 * ly / (2.0 * pi)
    a62 = 7.0 * ly / (2.0 * pi)
    a24 = 14.0 * ly / (4.0 * pi)
    a13 = a22 / 50.0
    a31 = a22 / 50.0

    def mode_dy(m: int, n: int, a_mn: float) -> np.ndarray:
        return (
            -a_mn
            * (pi * float(n) / ly)
            * np.sin(pi * float(m) * x / lx)
            * np.sin(pi * float(n) * y / ly)
        )

    forcing = np.zeros((problem.nx - 2, problem.ny - 2), dtype=float)
    forcing += mode_dy(2, 2, a22)
    forcing += mode_dy(4, 2, a42)
    forcing += mode_dy(6, 2, a62)
    forcing += mode_dy(2, 4, a24)
    forcing += mode_dy(1, 3, a13)
    forcing += mode_dy(3, 1, a31)
    return -forcing


def pack(psi: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.concatenate((psi[1:-1, 1:-1].ravel(), omega[1:-1, 1:-1].ravel()))


def unpack(z: np.ndarray, problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    ni = problem.nx - 2
    nj = problem.ny - 2
    n = ni * nj

    psi = np.zeros((problem.nx, problem.ny), dtype=float)
    omega = np.zeros((problem.nx, problem.ny), dtype=float)
    psi[1:-1, 1:-1] = z[:n].reshape(ni, nj)
    omega[1:-1, 1:-1] = z[n:].reshape(ni, nj)
    apply_thom_boundary(problem, psi, omega)
    return psi, omega


def unpack_tangent(dz: np.ndarray, problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    ni = problem.nx - 2
    nj = problem.ny - 2
    n = ni * nj

    dpsi = np.zeros((problem.nx, problem.ny), dtype=float)
    domega = np.zeros((problem.nx, problem.ny), dtype=float)
    dpsi[1:-1, 1:-1] = dz[:n].reshape(ni, nj)
    domega[1:-1, 1:-1] = dz[n:].reshape(ni, nj)
    apply_thom_boundary_tangent(problem, dpsi, domega)
    return dpsi, domega


def apply_thom_boundary(problem: Problem, psi: np.ndarray, omega: np.ndarray) -> None:
    dx = problem.dx
    dy = problem.dy
    bc = problem.bc

    omega[:, :] = omega
    omega[1:-1, 0] = -(2.0 * psi[1:-1, 1]) / (dy * dy) - 2.0 * bc.bottom.u / dy
    omega[1:-1, -1] = -(2.0 * psi[1:-1, -2]) / (dy * dy) - 2.0 * bc.top.u / dy
    omega[0, 1:-1] = -(2.0 * psi[1, 1:-1]) / (dx * dx) - 2.0 * bc.left.v / dx
    omega[-1, 1:-1] = -(2.0 * psi[-2, 1:-1]) / (dx * dx) - 2.0 * bc.right.v / dx
    omega[0, 0] = 0.0
    omega[0, -1] = 0.0
    omega[-1, 0] = 0.0
    omega[-1, -1] = 0.0


def apply_thom_boundary_tangent(problem: Problem, dpsi: np.ndarray, domega: np.ndarray) -> None:
    dx = problem.dx
    dy = problem.dy

    domega[1:-1, 0] = -(2.0 * dpsi[1:-1, 1]) / (dy * dy)
    domega[1:-1, -1] = -(2.0 * dpsi[1:-1, -2]) / (dy * dy)
    domega[0, 1:-1] = -(2.0 * dpsi[1, 1:-1]) / (dx * dx)
    domega[-1, 1:-1] = -(2.0 * dpsi[-2, 1:-1]) / (dx * dx)
    domega[0, 0] = 0.0
    domega[0, -1] = 0.0
    domega[-1, 0] = 0.0
    domega[-1, -1] = 0.0


def arakawa_jacobian(problem: Problem, psi: np.ndarray, omega: np.ndarray) -> np.ndarray:
    inv_4dxdy = 1.0 / (4.0 * problem.dx * problem.dy)

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


def laplacian_center(problem: Problem, field: np.ndarray) -> np.ndarray:
    dx2 = problem.dx * problem.dx
    dy2 = problem.dy * problem.dy
    return (
        (field[:-2, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[2:, 1:-1]) / dx2
        + (field[1:-1, :-2] - 2.0 * field[1:-1, 1:-1] + field[1:-1, 2:]) / dy2
    )


def velocities(problem: Problem, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    bc = problem.bc

    u[0, :] = bc.left.u
    v[0, :] = bc.left.v
    u[-1, :] = bc.right.u
    v[-1, :] = bc.right.v
    u[:, 0] = bc.bottom.u
    v[:, 0] = bc.bottom.v
    u[:, -1] = bc.top.u
    v[:, -1] = bc.top.v
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * problem.dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * problem.dx)
    return u, v


def make_residual(problem: Problem):
    forcing = omega_forcing(problem)

    def residual(z: np.ndarray) -> np.ndarray:
        psi, omega = unpack(z, problem)
        lap_psi = laplacian_center(problem, psi)
        lap_omega = laplacian_center(problem, omega)
        f_psi = lap_psi + omega[1:-1, 1:-1]
        f_omega = (1.0 / problem.re) * lap_omega - arakawa_jacobian(problem, psi, omega) - forcing
        return np.concatenate((f_psi.ravel(), f_omega.ravel()))

    return residual


def make_jacobian_matvec(problem: Problem, z: np.ndarray):
    psi, omega = unpack(z, problem)

    def matvec(dz: np.ndarray) -> np.ndarray:
        dpsi, domega = unpack_tangent(dz, problem)
        j_linearized = arakawa_jacobian(problem, dpsi, omega) + arakawa_jacobian(problem, psi, domega)
        df_psi = laplacian_center(problem, dpsi) + domega[1:-1, 1:-1]
        df_omega = (1.0 / problem.re) * laplacian_center(problem, domega) - j_linearized
        return np.concatenate((df_psi.ravel(), df_omega.ravel()))

    return matvec


def norm_info(vec: np.ndarray) -> tuple[float, float]:
    return float(np.linalg.norm(vec, ord=np.inf)), float(np.linalg.norm(vec) / math.sqrt(vec.size))


def make_stokes_preconditioner(problem: Problem):
    laplace_inverse = DirichletLaplaceInverse(problem)
    ni = problem.nx - 2
    nj = problem.ny - 2
    n = ni * nj

    def precondition(rhs: np.ndarray) -> np.ndarray:
        rhs_psi = rhs[:n].reshape(ni, nj)
        rhs_omega = rhs[n:].reshape(ni, nj)
        domega = problem.re * laplace_inverse.solve(rhs_omega)
        dpsi = laplace_inverse.solve(rhs_psi - domega)
        return np.concatenate((dpsi.ravel(), domega.ravel()))

    return precondition


def gmres_solve(
    matvec,
    b: np.ndarray,
    restart: int,
    max_iter: int,
    rel_tol: float,
    preconditioner=None,
) -> tuple[np.ndarray, int, float, float]:
    if preconditioner is None:
        linear_operator = matvec
        recover_solution = lambda y: y
    else:
        linear_operator = lambda y: matvec(preconditioner(y))
        recover_solution = preconditioner
    rhs = b

    x = np.zeros_like(b)
    b_norm = max(float(np.linalg.norm(rhs)), 1e-30)
    total = 0

    while total < max_iter:
        r = rhs - linear_operator(x)
        beta = float(np.linalg.norm(r))
        if beta / b_norm <= rel_tol:
            solution = recover_solution(x)
            true_rel = float(np.linalg.norm(b - matvec(solution)) / max(float(np.linalg.norm(b)), 1e-30))
            return solution, total, beta / b_norm, true_rel

        m = min(restart, max_iter - total)
        v = [r / beta]
        h = np.zeros((m + 1, m), dtype=float)
        krylov_rhs = np.zeros(m + 1, dtype=float)
        krylov_rhs[0] = beta

        best_x = x
        best_rel = beta / b_norm

        for j in range(m):
            w = linear_operator(v[j])
            for i in range(j + 1):
                h[i, j] = float(np.dot(v[i], w))
                w -= h[i, j] * v[i]
            h[j + 1, j] = float(np.linalg.norm(w))
            if h[j + 1, j] > 1e-30:
                v.append(w / h[j + 1, j])

            y, *_ = np.linalg.lstsq(h[: j + 2, : j + 1], krylov_rhs[: j + 2], rcond=None)
            candidate = x + np.column_stack(v[: j + 1]) @ y
            rel = float(np.linalg.norm(rhs - linear_operator(candidate)) / b_norm)
            best_x = candidate
            best_rel = rel
            total += 1
            if rel <= rel_tol or total >= max_iter:
                solution = recover_solution(best_x)
                true_rel = float(np.linalg.norm(b - matvec(solution)) / max(float(np.linalg.norm(b)), 1e-30))
                return solution, total, rel, true_rel

        x = best_x
        if best_rel <= rel_tol:
            solution = recover_solution(x)
            true_rel = float(np.linalg.norm(b - matvec(solution)) / max(float(np.linalg.norm(b)), 1e-30))
            return solution, total, best_rel, true_rel

    solution = recover_solution(x)
    true_rel = float(np.linalg.norm(b - matvec(solution)) / max(float(np.linalg.norm(b)), 1e-30))
    return solution, total, float(np.linalg.norm(rhs - linear_operator(x)) / b_norm), true_rel


def finite_difference_jv(residual, z: np.ndarray, fz: np.ndarray, fd_eps: float):
    z_norm = max(float(np.linalg.norm(z)), 1.0)

    def matvec(v: np.ndarray) -> np.ndarray:
        v_norm = max(float(np.linalg.norm(v)), 1e-30)
        h = fd_eps * z_norm / v_norm
        return (residual(z + h * v) - fz) / h

    return matvec


def validate_uniform_grid(problem: Problem) -> None:
    if problem.nx < 3 or problem.ny < 3:
        raise RuntimeError("Need at least 3 grid points in each direction")
    if not np.allclose(np.diff(problem.xs), problem.dx, rtol=1e-10, atol=1e-14):
        raise RuntimeError("CSV x grid is not uniform")
    if not np.allclose(np.diff(problem.ys), problem.dy, rtol=1e-10, atol=1e-14):
        raise RuntimeError("CSV y grid is not uniform")


def verify_jacobian_vector_product(
    residual,
    analytic_matvec,
    z: np.ndarray,
    fd_eps: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    dz = rng.standard_normal(z.size)
    dz /= max(float(np.linalg.norm(dz)), 1e-30)
    fz = residual(z)
    fd_matvec = finite_difference_jv(residual, z, fz, fd_eps)
    analytic = analytic_matvec(dz)
    finite_diff = fd_matvec(dz)
    err = analytic - finite_diff
    rel = float(np.linalg.norm(err) / max(float(np.linalg.norm(finite_diff)), 1e-30))
    print(
        "[newton] Jv check: "
        f"rel_l2_error={rel:.6e} "
        f"linf_error={np.linalg.norm(err, ord=np.inf):.6e}"
    )


def save_csv(path: Path, problem: Problem, psi: np.ndarray, omega: np.ndarray) -> None:
    u, v = velocities(problem, psi)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("x,y,psi,omega,u,v\n")
        for i, x in enumerate(problem.xs):
            for j, y in enumerate(problem.ys):
                f.write(
                    f"{x:.15g},{y:.15g},"
                    f"{psi[i, j]:.15g},{omega[i, j]:.15g},"
                    f"{u[i, j]:.15g},{v[i, j]:.15g}\n"
                )


def default_output(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_newton_equilibrium.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matrix-free Newton-Krylov solve of Alex's steady psi-omega equations from one CSV state."
    )
    parser.add_argument("csv", type=Path, help="Initial CSV state")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output equilibrium CSV")
    parser.add_argument("--config", type=Path, default=Path("alex/configs/config.cfg"), help="Alex config for Re and wall velocities")
    parser.add_argument("--re", type=float, default=None, help="Override Re from config")
    parser.add_argument("--max-newton", type=int, default=10)
    parser.add_argument("--newton-tol", type=float, default=1e-8, help="Stop when Linf residual is below this")
    parser.add_argument("--linear-tol", type=float, default=1e-3, help="Relative GMRES tolerance inside each Newton step")
    parser.add_argument("--gmres-restart", type=int, default=30)
    parser.add_argument("--gmres-max-iter", type=int, default=160)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--jacobian", choices=("exact", "finite-difference"), default="exact")
    parser.add_argument("--preconditioner", choices=("stokes", "none"), default="stokes")
    parser.add_argument("--verify-jv", action="store_true", help="Compare exact Jv with finite-difference Jv at the initial state")
    parser.add_argument("--verify-jv-seed", type=int, default=1)
    parser.add_argument("--line-search-steps", type=int, default=12)
    parser.add_argument("--check-only", action="store_true", help="Only print the initial steady residual")
    args = parser.parse_args()

    input_csv = args.csv.expanduser().resolve()
    output_csv = args.out.expanduser().resolve() if args.out is not None else default_output(input_csv)
    cfg = parse_config(args.config.expanduser().resolve())

    xs, ys, fields = load_csv(input_csv)
    if "psi" not in fields or "omega" not in fields:
        raise RuntimeError("Input CSV must contain psi and omega columns")

    bc = BoundaryConditions(
        left=WallVelocity(cfg_float(cfg, "bc.left.u"), cfg_float(cfg, "bc.left.v")),
        right=WallVelocity(cfg_float(cfg, "bc.right.u"), cfg_float(cfg, "bc.right.v")),
        bottom=WallVelocity(cfg_float(cfg, "bc.bottom.u"), cfg_float(cfg, "bc.bottom.v")),
        top=WallVelocity(cfg_float(cfg, "bc.top.u"), cfg_float(cfg, "bc.top.v")),
    )
    re_value = float(args.re if args.re is not None else cfg.get("Re", 100.0))
    problem = Problem(xs=xs, ys=ys, re=re_value, bc=bc)
    validate_uniform_grid(problem)

    z = pack(fields["psi"], fields["omega"])
    residual = make_residual(problem)
    fz = residual(z)
    linf, l2 = norm_info(fz)
    print(
        "[newton] initial: "
        f"Linf={linf:.6e} L2={l2:.6e} "
        f"unknowns={z.size} grid={problem.nx}x{problem.ny} Re={problem.re:g}"
    )

    if args.verify_jv:
        verify_jacobian_vector_product(
            residual,
            make_jacobian_matvec(problem, z),
            z,
            args.fd_eps,
            args.verify_jv_seed,
        )

    if args.check_only:
        return

    preconditioner = None
    if args.preconditioner == "stokes":
        preconditioner = make_stokes_preconditioner(problem)

    for it in range(args.max_newton):
        if linf <= args.newton_tol:
            print(f"[newton] converged before iteration {it}")
            break

        if args.jacobian == "exact":
            matvec = make_jacobian_matvec(problem, z)
        else:
            matvec = finite_difference_jv(residual, z, fz, args.fd_eps)

        dz, gmres_iters, gmres_rel, true_gmres_rel = gmres_solve(
            matvec,
            -fz,
            restart=args.gmres_restart,
            max_iter=args.gmres_max_iter,
            rel_tol=args.linear_tol,
            preconditioner=preconditioner,
        )
        print(
            f"[newton] iter={it} gmres_iters={gmres_iters} "
            f"gmres_rel={gmres_rel:.3e} true_rel={true_gmres_rel:.3e} "
            f"|dz|={np.linalg.norm(dz):.6e}"
        )

        accepted = False
        alpha = 1.0
        old_l2 = l2
        for _ in range(args.line_search_steps):
            trial_z = z + alpha * dz
            trial_f = residual(trial_z)
            trial_linf, trial_l2 = norm_info(trial_f)
            if trial_l2 < old_l2:
                z = trial_z
                fz = trial_f
                linf, l2 = trial_linf, trial_l2
                print(f"[newton] iter={it} alpha={alpha:.3e} Linf={linf:.6e} L2={l2:.6e}")
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            print("[newton] line search failed; saving the last accepted state")
            break

    psi, omega = unpack(z, problem)
    save_csv(output_csv, problem, psi, omega)
    print(f"[newton] saved: {output_csv}")


if __name__ == "__main__":
    main()
