import argparse
from pathlib import Path
import pickle

import numpy as np
from scipy import sparse as sp
from scipy.linalg import block_diag, eigvalsh, svdvals
from scipy.sparse.linalg import LinearOperator, eigs, splu

from simulation.projected_run import _oblique_projector
from simulation.steady import _validate_time_independent
from solver.config import SimConfig
from solver.lib import StokesMACLib, find_solver_lib
from solver.state_io import load_mac_state_pickle

root = Path(__file__).resolve().parent
cfg = SimConfig.from_yaml(root / 'pulsation.yaml')
_validate_time_independent(cfg)
output = root / cfg.output_dir
state, _ = load_mac_state_pickle(output / 'steady/state_internal.pkl', cfg)
eig_data = pickle.loads((output / 'linearized/eigenpairs.pkl').read_bytes())
if any(np.any(values) for values in cfg.make_bc_arrays().values()):
    raise ValueError("The calculation requires homogeneous boundary conditions")
nx, ny = cfg.nx, cfg.ny
dx, dy, dt = cfg.lx / nx, cfg.ly / ny, cfg.dt
nu = cfg.nu
n_u = (nx - 1) * ny
U = np.r_[state.u_vec, state.v_vec]
fu, fv = cfg.make_force_arrays(0)
F = np.r_[fu, np.zeros_like(state.v_vec) if fv is None else fv]


def face_diff(n, h):
    return sp.diags([np.ones(n - 1) / h, -np.ones(n - 1) / h], [0, -1], shape=(n, n - 1), format='csr')


def lap_1d(n, h, ghost=False):
    diag = np.full(n, 2.)
    if ghost:
        diag[[0, -1]] = 3
    return sp.diags([-np.ones(n - 1), diag, -np.ones(n - 1)], [-1, 0, 1], format='csr') / h**2


def centered(n, h, ghost=False):
    diag = np.zeros(n)
    if ghost:
        diag[[0, -1]] = [1, -1]
    return sp.diags([-np.ones(n - 1), diag, np.ones(n - 1)], [-1, 0, 1], format='csr') / (2 * h)


def edge_average(n):
    half = np.full(n - 1, 0.5)
    return sp.diags([half, half], [0, -1], shape=(n, n - 1), format='csr')


ix, iy, ixi, iyi = (sp.eye(n, format='csr') for n in (nx, ny, nx - 1, ny - 1))
curl = sp.vstack([sp.kron(face_diff(ny, dy), ixi), -sp.kron(iyi, face_diff(nx, dx))], format='csr')
identity = sp.eye(U.size, format='csr')
gram = (curl.T @ curl).tocsc()
Dxu = sp.kron(iy, centered(nx - 1, dx), format='csr')
Dyu = sp.kron(centered(ny, dy, True), ixi, format='csr')
Dxv = sp.kron(iyi, centered(nx, dx, True), format='csr')
Dyv = sp.kron(centered(ny - 1, dy), ix, format='csr')
Vu = sp.kron(edge_average(ny), edge_average(nx).T, format='csr')
Uv = Vu.T
K = sp.block_diag([
    sp.kronsum(lap_1d(nx - 1, dx), lap_1d(ny, dy, ghost=True)),
    sp.kronsum(lap_1d(nx, dx, ghost=True), lap_1d(ny - 1, dy)),
], format='csr')
mass_velocity = identity + (dt * nu) * K
mass = (curl.T @ mass_velocity @ curl).tocsc()


def advection(a, b):
    au, av = a[:n_u], a[n_u:]
    bu, bv = b[:n_u], b[n_u:]
    return np.concatenate((
        au * (Dxu @ bu) + (Vu @ av) * (Dyu @ bu),
        (Uv @ au) * (Dxv @ bv) + av * (Dyv @ bv),
    ))


def advection_jacobian(base):
    u, v = base[:n_u], base[n_u:]
    d = sp.diags
    return sp.bmat([
        [d(Dxu @ u) + d(u) @ Dxu + d(Vu @ v) @ Dyu, d(Dyu @ u) @ Vu],
        [d(Dxv @ v) @ Uv, d(Uv @ u) @ Dxv + d(Dyv @ v) + d(v) @ Dyv],
    ], format='csr')


def refine():
    gram_solve = splu(gram).solve
    base = curl @ gram_solve(curl.T @ U)
    for _ in range(3):
        residual = nu * (K @ base) + advection(base, base) - F
        jac = curl.T @ (nu * K + advection_jacobian(base)) @ curl
        base -= curl @ splu(jac.tocsc()).solve(curl.T @ residual)
    residual = nu * (K @ base) + advection(base, base) - F
    residual_sol = curl @ gram_solve(curl.T @ residual)
    print(f"stationary residual inf: {np.max(np.abs(residual_sol)):.3g}")
    print(f"change from saved velocity inf: {np.max(np.abs(base - U)):.3g}")
    return base


def slow_modes(mass, stationary_jacobian):
    solve = splu(stationary_jacobian).solve
    right_op = LinearOperator(mass.shape, matvec=lambda x: solve(mass @ x))
    left_op = LinearOperator(mass.shape, matvec=lambda x: solve(mass @ x, trans="T"))
    v0 = np.random.default_rng(5).standard_normal(mass.shape[0])
    inv_right, right = eigs(right_op, k=48, tol=1e-10, v0=v0, ncv=97)
    inv_left, left = eigs(left_op, k=48, tol=1e-10, v0=v0, ncv=97)
    values = 1 - dt / inv_right
    left_values = 1 - dt / inv_left
    right_columns, left_columns, blocks, unstable = [], [], [], []
    for index, value in enumerate(values):
        if value.imag < 0 and np.min(np.abs(values - value.conjugate())) < 1e-12:
            continue
        match = np.argmin(np.abs(left_values - value))
        if abs(left_values[match] - value) > 1e-8:
            raise ValueError(f"No matching left mode for {value}")
        vector = right[:, index]
        dual = mass @ left[:, match]
        size = np.sqrt(np.real(vector.conj() @ (mass @ vector)))
        right_columns.append(vector.real / size)
        left_columns.append(dual.real * size)
        unstable.append(abs(value) > 1)
        if value.imag:
            right_columns.append(vector.imag / size)
            left_columns.append(dual.imag * size)
            unstable.append(unstable[-1])
            blocks.append([[value.real, value.imag], [-value.imag, value.real]])
        else:
            blocks.append([[value.real]])
    V = np.column_stack(right_columns)
    left_modes = np.column_stack(left_columns)
    Z = np.linalg.solve(left_modes.T @ V, left_modes.T).T
    unstable = np.array(unstable)
    if np.linalg.norm(Z.T @ V - np.eye(V.shape[1]), np.inf) > 1e-10:
        raise ValueError("Biorthogonality check failed")
    print("multipliers near one:", sorted(values, key=lambda x: -abs(x))[:6])
    return V, Z, block_diag(*blocks), unstable


def check_fast_block(numerator, Z, q):
    bordered_base = sp.bmat([[q**2 * mass, numerator.T], [numerator, mass]], format="csc")
    factor = splu(bordered_base, diag_pivot_thresh=0, permc_spec="MMD_AT_PLUS_A",
                  options={"SymmetricMode": True, "Equil": False})
    if not np.array_equal(factor.perm_r, factor.perm_c):
        raise ValueError("The row and column permutations differ")
    pivots = factor.U.diagonal()
    diag_factor = sp.diags(pivots) @ factor.L.T
    asymmetry = sp.linalg.norm(factor.U - diag_factor, np.inf) / sp.linalg.norm(factor.U, np.inf)
    E = np.vstack((Z, np.zeros_like(Z)))
    solution = factor.solve(E)
    solve_error = np.linalg.norm(bordered_base @ solution - E) / np.linalg.norm(E)
    complement = -(E.T @ solution)
    complement_values = eigvalsh((complement + complement.T) / 2)
    signs = np.count_nonzero(pivots < 0), np.count_nonzero(complement_values < 0)
    print("negative counts and constraint rank:", *signs, Z.shape[1])
    print(f"minimum pivot modulus: {min(abs(pivots)):.3g}")
    print(f"minimum complement eigenvalue modulus: {min(abs(complement_values)):.3g}")
    print(f"relative LDU defect: {asymmetry:.3g}, solve defect: {solve_error:.3g}")
    if sum(signs) != Z.shape[1] or not np.all(pivots) or not np.all(complement_values):
        raise ValueError("The complement contraction check failed")
    if max(asymmetry, solve_error) > 1e-7:
        raise ValueError("The sparse factorization residual exceeds the tolerance")


def velocity_step_check(base, witness, nonlinear_velocity):
    with StokesMACLib(find_solver_lib(root), nx, ny, cfg.lx, cfg.ly, nu, dt) as solver:
        solver.set_bc_arrays(cfg.make_bc_arrays())

        def step(velocity):
            solver.set_state(velocity[:n_u], velocity[n_u:])
            solver.run_steps_with_force(0., 1, fu, fv)
            return np.concatenate(solver.get_state()[:2])

        fixed_step = step(base)
        curvature = (step(base + witness) + step(base - witness)) / 2 - fixed_step
    fixed_defect = np.max(np.abs(fixed_step - base))
    remainder_defect = np.linalg.norm(curvature - nonlinear_velocity)
    print(f"C++ fixed-point defect inf: {fixed_defect:.3g}")
    print(f"C++ quadratic remainder comparison 2: {remainder_defect:.3g}")
    if max(fixed_defect, remainder_defect) > 1e-11:
        raise ValueError("The matrix model disagrees with the C++ step")


def operator_norm(small_gram):
    return np.sqrt(max(0., eigvalsh((small_gram + small_gram.T) / 2)[-1]))


def check_initial_correction(base, direction, stable_size, quadratic_velocity, unstable_coordinates, long_run):
    scale = 3e-8 / stable_size
    initial = base + scale * direction
    correction = scale**2 * quadratic_velocity
    corrected = initial + correction
    print("small perturbation velocity norms (2, inf):", np.linalg.norm(scale * direction),
          np.max(np.abs(scale * direction)))
    print("small correction velocity norms (2, inf):", np.linalg.norm(correction), np.max(np.abs(correction)))
    print("velocity entries changed by the small correction:", np.count_nonzero(corrected != initial))

    with StokesMACLib(find_solver_lib(root), nx, ny, cfg.lx, cfg.ly, nu, dt) as solver:
        solver.set_bc_arrays(cfg.make_bc_arrays())
        outputs = []
        for field in (base, initial, corrected):
            solver.set_state(field[:n_u], field[n_u:])
            solver.run_steps_with_force(0., 1, fu, fv)
            outputs.append(np.concatenate(solver.get_state()[:2]))
    reference, uncorrected_step, corrected_step = outputs
    print("small perturbation signal and reference drift after one step (2):",
          np.linalg.norm(uncorrected_step - reference), np.linalg.norm(reference - base))
    print("small correction signal after one step (2):", np.linalg.norm(corrected_step - uncorrected_step))
    if not long_run:
        return

    amplitude = 0.05
    for name, field in (
        ("linear", base + amplitude * direction),
        ("quadratic", base + amplitude * direction + amplitude**2 * quadratic_velocity),
    ):
        with StokesMACLib(find_solver_lib(root), nx, ny, cfg.lx, cfg.ly, nu, dt) as solver:
            solver.set_bc_arrays(cfg.make_bc_arrays())
            solver.set_linear_solver("fast", cfg.linear_solver_fast_tol,
                                     cfg.linear_solver_fast_extrapolation, cfg.linear_solver_fast_parallel)
            solver.set_state(field[:n_u], field[n_u:])
            previous = 0
            for step in (5000, 15000, 30000, 40000):
                solver.run_steps_with_force(previous * dt, step - previous, fu, fv)
                deviation = np.concatenate(solver.get_state()[:2]) - base
                print(name, step, "velocity norm:", np.linalg.norm(deviation),
                      "unstable coordinate norm:", np.linalg.norm(unstable_coordinates.T @ deviation), flush=True)
                previous = step


def main(long_run=False):
    base = refine()
    stationary_jacobian = (curl.T @ (nu * K + advection_jacobian(base)) @ curl).tocsc()
    numerator = mass - dt * stationary_jacobian
    V, Z, blocks, unstable = slow_modes(mass, stationary_jacobian)
    stable = ~unstable
    if np.count_nonzero(unstable) != 2:
        raise ValueError("The reference calculation expects one unstable conjugate pair")
    q = 0.9998
    check_fast_block(numerator, Z, q)
    solve = splu(mass).solve
    Z_dual = solve(Z)

    def h_norm(x):
        return np.sqrt(x @ (mass @ x))

    def primal(matrix):
        return operator_norm(matrix.T @ (mass @ matrix))

    def dual(matrix):
        return operator_norm(matrix.T @ solve(matrix))

    def fast(x):
        return x - V @ (Z.T @ x)

    def star(x):
        c = Z.T @ x
        return max(np.linalg.norm(c[stable]), np.linalg.norm(c[unstable]), h_norm(fast(x)))

    def linear_step(x):
        return x - dt * solve(stationary_jacobian @ x)

    def quadratic(x):
        velocity = curl @ x
        return -dt * solve(curl.T @ advection(velocity, velocity))

    left_defect = -dt * (stationary_jacobian.T @ Z_dual) + Z @ (np.eye(len(blocks)) - blocks).T
    right_defect = linear_step(V) - V @ blocks
    stable_basis = V[:, stable]
    unstable_basis = V[:, unstable]
    projector_bound = 1 + primal(V) * dual(Z)
    c_in = max(dual(Z[:, stable]), dual(Z[:, unstable]), projector_bound)
    c_out = 1 + primal(stable_basis) + primal(unstable_basis)
    q_fast = q + primal(V) * dual(left_defect)
    a = max(svdvals(blocks[np.ix_(stable, stable)])[0], q_fast)
    inverse_unstable = np.linalg.inv(blocks[np.ix_(unstable, unstable)])
    b = svdvals(inverse_unstable)[0]
    delta = max(dual(left_defect) * c_out,
                projector_bound * (primal(right_defect[:, stable]) + primal(right_defect[:, unstable])))
    k_H = dt * (1 / dx + 1 / dy)
    k = k_H * c_in * c_out**2
    steady_residual = curl.T @ (nu * (K @ base) + advection(base, base) - F)
    center_update = splu(stationary_jacobian).solve(steady_residual)
    center_correction = star(center_update)
    print("linear and nonlinear estimates:", a, b, delta, k)
    print("norm equivalence:", c_in, c_out)
    print("center correction in block and H norms:", center_correction, h_norm(center_update))

    values = eig_data["eigenvalues"]
    modes = values.real > cfg.projected_real_threshold
    Qc, Wc = _oblique_projector(values[modes], eig_data["eigenvectors"][:U.size, modes],
                                eig_data["adjoint_vectors"][:, modes])
    gain = solve(curl.T @ Qc)
    observation = curl.T @ Wc
    gain_modes = Z.T @ gain
    obs_modes = observation.T @ V
    outputs = np.array([svdvals(gain_modes[stable])[0], svdvals(gain_modes[unstable])[0], primal(fast(gain))])
    inputs = np.array([svdvals(obs_modes[:, stable])[0], svdvals(obs_modes[:, unstable])[0],
                       dual(observation - Z @ obs_modes.T)])
    controller = np.diag([a, 0., q_fast]) + np.outer(outputs, inputs)
    controller[1, 1] = svdvals(blocks[np.ix_(unstable, unstable)] -
                              gain_modes[unstable] @ obs_modes[:, unstable])[0]
    weights = np.array([1., 0.01, 1.])
    gamma = np.max((controller @ weights + delta) / weights)
    print("feedback norm comparison rows:", (controller @ weights + delta) / weights)

    a_limit, b_limit, delta_limit, k_limit, theta = 0.999894, 0.999941, 1.2e-8, 112., 0.99995
    center_radius = 3e-13
    if not (a <= a_limit and b <= b_limit and k <= k_limit and
            delta + 2 * k_limit * center_radius <= delta_limit and
            center_correction <= 2.5e-13 and gamma + 2 * k_limit * center_radius / min(weights) <= 0.9999012):
        raise ValueError("The documented bounds do not hold for the computed matrices")
    m1 = max(1 / (1 - a_limit), b_limit / (1 - b_limit))
    m = max(m1, 1 / (theta - a_limit), b_limit / (1 - b_limit * theta))
    inverse_bound = m1 / (1 - m1 * delta)
    center_remainder = inverse_bound * k_limit * center_radius**2
    if (m * delta_limit > 1 / 8 or center_correction + center_remainder > center_radius
            or 2 * inverse_bound * k_limit * center_radius >= 1):
        raise ValueError("The fixed-point and stable manifold conditions failed")
    print(f"graph domain radius: {1 / (16 * k_limit * m):.3g}")
    print(f"repeated projection radius: {(1 - a_limit - delta_limit) / (2 * k_limit):.3g}")
    print(f"feedback radius: {(1 - 0.9999012) * min(weights) / (2 * k_limit):.3g}")

    x = np.arange(1, nx) * dx
    y = np.arange(1, ny) * dy
    trial = np.outer(np.sin(np.pi * y / cfg.ly), np.sin(np.pi * x / cfg.lx)).ravel()

    def project_stable(psi):
        return psi - unstable_basis @ (Z[:, unstable].T @ psi)

    trial = project_stable(trial)
    trial /= np.linalg.norm(curl @ trial)
    remainder = quadratic(trial)
    unstable_remainder = Z[:, unstable].T @ remainder
    unstable_norm = np.linalg.norm(unstable_remainder)
    defect_u = dual(left_defect[:, unstable])
    observation_u = dual(Z[:, unstable])
    center_h = h_norm(center_update) + c_out * center_remainder
    stable_to_unstable = (defect_u + observation_u * 2 * k_H * center_h) * c_out
    unstable_resolvent = b_limit / (1 - b_limit - b_limit * delta_limit)
    slope = unstable_resolvent * stable_to_unstable / (1 - m1 * delta_limit)
    witness_change = primal(unstable_basis) * slope * star(trial)
    quadratic_change = k_H * witness_change * (2 * h_norm(trial) + witness_change)
    exact_graph_defect = unstable_norm - observation_u * quadratic_change
    exact_graph_defect -= slope * c_in * (h_norm(remainder) + quadratic_change)
    if exact_graph_defect <= 0:
        raise ValueError("The quadratic witness is inconclusive after bounding projector errors")

    print("witness initial unstable norm:", np.linalg.norm(Z[:, unstable].T @ trial))
    print("witness coordinate and H norms:", star(trial), h_norm(trial))
    print("quadratic unstable coordinate norm:", unstable_norm)
    print("quadratic unstable velocity norm:", np.linalg.norm(curl @ (unstable_basis @ unstable_remainder)))
    print("quadratic H norm:", h_norm(remainder))
    print("quadratic defect from the exact stable space, lower estimate:", exact_graph_defect)
    velocity_step_check(base, curl @ trial, curl @ remainder)

    initial_direction = curl @ trial
    stable_size = star(trial)
    coefficient = np.zeros(np.count_nonzero(unstable))
    backward = inverse_unstable.copy()
    projected_dual = (curl @ Z_dual[:, unstable]).T
    for step in range(20001):
        velocity = curl @ trial
        coefficient += dt * (backward @ (projected_dual @ advection(velocity, velocity)))
        if step in (10000, 20000):
            print(f"quadratic graph at {step} steps:", coefficient, flush=True)
        trial = project_stable(linear_step(trial))
        backward = backward @ inverse_unstable
    unstable_coordinates = curl @ splu(gram).solve(Z[:, unstable])
    check_initial_correction(base, initial_direction, stable_size, curl @ (unstable_basis @ coefficient),
                             unstable_coordinates, long_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the local stable manifold for pulsation.yaml.")
    parser.add_argument("--long-run", action="store_true",
                        help="compare two 40,000-step trajectories using the macOS fast solver")
    main(parser.parse_args().long_run)
