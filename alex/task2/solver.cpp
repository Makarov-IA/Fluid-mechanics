#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

Task2Solver::Task2Solver(Task2Params params) : params_(std::move(params)) {}

int Task2Solver::idx(int i, int j, int Nx) {
    return j * Nx + i;
}

double Task2Solver::max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double mx = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
        mx = std::max(mx, std::abs(a[k] - b[k]));
    }
    return mx;
}

void Task2Solver::apply_stream_boundary(std::vector<double>& psi, int Nx, int Ny) const {
    for (int i = 0; i < Nx; ++i) {
        psi[idx(i, 0, Nx)] = 0.0;
        psi[idx(i, Ny - 1, Nx)] = 0.0;
    }
    for (int j = 0; j < Ny; ++j) {
        psi[idx(0, j, Nx)] = 0.0;
        psi[idx(Nx - 1, j, Nx)] = 0.0;
    }
}

void Task2Solver::apply_vorticity_boundary(const std::vector<double>& psi, std::vector<double>& omega,
                                           int Nx, int Ny, double dx, double dy) const {
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);

    for (int i = 1; i < Nx - 1; ++i) {
        omega[idx(i, 0, Nx)] = -2.0 * psi[idx(i, 1, Nx)] * inv_dy2;
        omega[idx(i, Ny - 1, Nx)] = -2.0 * psi[idx(i, Ny - 2, Nx)] * inv_dy2 - 2.0 * params_.lid_velocity / dy;
    }

    for (int j = 1; j < Ny - 1; ++j) {
        omega[idx(0, j, Nx)] = -2.0 * psi[idx(1, j, Nx)] * inv_dx2;
        omega[idx(Nx - 1, j, Nx)] = -2.0 * psi[idx(Nx - 2, j, Nx)] * inv_dx2;
    }

    omega[idx(0, 0, Nx)] = 0.5 * (omega[idx(1, 0, Nx)] + omega[idx(0, 1, Nx)]);
    omega[idx(Nx - 1, 0, Nx)] = 0.5 * (omega[idx(Nx - 2, 0, Nx)] + omega[idx(Nx - 1, 1, Nx)]);
    omega[idx(0, Ny - 1, Nx)] = 0.5 * (omega[idx(1, Ny - 1, Nx)] + omega[idx(0, Ny - 2, Nx)]);
    omega[idx(Nx - 1, Ny - 1, Nx)] =
        0.5 * (omega[idx(Nx - 2, Ny - 1, Nx)] + omega[idx(Nx - 1, Ny - 2, Nx)]);
}

double Task2Solver::solve_poisson(std::vector<double>& psi, const std::vector<double>& omega,
                                  int Nx, int Ny, double dx, double dy) const {
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double denom = 2.0 * (inv_dx2 + inv_dy2);

    double max_delta = 0.0;
    for (int it = 0; it < params_.poisson_max_iter; ++it) {
        max_delta = 0.0;
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                const int p = idx(i, j, Nx);
                const double old = psi[p];
                const double rhs = (psi[idx(i + 1, j, Nx)] + psi[idx(i - 1, j, Nx)]) * inv_dx2 +
                                   (psi[idx(i, j + 1, Nx)] + psi[idx(i, j - 1, Nx)]) * inv_dy2 + omega[p];
                psi[p] = rhs / denom;
                max_delta = std::max(max_delta, std::abs(psi[p] - old));
            }
        }
        apply_stream_boundary(psi, Nx, Ny);
        if (max_delta < params_.tol * 0.1) {
            break;
        }
    }

    return max_delta;
}

void Task2Solver::compute_velocity(const std::vector<double>& psi, std::vector<double>& u, std::vector<double>& v,
                                   int Nx, int Ny, double dx, double dy) const {
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            const int p = idx(i, j, Nx);
            u[p] = (psi[idx(i, j + 1, Nx)] - psi[idx(i, j - 1, Nx)]) / (2.0 * dy);
            v[p] = -(psi[idx(i + 1, j, Nx)] - psi[idx(i - 1, j, Nx)]) / (2.0 * dx);
        }
    }

    for (int i = 0; i < Nx; ++i) {
        u[idx(i, 0, Nx)] = 0.0;
        v[idx(i, 0, Nx)] = 0.0;
        u[idx(i, Ny - 1, Nx)] = params_.lid_velocity;
        v[idx(i, Ny - 1, Nx)] = 0.0;
    }
    for (int j = 0; j < Ny; ++j) {
        u[idx(0, j, Nx)] = 0.0;
        v[idx(0, j, Nx)] = 0.0;
        u[idx(Nx - 1, j, Nx)] = 0.0;
        v[idx(Nx - 1, j, Nx)] = 0.0;
    }
}

double Task2Solver::compute_stable_dt(double dx, double dy) const {
    const double denom = params_.nu * (2.0 / (dx * dx) + 2.0 / (dy * dy));
    const double dt_diff = params_.cfl_diff / (denom + 1e-12);
    return std::min(params_.dt_max, dt_diff);
}

void Task2Solver::fill_forcing(std::vector<double>& g, int Nx, int Ny, double t) const {
    const double phase = (params_.forcing_omega_t != 0.0) ? std::cos(params_.forcing_omega_t * t) : 1.0;
    static constexpr double kPi = 3.14159265358979323846;

    if (params_.forcing == "zero") {
        std::fill(g.begin(), g.end(), 0.0);
        return;
    }

    if (params_.forcing == "sin") {
        for (int j = 0; j < Ny; ++j) {
            const double y = static_cast<double>(j) / static_cast<double>(Ny - 1);
            for (int i = 0; i < Nx; ++i) {
                const double x = static_cast<double>(i) / static_cast<double>(Nx - 1);
                g[idx(i, j, Nx)] = params_.forcing_amp * std::sin(kPi * x) * std::sin(kPi * y) * phase;
            }
        }
        return;
    }

    std::fill(g.begin(), g.end(), 0.0);
}

Task2Result Task2Solver::solve() {
    Task2Result result;

    const int Nx = params_.Nx;
    const int Ny = params_.Ny;
    if (Nx < 3 || Ny < 3) {
        std::cerr << "Nx и Ny должны быть >= 3" << std::endl;
        return result;
    }
    if (params_.nu <= 0.0) {
        std::cerr << "nu должно быть > 0" << std::endl;
        return result;
    }

    const double dx = 1.0 / static_cast<double>(Nx - 1);
    const double dy = 1.0 / static_cast<double>(Ny - 1);
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);

    std::vector<double> psi(Nx * Ny, 0.0);
    std::vector<double> omega(Nx * Ny, 0.0);
    std::vector<double> omega_new = omega;
    std::vector<double> u(Nx * Ny, 0.0);
    std::vector<double> v(Nx * Ny, 0.0);
    std::vector<double> g(Nx * Ny, 0.0);

    apply_stream_boundary(psi, Nx, Ny);
    apply_vorticity_boundary(psi, omega, Nx, Ny, dx, dy);

    auto t0 = std::chrono::high_resolution_clock::now();
    double t = 0.0;
    double residual = 1.0;

    for (int iter = 1; iter <= params_.max_iter; ++iter) {
        fill_forcing(g, Nx, Ny, t);
        const double poisson_res = solve_poisson(psi, omega, Nx, Ny, dx, dy);
        const double dt = compute_stable_dt(dx, dy);

        omega_new = omega;
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                const int p = idx(i, j, Nx);
                const double lap_w = (omega[idx(i + 1, j, Nx)] - 2.0 * omega[p] + omega[idx(i - 1, j, Nx)]) * inv_dx2 +
                                     (omega[idx(i, j + 1, Nx)] - 2.0 * omega[p] + omega[idx(i, j - 1, Nx)]) * inv_dy2;
                omega_new[p] = omega[p] + dt * (params_.nu * lap_w + g[p]);
            }
        }

        apply_vorticity_boundary(psi, omega_new, Nx, Ny, dx, dy);
        const double delta_omega = max_abs_diff(omega_new, omega);
        residual = std::max(delta_omega, poisson_res);
        omega.swap(omega_new);
        t += dt;

        if (params_.verbose && (iter % params_.log_every == 0 || iter == 1)) {
            std::cout << "iter=" << iter << " residual=" << std::scientific << std::setprecision(3) << residual
                      << " dt=" << std::fixed << std::setprecision(6) << dt << " t=" << std::setprecision(5) << t
                      << std::endl;
        }

        if (residual < params_.tol) {
            result.converged = true;
            result.iterations = iter;
            break;
        }
        result.iterations = iter;
    }

    compute_velocity(psi, u, v, Nx, Ny, dx, dy);

    auto t1 = std::chrono::high_resolution_clock::now();

    result.final_residual = residual;
    result.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    result.final_time = t;
    result.Nx = Nx;
    result.Ny = Ny;
    result.dx = dx;
    result.dy = dy;
    result.psi = std::move(psi);
    result.omega = std::move(omega);
    result.u = std::move(u);
    result.v = std::move(v);
    result.g = std::move(g);

    return result;
}

bool Task2Solver::save_fields(const Task2Result& result) const {
    namespace fs = std::filesystem;

    fs::create_directories(params_.output_dir);

    auto write_field = [&](const std::string& name, const std::vector<double>& f) -> bool {
        const fs::path path = fs::path(params_.output_dir) / name;
        std::ofstream out(path);
        if (!out.is_open()) {
            std::cerr << "Не удалось открыть файл: " << path.string() << std::endl;
            return false;
        }

        out << std::setprecision(10);
        for (int j = 0; j < result.Ny; ++j) {
            for (int i = 0; i < result.Nx; ++i) {
                out << f[idx(i, j, result.Nx)];
                if (i + 1 < result.Nx) {
                    out << ' ';
                }
            }
            out << '\n';
        }

        return true;
    };

    bool ok = true;
    ok = ok && write_field("psi.txt", result.psi);
    ok = ok && write_field("omega.txt", result.omega);
    ok = ok && write_field("u.txt", result.u);
    ok = ok && write_field("v.txt", result.v);
    ok = ok && write_field("g.txt", result.g);

    const fs::path meta_path = fs::path(params_.output_dir) / "meta.txt";
    std::ofstream meta(meta_path);
    if (meta.is_open()) {
        meta << "Nx " << result.Nx << '\n';
        meta << "Ny " << result.Ny << '\n';
        meta << "dx " << result.dx << '\n';
        meta << "dy " << result.dy << '\n';
        meta << "nu " << params_.nu << '\n';
        meta << "lid_velocity " << params_.lid_velocity << '\n';
        meta << "forcing " << params_.forcing << '\n';
        meta << "forcing_amp " << params_.forcing_amp << '\n';
        meta << "forcing_omega_t " << params_.forcing_omega_t << '\n';
        meta << "iterations " << result.iterations << '\n';
        meta << "converged " << (result.converged ? 1 : 0) << '\n';
        meta << "residual " << std::setprecision(10) << result.final_residual << '\n';
        meta << "elapsed_seconds " << result.elapsed_seconds << '\n';
        meta << "final_time " << result.final_time << '\n';
    }

    return ok;
}
