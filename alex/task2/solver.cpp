#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

Task2Solver::Task2Solver(Task2Params params) : params_(std::move(params)) {}

int Task2Solver::idx(int i, int j, int Nx) {
    return j * Nx + i;
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

double Task2Solver::top_velocity(double x, double t) const {
    static constexpr double kPi = 3.14159265358979323846;
    if (params_.forcing == "sin") {
        return params_.lid_velocity * (1.0 + 0.2 * std::sin(kPi * x) * std::cos(params_.forcing_omega_t * t));
    }
    return params_.lid_velocity;
}

void Task2Solver::apply_vorticity_boundary(const std::vector<double>& psi, std::vector<double>& omega,
                                           int Nx, int Ny, double dx, double dy, double t) const {
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);

    for (int i = 1; i < Nx - 1; ++i) {
        const double x = static_cast<double>(i) / static_cast<double>(Nx - 1);
        const double u_top = top_velocity(x, t);

        omega[idx(i, 0, Nx)] = 2.0 * psi[idx(i, 1, Nx)] * inv_dy2;
        omega[idx(i, Ny - 1, Nx)] = 2.0 * psi[idx(i, Ny - 2, Nx)] * inv_dy2 + 2.0 * u_top / dy;
    }

    for (int j = 1; j < Ny - 1; ++j) {
        omega[idx(0, j, Nx)] = 2.0 * psi[idx(1, j, Nx)] * inv_dx2;
        omega[idx(Nx - 1, j, Nx)] = 2.0 * psi[idx(Nx - 2, j, Nx)] * inv_dx2;
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
                                   (psi[idx(i, j + 1, Nx)] + psi[idx(i, j - 1, Nx)]) * inv_dy2 - omega[p];
                psi[p] = rhs / denom;
                max_delta = std::max(max_delta, std::abs(psi[p] - old));
            }
        }

        apply_stream_boundary(psi, Nx, Ny);
        if (max_delta < params_.poisson_tol) {
            break;
        }
    }

    return max_delta;
}

double Task2Solver::solve_omega_implicit(std::vector<double>& omega_new, const std::vector<double>& omega_old,
                                         const std::vector<double>& g, int Nx, int Ny,
                                         double dx, double dy, double dt) const {
    const double rx = params_.nu * dt / (dx * dx);
    const double ry = params_.nu * dt / (dy * dy);
    const double denom = 1.0 + 2.0 * rx + 2.0 * ry;

    double max_delta = 0.0;
    for (int it = 0; it < params_.omega_max_iter; ++it) {
        max_delta = 0.0;

        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                const int p = idx(i, j, Nx);
                const double old = omega_new[p];
                const double rhs = omega_old[p] + dt * g[p] +
                                   rx * (omega_new[idx(i + 1, j, Nx)] + omega_new[idx(i - 1, j, Nx)]) +
                                   ry * (omega_new[idx(i, j + 1, Nx)] + omega_new[idx(i, j - 1, Nx)]);
                omega_new[p] = rhs / denom;
                max_delta = std::max(max_delta, std::abs(omega_new[p] - old));
            }
        }

        if (max_delta < params_.omega_tol) {
            break;
        }
    }

    return max_delta;
}

void Task2Solver::fill_forcing(std::vector<double>& g, int Nx, int Ny, double t) const {
    static constexpr double kPi = 3.14159265358979323846;

    if (params_.forcing == "zero") {
        std::fill(g.begin(), g.end(), 0.0);
        return;
    }

    if (params_.forcing == "sin") {
        const double phase = std::cos(params_.forcing_omega_t * t);
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
    const int Nt = params_.Nt;

    if (Nx < 3 || Ny < 3 || Nt < 1) {
        std::cerr << "Nx, Ny must be >= 3 and Nt must be >= 1" << std::endl;
        return result;
    }
    if (params_.nu <= 0.0) {
        std::cerr << "nu must be > 0" << std::endl;
        return result;
    }

    const double T = 1.0;
    const double dt = T / static_cast<double>(Nt);
    const double dx = 1.0 / static_cast<double>(Nx - 1);
    const double dy = 1.0 / static_cast<double>(Ny - 1);

    std::vector<double> psi(Nx * Ny, 0.0);
    std::vector<double> omega(Nx * Ny, 0.0);
    std::vector<double> omega_new(Nx * Ny, 0.0);
    std::vector<double> g(Nx * Ny, 0.0);

    auto save_snapshot = [&](int step, double t_now) {
        result.snapshot_steps.push_back(step);
        result.snapshot_times.push_back(t_now);
        result.psi_snapshots.push_back(psi);
        result.omega_snapshots.push_back(omega);
    };

    apply_stream_boundary(psi, Nx, Ny);
    apply_vorticity_boundary(psi, omega, Nx, Ny, dx, dy, 0.0);
    solve_poisson(psi, omega, Nx, Ny, dx, dy);
    apply_vorticity_boundary(psi, omega, Nx, Ny, dx, dy, 0.0);

    save_snapshot(0, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    double last_omega_res = 0.0;
    double last_poisson_res = 0.0;

    for (int n = 0; n < Nt; ++n) {
        const int step = n + 1;
        const double t_new = static_cast<double>(step) * dt;

        fill_forcing(g, Nx, Ny, t_new);
        omega_new = omega;

        apply_vorticity_boundary(psi, omega_new, Nx, Ny, dx, dy, t_new);
        const double omega_res_1 = solve_omega_implicit(omega_new, omega, g, Nx, Ny, dx, dy, dt);

        const double poisson_res_1 = solve_poisson(psi, omega_new, Nx, Ny, dx, dy);
        apply_vorticity_boundary(psi, omega_new, Nx, Ny, dx, dy, t_new);

        const double omega_res_2 = solve_omega_implicit(omega_new, omega, g, Nx, Ny, dx, dy, dt);
        const double poisson_res_2 = solve_poisson(psi, omega_new, Nx, Ny, dx, dy);
        apply_vorticity_boundary(psi, omega_new, Nx, Ny, dx, dy, t_new);

        omega.swap(omega_new);
        t = t_new;

        last_omega_res = std::max(omega_res_1, omega_res_2);
        last_poisson_res = std::max(poisson_res_1, poisson_res_2);

        if (params_.verbose && (step % params_.log_every == 0 || step == 1 || step == Nt)) {
            std::cout << "step=" << step << "/" << Nt
                      << " t=" << std::fixed << std::setprecision(5) << t
                      << " omega_res=" << std::scientific << std::setprecision(3) << last_omega_res
                      << " psi_res=" << std::scientific << std::setprecision(3) << last_poisson_res
                      << std::endl;
        }

        if (params_.save_every > 0 && (step % params_.save_every == 0 || step == Nt)) {
            save_snapshot(step, t);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    result.success = true;
    result.steps_completed = Nt;
    result.final_time = t;
    result.dt = dt;
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    result.final_omega_residual = last_omega_res;
    result.final_poisson_residual = last_poisson_res;
    result.Nx = Nx;
    result.Ny = Ny;
    result.dx = dx;
    result.dy = dy;
    result.psi = std::move(psi);
    result.omega = std::move(omega);

    return result;
}

bool Task2Solver::save_fields(const Task2Result& result) const {
    namespace fs = std::filesystem;
    fs::create_directories(params_.output_dir);
    fs::create_directories(fs::path(params_.output_dir) / "snapshots");

    auto write_field = [&](const fs::path& path, const std::vector<double>& f) -> bool {
        std::ofstream out(path);
        if (!out.is_open()) {
            std::cerr << "Failed to open: " << path.string() << std::endl;
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
    ok = ok && write_field(fs::path(params_.output_dir) / "psi.txt", result.psi);
    ok = ok && write_field(fs::path(params_.output_dir) / "omega.txt", result.omega);

    const fs::path index_path = fs::path(params_.output_dir) / "snapshots" / "index.csv";
    std::ofstream index_out(index_path);
    if (!index_out.is_open()) {
        std::cerr << "Failed to open: " << index_path.string() << std::endl;
        return false;
    }

    index_out << "index,step,time,psi_file,omega_file\n";
    for (size_t k = 0; k < result.snapshot_steps.size(); ++k) {
        const int step = result.snapshot_steps[k];
        const double t = result.snapshot_times[k];

        std::ostringstream sstep;
        sstep << std::setw(5) << std::setfill('0') << step;

        const fs::path psi_rel = fs::path("snapshots") / ("psi_step" + sstep.str() + ".txt");
        const fs::path omega_rel = fs::path("snapshots") / ("omega_step" + sstep.str() + ".txt");

        ok = ok && write_field(fs::path(params_.output_dir) / psi_rel, result.psi_snapshots[k]);
        ok = ok && write_field(fs::path(params_.output_dir) / omega_rel, result.omega_snapshots[k]);

        index_out << k << ',' << step << ',' << std::setprecision(12) << t << ','
                  << psi_rel.generic_string() << ',' << omega_rel.generic_string() << '\n';
    }

    const fs::path meta_path = fs::path(params_.output_dir) / "meta.txt";
    std::ofstream meta(meta_path);
    if (meta.is_open()) {
        meta << "scheme backward_euler_implicit_fd\n";
        meta << "time_interval_start 0\n";
        meta << "time_interval_end 1\n";
        meta << "Nx " << result.Nx << '\n';
        meta << "Ny " << result.Ny << '\n';
        meta << "Nt " << result.steps_completed << '\n';
        meta << "dx " << result.dx << '\n';
        meta << "dy " << result.dy << '\n';
        meta << "dt " << result.dt << '\n';
        meta << "nu " << params_.nu << '\n';
        meta << "lid_velocity " << params_.lid_velocity << '\n';
        meta << "forcing " << params_.forcing << '\n';
        meta << "forcing_amp " << params_.forcing_amp << '\n';
        meta << "forcing_omega_t " << params_.forcing_omega_t << '\n';
        meta << "omega_max_iter " << params_.omega_max_iter << '\n';
        meta << "omega_tol " << params_.omega_tol << '\n';
        meta << "poisson_max_iter " << params_.poisson_max_iter << '\n';
        meta << "poisson_tol " << params_.poisson_tol << '\n';
        meta << "save_every " << params_.save_every << '\n';
        meta << "snapshot_count " << result.snapshot_steps.size() << '\n';
        meta << "final_omega_residual " << std::setprecision(12) << result.final_omega_residual << '\n';
        meta << "final_poisson_residual " << std::setprecision(12) << result.final_poisson_residual << '\n';
        meta << "elapsed_seconds " << result.elapsed_seconds << '\n';
        meta << "final_time " << result.final_time << '\n';
    }

    return ok;
}
