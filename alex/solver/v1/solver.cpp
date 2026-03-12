#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <utility>
#include <vector>

Solver::Solver(const Config& cfg) : cfg_(cfg)
{
    ValidateConfig();

    nx_ = cfg_.nx;
    ny_ = cfg_.ny;
    n_  = nx_ * ny_;
    dx_ = 1.0 / (nx_ - 1);
    dy_ = 1.0 / (ny_ - 1);
    dt_ = cfg_.t_max / cfg_.n_time_steps;
    time_ = 0.0;
    step_ = 0;

    g_ = BuildGFunction(cfg_);

    psi_       = Vector::Zero(n_);
    omega_     = Vector::Zero(n_);
    u_         = Vector::Zero(n_);
    v_         = Vector::Zero(n_);
    rhs_psi_   = Vector::Zero(n_);
    rhs_omega_ = Vector::Zero(n_);

    poisson_matrix_.resize(n_, n_);
    omega_matrix_.resize(n_, n_);

    // Инициализация граничных условий скоростей
    for (int i = 0; i < nx_; ++i)
    {
        u_[Idx(i, 0)]      = cfg_.u[2]; // bottom
        v_[Idx(i, 0)]      = cfg_.v[2]; 
        u_[Idx(i, ny_ - 1)] = cfg_.u[0]; // top
        v_[Idx(i, ny_ - 1)] = cfg_.v[0];
    }
    for (int j = 0; j < ny_; ++j)
    {
        u_[Idx(0, j)]      = cfg_.u[3]; // left
        v_[Idx(0, j)]      = cfg_.v[3];
        u_[Idx(nx_ - 1, j)] = cfg_.u[1]; // right
        v_[Idx(nx_ - 1, j)] = cfg_.v[1];
    }

    // Построение матриц
    std::vector<Eigen::Triplet<double>> p_triplets;
    std::vector<Eigen::Triplet<double>> o_triplets;
    p_triplets.reserve(5 * n_);
    o_triplets.reserve(5 * n_);

    double dx2 = dx_ * dx_;
    double dy2 = dy_ * dy_;

    const double omega_ax = dt_ * cfg_.nu / dx2;
    const double omega_ay = dt_ * cfg_.nu / dy2;

    for (int i = 0; i < nx_; ++i)
    {
        for (int j = 0; j < ny_; ++j)
        {
            int idx = Idx(i, j);

            // Границы
            if (i == 0 || i == nx_ - 1 || j == 0 || j == ny_ - 1)
            {
                p_triplets.push_back({idx, idx, 1.0});
                o_triplets.push_back({idx, idx, 1.0});
            }
            // Внутренние узлы
            else
            {
                // Матрица Пуассона (Laplace(psi) = -omega)
                p_triplets.push_back({idx, idx, -2.0 / dx2 - 2.0 / dy2});
                // Для Dirichlet-границ не связываем внутренние строки с граничными
                // неизвестными: вклад границы переносится в RHS.
                if (i > 1)
                {
                    p_triplets.push_back({idx, Idx(i - 1, j), 1.0 / dx2});
                }
                if (i < nx_ - 2)
                {
                    p_triplets.push_back({idx, Idx(i + 1, j), 1.0 / dx2});
                }
                if (j > 1)
                {
                    p_triplets.push_back({idx, Idx(i, j - 1), 1.0 / dy2});
                }
                if (j < ny_ - 2)
                {
                    p_triplets.push_back({idx, Idx(i, j + 1), 1.0 / dy2});
                }

                // Матрица завихренности (Неявная диффузионная часть)
                // (I - dt * nu * Laplace) omega = ...
                o_triplets.push_back({idx, idx, 1.0 + 2.0 * omega_ax + 2.0 * omega_ay});
                if (i > 1)
                {
                    o_triplets.push_back({idx, Idx(i - 1, j), -omega_ax});
                }
                if (i < nx_ - 2)
                {
                    o_triplets.push_back({idx, Idx(i + 1, j), -omega_ax});
                }
                if (j > 1)
                {
                    o_triplets.push_back({idx, Idx(i, j - 1), -omega_ay});
                }
                if (j < ny_ - 2)
                {
                    o_triplets.push_back({idx, Idx(i, j + 1), -omega_ay});
                }
            }
        }
    }

    poisson_matrix_.setFromTriplets(p_triplets.begin(), p_triplets.end());
    omega_matrix_.setFromTriplets(o_triplets.begin(), o_triplets.end());

    poisson_ldlt_.compute(poisson_matrix_);
    omega_ldlt_.compute(omega_matrix_);

    if (poisson_ldlt_.info() != Eigen::Success)
    {
        throw std::runtime_error("Poisson factorization failed.");
    }
    if (omega_ldlt_.info() != Eigen::Success)
    {
        throw std::runtime_error("Omega factorization failed.");
    }
}

Solver::Solver(const std::string& config_path) : Solver(LoadConfigFromFile(config_path))
{
}

void Solver::solve()
{
    const double dx2 = dx_ * dx_;
    const double dy2 = dy_ * dy_;
    const double omega_ax = dt_ * cfg_.nu / dx2;
    const double omega_ay = dt_ * cfg_.nu / dy2;
    std::vector<double> psi_residual_history;
    std::vector<double> omega_residual_history;
    std::vector<double> max_residual_history;
    std::vector<double> time_history;
    std::vector<int> step_history;

    psi_residual_history.reserve(static_cast<std::size_t>(cfg_.n_time_steps));
    omega_residual_history.reserve(static_cast<std::size_t>(cfg_.n_time_steps));
    max_residual_history.reserve(static_cast<std::size_t>(cfg_.n_time_steps));
    time_history.reserve(static_cast<std::size_t>(cfg_.n_time_steps));
    step_history.reserve(static_cast<std::size_t>(cfg_.n_time_steps));

    auto apply_omega_boundary_from_psi = [&]()
    {
        for (int i = 0; i < nx_; ++i)
        {
            // omega = Delta(psi)
            omega_[Idx(i, 0)]       =  2.0 * psi_[Idx(i, 1)] / dy2 - 2.0 * cfg_.u[2] / dy_;           // bottom
            omega_[Idx(i, ny_ - 1)] =  2.0 * psi_[Idx(i, ny_ - 2)] / dy2 + 2.0 * cfg_.u[0] / dy_;     // top
        }
        for (int j = 1; j < ny_ - 1; ++j)
        {
            omega_[Idx(0, j)]       =  2.0 * psi_[Idx(1, j)] / dx2 + 2.0 * cfg_.v[3] / dx_;           // left
            omega_[Idx(nx_ - 1, j)] =  2.0 * psi_[Idx(nx_ - 2, j)] / dx2 - 2.0 * cfg_.v[1] / dx_;     // right
        }
    };

    auto update_velocity_from_psi = [&]()
    {
        for (int i = 1; i < nx_ - 1; ++i)
        {
            for (int j = 1; j < ny_ - 1; ++j)
            {
                u_[Idx(i, j)] =  (psi_[Idx(i, j + 1)] - psi_[Idx(i, j - 1)]) / (2.0 * dy_);
                v_[Idx(i, j)] = -(psi_[Idx(i + 1, j)] - psi_[Idx(i - 1, j)]) / (2.0 * dx_);
            }
        }

        for (int i = 0; i < nx_; ++i)
        {
            u_[Idx(i, 0)]       = cfg_.u[2];
            v_[Idx(i, 0)]       = cfg_.v[2];
            u_[Idx(i, ny_ - 1)] = cfg_.u[0];
            v_[Idx(i, ny_ - 1)] = cfg_.v[0];
        }
        for (int j = 0; j < ny_; ++j)
        {
            u_[Idx(0, j)]       = cfg_.u[3];
            v_[Idx(0, j)]       = cfg_.v[3];
            u_[Idx(nx_ - 1, j)] = cfg_.u[1];
            v_[Idx(nx_ - 1, j)] = cfg_.v[1];
        }
    };

    auto compute_jacobian = [&](int i, int j) -> double
    {
        const double psi_x = (psi_[Idx(i + 1, j)] - psi_[Idx(i - 1, j)]) / (2.0 * dx_);
        const double psi_y = (psi_[Idx(i, j + 1)] - psi_[Idx(i, j - 1)]) / (2.0 * dy_);
        const double omega_x = (omega_[Idx(i + 1, j)] - omega_[Idx(i - 1, j)]) / (2.0 * dx_);
        const double omega_y = (omega_[Idx(i, j + 1)] - omega_[Idx(i, j - 1)]) / (2.0 * dy_);

        return psi_y * omega_x - psi_x * omega_y;
    };

    auto compute_stationary_residuals = [&]() -> std::pair<double, double>
    {
        double psi_residual = 0.0;
        double omega_residual = 0.0;

        for (int i = 1; i < nx_ - 1; ++i)
        {
            for (int j = 1; j < ny_ - 1; ++j)
            {
                const int idx = Idx(i, j);
                const double laplace_psi =
                    (psi_[Idx(i + 1, j)] - 2.0 * psi_[idx] + psi_[Idx(i - 1, j)]) / dx2 +
                    (psi_[Idx(i, j + 1)] - 2.0 * psi_[idx] + psi_[Idx(i, j - 1)]) / dy2;
                const double laplace_omega =
                    (omega_[Idx(i + 1, j)] - 2.0 * omega_[idx] + omega_[Idx(i - 1, j)]) / dx2 +
                    (omega_[Idx(i, j + 1)] - 2.0 * omega_[idx] + omega_[Idx(i, j - 1)]) / dy2;

                double forcing = 0.0;
                if (g_)
                {
                    forcing = g_(i * dx_, j * dy_, time_);
                }

                psi_residual = std::max(psi_residual, std::abs(laplace_psi - omega_[idx]));
                omega_residual = std::max(
                    omega_residual,
                    std::abs(cfg_.nu * laplace_omega + forcing - compute_jacobian(i, j)));
            }
        }

        return {psi_residual, omega_residual};
    };

    auto write_residual_history = [&]()
    {
        const std::string filename = cfg_.save_dir + "/residual_history.csv";
        std::ofstream out(filename);
        if (!out.is_open())
        {
            std::cerr << "[Solver] Error: Cannot open file " << filename << " for writing.\n";
            return;
        }

        out << "step,time,psi_residual,omega_residual,max_residual\n";
        out << std::scientific << std::setprecision(16);

        for (std::size_t k = 0; k < step_history.size(); ++k)
        {
            out << step_history[k] << ","
                << time_history[k] << ","
                << psi_residual_history[k] << ","
                << omega_residual_history[k] << ","
                << max_residual_history[k] << "\n";
        }
    };

    // Инициализируем граничную завихренность из текущей psi (начальный момент).
    apply_omega_boundary_from_psi();
    update_velocity_from_psi();

    bool converged = false;

    for (int iter = 1; iter <= cfg_.n_time_steps; ++iter)
    {
        step_ = iter;
        time_ += dt_;

        // 1. Граничные значения omega считаем известными (Dirichlet) из предыдущего шага.
        for (int i = 0; i < nx_; ++i)
        {
            for (int j = 0; j < ny_; ++j)
            {
                if (i == 0 || i == nx_ - 1 || j == 0 || j == ny_ - 1)
                {
                    rhs_omega_[Idx(i, j)] = omega_[Idx(i, j)];
                }
            }
        }

        // 2. Внутренние узлы RHS: неявная диффузия + явная конвекция + источник g.
        for (int i = 1; i < nx_ - 1; ++i)
        {
            for (int j = 1; j < ny_ - 1; ++j)
            {
                const int idx = Idx(i, j);

                double forcing = 0.0;
                if (g_)
                {
                    forcing = g_(i * dx_, j * dy_, time_);
                }
                const double jacobian = compute_jacobian(i, j);

                double boundary_term = 0.0;
                if (i == 1)
                {
                    boundary_term += omega_ax * rhs_omega_[Idx(0, j)];
                }
                if (i == nx_ - 2)
                {
                    boundary_term += omega_ax * rhs_omega_[Idx(nx_ - 1, j)];
                }
                if (j == 1)
                {
                    boundary_term += omega_ay * rhs_omega_[Idx(i, 0)];
                }
                if (j == ny_ - 2)
                {
                    boundary_term += omega_ay * rhs_omega_[Idx(i, ny_ - 1)];
                }

                rhs_omega_[idx] = omega_[idx] + dt_ * (forcing - jacobian) + boundary_term;
            }
        }

        // 3. Решение уравнения для завихренности
        omega_ = omega_ldlt_.solve(rhs_omega_);
        if (omega_ldlt_.info() != Eigen::Success)
        {
            throw std::runtime_error("Omega linear solve failed.");
        }

        // 4. Формирование правой части для функции тока
        for (int i = 0; i < nx_; ++i)
        {
            for (int j = 0; j < ny_; ++j)
            {
                int idx = Idx(i, j);
                if (i == 0 || i == nx_ - 1 || j == 0 || j == ny_ - 1)
                {
                    rhs_psi_[idx] = 0.0; // Граничные условия Дирихле (стенки не проницаемы)
                }
                else
                {
                    rhs_psi_[idx] = omega_[idx];
                }
            }
        }

        // 5. Решение уравнения Пуассона для функции тока
        psi_ = poisson_ldlt_.solve(rhs_psi_);
        if (poisson_ldlt_.info() != Eigen::Success)
        {
            throw std::runtime_error("Poisson linear solve failed.");
        }

        // 6. Обновление граничной omega по новой psi (формула Тома).
        apply_omega_boundary_from_psi();

        // 7. Обновление полей скорости во внутренних узлах
        update_velocity_from_psi();

        const auto [psi_residual, omega_residual] = compute_stationary_residuals();
        const double max_residual = std::max(psi_residual, omega_residual);

        step_history.push_back(step_);
        time_history.push_back(time_);
        psi_residual_history.push_back(psi_residual);
        omega_residual_history.push_back(omega_residual);
        max_residual_history.push_back(max_residual);
        converged = max_residual < cfg_.steady_tolerance;

        // Логирование
        if (cfg_.log_info.enabled &&
            (step_ % cfg_.log_info.print_every_step == 0 || step_ == 1 || converged))
        {
            std::cout << "[Solver] Step: " << step_ << " / " << cfg_.n_time_steps 
                      << " | Time: " << time_
                      << " | max residual: " << std::scientific << max_residual
                      << std::defaultfloat << "\n";
        }

        if (step_ % cfg_.save_every_step == 0 || converged || step_ == cfg_.n_time_steps)
        {
            save(cfg_.save_dir);
        }

        if (converged)
        {
            break;
        }
    }

    write_residual_history();

    if (cfg_.log_info.enabled && !max_residual_history.empty())
    {
        std::cout << "[Solver] Finished at step " << step_
                  << " | time " << time_
                  << " | max residual " << std::scientific << max_residual_history.back()
                  << std::defaultfloat << "\n";
    }
}

const Config& Solver::config() const noexcept
{
    return cfg_;
}

double Solver::time() const noexcept
{
    return time_;
}

int Solver::step() const noexcept
{
    return step_;
}

const Solver::Vector& Solver::psi() const noexcept
{
    return psi_;
}

const Solver::Vector& Solver::omega() const noexcept
{
    return omega_;
}

const Solver::Vector& Solver::u() const noexcept
{
    return u_;
}

const Solver::Vector& Solver::v() const noexcept
{
    return v_;
}

void Solver::save(const std::string& directory) const
{
    std::filesystem::create_directories(directory);

    std::string filename = directory + "/result_" + std::to_string(step_) + ".csv";
    std::ofstream out(filename);
    
    if (!out.is_open())
    {
        std::cerr << "[Solver] Error: Cannot open file " << filename << " for writing.\n";
        return;
    }

    // Записываем заголовок CSV
    out << "x,y,psi,omega,u,v\n";

    for (int i = 0; i < nx_; ++i)
    {
        for (int j = 0; j < ny_; ++j)
        {
            int idx = Idx(i, j);
            double x = i * dx_;
            double y = j * dy_;
            
            out << std::fixed << std::setprecision(6)
                << x << ","
                << y << ","
                << psi_[idx] << ","
                << omega_[idx] << ","
                << u_[idx] << ","
                << v_[idx] << "\n";
        }
    }
    
    out.close();
    if (cfg_.log_info.enabled)
    {
        std::cout << "[Solver] Results saved to: " << filename << "\n";
    }
}

int Solver::Idx(int i, int j) const
{
    return i * ny_ + j;
}

void Solver::ValidateConfig() const
{
    // Большая часть валидации уже выполнена в LoadConfigFromFile,
    // но здесь можно добавить проверки специфичные для солвера, если необходимо.
    if (cfg_.nx <= 2 || cfg_.ny <= 2)
    {
        throw std::runtime_error("Grid size too small for reasonable discretization.");
    }

    // В текущей постановке psi задана константой на всех стенках (непроницаемые стенки).
    // Это требует нулевой нормальной скорости на границе.
    const double eps = 1e-12;
    const bool has_nonzero_normal_velocity =
        std::abs(cfg_.v[0]) > eps || // top normal component
        std::abs(cfg_.v[2]) > eps || // bottom normal component
        std::abs(cfg_.u[1]) > eps || // right normal component
        std::abs(cfg_.u[3]) > eps;   // left normal component

    if (has_nonzero_normal_velocity)
    {
        throw std::runtime_error(
            "Incompatible boundary conditions for current psi-omega setup: "
            "normal boundary velocity must be zero "
            "(require v[top]=v[bottom]=0 and u[right]=u[left]=0).");
    }

    const double dx = 1.0 / (cfg_.nx - 1);
    const double dy = 1.0 / (cfg_.ny - 1);
    const double min_h2 = std::min(dx * dx, dy * dy);
    const double dt = cfg_.t_max / cfg_.n_time_steps;
    const double alpha = cfg_.nu * dt / min_h2;
    constexpr double alpha_limit = 0.25;

    if (alpha > alpha_limit)
    {
        throw std::runtime_error(
            "Time step is too large for this psi-omega discretization with Thom boundary update: "
            "nu*dt/min(h^2) = " + std::to_string(alpha) +
            " > " + std::to_string(alpha_limit) +
            ". Increase n_time_steps or reduce t_max.");
    }
}
