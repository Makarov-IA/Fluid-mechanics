#include "solver.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
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
                p_triplets.push_back({idx, Idx(i - 1, j), 1.0 / dx2});
                p_triplets.push_back({idx, Idx(i + 1, j), 1.0 / dx2});
                p_triplets.push_back({idx, Idx(i, j - 1), 1.0 / dy2});
                p_triplets.push_back({idx, Idx(i, j + 1), 1.0 / dy2});

                // Матрица завихренности (Неявная диффузионная часть)
                // (I - dt * nu * Laplace) omega = ...
                o_triplets.push_back({idx, idx, 1.0 + 2.0 * dt_ * cfg_.nu / dx2 + 2.0 * dt_ * cfg_.nu / dy2});
                o_triplets.push_back({idx, Idx(i - 1, j), -dt_ * cfg_.nu / dx2});
                o_triplets.push_back({idx, Idx(i + 1, j), -dt_ * cfg_.nu / dx2});
                o_triplets.push_back({idx, Idx(i, j - 1), -dt_ * cfg_.nu / dy2});
                o_triplets.push_back({idx, Idx(i, j + 1), -dt_ * cfg_.nu / dy2});
            }
        }
    }

    poisson_matrix_.setFromTriplets(p_triplets.begin(), p_triplets.end());
    omega_matrix_.setFromTriplets(o_triplets.begin(), o_triplets.end());

    poisson_ldlt_.compute(poisson_matrix_);
    omega_ldlt_.compute(omega_matrix_);
}

Solver::Solver(const std::string& config_path) : Solver(LoadConfigFromFile(config_path))
{
}

void Solver::solve()
{
    double dx2 = dx_ * dx_;
    double dy2 = dy_ * dy_;

    for (step_ = 1; step_ <= cfg_.n_time_steps; ++step_)
    {
        time_ += dt_;

        // 1. Формирование правой части для уравнения завихренности
        for (int i = 0; i < nx_; ++i)
        {
            for (int j = 0; j < ny_; ++j)
            {
                int idx = Idx(i, j);

                // Граничные условия для завихренности (Формула Тома)
                if (j == 0) // Bottom
                {
                    rhs_omega_[idx] = -2.0 * psi_[Idx(i, 1)] / dy2 + 2.0 * cfg_.u[2] / dy_;
                }
                else if (j == ny_ - 1) // Top
                {
                    rhs_omega_[idx] = -2.0 * psi_[Idx(i, ny_ - 2)] / dy2 - 2.0 * cfg_.u[0] / dy_;
                }
                else if (i == 0) // Left
                {
                    rhs_omega_[idx] = -2.0 * psi_[Idx(1, j)] / dx2 - 2.0 * cfg_.v[3] / dx_;
                }
                else if (i == nx_ - 1) // Right
                {
                    rhs_omega_[idx] = -2.0 * psi_[Idx(nx_ - 2, j)] / dx2 + 2.0 * cfg_.v[1] / dx_;
                }
                // Внутренние узлы (явный перенос и источники)
                else
                {
                    double d_omega_dx = (omega_[Idx(i + 1, j)] - omega_[Idx(i - 1, j)]) / (2.0 * dx_);
                    double d_omega_dy = (omega_[Idx(i, j + 1)] - omega_[Idx(i, j - 1)]) / (2.0 * dy_);
                    
                    double conv = u_[idx] * d_omega_dx + v_[idx] * d_omega_dy;
                    
                    double forcing = 0.0;
                    if (g_)
                    {
                        forcing = g_(i * dx_, j * dy_); // Координаты узла
                    }

                    rhs_omega_[idx] = omega_[idx] - dt_ * conv + dt_ * forcing;
                }
            }
        }

        // 2. Решение уравнения для завихренности
        omega_ = omega_ldlt_.solve(rhs_omega_);

        // 3. Формирование правой части для функции тока
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
                    rhs_psi_[idx] = -omega_[idx];
                }
            }
        }

        // 4. Решение уравнения Пуассона для функции тока
        psi_ = poisson_ldlt_.solve(rhs_psi_);

        // 5. Обновление полей скорости во внутренних узлах
        for (int i = 1; i < nx_ - 1; ++i)
        {
            for (int j = 1; j < ny_ - 1; ++j)
            {
                u_[Idx(i, j)] =  (psi_[Idx(i, j + 1)] - psi_[Idx(i, j - 1)]) / (2.0 * dy_);
                v_[Idx(i, j)] = -(psi_[Idx(i + 1, j)] - psi_[Idx(i - 1, j)]) / (2.0 * dx_);
            }
        }

        // Логирование
        if (cfg_.log_info.enabled && step_ % cfg_.log_info.print_every_step == 0)
        {
            std::cout << "[Solver] Step: " << step_ << " / " << cfg_.n_time_steps 
                      << " | Time: " << time_ << "\n";
        }

        if (step_ % cfg_.save_every_step == 0 || step_ == cfg_.n_time_steps)
        {
            save(cfg_.save_dir);
        }
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
}
