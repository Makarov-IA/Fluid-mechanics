#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

Solver::Solver(const Config& cfg) : cfg_(cfg)
{
    nx_ = cfg.nx;
    ny_ = cfg.ny;
    t_max_ = cfg.t_max;
    n_time_steps_ = cfg.n_time_steps;

    dx_ = 1.0 / (nx_ - 1);
    dy_ = 1.0 / (ny_ - 1);
    dt_ = t_max_ / n_time_steps_;

    Re_ = cfg.Re;

    psi_ = Eigen::MatrixXd::Zero(nx_, ny_);
    omega_ = Eigen::MatrixXd::Zero(nx_, ny_);
    u_ = Eigen::MatrixXd::Zero(nx_, ny_);
    v_ = Eigen::MatrixXd::Zero(nx_, ny_);

    f_ = Eigen::MatrixXd::Zero(nx_, ny_);
    g_ = Eigen::MatrixXd::Zero(nx_, ny_);
}

void Solver::updateVelocities() {
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {

            if (i == 0 || i == nx_ - 1) {
                u_(i, j) = 0.0;
                v_(i, j) = 0.0;
                continue;
            }

            if (j == 0) {
                u_(i, j) = 0.0;
                v_(i, j) = 0.0;
                continue;
            }

            if (j == ny_ - 1){
                u_(i, j) = 1.0;
                v_(i, j) = 0.0;
                continue;
            }

            u_(i, j) = (psi_(i, j+1) - psi_(i, j-1)) / (2.0 * dy_);
            v_(i, j) = -(psi_(i+1, j) - psi_(i-1, j)) / (2.0 * dx_);
        }
    }
}

void Solver::ApplyThomBoundary() {

    for (int i = 1; i < nx_ - 1; ++i) {
        omega_(i, 0) = -(2.0 * psi_(i, 1)) / (dy_ * dy_);
        omega_(i, ny_ - 1) = - (2.0 * psi_(i, ny_ - 2)) / (dy_ * dy_) - 2. / dy_;
    }

    for (int j = 1; j < ny_ - 1; ++j) {
        omega_(0, j) = - (2.0 * psi_(1, j)) / (dx_ * dx_);
        omega_(nx_ - 1, j) = - (2.0 * psi_(nx_ - 2, j)) / (dx_ * dx_);
    }
}

void Solver::computeRHS(Eigen::MatrixXd& rhs) const {
    if (rhs.rows() != nx_ || rhs.cols() != ny_) {
        throw std::runtime_error("Error in compute RHS: wrong matrix size, need" + std::to_string(nx_) + "x" + std::to_string(ny_) + " but got " + std::to_string(rhs.rows()) + "x" + std::to_string(rhs.cols()) + ".");
    }

    double tmp1, tmp2, tmp3;
    double term;

    for (int i = 1; i < nx_ - 1; ++i){
        for (int j = 1; j < ny_ - 1; ++j){
            tmp1 = (psi_(i-1, j-1) - 2.0 * psi_(i-1, j) + psi_(i-1, j+1)) / (dy_ * dy_);
            tmp2 = (psi_(i, j-1) - 2.0 * psi_(i, j) + psi_(i, j+1)) / (dy_ * dy_);
            tmp3 = (psi_(i+1, j-1) - 2.0 * psi_(i+1, j) + psi_(i+1, j+1)) / (dy_ * dy_);

            term = (tmp1 - 2 * tmp2 + tmp3) / (dx_ * dx_);

            rhs(i, j) = psi_(i, j) + dt_ * omega_(i, j) + dt_*dt_*term;
        }
    }

}

void Solver::computeOmegaRHS(Eigen::MatrixXd& rhs) const {

    if (rhs.rows() != nx_ || rhs.cols() != ny_) {
        throw std::runtime_error("Error in compute RHS: wrong matrix size, need" + std::to_string(nx_) + "x" + std::to_string(ny_) + " but got " + std::to_string(rhs.rows()) + "x" + std::to_string(rhs.cols()) + ".");
    }

    Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(nx_, ny_);

    for (int i = 1; i < nx_ - 1; ++i) {
        for (int j = 1; j < ny_ - 1; ++j) {
            double d2y = (omega_(i, j-1) - 2.0 * omega_(i, j) + omega_(i, j+1)) / (dy_ * dy_);
            double d1y = (omega_(i, j+1) - omega_(i, j-1)) / (2.0 * dy_);
            double dPsix = (psi_(i+1, j) - psi_(i-1, j)) / (2.0 * dx_);

            tmp(i, j) = (dt_ / Re_) * d2y + dt_ * dPsix * d1y;
        }
    }
    for (int j = 1; j < ny_ - 1; ++j) {
        double d2y_left = (omega_(0, j-1) - 2.0 * omega_(0, j) + omega_(0, j+1)) / (dy_ * dy_);
        tmp(0, j) = (dt_ / Re_) * d2y_left;

        double d2y_right = (omega_(nx_ - 1, j-1) - 2.0 * omega_(nx_ - 1, j) + omega_(nx_ - 1, j+1)) / (dy_ * dy_);
        tmp(nx_ - 1, j) = (dt_ / Re_) * d2y_right;
    }

    for (int i = 1; i < nx_ - 1; ++i) {
        for (int j = 1; j < ny_ - 1; ++j) {
            double d2x_tmp = (tmp(i-1, j) - 2.0 * tmp(i, j) + tmp(i+1, j)) / (dx_ * dx_);
            double d1x_tmp = (tmp(i+1, j) - tmp(i-1, j)) / (2.0 * dx_);
            double dPsiy = (psi_(i, j+1) - psi_(i, j-1)) / (2.0 * dy_);

            double Lx_tmp = (dt_ / Re_) * d2x_tmp - dt_ * dPsiy * d1x_tmp;

            rhs(i, j) = omega_(i, j) + Lx_tmp;
        }
    }
}

void Solver::solvePsi() {
    computeRHS(f_);

    double x_term = dt_ / (dx_ * dx_);
    double y_term = dt_ / (dy_ * dy_);

    Eigen::VectorXd a_x = Eigen::VectorXd::Zero(nx_);
    Eigen::VectorXd b_x = Eigen::VectorXd::Zero(nx_);
    Eigen::VectorXd c_x = Eigen::VectorXd::Zero(nx_);
    Eigen::VectorXd res_x = Eigen::VectorXd::Zero(nx_);

    for (int i = 1; i < nx_ - 1; ++i) {
        a_x(i) = -x_term;
        c_x(i) = 1.0 + 2.0 * x_term;
        b_x(i) = -x_term;
    }
    c_x(0) = 1.0;
    c_x(nx_ - 1) = 1.0;

    for (int j = 1; j < ny_ - 1; ++j) {
        Eigen::VectorXd d = f_.col(j);
        d(0) = 0.0;
        d(nx_ - 1) = 0.0;

        Progonka(a_x, b_x, c_x, d, res_x);
        f_.col(j) = res_x;
    }

    Eigen::VectorXd a_y = Eigen::VectorXd::Zero(ny_);
    Eigen::VectorXd b_y = Eigen::VectorXd::Zero(ny_);
    Eigen::VectorXd c_y = Eigen::VectorXd::Zero(ny_);
    Eigen::VectorXd res_y = Eigen::VectorXd::Zero(ny_);

    for (int j = 1; j < ny_ - 1; ++j) {
        a_y(j) = -y_term;
        c_y(j) = 1.0 + 2.0 * y_term;
        b_y(j) = -y_term;
    }
    c_y(0) = 1.0;
    c_y(ny_ - 1) = 1.0;

    for (int i = 1; i < nx_ - 1; ++i) {
        Eigen::VectorXd d = f_.row(i);
        d(0) = 0.0;
        d(ny_ - 1) = 0.0;

        Progonka(a_y, b_y, c_y, d, res_y);
        psi_.row(i) = res_y;
    }

    for (int i = 1; i < nx_ - 1; ++i) {
        psi_(i, 0) = 0.0;
        psi_(i, ny_ - 1) = 0.0;
    }
    for(int j = 1; j < ny_ - 1; ++j) {
        psi_(0, j) = 0.0;
        psi_(nx_ - 1, j) = 0.0;
    }
}

void Solver::solveOmega() {
    computeOmegaRHS(g_);

    double x_term = dt_ / (Re_ * dx_ * dx_);
    double y_term = dt_ / (Re_ * dy_ * dy_);

    Eigen::VectorXd a_x = Eigen::VectorXd::Zero(nx_);
    Eigen::VectorXd b_x = Eigen::VectorXd::Zero(nx_);
    Eigen::VectorXd c_x = Eigen::VectorXd::Zero(nx_);
    Eigen::VectorXd res_x = Eigen::VectorXd::Zero(nx_);

    for (int j = 1; j < ny_ - 1; ++j) {
        for (int i = 1; i < nx_ - 1; ++i) {

            double psi_y = (psi_(i, j+1) - psi_(i, j-1)) / (2.0 * dy_);

            double conv_x = (dt_ * psi_y) / (2.0 * dx_);

            a_x(i) = -x_term - conv_x;
            c_x(i) = 1.0 + 2.0 * x_term;
            b_x(i) = -x_term + conv_x;
        }

        c_x(0) = 1.0;
        c_x(nx_ - 1) = 1.0;

        Eigen::VectorXd d = g_.col(j);

        Progonka(a_x, b_x, c_x, d, res_x);
        g_.col(j) = res_x;
    }

    Eigen::VectorXd a_y = Eigen::VectorXd::Zero(ny_);
    Eigen::VectorXd b_y = Eigen::VectorXd::Zero(ny_);
    Eigen::VectorXd c_y = Eigen::VectorXd::Zero(ny_);
    Eigen::VectorXd res_y = Eigen::VectorXd::Zero(ny_);


    for (int i = 1; i < nx_ - 1; ++i) {
        for (int j = 1; j < ny_ - 1; ++j) {
            double psi_x = (psi_(i+1, j) - psi_(i-1, j)) / (2.0 * dx_);

            double conv_y = (dt_ * psi_x) / (2.0 * dy_);

            a_y(j) = -y_term + conv_y;
            c_y(j) = 1.0 + 2.0 * y_term;
            b_y(j) = -y_term - conv_y;
        }

        c_y(0) = 1.0;
        c_y(ny_ - 1) = 1.0;

        Eigen::VectorXd d = g_.row(i);

        d(0) = omega_(i, 0);
        d(ny_ - 1) = omega_(i, ny_ - 1);

        Progonka(a_y, b_y, c_y, d, res_y);
        omega_.row(i) = res_y;
    }

}

void Solver::computeResiduals() {

    double tmp = 0;

    for (int i = 1; i < nx_ - 1; ++i) {
        for (int j = 1; j < ny_ - 1; ++j) {
            residual_.psi_res = std::max(residual_.psi_res, std::abs((psi_(i-1, j) - 2.0 * psi_(i, j) + psi_(i+1, j)) / (dx_ * dx_) + (psi_(i, j-1) - 2.0 * psi_(i, j) + psi_(i, j+1))/(dy_ * dy_) + omega_(i, j)));
            tmp = 1. / Re_ * (omega_(i+1, j) - 2.0 * omega_(i, j) + omega_(i-1, j)) / (dx_ * dx_) + 1. / Re_ * (omega_(i, j+1) - 2.0 * omega_(i, j) + omega_(i, j-1)) / (dy_ * dy_);
            tmp += -(psi_(i, j+1) - psi_(i, j-1)) / (2.0 * dy_) * (omega_(i+1, j) - omega_(i-1, j)) / (2.0 * dx_);
            tmp += (psi_(i+1, j) - psi_(i-1, j)) / (2.0 * dx_) * (omega_(i, j+1) - omega_(i, j-1)) / (2.0 * dy_);
            residual_.omega_res = std::max(residual_.omega_res, std::abs(tmp));
        }
    }
}

void Solver::step() {
    solvePsi();
    ApplyThomBoundary();
    solveOmega();
    computeResiduals();
}

void Solver::solve() {
    const auto start_time = std::chrono::steady_clock::now();
    std::filesystem::create_directories(cfg_.save_dir);

    std::ofstream residual_history(cfg_.save_dir + "/residual_history.csv");
    if (!residual_history.is_open()) {
        throw std::runtime_error("Cannot open residual_history.csv for writing");
    }
    residual_history << "step,time,psi_res,omega_res,max_residual\n";

    if (cfg_.mode == "fixed_steps") {
        for (int i = 0; i < n_time_steps_; ++i) {
            step_ = i;
            residual_ = Residual{};
            const auto step_start_time = std::chrono::steady_clock::now();
            step();
            const double step_elapsed_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - step_start_time).count();
            if (i % cfg_.save_every_step == 0) {
                updateVelocities();
                save(cfg_.save_dir);
                const double pseudo_time = (step_ + 1) * dt_;
                const double max_residual = std::max(residual_.psi_res, residual_.omega_res);
                residual_history << step_ << ','
                                 << pseudo_time << ','
                                 << residual_.psi_res << ','
                                 << residual_.omega_res << ','
                                 << max_residual << '\n';
                std::cout << "\r[Solver] step=" << step_
                          << " psi_res=" << residual_.psi_res
                          << " omega_res=" << residual_.omega_res
                          << " step_time=" << step_elapsed_time << " s"
                          << std::flush;
            }
        }
    } else if (cfg_.mode == "till_converges") {
        while (residual_.psi_res > cfg_.steady_tolerance ||
               residual_.omega_res > cfg_.steady_tolerance ||
               step_ == 0) {
            residual_ = Residual{};
            const auto step_start_time = std::chrono::steady_clock::now();
            step();
            const double step_elapsed_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - step_start_time).count();
            if (step_ % cfg_.save_every_step == 0) {
                updateVelocities();
                save(cfg_.save_dir);
                const double pseudo_time = (step_ + 1) * dt_;
                const double max_residual = std::max(residual_.psi_res, residual_.omega_res);
                residual_history << step_ << ','
                                 << pseudo_time << ','
                                 << residual_.psi_res << ','
                                 << residual_.omega_res << ','
                                 << max_residual << '\n';
                std::cout << "\r[Solver] step=" << step_
                          << " psi_res=" << residual_.psi_res
                          << " omega_res=" << residual_.omega_res
                          << " step_time=" << step_elapsed_time << " s"
                          << std::flush;
            }
            ++step_;
        }
    }

    residual_history.close();
    std::cout << '\n';

    elapsed_time_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
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

    out << "x,y,psi,omega,u,v\n";

    for (int i = 0; i < nx_; ++i)
    {
        for (int j = 0; j < ny_; ++j)
        {
            const double x = i * dx_;
            const double y = j * dy_;

            out << std::fixed << std::setprecision(6)
                << x << ","
                << y << ","
                << psi_(i, j) << ","
                << omega_(i, j) << ","
                << u_(i, j) << ","
                << v_(i, j) << "\n";
        }
    }

    out.close();
    if (cfg_.log_info.enabled)
    {
        std::cout << "[Solver] Results saved to: " << filename << "\n";
    }
}

double Solver::time() const
{
    return elapsed_time_;
}
