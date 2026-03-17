#ifndef SOLVER__SOLVER_HPP
#define SOLVER__SOLVER_HPP

#include <string>

#include "../../External_libs/Eigen/Dense"
#include "../utils/config.hpp"

#include "solver_utils.hpp"

struct Residual {
    double psi_res = 0.0;
    double omega_res = 0.0;
};

class Solver
{
private:
    Config cfg_;

    int nx_;
    int ny_;
    double t_max_;
    int n_time_steps_;

    double dx_;
    double dy_;
    double dt_;

    int Re_;
    int step_ = 0;
    double elapsed_time_ = 0.0;

    Residual residual_;

    Eigen::MatrixXd psi_;
    Eigen::MatrixXd omega_;
    Eigen::MatrixXd u_;
    Eigen::MatrixXd v_;

    Eigen::MatrixXd f_;
    Eigen::MatrixXd g_;

    void updateVelocities();
    void ApplyThomBoundary();
    void solvePsi();
    void solveOmega();
    void computeRHS(Eigen::MatrixXd& rhs) const;
    void computeOmegaRHS(Eigen::MatrixXd& rhs) const;
    void computeResiduals();

public:
    Solver(const Config& cfg);

    void step();
    void solve();
    double time() const;
    void save(const std::string& directory) const;
};

#endif // SOLVER__SOLVER_HPP
