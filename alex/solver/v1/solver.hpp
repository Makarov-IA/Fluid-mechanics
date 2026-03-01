#ifndef SOLVER_V1_SOLVER_HPP
#define SOLVER_V1_SOLVER_HPP

#include <string>

#include "../../../External_libs/Eigen/Sparse"
#include "../../../External_libs/Eigen/SparseCholesky"
#include "../../utils/config.hpp"

class Solver
{
public:
    using SparseMatrix = Eigen::SparseMatrix<double>;
    using Vector = Eigen::VectorXd;
    using SparseLDLT = Eigen::SimplicialLDLT<SparseMatrix>;

    explicit Solver(const Config& cfg);
    explicit Solver(const std::string& config_path);

    void solve();

    const Config& config() const noexcept;
    double time() const noexcept;
    int step() const noexcept;

    const Vector& psi() const noexcept;
    const Vector& omega() const noexcept;
    const Vector& u() const noexcept;
    const Vector& v() const noexcept;

    void save(const std::string& directory) const;

private:
    int Idx(int i, int j) const;
    void ValidateConfig() const;

private:
    Config cfg_{};
    GFunction g_{};

    int nx_{0};
    int ny_{0};
    int n_{0};
    int step_{0};
    double dx_{0.0};
    double dy_{0.0};
    double dt_{0.0};
    double time_{0.0};

    Vector psi_;
    Vector omega_;
    Vector u_;
    Vector v_;

    SparseMatrix poisson_matrix_;
    SparseMatrix omega_matrix_;
    Vector rhs_psi_;
    Vector rhs_omega_;
    SparseLDLT poisson_ldlt_;
    SparseLDLT omega_ldlt_;
};

#endif // SOLVER_V1_SOLVER_HPP
