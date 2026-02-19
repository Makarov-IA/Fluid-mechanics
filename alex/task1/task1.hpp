#ifndef TASK1_HPP
#define TASK1_HPP

#include "../../External_libs/Eigen/Dense"
#include "../../External_libs/Eigen/Sparse"
#include <vector>

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using VectorXd = Eigen::VectorXd;

struct GridData {
    int Nx;
    int Ny;
    int total_points;
    double hx;
    double hy;
    SparseMatrix A;
    VectorXd b;
};

struct Task1Params {
    int Nx = 50;
    int Ny = 50;
    int problem_type = 1; // 1 - Пуассон, 2 - Другое
};

class Task1 {
public:
    Task1(const Task1Params& params);
    GridData generate() const;
    const Task1Params& getParams() const { return params_; }

private:
    Task1Params params_;
    double hx_;
    double hy_;
    
    double f_func(double x, double y) const;
};

#endif // TASK1_HPP
