#ifndef SOLVER_UTILS
#define SOLVER_UTILS

#include "../../External_libs/Eigen/Dense"

// c - diag
// b - minus upper diag
// a - minues lower diag
// d - right side of eq
inline void Progonka(const Eigen::VectorXd& a,
                     const Eigen::VectorXd& b,
                     const Eigen::VectorXd& c,
                     const Eigen::VectorXd& d,
                     Eigen::VectorXd& res) {

    int n = d.size();
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(n);
    res = Eigen::VectorXd::Zero(n);

    alpha(0) = -b(0) / c(0);
    beta(0) = d(0) / c(0);

    for (int i = 1; i < n-1; i++) {
        alpha(i) = -b(i) / (c(i) + a(i) * alpha(i-1));
        beta(i) = (d(i) - a(i) * beta(i-1)) / (c(i) + a(i) * alpha(i-1));
    }

    res(n-1) = (d(n-1) - a(n-1) * beta(n-2)) / (c(n-1) + a(n-1) * alpha(n-2));

    for (int i = n-2; i >= 0; i--) {
        res(i) = alpha(i) * res(i+1) + beta(i);
    }
}

#endif // !SOLVER_UTILS
