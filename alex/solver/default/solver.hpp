#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "../../task1/task1.hpp"
#include <chrono>
#include <string>

struct SolverParams {
    std::string method = "ldlt"; // ldlt, lu, gmres
    bool save_to_file = true;
    std::string output_filename = "solution.txt";
    bool verbose = true;
};

struct SolverResult {
    VectorXd u;
    double solve_time;
    double total_time;
    bool success;
    std::string error_message;
    int iterations;
};

class Solver {
public:
    Solver(const SolverParams& params);
    SolverResult solve(const GridData& data);
    void print_report(const GridData& data, const SolverResult& result) const;
    void save_to_file(const VectorXd& u, const GridData& data, const std::string& filename) const;

private:
    SolverParams params_;
    double get_exact_solution(double x, double y) const;
};

#endif // SOLVER_HPP
