#ifndef TASK2_SOLVER_HPP
#define TASK2_SOLVER_HPP

#include <string>
#include <vector>

struct Task2Params {
    int Nx = 81;
    int Ny = 81;

    double nu = 0.01;
    double lid_velocity = 1.0;

    int max_iter = 50000;
    int poisson_max_iter = 300;
    double tol = 1e-6;

    double cfl_diff = 0.45;
    double dt_max = 1e-2;

    std::string forcing = "zero"; // zero | sin
    double forcing_amp = 0.0;
    double forcing_omega_t = 0.0;

    bool verbose = true;
    int log_every = 200;

    std::string output_dir = "results";
};

struct Task2Result {
    bool converged = false;
    int iterations = 0;
    double final_residual = 0.0;
    double elapsed_seconds = 0.0;
    double final_time = 0.0;

    int Nx = 0;
    int Ny = 0;
    double dx = 0.0;
    double dy = 0.0;

    std::vector<double> psi;
    std::vector<double> omega;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> g;
};

class Task2Solver {
public:
    explicit Task2Solver(Task2Params params);

    Task2Result solve();
    bool save_fields(const Task2Result& result) const;

private:
    Task2Params params_;

    static int idx(int i, int j, int Nx);
    static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b);

    void apply_stream_boundary(std::vector<double>& psi, int Nx, int Ny) const;
    void apply_vorticity_boundary(const std::vector<double>& psi, std::vector<double>& omega,
                                  int Nx, int Ny, double dx, double dy) const;

    double solve_poisson(std::vector<double>& psi, const std::vector<double>& omega,
                         int Nx, int Ny, double dx, double dy) const;

    void compute_velocity(const std::vector<double>& psi, std::vector<double>& u, std::vector<double>& v,
                          int Nx, int Ny, double dx, double dy) const;

    double compute_stable_dt(double dx, double dy) const;
    void fill_forcing(std::vector<double>& g, int Nx, int Ny, double t) const;
};

#endif
