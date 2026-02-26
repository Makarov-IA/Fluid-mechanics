#ifndef TASK2_SOLVER_HPP
#define TASK2_SOLVER_HPP

#include <string>
#include <vector>

struct Task2Params {
    int Nx = 81;
    int Ny = 81;
    int Nt = 200; // t in [0, 1], dt = 1 / Nt

    double nu = 0.01;
    double lid_velocity = 1.0;

    // Implicit omega solver: backward Euler + finite differences
    int omega_max_iter = 400;
    double omega_tol = 1e-7;

    // Poisson solver for Delta(psi) = omega
    int poisson_max_iter = 400;
    double poisson_tol = 1e-7;

    std::string forcing = "zero"; // zero | sin
    double forcing_amp = 0.0;
    double forcing_omega_t = 0.0;

    int save_every = 20;
    bool verbose = true;
    int log_every = 20;

    std::string output_dir = "results";
};

struct Task2Result {
    bool success = false;
    int steps_completed = 0;

    double elapsed_seconds = 0.0;
    double final_time = 0.0;
    double dt = 0.0;

    double final_omega_residual = 0.0;
    double final_poisson_residual = 0.0;

    int Nx = 0;
    int Ny = 0;
    double dx = 0.0;
    double dy = 0.0;

    std::vector<double> psi;
    std::vector<double> omega;

    std::vector<int> snapshot_steps;
    std::vector<double> snapshot_times;
    std::vector<std::vector<double>> psi_snapshots;
    std::vector<std::vector<double>> omega_snapshots;
};

class Task2Solver {
public:
    explicit Task2Solver(Task2Params params);

    Task2Result solve();
    bool save_fields(const Task2Result& result) const;

private:
    Task2Params params_;

    static int idx(int i, int j, int Nx);

    void apply_stream_boundary(std::vector<double>& psi, int Nx, int Ny) const;
    void apply_vorticity_boundary(const std::vector<double>& psi, std::vector<double>& omega,
                                  int Nx, int Ny, double dx, double dy, double t) const;

    double solve_poisson(std::vector<double>& psi, const std::vector<double>& omega,
                         int Nx, int Ny, double dx, double dy) const;

    double solve_omega_implicit(std::vector<double>& omega_new, const std::vector<double>& omega_old,
                                const std::vector<double>& g, int Nx, int Ny,
                                double dx, double dy, double dt) const;

    void fill_forcing(std::vector<double>& g, int Nx, int Ny, double t) const;
    double top_velocity(double x, double t) const;
};

#endif
