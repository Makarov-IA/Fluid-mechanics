#include "solver.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

struct Config {
    Task2Params params;
};

void print_usage() {
    std::cout << "Usage: ./task2 [options]\n";
    std::cout << "Time domain is fixed: t in [0, 1]\n";
    std::cout << "  --Nx N                  Grid nodes in x (default 81)\n";
    std::cout << "  --Ny N                  Grid nodes in y (default 81)\n";
    std::cout << "  --Nt N                  Time steps on [0,1] (default 200)\n";
    std::cout << "  --nu V                  Kinematic viscosity nu (default 0.01)\n";
    std::cout << "  --lid U                 Top boundary velocity (default 1)\n";
    std::cout << "  --forcing MODE          zero | sin (default zero)\n";
    std::cout << "  --forcing-amp A         Forcing amplitude (default 0)\n";
    std::cout << "  --forcing-omega W       Forcing frequency in time (default 0)\n";
    std::cout << "  --omega-iter N          Max iterations for implicit omega step (default 400)\n";
    std::cout << "  --omega-tol E           Tolerance for implicit omega step (default 1e-7)\n";
    std::cout << "  --poisson-iter N        Max iterations for Poisson solver (default 400)\n";
    std::cout << "  --poisson-tol E         Tolerance for Poisson solver (default 1e-7)\n";
    std::cout << "  --save-every N          Save snapshot each N steps (default 20)\n";
    std::cout << "  --output-dir PATH       Output directory (default results)\n";
    std::cout << "  --quiet                 Quiet mode\n";
    std::cout << "  --help                  Show help\n";
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage();
            std::exit(0);
        } else if (arg == "--Nx" && i + 1 < argc) {
            cfg.params.Nx = std::stoi(argv[++i]);
        } else if (arg == "--Ny" && i + 1 < argc) {
            cfg.params.Ny = std::stoi(argv[++i]);
        } else if (arg == "--Nt" && i + 1 < argc) {
            cfg.params.Nt = std::stoi(argv[++i]);
        } else if (arg == "--nu" && i + 1 < argc) {
            cfg.params.nu = std::stod(argv[++i]);
        } else if (arg == "--lid" && i + 1 < argc) {
            cfg.params.lid_velocity = std::stod(argv[++i]);
        } else if (arg == "--forcing" && i + 1 < argc) {
            cfg.params.forcing = argv[++i];
        } else if (arg == "--forcing-amp" && i + 1 < argc) {
            cfg.params.forcing_amp = std::stod(argv[++i]);
        } else if (arg == "--forcing-omega" && i + 1 < argc) {
            cfg.params.forcing_omega_t = std::stod(argv[++i]);
        } else if (arg == "--omega-iter" && i + 1 < argc) {
            cfg.params.omega_max_iter = std::stoi(argv[++i]);
        } else if (arg == "--omega-tol" && i + 1 < argc) {
            cfg.params.omega_tol = std::stod(argv[++i]);
        } else if (arg == "--poisson-iter" && i + 1 < argc) {
            cfg.params.poisson_max_iter = std::stoi(argv[++i]);
        } else if (arg == "--poisson-tol" && i + 1 < argc) {
            cfg.params.poisson_tol = std::stod(argv[++i]);
        } else if (arg == "--save-every" && i + 1 < argc) {
            cfg.params.save_every = std::stoi(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            cfg.params.output_dir = argv[++i];
        } else if (arg == "--quiet") {
            cfg.params.verbose = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n\n";
            print_usage();
            std::exit(1);
        }
    }

    return cfg;
}

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    std::cout << "=== Task2: implicit FD solver ===" << std::endl;
    std::cout << "PDE: omega_t - nu*Delta(omega) = g(x,y,t),  Delta(psi) = omega" << std::endl;
    std::cout << "Domain: (x,y,t) in [0,1] x [0,1] x [0,1]" << std::endl;
    std::cout << "Grid: " << cfg.params.Nx << "x" << cfg.params.Ny << ", Nt=" << cfg.params.Nt
              << ", nu=" << cfg.params.nu << std::endl;

    Task2Solver solver(cfg.params);
    Task2Result result = solver.solve();

    if (!result.success) {
        std::cerr << "Solve failed" << std::endl;
        return 2;
    }

    std::cout << "Steps completed: " << result.steps_completed << std::endl;
    std::cout << "dt: " << result.dt << ", final t: " << result.final_time << std::endl;
    std::cout << "final omega residual: " << result.final_omega_residual << std::endl;
    std::cout << "final psi residual: " << result.final_poisson_residual << std::endl;
    std::cout << "elapsed: " << result.elapsed_seconds << " s" << std::endl;

    if (!solver.save_fields(result)) {
        return 3;
    }

    std::cout << "Fields saved to: " << cfg.params.output_dir << std::endl;
    return 0;
}
