#include "solver.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

struct Config {
    Task2Params params;
};

void print_usage() {
    std::cout << "Usage: ./task2 [options]\n";
    std::cout << "  --Nx N                Grid nodes in x (default 81)\n";
    std::cout << "  --Ny N                Grid nodes in y (default 81)\n";
    std::cout << "  --nu V                Kinematic viscosity nu (default 0.01)\n";
    std::cout << "  --lid U               Top boundary velocity d(psi)/dy at y=1 (default 1)\n";
    std::cout << "  --forcing MODE        zero | sin (default zero)\n";
    std::cout << "  --forcing-amp A       Forcing amplitude g (default 0)\n";
    std::cout << "  --forcing-omega W     Time frequency for forcing (default 0)\n";
    std::cout << "  --max-iter N          Max time iterations (default 50000)\n";
    std::cout << "  --poisson-iter N      Poisson sweeps per step (default 300)\n";
    std::cout << "  --tol E               Convergence tolerance (default 1e-6)\n";
    std::cout << "  --output-dir PATH     Output directory (default results)\n";
    std::cout << "  --quiet               Quiet mode\n";
    std::cout << "  --help                Show help\n";
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
        } else if (arg == "--max-iter" && i + 1 < argc) {
            cfg.params.max_iter = std::stoi(argv[++i]);
        } else if (arg == "--poisson-iter" && i + 1 < argc) {
            cfg.params.poisson_max_iter = std::stoi(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            cfg.params.tol = std::stod(argv[++i]);
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

    std::cout << "=== Task2: omega_t - nu*Delta(omega) = g, Delta(psi) = omega ===" << std::endl;
    std::cout << "Grid: " << cfg.params.Nx << "x" << cfg.params.Ny
              << ", nu=" << cfg.params.nu
              << ", lid=" << cfg.params.lid_velocity
              << ", forcing=" << cfg.params.forcing
              << ", forcing_amp=" << cfg.params.forcing_amp << std::endl;

    Task2Solver solver(cfg.params);
    Task2Result result = solver.solve();

    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Residual: " << result.final_residual << std::endl;
    std::cout << "Final time: " << result.final_time << std::endl;
    std::cout << "Elapsed: " << result.elapsed_seconds << " s" << std::endl;
    std::cout << "Converged: " << (result.converged ? "yes" : "no") << std::endl;

    if (!solver.save_fields(result)) {
        return 2;
    }

    std::cout << "Fields saved to: " << cfg.params.output_dir << std::endl;
    return result.converged ? 0 : 1;
}
