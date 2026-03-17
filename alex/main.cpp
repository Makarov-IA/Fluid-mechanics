#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "solver/solver.hpp"

int main(int argc, char* argv[])
{
    try
    {
        const std::string config_path = (argc > 1) ? argv[1] : "configs/config.cfg";
        Config cfg = LoadConfigFromFile(config_path);
        if (argc > 2)
        {
            cfg.save_dir = argv[2];
        }

        std::filesystem::create_directories(cfg.save_dir);

        Solver solver(cfg);
        solver.solve();

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Main] Error: " << ex.what() << "\n";
        return 1;
    }
}
