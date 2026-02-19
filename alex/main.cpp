#include <iostream>
#include <string>
#include "./task1/task1.hpp"
#include "./solver/default/solver.hpp"

struct Config {
    // Параметры сетки
    int Nx = 50;
    int Ny = 50;
    int problem_type = 1;
    
    // Параметры решателя
    std::string method = "ldlt";
    bool save_to_file = true;
    std::string output_filename = "solution.txt";
    bool verbose = true;
};

void print_usage() {
    std::cout << "Использование: ./a [параметры]" << std::endl;
    std::cout << "Параметры:" << std::endl;
    std::cout << "  --Nx N        Количество узлов по X (по умолчанию 50)" << std::endl;
    std::cout << "  --Ny N        Количество узлов по Y (по умолчанию 50)" << std::endl;
    std::cout << "  --method M    Метод решения: ldlt, lu (по умолчанию ldlt)" << std::endl;
    std::cout << "  --output F    Имя файла вывода (по умолчанию solution.txt)" << std::endl;
    std::cout << "  --quiet       Тихий режим (без подробного вывода)" << std::endl;
    std::cout << "  --help        Показать эту справку" << std::endl;
}

Config parse_args(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage();
            exit(0);
        }
        else if (arg == "--Nx" && i + 1 < argc) {
            config.Nx = std::stoi(argv[++i]);
        }
        else if (arg == "--Ny" && i + 1 < argc) {
            config.Ny = std::stoi(argv[++i]);
        }
        else if (arg == "--method" && i + 1 < argc) {
            config.method = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_filename = argv[++i];
        }
        else if (arg == "--quiet") {
            config.verbose = false;
        }
        else {
            std::cerr << "Неизвестный параметр: " << arg << std::endl;
            print_usage();
            exit(1);
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Fluid Mechanics Solver ===" << std::endl;
    
    // Парсинг аргументов командной строки
    Config config = parse_args(argc, argv);
    
    // Настройка параметров задачи
    Task1Params task_params;
    task_params.Nx = config.Nx;
    task_params.Ny = config.Ny;
    task_params.problem_type = config.problem_type;
    
    // Настройка параметров решателя
    SolverParams solver_params;
    solver_params.method = config.method;
    solver_params.save_to_file = config.save_to_file;
    solver_params.output_filename = config.output_filename;
    solver_params.verbose = config.verbose;
    
    // Создание и генерация сетки
    Task1 task(task_params);
    GridData data = task.generate();
    
    // Создание решателя и решение
    Solver solver(solver_params);
    SolverResult result = solver.solve(data);
    
    // Вывод отчета
    solver.print_report(data, result);
    
    // Сохранение результата
    if (result.success && config.save_to_file) {
        solver.save_to_file(result.u, data, config.output_filename);
    }
    
    return result.success ? 0 : -1;
}
