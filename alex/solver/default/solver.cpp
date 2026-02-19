#include "solver.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

Solver::Solver(const SolverParams& params) : params_(params) {}

SolverResult Solver::solve(const GridData& data) {
    SolverResult result;
    result.success = false;
    result.solve_time = 0.0;
    result.total_time = 0.0;
    result.iterations = 0;

    auto start = std::chrono::high_resolution_clock::now();

    if (params_.method == "ldlt") {
        Eigen::SimplicialLDLT<SparseMatrix> solver;
        solver.compute(data.A);

        if (solver.info() != Eigen::Success) {
            result.error_message = "LDLT: Ошибка разложения матрицы!";
            auto end = std::chrono::high_resolution_clock::now();
            result.solve_time = std::chrono::duration<double>(end - start).count();
            return result;
        }

        result.u = solver.solve(data.b);
        result.iterations = -1;

        if (solver.info() != Eigen::Success) {
            result.error_message = "LDLT: Ошибка решения системы!";
            auto end = std::chrono::high_resolution_clock::now();
            result.solve_time = std::chrono::duration<double>(end - start).count();
            return result;
        }
    }
    else if (params_.method == "lu") {
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(data.A);

        if (solver.info() != Eigen::Success) {
            result.error_message = "LU: Ошибка разложения матрицы!";
            auto end = std::chrono::high_resolution_clock::now();
            result.solve_time = std::chrono::duration<double>(end - start).count();
            return result;
        }

        result.u = solver.solve(data.b);
        result.iterations = -1;

        if (solver.info() != Eigen::Success) {
            result.error_message = "LU: Ошибка решения системы!";
            auto end = std::chrono::high_resolution_clock::now();
            result.solve_time = std::chrono::duration<double>(end - start).count();
            return result;
        }
    }
    else {
        result.error_message = "Неизвестный метод: " + params_.method;
        auto end = std::chrono::high_resolution_clock::now();
        result.solve_time = std::chrono::duration<double>(end - start).count();
        return result;
    }

    result.success = true;
    auto end = std::chrono::high_resolution_clock::now();
    result.solve_time = std::chrono::duration<double>(end - start).count();

    return result;
}

double Solver::get_exact_solution(double x, double y) const {
    return std::sin(M_PI * x) * std::sin(M_PI * y);
}

void Solver::print_report(const GridData& data, const SolverResult& result) const {
    if (!params_.verbose) return;

    std::cout << "=== Отчет решателя ===" << std::endl;

    if (!result.success) {
        std::cerr << "Ошибка: " << result.error_message << std::endl;
        return;
    }

    std::cout << "Метод: " << params_.method << std::endl;
    std::cout << "Время решения: " << std::fixed << std::setprecision(6) 
              << result.solve_time << " с" << std::endl;
    if (result.iterations >= 0) {
        std::cout << "Итераций: " << result.iterations << std::endl;
    } else {
        std::cout << "Итераций: n/a (прямой метод)" << std::endl;
    }

    int center_idx = data.total_points / 2;
    int j_center = center_idx / data.Nx;
    int i_center = center_idx % data.Nx;
    double x_c = (i_center + 1) * data.hx;
    double y_c = (j_center + 1) * data.hy;

    double u_calc = result.u(center_idx);
    double u_exact = get_exact_solution(x_c, y_c);
    double error = std::abs(u_calc - u_exact);

    std::cout << "Точка центра: (" << x_c << ", " << y_c << ")" << std::endl;
    std::cout << "u(расч): " << u_calc << std::endl;
    std::cout << "u(точн): " << u_exact << std::endl;
    std::cout << "Погрешность: " << error << std::endl;
    std::cout << "========================" << std::endl;
}

void Solver::save_to_file(const VectorXd& u, const GridData& data, const std::string& filename) const {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Не удалось открыть файл " << filename << std::endl;
        return;
    }

    for (int j = 0; j < data.Ny; ++j) {
        for (int i = 0; i < data.Nx; ++i) {
            outfile << std::setprecision(10) << u(j * data.Nx + i) << " ";
        }
        outfile << "\n";
    }
    outfile.close();
    std::cout << "Результат сохранен в " << filename << std::endl;
}
