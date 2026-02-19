#include "task1.hpp"
#include <cmath>
#include <iostream>

Task1::Task1(const Task1Params& params) : params_(params) {
    hx_ = 1.0 / (params_.Nx + 1);
    hy_ = 1.0 / (params_.Ny + 1);
}

double Task1::f_func(double x, double y) const {
    switch (params_.problem_type) {
        case 1:
        default:
            return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    }
}

GridData Task1::generate() const {
    GridData data;
    data.Nx = params_.Nx;
    data.Ny = params_.Ny;
    data.total_points = params_.Nx * params_.Ny;
    data.hx = hx_;
    data.hy = hy_;

    std::cout << "=== Настройка сетки ===" << std::endl;
    std::cout << "Размерность: " << params_.Nx << "x" << params_.Ny << std::endl;
    std::cout << "Шаги: hx=" << hx_ << ", hy=" << hy_ << std::endl;
    std::cout << "Неизвестных: " << data.total_points << std::endl;

    data.b = VectorXd::Zero(data.total_points);
    std::vector<Triplet> tripletList;
    tripletList.reserve(data.total_points * 5);

    double inv_hx2 = 1.0 / (hx_ * hx_);
    double inv_hy2 = 1.0 / (hy_ * hy_);
    double center_coeff = 2.0 * inv_hx2 + 2.0 * inv_hy2;

    for (int j = 0; j < params_.Ny; ++j) {
        for (int i = 0; i < params_.Nx; ++i) {
            int idx = j * params_.Nx + i;
            double x = (i + 1) * hx_;
            double y = (j + 1) * hy_;

            data.b(idx) = f_func(x, y);
            tripletList.push_back(Triplet(idx, idx, center_coeff));

            if (i > 0)
                tripletList.push_back(Triplet(idx, idx - 1, -inv_hx2));
            if (i < params_.Nx - 1)
                tripletList.push_back(Triplet(idx, idx + 1, -inv_hx2));
            if (j > 0)
                tripletList.push_back(Triplet(idx, idx - params_.Nx, -inv_hy2));
            if (j < params_.Ny - 1)
                tripletList.push_back(Triplet(idx, idx + params_.Nx, -inv_hy2));
        }
    }

    data.A.resize(data.total_points, data.total_points);
    data.A.setFromTriplets(tripletList.begin(), tripletList.end());
    data.A.makeCompressed();

    std::cout << "Матрица создана. Ненулевых элементов: " << data.A.nonZeros() << std::endl;
    std::cout << "=======================" << std::endl;

    return data;
}