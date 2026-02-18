#include <math.h>
#include <grid.h>
#include <solver.h>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

double* solve_dirichlet(Point sp, int Nx, int Ny, double hx, double hy, Func2D f, Func2D g) {

    int Nx_internal = Nx - 2;
    int Ny_internal = Ny - 2;
    int N_internal  = Nx_internal * Ny_internal; // количество внутренних точек

    double a = 1.0 / (hx * hx);
    double b = 1.0 / (hy * hy);
    double c = 2.0 * (a + b);

    double* u = new double[Nx * Ny]; // массив решения

    auto id  = [Nx](int i, int j) { return i + j * Nx; };
    auto idx = [Nx_internal](int i, int j) { return (i - 1) + (j - 1) * Nx_internal; };

    // граница (Дирихле) + начальное заполнение внутри
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {

            double x = uniform_grid(sp, i, 0, hx).x;
            double y = uniform_grid(sp, 0, j, hy).y;

            if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1)
                u[id(i, j)] = g(x, y);
            else
                u[id(i, j)] = 0.0;
        }
    }

    using SpMat = Eigen::SparseMatrix<double>;
    using Trip  = Eigen::Triplet<double>;
    using Vec   = Eigen::VectorXd;

    std::vector<Trip> T;
    T.reserve(5 * (size_t)N_internal);

    Vec rhs = Vec::Zero(N_internal);

    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {

            int k = idx(i, j);

            double x = uniform_grid(sp, i, 0, hx).x;
            double y = uniform_grid(sp, 0, j, hy).y;

            T.emplace_back(k, k, c);
            rhs[k] += f(x, y);

            // левый сосед
            if (i > 1) T.emplace_back(k, idx(i - 1, j), -a);
            else       rhs[k] += a * u[id(0, j)];

            // правый сосед
            if (i < Nx - 2) T.emplace_back(k, idx(i + 1, j), -a);
            else            rhs[k] += a * u[id(Nx - 1, j)];

            // нижний сосед
            if (j > 1) T.emplace_back(k, idx(i, j - 1), -b);
            else       rhs[k] += b * u[id(i, 0)];

            // верхний сосед
            if (j < Ny - 2) T.emplace_back(k, idx(i, j + 1), -b);
            else            rhs[k] += b * u[id(i, Ny - 1)];
        }
    }

    SpMat A(N_internal, N_internal);
    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();

    Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver;
    solver.setTolerance(1e-10);
    solver.setMaxIterations(5000);
    solver.compute(A);

    Vec sol = solver.solve(rhs);

    // переносим решение во внутренние узлы
    for (int j = 1; j < Ny - 1; ++j)
        for (int i = 1; i < Nx - 1; ++i)
            u[id(i, j)] = sol[idx(i, j)];

    return u;
}

extern "C" double* solve_dirichlet_c(double spx, double spy, int Nx, int Ny, double hx, double hy, Func2D_C f, Func2D_C g) {
    Point sp{spx, spy};
    return solve_dirichlet(sp, Nx, Ny, hx, hy, f, g);
}

extern "C" void solver_free(double* ptr) {
    delete[] ptr;
}
