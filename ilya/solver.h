#pragma once

#include <grid.h>

using Func2D = double (*)(double, double);

double* solve_dirichlet(Point sp, int Nx, int Ny, double hx, double hy, Func2D f, Func2D g);

extern "C" {
    typedef double (*Func2D_C)(double, double);

    double* solve_dirichlet_c(double spx, double spy, int Nx, int Ny, double hx, double hy, Func2D_C f, Func2D_C g);
    void solver_free(double* ptr);
}
