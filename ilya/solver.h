#pragma once

#include <grid.h>

using Func2D = double (*)(double, double);

double* solve_dirichlet(Point sp, int Nx, int Ny, double hx, double hy, Func2D f, Func2D g);
