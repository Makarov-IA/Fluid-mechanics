#ifndef FORCING__OMEGA_FORCING_HPP
#define FORCING__OMEGA_FORCING_HPP

#include <cmath>

inline double OmegaForcing(double x, double y, double lx, double ly)
{
    constexpr double kPi = 3.14159265358979323846;
    const double a22 = -19.0 * ly / (2.0 * kPi);
    const double a42 = -6.0 * ly / (2.0 * kPi);
    const double a62 = 7.0 * ly / (2.0 * kPi);
    const double a24 = 14.0 * ly / (4.0 * kPi);
    const double a13 = a22 / 50.0;
    const double a31 = a22 / 50.0;

    const auto mode = [&](int m, int n, double a_mn) {
        return a_mn * std::sin(kPi * static_cast<double>(m) * x / lx) *
               std::cos(kPi * static_cast<double>(n) * y / ly);
    };

    double f1 = 0.0;
    f1 += mode(2, 2, a22);
    f1 += mode(4, 2, a42);
    f1 += mode(6, 2, a62);
    f1 += mode(2, 4, a24);
    f1 += mode(1, 3, a13);
    f1 += mode(3, 1, a31);

    return f1;
}

#endif // FORCING__OMEGA_FORCING_HPP
