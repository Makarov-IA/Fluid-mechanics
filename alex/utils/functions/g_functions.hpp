#ifndef G_FUNCTIONS_HPP
#define G_FUNCTIONS_HPP

#include <cmath>
#include <functional>
#include <stdexcept>

struct GFunction
{
    std::function<double(double, double, double)> fn{};

    double operator()(double x, double y, double t) const
    {
        if (!fn)
        {
            throw std::runtime_error("GFunction is not set");
        }
        return fn(x, y, t);
    }

    double operator()(double x, double y) const
    {
        if (!fn)
        {
            throw std::runtime_error("GFunction is not set");
        }
        return fn(x, y, 0.0);
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(fn);
    }
};

inline GFunction GetGFunctionById(int id)
{
    constexpr double pi = 3.14159265358979323846;

    switch (id)
    {
    case 0:
        // g(x,y,t) = 0
        return GFunction{[](double, double, double) { return 0.0; }};

    case 1:
        // g(x,y,t) = sin(pi*x) * sin(pi*y)
        return GFunction{[pi](double x, double y, double)
        {
            return std::sin(pi * x) * std::sin(pi * y);
        }};

    case 2:
        // g(x,y,t) = sin(2*pi*x) * sin(pi*y) * cos(t)
        return GFunction{[pi](double x, double y, double t)
        {
            return std::sin(2.0 * pi * x) * std::sin(pi * y) * std::cos(t);
        }};

    case 3:
        // Gaussian source centered at (0.5, 0.5), sigma=0.1
        return GFunction{[](double x, double y, double)
        {
            const double dx = x - 0.5;
            const double dy = y - 0.5;
            const double sigma = 0.1;
            const double r2 = dx * dx + dy * dy;
            return std::exp(-r2 / (2.0 * sigma * sigma));
        }};

    case 4:
        // Dipole forcing: two opposite Gaussian lobes
        return GFunction{[](double x, double y, double)
        {
            const double sigma = 0.08;
            const double s2 = 2.0 * sigma * sigma;

            const double dx1 = x - 0.35;
            const double dy1 = y - 0.5;
            const double g1 = std::exp(-(dx1 * dx1 + dy1 * dy1) / s2);

            const double dx2 = x - 0.65;
            const double dy2 = y - 0.5;
            const double g2 = std::exp(-(dx2 * dx2 + dy2 * dy2) / s2);

            return 2.0 * (g1 - g2);
        }};

    case 5:
        // Traveling-wave forcing
        // g = A * sin(2*pi*x - w*t) * sin(pi*y), A=1.5, w=6
        return GFunction{[pi](double x, double y, double t)
        {
            const double A = 1.5;
            const double w = 6.0;
            return A * std::sin(2.0 * pi * x - w * t) * std::sin(pi * y);
        }};

    default:
        throw std::runtime_error("Unknown g_function_id. Supported ids: 0, 1, 2, 3, 4, 5");
    }
}

#endif // G_FUNCTIONS_HPP
