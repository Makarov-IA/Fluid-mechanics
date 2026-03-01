#include "stokes_mac.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

StokesMac2D::StokesMac2D(
    int nx, int ny, double lx, double ly, double nu, double dt, int poisson_max_iter, double poisson_tol
)
    : nx_(nx),
      ny_(ny),
      lx_(lx),
      ly_(ly),
      nu_(nu),
      dt_(dt),
      poisson_max_iter_(poisson_max_iter),
      poisson_tol_(poisson_tol),
      dx_(lx / static_cast<double>(nx)),
      dy_(ly / static_cast<double>(ny)),
      dx2_(dx_ * dx_),
      dy2_(dy_ * dy_),
      p_(static_cast<size_t>(nx) * static_cast<size_t>(ny), 0.0),
      u_(static_cast<size_t>(nx + 1) * static_cast<size_t>(ny), 0.0),
      v_(static_cast<size_t>(nx) * static_cast<size_t>(ny + 1), 0.0) {
    if (nx <= 1 || ny <= 1) {
        throw std::invalid_argument("Nx and Ny must be > 1");
    }
    if (dt <= 0.0 || lx <= 0.0 || ly <= 0.0 || nu < 0.0) {
        throw std::invalid_argument("Invalid physical parameters");
    }
    apply_velocity_bc(u_, v_);
}

void StokesMac2D::apply_velocity_bc(std::vector<double>& u, std::vector<double>& v) const {
    for (int j = 0; j < ny_; ++j) {
        u[u_idx(0, j)] = 0.0;
        u[u_idx(nx_, j)] = 0.0;
    }
    for (int i = 0; i <= nx_; ++i) {
        u[u_idx(i, 0)] = 0.0;
        u[u_idx(i, ny_ - 1)] = 0.0;
    }

    for (int i = 0; i < nx_; ++i) {
        v[v_idx(i, 0)] = 0.0;
        v[v_idx(i, ny_)] = 0.0;
    }
    for (int j = 0; j <= ny_; ++j) {
        v[v_idx(0, j)] = 0.0;
        v[v_idx(nx_ - 1, j)] = 0.0;
    }
}

void StokesMac2D::compute_laplacian_u(const std::vector<double>& u, std::vector<double>& lap_u) const {
    std::fill(lap_u.begin(), lap_u.end(), 0.0);
    for (int i = 1; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            const double uij = u[u_idx(i, j)];
            const double uim = u[u_idx(i - 1, j)];
            const double uip = u[u_idx(i + 1, j)];
            const double ujm = (j > 0) ? u[u_idx(i, j - 1)] : -uij;
            const double ujp = (j < ny_ - 1) ? u[u_idx(i, j + 1)] : -uij;

            const double u_xx = (uip - 2.0 * uij + uim) / dx2_;
            const double u_yy = (ujp - 2.0 * uij + ujm) / dy2_;
            lap_u[u_idx(i, j)] = u_xx + u_yy;
        }
    }
}

void StokesMac2D::compute_laplacian_v(const std::vector<double>& v, std::vector<double>& lap_v) const {
    std::fill(lap_v.begin(), lap_v.end(), 0.0);
    for (int i = 0; i < nx_; ++i) {
        for (int j = 1; j < ny_; ++j) {
            const double vij = v[v_idx(i, j)];
            const double vim = (i > 0) ? v[v_idx(i - 1, j)] : -vij;
            const double vip = (i < nx_ - 1) ? v[v_idx(i + 1, j)] : -vij;
            const double vjm = v[v_idx(i, j - 1)];
            const double vjp = v[v_idx(i, j + 1)];

            const double v_xx = (vip - 2.0 * vij + vim) / dx2_;
            const double v_yy = (vjp - 2.0 * vij + vjm) / dy2_;
            lap_v[v_idx(i, j)] = v_xx + v_yy;
        }
    }
}

void StokesMac2D::compute_divergence(const std::vector<double>& u, const std::vector<double>& v, std::vector<double>& div) const {
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double du_dx = (u[u_idx(i + 1, j)] - u[u_idx(i, j)]) / dx_;
            const double dv_dy = (v[v_idx(i, j + 1)] - v[v_idx(i, j)]) / dy_;
            div[p_idx(i, j)] = du_dx + dv_dy;
        }
    }
}

void StokesMac2D::solve_poisson_jacobi(const std::vector<double>& rhs, std::vector<double>& p) const {
    std::vector<double> p_old = p;
    std::vector<double> p_new = p;

    const double coef = 2.0 / dx2_ + 2.0 / dy2_;
    for (int iter = 0; iter < poisson_max_iter_; ++iter) {
        double max_delta = 0.0;
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                const double pw = (i > 0) ? p_old[p_idx(i - 1, j)] : p_old[p_idx(i, j)];
                const double pe = (i < nx_ - 1) ? p_old[p_idx(i + 1, j)] : p_old[p_idx(i, j)];
                const double ps = (j > 0) ? p_old[p_idx(i, j - 1)] : p_old[p_idx(i, j)];
                const double pn = (j < ny_ - 1) ? p_old[p_idx(i, j + 1)] : p_old[p_idx(i, j)];

                p_new[p_idx(i, j)] = ((pe + pw) / dx2_ + (pn + ps) / dy2_ - rhs[p_idx(i, j)]) / coef;
            }
        }

        // Neumann BC and gauge fix.
        for (int j = 0; j < ny_; ++j) {
            p_new[p_idx(0, j)] = p_new[p_idx(1, j)];
            p_new[p_idx(nx_ - 1, j)] = p_new[p_idx(nx_ - 2, j)];
        }
        for (int i = 0; i < nx_; ++i) {
            p_new[p_idx(i, 0)] = p_new[p_idx(i, 1)];
            p_new[p_idx(i, ny_ - 1)] = p_new[p_idx(i, ny_ - 2)];
        }
        p_new[p_idx(0, 0)] = 0.0;

        for (size_t k = 0; k < p_new.size(); ++k) {
            max_delta = std::max(max_delta, std::abs(p_new[k] - p_old[k]));
        }
        p_old.swap(p_new);
        if (max_delta < poisson_tol_) {
            break;
        }
    }
    p = p_old;
}

double StokesMac2D::step(double t, ForceFn f1, ForceFn f2) {
    std::vector<double> lap_u(u_.size(), 0.0);
    std::vector<double> lap_v(v_.size(), 0.0);
    std::vector<double> u_star = u_;
    std::vector<double> v_star = v_;
    std::vector<double> rhs(p_.size(), 0.0);

    compute_laplacian_u(u_, lap_u);
    compute_laplacian_v(v_, lap_v);

    for (int j = 0; j < ny_; ++j) {
        const double y = (static_cast<double>(j) + 0.5) * dy_;
        for (int i = 0; i <= nx_; ++i) {
            const double x = static_cast<double>(i) * dx_;
            const double force = (f1 != nullptr) ? f1(x, y, t) : 0.0;
            u_star[u_idx(i, j)] = u_[u_idx(i, j)] + dt_ * (nu_ * lap_u[u_idx(i, j)] + force);
        }
    }

    for (int j = 0; j <= ny_; ++j) {
        const double y = static_cast<double>(j) * dy_;
        for (int i = 0; i < nx_; ++i) {
            const double x = (static_cast<double>(i) + 0.5) * dx_;
            const double force = (f2 != nullptr) ? f2(x, y, t) : 0.0;
            v_star[v_idx(i, j)] = v_[v_idx(i, j)] + dt_ * (nu_ * lap_v[v_idx(i, j)] + force);
        }
    }

    apply_velocity_bc(u_star, v_star);

    compute_divergence(u_star, v_star, rhs);
    for (double& val : rhs) {
        val /= dt_;
    }

    double rhs_mean = 0.0;
    for (double val : rhs) {
        rhs_mean += val;
    }
    rhs_mean /= static_cast<double>(rhs.size());
    for (double& val : rhs) {
        val -= rhs_mean;
    }

    std::vector<double> p_new = p_;
    solve_poisson_jacobi(rhs, p_new);

    std::vector<double> u_new = u_star;
    std::vector<double> v_new = v_star;

    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const double dp_dx = (p_new[p_idx(i, j)] - p_new[p_idx(i - 1, j)]) / dx_;
            u_new[u_idx(i, j)] = u_star[u_idx(i, j)] - dt_ * dp_dx;
        }
    }
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double dp_dy = (p_new[p_idx(i, j)] - p_new[p_idx(i, j - 1)]) / dy_;
            v_new[v_idx(i, j)] = v_star[v_idx(i, j)] - dt_ * dp_dy;
        }
    }

    apply_velocity_bc(u_new, v_new);

    p_.swap(p_new);
    u_.swap(u_new);
    v_.swap(v_new);

    std::vector<double> div(p_.size(), 0.0);
    compute_divergence(u_, v_, div);
    double max_div = 0.0;
    for (double val : div) {
        max_div = std::max(max_div, std::abs(val));
    }
    return max_div;
}

extern "C" void* stokes_mac_create_c(
    int Nx, int Ny, double Lx, double Ly, double nu, double dt, int poisson_max_iter, double poisson_tol
) {
    try {
        return new StokesMac2D(Nx, Ny, Lx, Ly, nu, dt, poisson_max_iter, poisson_tol);
    } catch (...) {
        return nullptr;
    }
}

extern "C" void stokes_mac_free_c(void* handle) {
    if (handle == nullptr) {
        return;
    }
    delete reinterpret_cast<StokesMac2D*>(handle);
}

extern "C" double stokes_mac_step_c(void* handle, double t, Force2DTime_C f1, Force2DTime_C f2) {
    if (handle == nullptr) {
        return -1.0;
    }
    return reinterpret_cast<StokesMac2D*>(handle)->step(t, f1, f2);
}

extern "C" const double* stokes_mac_get_p_c(void* handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<StokesMac2D*>(handle)->p_data();
}

extern "C" const double* stokes_mac_get_u_c(void* handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<StokesMac2D*>(handle)->u_data();
}

extern "C" const double* stokes_mac_get_v_c(void* handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<StokesMac2D*>(handle)->v_data();
}
