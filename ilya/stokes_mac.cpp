#include "stokes_mac.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

StokesMac2D::StokesMac2D(int nx, int ny, double lx, double ly, double nu, double dt)
    : nx_(nx),
      ny_(ny),
      lx_(lx),
      ly_(ly),
      nu_(nu),
      dt_(dt),
      dx_(lx / static_cast<double>(nx)),
      dy_(ly / static_cast<double>(ny)),
      dx2_(dx_ * dx_),
      dy2_(dy_ * dy_),
      u_lid_top_(1.0),
      p_(static_cast<size_t>(nx) * static_cast<size_t>(ny), 0.0),
      u_(static_cast<size_t>(nx + 1) * static_cast<size_t>(ny), 0.0),
      v_(static_cast<size_t>(nx) * static_cast<size_t>(ny + 1), 0.0),
      nu_unknowns_((nx - 1) * ny),
      nv_unknowns_(nx * (ny - 1)),
      np_unknowns_(nx * ny),
      total_unknowns_(nu_unknowns_ + nv_unknowns_ + np_unknowns_),
      system_mat_(total_unknowns_, total_unknowns_) {
    if (nx <= 1 || ny <= 1) {
        throw std::invalid_argument("Nx and Ny must be > 1");
    }
    if (dt <= 0.0 || lx <= 0.0 || ly <= 0.0 || nu < 0.0) {
        throw std::invalid_argument("Invalid physical parameters");
    }

    build_monolithic_system();
    system_solver_.analyzePattern(system_mat_);
    system_solver_.factorize(system_mat_);
    if (system_solver_.info() != Eigen::Success) {
        throw std::runtime_error("Monolithic matrix factorization failed");
    }

    apply_velocity_bc(u_, v_);
}

void StokesMac2D::apply_velocity_bc(std::vector<double>& u, std::vector<double>& v) const {
    // u-normal on vertical walls.
    for (int j = 0; j < ny_; ++j) {
        u[u_idx(0, j)] = 0.0;
        u[u_idx(nx_, j)] = 0.0;
    }
    // v-normal on horizontal walls.
    for (int i = 0; i < nx_; ++i) {
        v[v_idx(i, 0)] = 0.0;
        v[v_idx(i, ny_)] = 0.0;
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

void StokesMac2D::build_monolithic_system() {
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> t;
    t.reserve(static_cast<size_t>(total_unknowns_) * 8U);

    const double inv_dt = 1.0 / dt_;

    // Momentum for u(i,j), i=1..Nx-1, j=0..Ny-1
    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const int row = u_unknown_idx(i, j);
            double diag = inv_dt;

            // x-part of Laplacian
            diag += 2.0 * nu_ / dx2_;
            if (i - 1 >= 1) {
                t.emplace_back(row, u_unknown_idx(i - 1, j), -nu_ / dx2_);
            }
            if (i + 1 <= nx_ - 1) {
                t.emplace_back(row, u_unknown_idx(i + 1, j), -nu_ / dx2_);
            }

            // y-part with no-slip wall via ghost nodes
            if (j == 0) {
                diag += 3.0 * nu_ / dy2_;
                if (j + 1 <= ny_ - 1) {
                    t.emplace_back(row, u_unknown_idx(i, j + 1), -nu_ / dy2_);
                }
            } else if (j == ny_ - 1) {
                diag += 3.0 * nu_ / dy2_;
                t.emplace_back(row, u_unknown_idx(i, j - 1), -nu_ / dy2_);
            } else {
                diag += 2.0 * nu_ / dy2_;
                t.emplace_back(row, u_unknown_idx(i, j - 1), -nu_ / dy2_);
                t.emplace_back(row, u_unknown_idx(i, j + 1), -nu_ / dy2_);
            }

            t.emplace_back(row, row, diag);

            // +dp/dx at u location
            t.emplace_back(row, p_unknown_idx(i, j), +1.0 / dx_);
            t.emplace_back(row, p_unknown_idx(i - 1, j), -1.0 / dx_);
        }
    }

    // Momentum for v(i,j), i=0..Nx-1, j=1..Ny-1
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const int row = v_unknown_idx(i, j);
            double diag = inv_dt;

            // x-part with no-slip via ghost at side walls
            if (i == 0 || i == nx_ - 1) {
                diag += 3.0 * nu_ / dx2_;
            } else {
                diag += 2.0 * nu_ / dx2_;
            }
            if (i > 0) {
                t.emplace_back(row, v_unknown_idx(i - 1, j), -nu_ / dx2_);
            }
            if (i < nx_ - 1) {
                t.emplace_back(row, v_unknown_idx(i + 1, j), -nu_ / dx2_);
            }

            // y-part (j=0,Ny are Dirichlet boundaries v=0)
            diag += 2.0 * nu_ / dy2_;
            if (j - 1 >= 1) {
                t.emplace_back(row, v_unknown_idx(i, j - 1), -nu_ / dy2_);
            }
            if (j + 1 <= ny_ - 1) {
                t.emplace_back(row, v_unknown_idx(i, j + 1), -nu_ / dy2_);
            }

            t.emplace_back(row, row, diag);

            // +dp/dy at v location
            t.emplace_back(row, p_unknown_idx(i, j), +1.0 / dy_);
            t.emplace_back(row, p_unknown_idx(i, j - 1), -1.0 / dy_);
        }
    }

    // Continuity at p-cells.
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const int row = nu_unknowns_ + nv_unknowns_ + p_idx(i, j);
            if (i == 0 && j == 0) {
                // Pressure gauge.
                t.emplace_back(row, p_unknown_idx(0, 0), 1.0);
                continue;
            }

            // (u(i+1,j)-u(i,j))/dx
            if (i + 1 <= nx_ - 1) {
                t.emplace_back(row, u_unknown_idx(i + 1, j), +1.0 / dx_);
            }
            if (i >= 1) {
                t.emplace_back(row, u_unknown_idx(i, j), -1.0 / dx_);
            }

            // (v(i,j+1)-v(i,j))/dy
            if (j + 1 <= ny_ - 1) {
                t.emplace_back(row, v_unknown_idx(i, j + 1), +1.0 / dy_);
            }
            if (j >= 1) {
                t.emplace_back(row, v_unknown_idx(i, j), -1.0 / dy_);
            }
        }
    }

    system_mat_.setFromTriplets(t.begin(), t.end());
    system_mat_.makeCompressed();
}

double StokesMac2D::step(double t, ForceFn f1, ForceFn f2) {
    const double inv_dt = 1.0 / dt_;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(total_unknowns_);

    // u-equations RHS 
    for (int j = 0; j < ny_; ++j) {
        const double y = (static_cast<double>(j) + 0.5) * dy_;
        for (int i = 1; i < nx_; ++i) {
            const int row = u_unknown_idx(i, j);
            const double x = static_cast<double>(i) * dx_;
            const double force = (f1 != nullptr) ? f1(x, y, t) : 0.0;

            double b = inv_dt * u_[u_idx(i, j)] + force;
            if (j == ny_ - 1) {
                // Top moving lid contributes via ghost: u_ghost = 2*U_lid - u_inside.
                b += 2.0 * nu_ * u_lid_top_ / dy2_;
            }
            rhs[row] = b;
        }
    }

    // v-equations RHS
    for (int j = 1; j < ny_; ++j) {
        const double y = static_cast<double>(j) * dy_;
        for (int i = 0; i < nx_; ++i) {
            const int row = v_unknown_idx(i, j);
            const double x = (static_cast<double>(i) + 0.5) * dx_;
            const double force = (f2 != nullptr) ? f2(x, y, t) : 0.0;
            rhs[row] = inv_dt * v_[v_idx(i, j)] + force;
        }
    }

    // Continuity RHS is zero, except gauge equation.
    rhs[nu_unknowns_ + nv_unknowns_ + p_idx(0, 0)] = 0.0;

    Eigen::VectorXd sol = system_solver_.solve(rhs);
    if (system_solver_.info() != Eigen::Success) {
        throw std::runtime_error("Monolithic linear solve failed");
    }

    std::vector<double> u_new = u_;
    std::vector<double> v_new = v_;
    std::vector<double> p_new = p_;

    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            u_new[u_idx(i, j)] = sol[u_unknown_idx(i, j)];
        }
    }
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            v_new[v_idx(i, j)] = sol[v_unknown_idx(i, j)];
        }
    }
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            p_new[p_idx(i, j)] = sol[p_unknown_idx(i, j)];
        }
    }

    apply_velocity_bc(u_new, v_new);

    u_.swap(u_new);
    v_.swap(v_new);
    p_.swap(p_new);

    std::vector<double> div(p_.size(), 0.0);
    compute_divergence(u_, v_, div);
    double max_div = 0.0;
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            if (i == 0 && j == 0) {
                continue;
            }
            max_div = std::max(max_div, std::abs(div[p_idx(i, j)]));
        }
    }
    return max_div;
}

extern "C" void* stokes_mac_create_c(int Nx, int Ny, double Lx, double Ly, double nu, double dt) {
    try {
        return new StokesMac2D(Nx, Ny, Lx, Ly, nu, dt);
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
