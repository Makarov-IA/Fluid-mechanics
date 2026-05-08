#include "stokes_mac.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <Eigen/Eigenvalues>
#ifdef _OPENMP
#  include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

StokesMac2D::StokesMac2D(int nx, int ny, double lx, double ly,
                         double nu, double dt)
    : nx_(nx), ny_(ny),
      lx_(lx), ly_(ly),
      nu_(nu), dt_(dt),
      dx_(lx / nx), dy_(ly / ny),
      dx2_(dx_ * dx_), dy2_(dy_ * dy_),
      p_(static_cast<size_t>(nx)   *  ny,      0.0),
      u_(static_cast<size_t>(nx+1) *  ny,      0.0),
      v_(static_cast<size_t>(nx)   * (ny + 1), 0.0),
      nu_unknowns_((nx - 1) * ny),
      nv_unknowns_(nx * (ny - 1)),
      np_unknowns_(nx * ny),
      total_unknowns_((nx - 1) * ny + nx * (ny - 1) + nx * ny),
      system_mat_(total_unknowns_, total_unknowns_)
{
    if (nx <= 1 || ny <= 1)
        throw std::invalid_argument("nx and ny must be > 1");
    if (dt <= 0.0 || lx <= 0.0 || ly <= 0.0 || nu < 0.0)
        throw std::invalid_argument("Invalid physical parameters");

    // Pre-allocate work buffers (reused every step, no heap alloc at runtime)
    adv_u_.assign(u_.size(), 0.0);
    adv_v_.assign(v_.size(), 0.0);
    rhs_.setZero(total_unknowns_);
    sol_.setZero(total_unknowns_);

    // Boundary condition arrays — all zero by default (set via set_bc_arrays)
    bc_u_top_.assign(nx_ - 1, 0.0);
    bc_u_bot_.assign(nx_ - 1, 0.0);
    bc_v_left_.assign(ny_ - 1, 0.0);
    bc_v_right_.assign(ny_ - 1, 0.0);
    bc_u_left_.assign(ny_, 0.0);
    bc_u_right_.assign(ny_, 0.0);
    bc_v_bot_.assign(nx_, 0.0);
    bc_v_top_.assign(nx_, 0.0);

    build_monolithic_system();
    system_solver_.analyzePattern(system_mat_);
    system_solver_.factorize(system_mat_);
    if (system_solver_.info() != Eigen::Success)
        throw std::runtime_error("Monolithic matrix factorisation failed");

    apply_velocity_bc(u_, v_);
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

void StokesMac2D::apply_velocity_bc(std::vector<double>& u,
                                     std::vector<double>& v) const {
    // u on left and right walls (vertical faces i=0, i=Nx)
    for (int j = 0; j < ny_; ++j) {
        u[u_idx(0,   j)] = bc_u_left_[j];
        u[u_idx(nx_, j)] = bc_u_right_[j];
    }
    // v on bottom and top walls (horizontal faces j=0, j=Ny)
    for (int i = 0; i < nx_; ++i) {
        v[v_idx(i, 0  )] = bc_v_bot_[i];
        v[v_idx(i, ny_)] = bc_v_top_[i];
    }
}

// Ghost-node extension of u across horizontal walls (used in advection).
//   j == -1  : bottom wall → u_ghost = 2·bc_u_bot − u(i,0)
//   j == Ny  : top wall    → u_ghost = 2·bc_u_top − u(i,Ny-1)
//   Corner nodes (i=0 or i=Nx) inherit the Dirichlet face value.
double StokesMac2D::u_ghost(const std::vector<double>& u, int i, int j) const {
    if (j >= 0 && j < ny_) return u[u_idx(i, j)];
    if (j == -1) {
        const double bc = (i > 0 && i < nx_) ? bc_u_bot_[i - 1] : 0.0;
        return 2.0 * bc - u[u_idx(i, 0)];
    }
    // j == ny_
    if (i == 0 || i == nx_) return 0.0;
    return 2.0 * bc_u_top_[i - 1] - u[u_idx(i, ny_ - 1)];
}

// Ghost-node extension of v across vertical walls.
//   i == -1  : left wall  → v_ghost = 2·bc_v_left  − v(0,j)
//   i == Nx  : right wall → v_ghost = 2·bc_v_right − v(Nx-1,j)
//   Corner nodes (j=0 or j=Ny) inherit the Dirichlet face value (0 by default).
double StokesMac2D::v_ghost(const std::vector<double>& v, int i, int j) const {
    if (i >= 0 && i < nx_) return v[v_idx(i, j)];
    if (i == -1) {
        const double bc = (j > 0 && j < ny_) ? bc_v_left_[j - 1] : 0.0;
        return 2.0 * bc - v[v_idx(0, j)];
    }
    // i == nx_
    const double bc = (j > 0 && j < ny_) ? bc_v_right_[j - 1] : 0.0;
    return 2.0 * bc - v[v_idx(nx_ - 1, j)];
}

double StokesMac2D::u_ghost_zero_bc(const std::vector<double>& u, int i, int j) const {
    if (j >= 0 && j < ny_) return u[u_idx(i, j)];
    if (j == -1) return -u[u_idx(i, 0)];
    if (i == 0 || i == nx_) return 0.0;
    return -u[u_idx(i, ny_ - 1)];
}

double StokesMac2D::v_ghost_zero_bc(const std::vector<double>& v, int i, int j) const {
    if (i >= 0 && i < nx_) return v[v_idx(i, j)];
    if (i == -1) return -v[v_idx(0, j)];
    return -v[v_idx(nx_ - 1, j)];
}

// ---------------------------------------------------------------------------
// Advection  (explicit, central differences, writes into adv_u_ / adv_v_)
// ---------------------------------------------------------------------------

void StokesMac2D::compute_advection(const std::vector<double>& u,
                                     const std::vector<double>& v) {
    std::fill(adv_u_.begin(), adv_u_.end(), 0.0);
    std::fill(adv_v_.begin(), adv_v_.end(), 0.0);

    // N_u(i,j) = u·∂u/∂x + v_at_u·∂u/∂y   at interior u-faces (i=1..Nx-1)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const double u_ij   = u[u_idx(i, j)];
            const double du_dx  = (u[u_idx(i+1,j)] - u[u_idx(i-1,j)]) / (2.0*dx_);
            const double du_dy  = (u_ghost(u,i,j+1) - u_ghost(u,i,j-1)) / (2.0*dy_);
            // v interpolated to the u-face by bilinear averaging of 4 surrounding v-faces
            const double v_at_u = 0.25 * (v[v_idx(i-1,j  )] + v[v_idx(i,j  )]
                                         + v[v_idx(i-1,j+1)] + v[v_idx(i,j+1)]);
            adv_u_[u_idx(i,j)] = u_ij*du_dx + v_at_u*du_dy;
        }
    }

    // N_v(i,j) = u_at_v·∂v/∂x + v·∂v/∂y   at interior v-faces (j=1..Ny-1)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double v_ij   = v[v_idx(i, j)];
            const double dv_dx  = (v_ghost(v,i+1,j) - v_ghost(v,i-1,j)) / (2.0*dx_);
            const double dv_dy  = (v[v_idx(i,j+1)] - v[v_idx(i,j-1)]) / (2.0*dy_);
            // u interpolated to the v-face by bilinear averaging of 4 surrounding u-faces
            const double u_at_v = 0.25 * (u[u_idx(i,  j-1)] + u[u_idx(i+1,j-1)]
                                         + u[u_idx(i,  j  )] + u[u_idx(i+1,j  )]);
            adv_v_[v_idx(i,j)] = u_at_v*dv_dx + v_ij*dv_dy;
        }
    }
}

double StokesMac2D::scatter_solution_and_measure_velocity_change() {
    double max_u_change = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(max:max_u_change)
    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const double new_u = sol_[u_unknown_idx(i, j)];
            max_u_change = std::max(max_u_change, std::abs(new_u - u_[u_idx(i, j)]));
            u_[u_idx(i, j)] = new_u;
        }
    }

    double max_v_change = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(max:max_v_change)
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double new_v = sol_[v_unknown_idx(i, j)];
            max_v_change = std::max(max_v_change, std::abs(new_v - v_[v_idx(i, j)]));
            v_[v_idx(i, j)] = new_v;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            p_[p_idx(i, j)] = sol_[p_unknown_idx(i, j)];
        }
    }

    apply_velocity_bc(u_, v_);
    return std::max(max_u_change, max_v_change);
}

// ---------------------------------------------------------------------------
// Build the monolithic linear system  A·x = b  (called once in constructor)
//
// Unknown ordering:
//   rows  0 .. nu_unknowns-1       — u-momentum  (interior u-faces)
//   rows  nu_unknowns .. nv_off-1  — v-momentum  (interior v-faces)
//   rows  nv_off .. total-1        — continuity / pressure gauge
//
// A is constant (IMEX with fixed geometry) → factorised once.
// ---------------------------------------------------------------------------

void StokesMac2D::build_monolithic_system() {
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(static_cast<size_t>(total_unknowns_) * 8);

    const double inv_dt = 1.0 / dt_;

    // -------------------------------------------------------------------
    // Block 1: u-momentum   i=1..Nx-1, j=0..Ny-1
    //
    //   (1/dt − ν∇²) u  +  ∂p/∂x  =  rhs_u
    //
    // Ghost-node treatment at horizontal walls:
    //   j=0    (bottom): u_ghost(i,-1) = −u(i,0)                → 3ν/dy² on diagonal
    //   j=Ny-1 (top):    u_ghost(i,Ny) = 2·bc_u_top−u(i,Ny-1) → 3ν/dy² on diagonal;
    //                    wall term  2·ν·bc_u_top/dy²  moves to RHS
    // -------------------------------------------------------------------
    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const int row = u_unknown_idx(i, j);
            double diag = inv_dt;

            // x-diffusion
            diag += 2.0 * nu_ / dx2_;
            if (i-1 >= 1    ) trips.emplace_back(row, u_unknown_idx(i-1,j), -nu_/dx2_);
            if (i+1 <= nx_-1) trips.emplace_back(row, u_unknown_idx(i+1,j), -nu_/dx2_);

            // y-diffusion with ghost-node BC at horizontal walls
            const bool at_bottom = (j == 0);
            const bool at_top    = (j == ny_-1);
            diag += (at_bottom || at_top) ? 3.0*nu_/dy2_ : 2.0*nu_/dy2_;
            if (!at_bottom) trips.emplace_back(row, u_unknown_idx(i,j-1), -nu_/dy2_);
            if (!at_top)    trips.emplace_back(row, u_unknown_idx(i,j+1), -nu_/dy2_);

            trips.emplace_back(row, row, diag);

            // Pressure gradient:  (p(i,j) − p(i-1,j)) / dx
            trips.emplace_back(row, p_unknown_idx(i,  j), +1.0/dx_);
            trips.emplace_back(row, p_unknown_idx(i-1,j), -1.0/dx_);
        }
    }

    // -------------------------------------------------------------------
    // Block 2: v-momentum   i=0..Nx-1, j=1..Ny-1
    //
    //   (1/dt − ν∇²) v  +  ∂p/∂y  =  rhs_v
    //
    // Ghost-node treatment at vertical walls:
    //   i=0    (left):  v_ghost(-1,j) = −v(0,j)      → 3ν/dx² on diagonal
    //   i=Nx-1 (right): v_ghost(Nx,j) = −v(Nx-1,j)   → 3ν/dx² on diagonal
    // j=0 and j=Ny are Dirichlet (v=0) → not unknowns → not included.
    // -------------------------------------------------------------------
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const int row = v_unknown_idx(i, j);
            double diag = inv_dt;

            // x-diffusion with ghost-node BC at vertical walls
            const bool at_left  = (i == 0);
            const bool at_right = (i == nx_-1);
            diag += (at_left || at_right) ? 3.0*nu_/dx2_ : 2.0*nu_/dx2_;
            if (!at_left)  trips.emplace_back(row, v_unknown_idx(i-1,j), -nu_/dx2_);
            if (!at_right) trips.emplace_back(row, v_unknown_idx(i+1,j), -nu_/dx2_);

            // y-diffusion (j=0, j=Ny are Dirichlet → those nodes absent)
            diag += 2.0*nu_/dy2_;
            if (j-1 >= 1    ) trips.emplace_back(row, v_unknown_idx(i,j-1), -nu_/dy2_);
            if (j+1 <= ny_-1) trips.emplace_back(row, v_unknown_idx(i,j+1), -nu_/dy2_);

            trips.emplace_back(row, row, diag);

            // Pressure gradient:  (p(i,j) − p(i,j-1)) / dy
            trips.emplace_back(row, p_unknown_idx(i,j  ), +1.0/dy_);
            trips.emplace_back(row, p_unknown_idx(i,j-1), -1.0/dy_);
        }
    }

    // -------------------------------------------------------------------
    // Block 3: incompressibility  +  pressure gauge at (0,0)
    //
    //   (u(i+1,j)−u(i,j))/dx + (v(i,j+1)−v(i,j))/dy = 0
    //
    // Boundary faces (i=0, i=Nx for u; j=0, j=Ny for v) are prescribed = 0
    // by the Dirichlet BC → their columns are absent from the unknowns.
    // -------------------------------------------------------------------
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const int row = p_unknown_idx(i, j);

            if (i == 0 && j == 0) {
                // Pressure gauge: p(0,0) = 0
                trips.emplace_back(row, p_unknown_idx(0,0), 1.0);
                continue;
            }

            // du/dx contribution at this cell
            if (i+1 <= nx_-1) trips.emplace_back(row, u_unknown_idx(i+1,j), +1.0/dx_);
            if (i    >= 1    ) trips.emplace_back(row, u_unknown_idx(i,  j), -1.0/dx_);

            // dv/dy contribution at this cell
            if (j+1 <= ny_-1) trips.emplace_back(row, v_unknown_idx(i,j+1), +1.0/dy_);
            if (j    >= 1    ) trips.emplace_back(row, v_unknown_idx(i,j  ), -1.0/dy_);
        }
    }

    system_mat_.setFromTriplets(trips.begin(), trips.end());
    system_mat_.makeCompressed();
}

// ---------------------------------------------------------------------------
// Time step
// ---------------------------------------------------------------------------

double StokesMac2D::step(double t, ForceFn f1, ForceFn f2) {
    const double inv_dt = 1.0 / dt_;

    // Explicit convection from the current (old) velocity field → adv_u_, adv_v_
    compute_advection(u_, v_);

    // ------------------------------------------------------------------
    // Assemble RHS
    //   u-block : (1/dt)·u_old − N_u + f1  [+ lid correction at j=Ny-1]
    //   v-block : (1/dt)·v_old − N_v + f2
    //   p-block : 0  (continuity; gauge cell is 0 by construction)
    // ------------------------------------------------------------------
    rhs_.setZero();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const double x     = i * dx_;
            const double y     = (j + 0.5) * dy_;
            const double force = f1 ? f1(x, y, t) : 0.0;
            double b = inv_dt * u_[u_idx(i,j)] - adv_u_[u_idx(i,j)] + force;
            // Ghost-node BC corrections: 2·ν·bc/dy² at boundary rows
            if (j == ny_-1) b += 2.0 * nu_ * bc_u_top_[i-1] / dy2_;
            if (j == 0)     b += 2.0 * nu_ * bc_u_bot_[i-1] / dy2_;
            rhs_[u_unknown_idx(i, j)] = b;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double force = f2 ? f2((i + 0.5) * dx_, j * dy_, t) : 0.0;
            double b = inv_dt * v_[v_idx(i,j)] - adv_v_[v_idx(i,j)] + force;
            // Ghost-node BC corrections: 2·ν·bc/dx² at boundary columns
            if (i == 0)      b += 2.0 * nu_ * bc_v_left_[j-1]  / dx2_;
            if (i == nx_-1)  b += 2.0 * nu_ * bc_v_right_[j-1] / dx2_;
            rhs_[v_unknown_idx(i, j)] = b;
        }
    }
    // rhs_ for the gauge row stays 0 (p(0,0) = 0)

    // ------------------------------------------------------------------
    // Solve  A·x = rhs  (factorisation already done in constructor)
    // ------------------------------------------------------------------
    sol_.noalias() = system_solver_.solve(rhs_);
    if (system_solver_.info() != Eigen::Success)
        throw std::runtime_error("Monolithic linear solve failed");

    last_velocity_change_ = scatter_solution_and_measure_velocity_change();

    return max_divergence();
}

void StokesMac2D::run_steps(double t_start, int n_steps, double* div_out) {
    run_steps_diagnostics(t_start, n_steps, div_out, nullptr);
}

// ---------------------------------------------------------------------------
// step_with_force_arrays — same as step() but force comes from pre-evaluated arrays
// ---------------------------------------------------------------------------

double StokesMac2D::step_with_force_arrays(double t,
                                            const double* fu,
                                            const double* fv) {
    const double inv_dt = 1.0 / dt_;
    compute_advection(u_, v_);

    rhs_.setZero();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const double force = fu ? fu[j * (nx_ - 1) + (i - 1)] : 0.0;
            double b = inv_dt * u_[u_idx(i,j)] - adv_u_[u_idx(i,j)] + force;
            if (j == ny_-1) b += 2.0 * nu_ * bc_u_top_[i-1] / dy2_;
            if (j == 0)     b += 2.0 * nu_ * bc_u_bot_[i-1] / dy2_;
            rhs_[u_unknown_idx(i, j)] = b;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double force = fv ? fv[(j - 1) * nx_ + i] : 0.0;
            double b = inv_dt * v_[v_idx(i,j)] - adv_v_[v_idx(i,j)] + force;
            if (i == 0)      b += 2.0 * nu_ * bc_v_left_[j-1]  / dx2_;
            if (i == nx_-1)  b += 2.0 * nu_ * bc_v_right_[j-1] / dx2_;
            rhs_[v_unknown_idx(i, j)] = b;
        }
    }

    sol_.noalias() = system_solver_.solve(rhs_);
    if (system_solver_.info() != Eigen::Success)
        throw std::runtime_error("Monolithic linear solve failed");

    last_velocity_change_ = scatter_solution_and_measure_velocity_change();
    return max_divergence();
}

void StokesMac2D::run_steps_with_force(double t_start, int n_steps,
                                        const double* fu, const double* fv,
                                        double* div_out) {
    run_steps_with_force_diagnostics(t_start, n_steps, fu, fv, div_out, nullptr);
}

void StokesMac2D::run_steps_diagnostics(double t_start, int n_steps,
                                         double* div_out,
                                         double* change_out) {
    for (int k = 0; k < n_steps; ++k) {
        div_out[k] = step(t_start + (k + 1) * dt_, nullptr, nullptr);
        if (change_out) {
            change_out[k] = last_velocity_change_;
        }
    }
}

void StokesMac2D::run_steps_with_force_diagnostics(double t_start, int n_steps,
                                                    const double* fu, const double* fv,
                                                    double* div_out,
                                                    double* change_out) {
    for (int k = 0; k < n_steps; ++k) {
        div_out[k] = step_with_force_arrays(t_start + (k + 1) * dt_, fu, fv);
        if (change_out) {
            change_out[k] = last_velocity_change_;
        }
    }
}

// ---------------------------------------------------------------------------
// set_bc_arrays
// ---------------------------------------------------------------------------

void StokesMac2D::set_bc_arrays(const double* u_top,   const double* u_bot,
                                 const double* v_left,  const double* v_right,
                                 const double* u_left,  const double* u_right,
                                 const double* v_bot,   const double* v_top) {
    if (u_top)   std::copy(u_top,   u_top   + (nx_-1), bc_u_top_.begin());
    if (u_bot)   std::copy(u_bot,   u_bot   + (nx_-1), bc_u_bot_.begin());
    if (v_left)  std::copy(v_left,  v_left  + (ny_-1), bc_v_left_.begin());
    if (v_right) std::copy(v_right, v_right + (ny_-1), bc_v_right_.begin());
    if (u_left)  std::copy(u_left,  u_left  + ny_,     bc_u_left_.begin());
    if (u_right) std::copy(u_right, u_right + ny_,     bc_u_right_.begin());
    if (v_bot)   std::copy(v_bot,   v_bot   + nx_,     bc_v_bot_.begin());
    if (v_top)   std::copy(v_top,   v_top   + nx_,     bc_v_top_.begin());
    apply_velocity_bc(u_, v_);
}

// ---------------------------------------------------------------------------
// Nonlinear state import/export
// ---------------------------------------------------------------------------

void StokesMac2D::set_state_arrays(const double* u_interior,
                                    const double* v_interior,
                                    const double* p_cells) {
    if (!u_interior || !v_interior)
        throw std::invalid_argument("set_state_arrays requires u and v arrays");

    for (int j = 0; j < ny_; ++j)
        for (int i = 1; i < nx_; ++i)
            u_[u_idx(i, j)] = u_interior[u_unknown_idx(i, j)];

    for (int j = 1; j < ny_; ++j)
        for (int i = 0; i < nx_; ++i)
            v_[v_idx(i, j)] = v_interior[v_unknown_idx(i, j) - nu_unknowns_];

    if (p_cells) {
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i)
                p_[p_idx(i, j)] = p_cells[p_idx(i, j)];
    } else {
        std::fill(p_.begin(), p_.end(), 0.0);
    }

    apply_velocity_bc(u_, v_);
}

void StokesMac2D::get_state_arrays(double* u_interior,
                                    double* v_interior,
                                    double* p_cells) const {
    if (u_interior) {
        for (int j = 0; j < ny_; ++j)
            for (int i = 1; i < nx_; ++i)
                u_interior[u_unknown_idx(i, j)] = u_[u_idx(i, j)];
    }

    if (v_interior) {
        for (int j = 1; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i)
                v_interior[v_unknown_idx(i, j) - nu_unknowns_] = v_[v_idx(i, j)];
    }

    if (p_cells) {
        std::copy(p_.begin(), p_.end(), p_cells);
    }
}

// ---------------------------------------------------------------------------
// Divergence diagnostic
// ---------------------------------------------------------------------------

double StokesMac2D::max_divergence() const {
    double max_div = 0.0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(max:max_div)
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            if (i == 0 && j == 0) continue;  // gauge cell — skip
            const double div = (u_[u_idx(i+1,j)] - u_[u_idx(i,j)]) / dx_
                             + (v_[v_idx(i,j+1)] - v_[v_idx(i,j)]) / dy_;
            max_div = std::max(max_div, std::abs(div));
        }
    }
    return max_div;
}

double StokesMac2D::steady_residual_inf(const double* fu, const double* fv) const {
    double max_res = 0.0;

    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const double u_ij = u_[u_idx(i, j)];
            const double du_dx = (u_[u_idx(i + 1, j)] - u_[u_idx(i - 1, j)]) / (2.0 * dx_);
            const double du_dy = (u_ghost(u_, i, j + 1) - u_ghost(u_, i, j - 1)) / (2.0 * dy_);
            const double v_at_u = 0.25 * (
                v_[v_idx(i - 1, j)] + v_[v_idx(i, j)] +
                v_[v_idx(i - 1, j + 1)] + v_[v_idx(i, j + 1)]
            );
            const double adv = u_ij * du_dx + v_at_u * du_dy;
            const double lap = (
                (u_[u_idx(i + 1, j)] - 2.0 * u_ij + u_[u_idx(i - 1, j)]) / dx2_ +
                (u_ghost(u_, i, j + 1) - 2.0 * u_ij + u_ghost(u_, i, j - 1)) / dy2_
            );
            const double dp_dx = (p_[p_idx(i, j)] - p_[p_idx(i - 1, j)]) / dx_;
            const double force = fu ? fu[j * (nx_ - 1) + (i - 1)] : 0.0;
            max_res = std::max(max_res, std::abs(adv - nu_ * lap + dp_dx - force));
        }
    }

    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double v_ij = v_[v_idx(i, j)];
            const double dv_dx = (v_ghost(v_, i + 1, j) - v_ghost(v_, i - 1, j)) / (2.0 * dx_);
            const double dv_dy = (v_[v_idx(i, j + 1)] - v_[v_idx(i, j - 1)]) / (2.0 * dy_);
            const double u_at_v = 0.25 * (
                u_[u_idx(i, j - 1)] + u_[u_idx(i + 1, j - 1)] +
                u_[u_idx(i, j)] + u_[u_idx(i + 1, j)]
            );
            const double adv = u_at_v * dv_dx + v_ij * dv_dy;
            const double lap = (
                (v_ghost(v_, i + 1, j) - 2.0 * v_ij + v_ghost(v_, i - 1, j)) / dx2_ +
                (v_[v_idx(i, j + 1)] - 2.0 * v_ij + v_[v_idx(i, j - 1)]) / dy2_
            );
            const double dp_dy = (p_[p_idx(i, j)] - p_[p_idx(i, j - 1)]) / dy_;
            const double force = fv ? fv[(j - 1) * nx_ + i] : 0.0;
            max_res = std::max(max_res, std::abs(adv - nu_ * lap + dp_dy - force));
        }
    }

    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double value = (i == 0 && j == 0)
                ? p_[p_idx(0, 0)]
                : (u_[u_idx(i + 1, j)] - u_[u_idx(i, j)]) / dx_
                    + (v_[v_idx(i, j + 1)] - v_[v_idx(i, j)]) / dy_;
            max_res = std::max(max_res, std::abs(value));
        }
    }

    return max_res;
}

int StokesMac2D::solve_steady_newton(int max_newton_iters,
                                     double residual_tol,
                                     double krylov_tol,
                                     int krylov_maxiter,
                                     int krylov_restart,
                                     double jacobian_rdiff,
                                     const char* line_search,
                                     double min_step,
                                     const double* fu,
                                     const double* fv,
                                     int* newton_iters,
                                     double* residual_inf,
                                     double* max_div,
                                     int* converged,
                                     int* stop_code,
                                     double* iterate_change_inf,
                                     int* iterate_change_count) {
    try {
        const int velocity_size = nu_unknowns_ + nv_unknowns_;
        if (max_newton_iters <= 0 || krylov_maxiter <= 0 || krylov_restart <= 0) return -2;
        if (residual_tol <= 0.0 || krylov_tol <= 0.0 || jacobian_rdiff <= 0.0) return -2;
        if (!(min_step > 0.0 && min_step <= 1.0)) return -2;

        struct FixedPointEval {
            Eigen::VectorXd residual;
            Eigen::VectorXd next_state;
            Eigen::VectorXd pressure;
            double residual_inf = 0.0;
            double max_div = 0.0;
        };

        auto inf_norm = [](const Eigen::VectorXd& vec) -> double {
            return vec.size() == 0 ? 0.0 : vec.cwiseAbs().maxCoeff();
        };

        auto pack_velocity_state = [&]() -> Eigen::VectorXd {
            Eigen::VectorXd state(velocity_size);
            for (int j = 0; j < ny_; ++j) {
                for (int i = 1; i < nx_; ++i) {
                    state[u_unknown_idx(i, j)] = u_[u_idx(i, j)];
                }
            }
            for (int j = 1; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    state[v_unknown_idx(i, j)] = v_[v_idx(i, j)];
                }
            }
            return state;
        };

        auto set_velocity_state = [&](const Eigen::VectorXd& state) {
            for (int j = 0; j < ny_; ++j) {
                for (int i = 1; i < nx_; ++i) {
                    u_[u_idx(i, j)] = state[u_unknown_idx(i, j)];
                }
            }
            for (int j = 1; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    v_[v_idx(i, j)] = state[v_unknown_idx(i, j)];
                }
            }
            std::fill(p_.begin(), p_.end(), 0.0);
            apply_velocity_bc(u_, v_);
        };

        auto pack_pressure = [&]() -> Eigen::VectorXd {
            Eigen::VectorXd pressure(np_unknowns_);
            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    pressure[p_idx(i, j)] = p_[p_idx(i, j)];
                }
            }
            return pressure;
        };

        auto restore_eval = [&](const FixedPointEval& eval) {
            set_velocity_state(eval.next_state);
            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    p_[p_idx(i, j)] = eval.pressure[p_idx(i, j)];
                }
            }
            apply_velocity_bc(u_, v_);
        };

        auto evaluate = [&](const Eigen::VectorXd& state) -> FixedPointEval {
            set_velocity_state(state);
            const double div = step_with_force_arrays(0.0, fu, fv);
            FixedPointEval eval;
            eval.next_state = pack_velocity_state();
            eval.pressure = pack_pressure();
            eval.residual = eval.next_state - state;
            eval.residual_inf = inf_norm(eval.residual);
            eval.max_div = div;
            if (!eval.next_state.allFinite() || !eval.residual.allFinite() ||
                !eval.pressure.allFinite() || !std::isfinite(div)) {
                throw std::runtime_error("Non-finite value in fixed-point evaluation");
            }
            return eval;
        };

        auto jacobian_vec = [&](const Eigen::VectorXd& base_state,
                                const Eigen::VectorXd& base_residual,
                                const Eigen::VectorXd& vec) -> Eigen::VectorXd {
            const double vec_inf = inf_norm(vec);
            if (vec_inf < 1e-14) {
                return Eigen::VectorXd::Zero(velocity_size);
            }
            const double state_inf = std::max(inf_norm(base_state), 1.0);
            const double eps = jacobian_rdiff * state_inf / vec_inf;
            FixedPointEval trial = evaluate(base_state + eps * vec);
            return (trial.residual - base_residual) / eps;
        };

        auto gmres_solve = [&](const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& apply,
                               const Eigen::VectorXd& rhs,
                               Eigen::VectorXd& solution) -> int {
            const int n = static_cast<int>(rhs.size());
            const int restart = std::max(1, std::min(krylov_restart, n));
            solution = Eigen::VectorXd::Zero(n);

            const double rhs_norm = rhs.norm();
            if (rhs_norm == 0.0) {
                return 0;
            }

            const double target = krylov_tol * rhs_norm;
            Eigen::VectorXd residual = rhs;
            double beta = residual.norm();
            int iterations = 0;

            while (iterations < krylov_maxiter && beta > target) {
                const int inner_max = std::min(restart, krylov_maxiter - iterations);
                std::vector<Eigen::VectorXd> basis(static_cast<size_t>(inner_max + 1),
                                                   Eigen::VectorXd::Zero(n));
                Eigen::MatrixXd hessenberg = Eigen::MatrixXd::Zero(inner_max + 1, inner_max);
                Eigen::VectorXd rhs_small = Eigen::VectorXd::Zero(inner_max + 1);
                rhs_small[0] = beta;
                basis[0] = residual / beta;

                Eigen::VectorXd best_solution = solution;
                double best_residual = beta;

                for (int j = 0; j < inner_max; ++j) {
                    Eigen::VectorXd w = apply(basis[static_cast<size_t>(j)]);
                    if (!w.allFinite()) return -1;

                    for (int i = 0; i <= j; ++i) {
                        const double hij = basis[static_cast<size_t>(i)].dot(w);
                        hessenberg(i, j) = hij;
                        w -= hij * basis[static_cast<size_t>(i)];
                    }
                    // One re-orthogonalization pass is cheap here and keeps GMRES stable.
                    for (int i = 0; i <= j; ++i) {
                        const double corr = basis[static_cast<size_t>(i)].dot(w);
                        hessenberg(i, j) += corr;
                        w -= corr * basis[static_cast<size_t>(i)];
                    }

                    const double h_next = w.norm();
                    hessenberg(j + 1, j) = h_next;
                    if (h_next > 0.0 && j + 1 < inner_max + 1) {
                        basis[static_cast<size_t>(j + 1)] = w / h_next;
                    }

                    const Eigen::MatrixXd h_small = hessenberg.block(0, 0, j + 2, j + 1);
                    const Eigen::VectorXd g_small = rhs_small.head(j + 2);
                    const Eigen::VectorXd y = h_small.colPivHouseholderQr().solve(g_small);

                    Eigen::VectorXd candidate = solution;
                    for (int i = 0; i <= j; ++i) {
                        candidate += y[i] * basis[static_cast<size_t>(i)];
                    }

                    best_solution = candidate;
                    best_residual = (g_small - h_small * y).norm();
                    ++iterations;

                    if (best_residual <= target) {
                        solution = best_solution;
                        return 0;
                    }
                    if (h_next == 0.0) {
                        break;
                    }
                }

                solution = best_solution;
                residual = rhs - apply(solution);
                if (!residual.allFinite()) return -1;
                beta = residual.norm();
            }

            return beta <= target ? 0 : 1;
        };

        Eigen::VectorXd state = pack_velocity_state();
        FixedPointEval current = evaluate(state);
        int completed_newton_iters = 0;
        int change_count = 0;
        int local_converged = 0;
        int local_stop_code = 0;  // 1 converged, 2 GMRES failed, 3 line search failed, 4 iter limit.
        const bool no_line_search = line_search && std::strcmp(line_search, "none") == 0;

        for (int k = 1; k <= max_newton_iters; ++k) {
            if (current.residual_inf < residual_tol) {
                local_converged = 1;
                local_stop_code = 1;
                break;
            }

            const Eigen::VectorXd base_state = state;
            const Eigen::VectorXd base_residual = current.residual;
            const auto apply_jacobian = [&](const Eigen::VectorXd& vec) -> Eigen::VectorXd {
                return jacobian_vec(base_state, base_residual, vec);
            };

            Eigen::VectorXd delta;
            const int gmres_info = gmres_solve(apply_jacobian, -base_residual, delta);
            if (gmres_info < 0 || !delta.allFinite()) {
                local_stop_code = 2;
                break;
            }

            double alpha = 1.0;
            bool accepted = false;
            Eigen::VectorXd accepted_state;
            FixedPointEval accepted_eval;

            if (no_line_search) {
                accepted_state = base_state + delta;
                accepted_eval = evaluate(accepted_state);
                accepted = true;
            }

            while (!accepted && alpha >= min_step) {
                Eigen::VectorXd trial_state = base_state + alpha * delta;
                FixedPointEval trial_eval = evaluate(trial_state);
                const double target = (1.0 - 1e-4 * alpha) * current.residual_inf;
                if (trial_eval.residual_inf < target) {
                    accepted_state = std::move(trial_state);
                    accepted_eval = std::move(trial_eval);
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }

            if (!accepted) {
                local_stop_code = 3;
                break;
            }

            const double change_inf = inf_norm(accepted_state - base_state);
            state = std::move(accepted_state);
            current = std::move(accepted_eval);
            completed_newton_iters = k;
            if (iterate_change_inf && change_count < max_newton_iters) {
                iterate_change_inf[change_count] = change_inf;
            }
            ++change_count;
        }

        if (!local_converged && local_stop_code == 0) {
            if (current.residual_inf < residual_tol) {
                local_converged = 1;
                local_stop_code = 1;
            } else {
                local_stop_code = 4;
            }
        }

        restore_eval(current);
        if (newton_iters) *newton_iters = completed_newton_iters;
        if (residual_inf) *residual_inf = current.residual_inf;
        if (max_div) *max_div = current.max_div;
        if (converged) *converged = local_converged;
        if (stop_code) *stop_code = local_stop_code;
        if (iterate_change_count) *iterate_change_count = change_count;
        return 0;
    } catch (...) {
        return -99;
    }
}

Eigen::SparseMatrix<double> StokesMac2D::build_projection_matrix() const {
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(static_cast<size_t>(total_unknowns_) * 5);

    for (int row = 0; row < nu_unknowns_ + nv_unknowns_; ++row) {
        trips.emplace_back(row, row, 1.0);
    }

    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const int row = u_unknown_idx(i, j);
            trips.emplace_back(row, p_unknown_idx(i, j), +1.0 / dx_);
            trips.emplace_back(row, p_unknown_idx(i - 1, j), -1.0 / dx_);
        }
    }

    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const int row = v_unknown_idx(i, j);
            trips.emplace_back(row, p_unknown_idx(i, j), +1.0 / dy_);
            trips.emplace_back(row, p_unknown_idx(i, j - 1), -1.0 / dy_);
        }
    }

    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const int row = p_unknown_idx(i, j);
            if (i == 0 && j == 0) {
                trips.emplace_back(row, p_unknown_idx(0, 0), 1.0);
                continue;
            }
            if (i + 1 <= nx_ - 1) trips.emplace_back(row, u_unknown_idx(i + 1, j), +1.0 / dx_);
            if (i >= 1)           trips.emplace_back(row, u_unknown_idx(i, j),     -1.0 / dx_);
            if (j + 1 <= ny_ - 1) trips.emplace_back(row, v_unknown_idx(i, j + 1), +1.0 / dy_);
            if (j >= 1)           trips.emplace_back(row, v_unknown_idx(i, j),     -1.0 / dy_);
        }
    }

    Eigen::SparseMatrix<double> matrix(total_unknowns_, total_unknowns_);
    matrix.setFromTriplets(trips.begin(), trips.end());
    matrix.makeCompressed();
    return matrix;
}

Eigen::VectorXd StokesMac2D::project_velocity_rhs(
    const SparseSystemSolver& solver,
    const Eigen::VectorXd& raw,
    Eigen::VectorXd* pressure
) const {
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(total_unknowns_);
    rhs.head(nu_unknowns_ + nv_unknowns_) = raw;
    Eigen::VectorXd solved = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Pressure projection solve failed");
    }
    if (pressure) {
        *pressure = solved.tail(np_unknowns_);
    }
    return solved.head(nu_unknowns_ + nv_unknowns_);
}

Eigen::VectorXd StokesMac2D::linearized_raw_velocity_action(const Eigen::VectorXd& velocity) const {
    const int velocity_size = nu_unknowns_ + nv_unknowns_;
    if (velocity.size() != velocity_size) {
        throw std::invalid_argument("linearized_raw_velocity_action got wrong vector size");
    }

    std::vector<double> a(static_cast<size_t>(nx_ + 1) * ny_, 0.0);
    std::vector<double> b(static_cast<size_t>(nx_) * (ny_ + 1), 0.0);

    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            a[u_idx(i, j)] = velocity[u_unknown_idx(i, j)];
        }
    }
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            b[v_idx(i, j)] = velocity[v_unknown_idx(i, j) - nu_unknowns_];
        }
    }

    Eigen::VectorXd raw(velocity_size);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < ny_; ++j) {
        for (int i = 1; i < nx_; ++i) {
            const double a_ij = a[u_idx(i, j)];
            const double da_dx = (a[u_idx(i + 1, j)] - a[u_idx(i - 1, j)]) / (2.0 * dx_);
            const double da_dy = (
                u_ghost_zero_bc(a, i, j + 1) - u_ghost_zero_bc(a, i, j - 1)
            ) / (2.0 * dy_);
            const double b_at_u = 0.25 * (
                b[v_idx(i - 1, j)] + b[v_idx(i, j)] +
                b[v_idx(i - 1, j + 1)] + b[v_idx(i, j + 1)]
            );

            const double u0_ij = u_[u_idx(i, j)];
            const double du0_dx = (u_[u_idx(i + 1, j)] - u_[u_idx(i - 1, j)]) / (2.0 * dx_);
            const double du0_dy = (
                u_ghost(u_, i, j + 1) - u_ghost(u_, i, j - 1)
            ) / (2.0 * dy_);
            const double v0_at_u = 0.25 * (
                v_[v_idx(i - 1, j)] + v_[v_idx(i, j)] +
                v_[v_idx(i - 1, j + 1)] + v_[v_idx(i, j + 1)]
            );

            const double delta_adv = a_ij * du0_dx + u0_ij * da_dx
                                   + b_at_u * du0_dy + v0_at_u * da_dy;
            const double lap = (
                (a[u_idx(i + 1, j)] - 2.0 * a_ij + a[u_idx(i - 1, j)]) / dx2_ +
                (u_ghost_zero_bc(a, i, j + 1) - 2.0 * a_ij + u_ghost_zero_bc(a, i, j - 1)) / dy2_
            );
            raw[u_unknown_idx(i, j)] = -(delta_adv - nu_ * lap);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double b_ij = b[v_idx(i, j)];
            const double db_dx = (
                v_ghost_zero_bc(b, i + 1, j) - v_ghost_zero_bc(b, i - 1, j)
            ) / (2.0 * dx_);
            const double db_dy = (b[v_idx(i, j + 1)] - b[v_idx(i, j - 1)]) / (2.0 * dy_);
            const double a_at_v = 0.25 * (
                a[u_idx(i, j - 1)] + a[u_idx(i + 1, j - 1)] +
                a[u_idx(i, j)] + a[u_idx(i + 1, j)]
            );

            const double v0_ij = v_[v_idx(i, j)];
            const double dv0_dx = (
                v_ghost(v_, i + 1, j) - v_ghost(v_, i - 1, j)
            ) / (2.0 * dx_);
            const double dv0_dy = (v_[v_idx(i, j + 1)] - v_[v_idx(i, j - 1)]) / (2.0 * dy_);
            const double u0_at_v = 0.25 * (
                u_[u_idx(i, j - 1)] + u_[u_idx(i + 1, j - 1)] +
                u_[u_idx(i, j)] + u_[u_idx(i + 1, j)]
            );

            const double delta_adv = a_at_v * dv0_dx + u0_at_v * db_dx
                                   + b_ij * dv0_dy + v0_ij * db_dy;
            const double lap = (
                (v_ghost_zero_bc(b, i + 1, j) - 2.0 * b_ij + v_ghost_zero_bc(b, i - 1, j)) / dx2_ +
                (b[v_idx(i, j + 1)] - 2.0 * b_ij + b[v_idx(i, j - 1)]) / dy2_
            );
            raw[v_unknown_idx(i, j)] = -(delta_adv - nu_ * lap);
        }
    }

    return raw;
}

int StokesMac2D::solve_linearized_eigenmodes(int n_eigs,
                                             const char* which,
                                             const double* fu,
                                             const double* fv,
                                             double* eig_real,
                                             double* eig_imag,
                                             double* vec_real,
                                             double* vec_imag,
                                             double* base_residual_inf,
                                             long long* matvec_count,
                                             long long* dense_operator_bytes) const {
    try {
        const int velocity_size = nu_unknowns_ + nv_unknowns_;
        if (n_eigs <= 0 || n_eigs > velocity_size) return -2;
        if (!eig_real || !eig_imag || !vec_real || !vec_imag) return -3;
        const char* selector = which ? which : "LR";

        if (base_residual_inf) {
            *base_residual_inf = steady_residual_inf(fu, fv);
        }

        Eigen::SparseMatrix<double> projection_matrix = build_projection_matrix();
        SparseSystemSolver projection_solver;
        projection_solver.analyzePattern(projection_matrix);
        projection_solver.factorize(projection_matrix);
        if (projection_solver.info() != Eigen::Success) {
            return -4;
        }

        auto apply_operator = [&](const Eigen::VectorXd& vec) -> Eigen::VectorXd {
            Eigen::VectorXd raw = linearized_raw_velocity_action(vec);
            return project_velocity_rhs(projection_solver, raw, nullptr);
        };

        auto less_for_selector = [&](const Eigen::VectorXcd& values, int a, int b) {
            const std::complex<double> va = values[a];
            const std::complex<double> vb = values[b];
            if (std::strcmp(selector, "LM") == 0) return std::abs(va) > std::abs(vb);
            if (std::strcmp(selector, "SM") == 0) return std::abs(va) < std::abs(vb);
            if (std::strcmp(selector, "LR") == 0) return va.real() > vb.real();
            if (std::strcmp(selector, "SR") == 0) return va.real() < vb.real();
            if (std::strcmp(selector, "LI") == 0) return va.imag() > vb.imag();
            return va.imag() < vb.imag();
        };

        auto write_selected_modes = [&](const Eigen::VectorXcd& values,
                                        const Eigen::MatrixXcd& vectors) -> int {
            if (values.size() < n_eigs || vectors.cols() < n_eigs) {
                return -6;
            }

            std::vector<int> order(static_cast<size_t>(values.size()));
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return less_for_selector(values, a, b);
            });

            for (int mode = 0; mode < n_eigs; ++mode) {
                const int src = order[static_cast<size_t>(mode)];
                std::complex<double> lambda = values[src];
                Eigen::VectorXcd velocity_mode = vectors.col(src);

                double max_norm = 0.0;
                for (int row = 0; row < velocity_size; ++row) {
                    max_norm = std::max(max_norm, std::abs(velocity_mode[row]));
                }
                if (max_norm > 0.0) {
                    velocity_mode /= max_norm;
                }

                Eigen::VectorXd mode_real = velocity_mode.real();
                Eigen::VectorXd raw_real = linearized_raw_velocity_action(mode_real);
                Eigen::VectorXd pressure_real;
                project_velocity_rhs(projection_solver, raw_real, &pressure_real);

                Eigen::VectorXd pressure_imag = Eigen::VectorXd::Zero(np_unknowns_);
                bool has_imag = false;
                for (int row = 0; row < velocity_size; ++row) {
                    if (std::abs(velocity_mode[row].imag()) > 0.0) {
                        has_imag = true;
                        break;
                    }
                }
                if (has_imag) {
                    Eigen::VectorXd mode_imag = velocity_mode.imag();
                    Eigen::VectorXd raw_imag = linearized_raw_velocity_action(mode_imag);
                    project_velocity_rhs(projection_solver, raw_imag, &pressure_imag);
                }

                eig_real[mode] = lambda.real();
                eig_imag[mode] = lambda.imag();
                const int full_size = total_unknowns_;
                for (int row = 0; row < velocity_size; ++row) {
                    const int out = mode * full_size + row;
                    vec_real[out] = velocity_mode[row].real();
                    vec_imag[out] = velocity_mode[row].imag();
                }
                for (int row = 0; row < np_unknowns_; ++row) {
                    const int out = mode * full_size + velocity_size + row;
                    vec_real[out] = pressure_real[row];
                    vec_imag[out] = pressure_imag[row];
                }
            }
            return 0;
        };

        constexpr int dense_velocity_limit = 1200;
        long long count = 0;

        if (velocity_size <= dense_velocity_limit) {
            const long long dense_bytes = static_cast<long long>(velocity_size)
                                        * static_cast<long long>(velocity_size)
                                        * static_cast<long long>(sizeof(double));
            if (dense_operator_bytes) {
                *dense_operator_bytes = dense_bytes;
            }

            Eigen::MatrixXd dense(velocity_size, velocity_size);
            Eigen::VectorXd basis = Eigen::VectorXd::Zero(velocity_size);
            for (int col = 0; col < velocity_size; ++col) {
                basis[col] = 1.0;
                dense.col(col) = apply_operator(basis);
                basis[col] = 0.0;
                ++count;
            }
            if (matvec_count) {
                *matvec_count = count;
            }

            Eigen::EigenSolver<Eigen::MatrixXd> eig_solver(dense, true);
            if (eig_solver.info() != Eigen::Success) {
                return -5;
            }
            return write_selected_modes(eig_solver.eigenvalues(), eig_solver.eigenvectors());
        }

        const int krylov_dim = std::min(
            velocity_size,
            std::max(80, 4 * n_eigs + 20)
        );
        const long long arnoldi_bytes =
            static_cast<long long>(velocity_size) * static_cast<long long>(krylov_dim + 1)
            * static_cast<long long>(sizeof(double))
            + static_cast<long long>(krylov_dim + 1) * static_cast<long long>(krylov_dim)
            * static_cast<long long>(sizeof(double));
        if (dense_operator_bytes) {
            *dense_operator_bytes = arnoldi_bytes;
        }

        Eigen::MatrixXd basis(velocity_size, krylov_dim + 1);
        Eigen::MatrixXd hessenberg = Eigen::MatrixXd::Zero(krylov_dim + 1, krylov_dim);
        Eigen::VectorXd start(velocity_size);
        for (int i = 0; i < velocity_size; ++i) {
            start[i] = std::sin(0.17 * static_cast<double>(i + 1))
                     + 0.5 * std::cos(0.11 * static_cast<double>(i + 1));
        }
        double start_norm = start.norm();
        if (start_norm == 0.0) {
            start.setZero();
            start[0] = 1.0;
            start_norm = 1.0;
        }
        basis.col(0) = start / start_norm;

        int actual_dim = 0;
        for (int j = 0; j < krylov_dim; ++j) {
            Eigen::VectorXd w = apply_operator(basis.col(j));
            ++count;

            for (int i = 0; i <= j; ++i) {
                const double hij = basis.col(i).dot(w);
                hessenberg(i, j) = hij;
                w -= hij * basis.col(i);
            }
            // Re-orthogonalize once to reduce loss of orthogonality in long runs.
            for (int i = 0; i <= j; ++i) {
                const double corr = basis.col(i).dot(w);
                hessenberg(i, j) += corr;
                w -= corr * basis.col(i);
            }

            const double h_next = w.norm();
            hessenberg(j + 1, j) = h_next;
            actual_dim = j + 1;
            if (h_next < 1e-12) {
                break;
            }
            if (j + 1 < krylov_dim + 1) {
                basis.col(j + 1) = w / h_next;
            }
        }

        if (matvec_count) {
            *matvec_count = count;
        }
        if (actual_dim < n_eigs) {
            return -6;
        }

        Eigen::MatrixXd small_operator = hessenberg.block(0, 0, actual_dim, actual_dim);
        Eigen::EigenSolver<Eigen::MatrixXd> small_solver(small_operator, true);
        if (small_solver.info() != Eigen::Success) {
            return -5;
        }

        Eigen::MatrixXcd velocity_vectors =
            basis.leftCols(actual_dim).cast<std::complex<double>>() * small_solver.eigenvectors();
        return write_selected_modes(small_solver.eigenvalues(), velocity_vectors);
    } catch (...) {
        return -99;
    }
}

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" void* stokes_mac_create_c(int Nx, int Ny,
                                     double Lx, double Ly,
                                     double nu, double dt) {
    try {
        return new StokesMac2D(Nx, Ny, Lx, Ly, nu, dt);
    } catch (...) {
        return nullptr;
    }
}

extern "C" void stokes_mac_free_c(void* handle) {
    delete reinterpret_cast<StokesMac2D*>(handle);
}

extern "C" double stokes_mac_step_c(void* handle, double t,
                                    Force2DTime_C f1, Force2DTime_C f2) {
    if (!handle) return -1.0;
    return reinterpret_cast<StokesMac2D*>(handle)->step(t, f1, f2);
}

extern "C" void stokes_mac_run_steps_c(void* handle, double t_start,
                                       int n_steps, double* div_out) {
    if (!handle || !div_out) return;
    reinterpret_cast<StokesMac2D*>(handle)->run_steps(t_start, n_steps, div_out);
}

extern "C" void stokes_mac_run_steps_diagnostics_c(void* handle, double t_start,
                                                   int n_steps, double* div_out,
                                                   double* change_out) {
    if (!handle || !div_out || !change_out) return;
    reinterpret_cast<StokesMac2D*>(handle)->run_steps_diagnostics(
        t_start, n_steps, div_out, change_out
    );
}

extern "C" void stokes_mac_set_bc_c(void* handle,
                                    const double* u_top,   const double* u_bot,
                                    const double* v_left,  const double* v_right,
                                    const double* u_left,  const double* u_right,
                                    const double* v_bot,   const double* v_top) {
    if (!handle) return;
    reinterpret_cast<StokesMac2D*>(handle)->set_bc_arrays(
        u_top, u_bot, v_left, v_right, u_left, u_right, v_bot, v_top);
}

extern "C" void stokes_mac_run_steps_with_force_c(void* handle,
                                                   double t_start, int n_steps,
                                                   const double* fu, const double* fv,
                                                   double* div_out) {
    if (!handle || !div_out) return;
    reinterpret_cast<StokesMac2D*>(handle)->run_steps_with_force(
        t_start, n_steps, fu, fv, div_out);
}

extern "C" void stokes_mac_run_steps_with_force_diagnostics_c(
    void* handle,
    double t_start,
    int n_steps,
    const double* fu,
    const double* fv,
    double* div_out,
    double* change_out
) {
    if (!handle || !div_out || !change_out) return;
    reinterpret_cast<StokesMac2D*>(handle)->run_steps_with_force_diagnostics(
        t_start, n_steps, fu, fv, div_out, change_out
    );
}

extern "C" void stokes_mac_set_state_c(void* handle,
                                        const double* u_interior,
                                        const double* v_interior,
                                        const double* p_cells) {
    if (!handle) return;
    reinterpret_cast<StokesMac2D*>(handle)->set_state_arrays(
        u_interior, v_interior, p_cells);
}

extern "C" void stokes_mac_get_state_c(void* handle,
                                        double* u_interior,
                                        double* v_interior,
                                        double* p_cells) {
    if (!handle) return;
    reinterpret_cast<StokesMac2D*>(handle)->get_state_arrays(
        u_interior, v_interior, p_cells);
}

extern "C" const double* stokes_mac_get_p_c(void* handle) {
    if (!handle) return nullptr;
    return reinterpret_cast<StokesMac2D*>(handle)->p_data();
}

extern "C" const double* stokes_mac_get_u_c(void* handle) {
    if (!handle) return nullptr;
    return reinterpret_cast<StokesMac2D*>(handle)->u_data();
}

extern "C" const double* stokes_mac_get_v_c(void* handle) {
    if (!handle) return nullptr;
    return reinterpret_cast<StokesMac2D*>(handle)->v_data();
}

extern "C" int stokes_mac_linearized_eig_c(void* handle,
                                            int n_eigs,
                                            const char* which,
                                            const double* fu,
                                            const double* fv,
                                            double* eig_real,
                                            double* eig_imag,
                                            double* vec_real,
                                            double* vec_imag,
                                            double* base_residual_inf,
                                            long long* matvec_count,
                                            long long* dense_operator_bytes) {
    if (!handle) return -1;
    return reinterpret_cast<StokesMac2D*>(handle)->solve_linearized_eigenmodes(
        n_eigs,
        which,
        fu,
        fv,
        eig_real,
        eig_imag,
        vec_real,
        vec_imag,
        base_residual_inf,
        matvec_count,
        dense_operator_bytes
    );
}

extern "C" int stokes_mac_solve_steady_c(void* handle,
                                          int max_newton_iters,
                                          double residual_tol,
                                          double krylov_tol,
                                          int krylov_maxiter,
                                          int krylov_restart,
                                          double jacobian_rdiff,
                                          const char* line_search,
                                          double min_step,
                                          const double* fu,
                                          const double* fv,
                                          int* newton_iters,
                                          double* residual_inf,
                                          double* max_div,
                                          int* converged,
                                          int* stop_code,
                                          double* iterate_change_inf,
                                          int* iterate_change_count) {
    if (!handle) return -1;
    return reinterpret_cast<StokesMac2D*>(handle)->solve_steady_newton(
        max_newton_iters,
        residual_tol,
        krylov_tol,
        krylov_maxiter,
        krylov_restart,
        jacobian_rdiff,
        line_search,
        min_step,
        fu,
        fv,
        newton_iters,
        residual_inf,
        max_div,
        converged,
        stop_code,
        iterate_change_inf,
        iterate_change_count
    );
}
