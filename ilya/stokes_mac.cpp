#include "stokes_mac.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#  include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

StokesMac2D::StokesMac2D(int nx, int ny, double lx, double ly,
                         double nu, double dt, double u_lid)
    : nx_(nx), ny_(ny),
      lx_(lx), ly_(ly),
      nu_(nu), dt_(dt),
      dx_(lx / nx), dy_(ly / ny),
      dx2_(dx_ * dx_), dy2_(dy_ * dy_),
      u_lid_(u_lid),
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
    // u = 0 on left and right walls  (vertical faces i=0, i=Nx)
    for (int j = 0; j < ny_; ++j) {
        u[u_idx(0,   j)] = 0.0;
        u[u_idx(nx_, j)] = 0.0;
    }
    // v = 0 on bottom and top walls  (horizontal faces j=0, j=Ny)
    for (int i = 0; i < nx_; ++i) {
        v[v_idx(i, 0  )] = 0.0;
        v[v_idx(i, ny_)] = 0.0;
    }
}

// Ghost-node extension of u across horizontal walls (used in advection).
//   j == -1  : bottom wall  → u_ghost = −u(i,0)            (no-slip at j=−½)
//   j == Ny  : top lid      → u_ghost = 2·u_lid − u(i,Ny−1) (u=u_lid at j=Ny−½)
//              corner nodes (i=0 or i=Nx) inherit u=0.
double StokesMac2D::u_ghost(const std::vector<double>& u, int i, int j) const {
    if (j >= 0 && j < ny_) return u[u_idx(i, j)];
    if (j == -1)            return -u[u_idx(i, 0)];
    // j == ny_
    if (i == 0 || i == nx_) return 0.0;
    return 2.0 * u_lid_ - u[u_idx(i, ny_ - 1)];
}

// Ghost-node extension of v across vertical walls.
//   i == −1  : left wall  → v_ghost = −v(0,j)       (no-slip at i=−½)
//   i == Nx  : right wall → v_ghost = −v(Nx−1,j)    (no-slip at i=Nx−½)
double StokesMac2D::v_ghost(const std::vector<double>& v, int i, int j) const {
    if (i >= 0 && i < nx_) return v[v_idx(i, j)];
    if (i == -1)            return -v[v_idx(0,      j)];
    return                          -v[v_idx(nx_-1, j)];  // i == nx_
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
    //   j=0    (bottom): u_ghost(i,-1) = −u(i,0)          → 3ν/dy² on diagonal
    //   j=Ny-1 (top):    u_ghost(i,Ny) = 2·u_lid−u(i,Ny-1) → 3ν/dy² on diagonal;
    //                    lid term  2·ν·u_lid/dy²  moves to RHS
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
            const double y = (j + 0.5) * dy_;
            const double x     = i * dx_;
            const double force = f1 ? f1(x, y, t) : 0.0;
            double b = inv_dt * u_[u_idx(i,j)] - adv_u_[u_idx(i,j)] + force;
            // Lid correction: ghost u(i,Ny) = 2·u_lid − u(i,Ny-1)
            // contributes +2·ν·u_lid/dy² to RHS at the top row
            if (j == ny_-1) b += 2.0 * nu_ * u_lid_ / dy2_;
            rhs_[u_unknown_idx(i, j)] = b;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const double force = f2 ? f2((i + 0.5) * dx_, j * dy_, t) : 0.0;
            rhs_[v_unknown_idx(i, j)] = inv_dt * v_[v_idx(i,j)] - adv_v_[v_idx(i,j)] + force;
        }
    }
    // rhs_ for the gauge row stays 0 (p(0,0) = 0)

    // ------------------------------------------------------------------
    // Solve  A·x = rhs  (factorisation already done in constructor)
    // ------------------------------------------------------------------
    sol_.noalias() = system_solver_.solve(rhs_);
    if (system_solver_.info() != Eigen::Success)
        throw std::runtime_error("Monolithic linear solve failed");

    // ------------------------------------------------------------------
    // Scatter solution back into the field arrays (in-place update)
    // ------------------------------------------------------------------
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < ny_; ++j)
        for (int i = 1; i < nx_; ++i)
            u_[u_idx(i,j)] = sol_[u_unknown_idx(i,j)];

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny_; ++j)
        for (int i = 0; i < nx_; ++i)
            v_[v_idx(i,j)] = sol_[v_unknown_idx(i,j)];

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < ny_; ++j)
        for (int i = 0; i < nx_; ++i)
            p_[p_idx(i,j)] = sol_[p_unknown_idx(i,j)];

    apply_velocity_bc(u_, v_);

    return max_divergence();
}

void StokesMac2D::run_steps(double t_start, int n_steps, double* div_out) {
    for (int k = 0; k < n_steps; ++k)
        div_out[k] = step(t_start + (k + 1) * dt_, nullptr, nullptr);
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
