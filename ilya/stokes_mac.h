#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <vector>

// ---------------------------------------------------------------------------
// 2-D Navier-Stokes on a staggered MAC grid
//
// Grid layout (uniform, cell size dx × dy):
//
//   p[i,j]  at cell centres   (i=0..Nx-1, j=0..Ny-1)  — size  Nx    × Ny
//   u[i,j]  at vertical faces (i=0..Nx,   j=0..Ny-1)  — size (Nx+1) × Ny
//   v[i,j]  at horiz. faces   (i=0..Nx-1, j=0..Ny  )  — size  Nx    × (Ny+1)
//
// Lid-driven cavity boundary conditions:
//   Top wall   : u = u_lid, v = 0
//   Other walls: u = 0,     v = 0
//   Pressure gauge: p[0,0] = 0  (removes the constant-pressure null space)
//
// Time integration — IMEX (semi-implicit):
//   Convection : explicit central differences from the previous time layer
//   Viscosity  : implicit (backward Euler)
//   Pressure   : implicit
//
// The coefficient matrix is constant → factorised once at construction (SparseLU).
// ---------------------------------------------------------------------------

class StokesMac2D {
public:
    using ForceFn = double (*)(double x, double y, double t);

    StokesMac2D(int nx, int ny, double lx, double ly, double nu, double dt,
                double u_lid = 1.0);

    // Advance one time step; returns max|div u| after the update.
    [[nodiscard]] double step(double t, ForceFn f1, ForceFn f2);

    // Run n_steps with zero body force entirely in C++.
    // div_out must point to a caller-allocated array of at least n_steps doubles.
    void run_steps(double t_start, int n_steps, double* div_out);

    // Raw pointers to field storage (row-major, j-first indexing).
    [[nodiscard]] const double* p_data() const { return p_.data(); }
    [[nodiscard]] const double* u_data() const { return u_.data(); }
    [[nodiscard]] const double* v_data() const { return v_.data(); }

    int    nx() const { return nx_; }
    int    ny() const { return ny_; }
    double dt() const { return dt_; }

private:
    // -----------------------------------------------------------------------
    // Physical and grid parameters
    // -----------------------------------------------------------------------
    const int    nx_, ny_;
    const double lx_, ly_;
    const double nu_;
    const double dt_;
    const double dx_, dy_;
    const double dx2_, dy2_;   // dx², dy²
    const double u_lid_;       // top-wall velocity

    // -----------------------------------------------------------------------
    // Field storage (flat row-major arrays)
    //   p[i,j] = p_[ j*Nx + i ]
    //   u[i,j] = u_[ j*(Nx+1) + i ]
    //   v[i,j] = v_[ j*Nx + i ]
    // -----------------------------------------------------------------------
    std::vector<double> p_;
    std::vector<double> u_;
    std::vector<double> v_;

    // -----------------------------------------------------------------------
    // Linear system
    //
    // Unknown ordering:
    //   [0 .. nu_unknowns)            — interior u-faces: i=1..Nx-1, j=0..Ny-1
    //   [nu_unknowns .. nv_off)       — interior v-faces: i=0..Nx-1, j=1..Ny-1
    //   [nv_off .. total_unknowns)    — pressure cells:   i=0..Nx-1, j=0..Ny-1
    // -----------------------------------------------------------------------
    const int nu_unknowns_;   // (Nx-1)*Ny
    const int nv_unknowns_;   // Nx*(Ny-1)
    const int np_unknowns_;   // Nx*Ny
    const int total_unknowns_;

    Eigen::SparseMatrix<double>                  system_mat_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> system_solver_;

    // -----------------------------------------------------------------------
    // Pre-allocated work buffers — avoid per-step heap allocation
    // -----------------------------------------------------------------------
    std::vector<double> adv_u_;  // explicit advection term for u, size (Nx+1)*Ny
    std::vector<double> adv_v_;  // explicit advection term for v, size Nx*(Ny+1)
    Eigen::VectorXd     rhs_;    // RHS vector, length total_unknowns_
    Eigen::VectorXd     sol_;    // solution vector, length total_unknowns_

    // -----------------------------------------------------------------------
    // Index helpers
    // -----------------------------------------------------------------------

    // Storage indices (into the flat field vectors)
    int p_idx(int i, int j) const { return j * nx_ + i; }
    int u_idx(int i, int j) const { return j * (nx_ + 1) + i; }
    int v_idx(int i, int j) const { return j * nx_ + i; }

    // Global unknown indices (into the linear system vector)
    int u_unknown_idx(int i, int j) const { return j * (nx_ - 1) + (i - 1); }
    int v_unknown_idx(int i, int j) const { return nu_unknowns_ + (j - 1) * nx_ + i; }
    int p_unknown_idx(int i, int j) const { return nu_unknowns_ + nv_unknowns_ + p_idx(i, j); }

    // -----------------------------------------------------------------------
    // Private methods
    // -----------------------------------------------------------------------

    void build_monolithic_system();

    // Apply Dirichlet boundary conditions to the u and v fields.
    void apply_velocity_bc(std::vector<double>& u, std::vector<double>& v) const;

    // Ghost-node extensions of u and v across solid walls (for advection).
    double u_ghost(const std::vector<double>& u, int i, int j) const;
    double v_ghost(const std::vector<double>& v, int i, int j) const;

    // Compute explicit convective terms; results written into adv_u_, adv_v_.
    void compute_advection(const std::vector<double>& u,
                           const std::vector<double>& v);

    // Return max|div u| over all non-gauge pressure cells.
    double max_divergence() const;
};


// ---------------------------------------------------------------------------
// C API — called from Python via ctypes
// ---------------------------------------------------------------------------
extern "C" {
    typedef double (*Force2DTime_C)(double x, double y, double t);

    void*        stokes_mac_create_c   (int Nx, int Ny, double Lx, double Ly,
                                        double nu, double dt);
    void         stokes_mac_free_c     (void* handle);
    // Single step with optional body force (pass NULL for zero force)
    double       stokes_mac_step_c     (void* handle, double t,
                                        Force2DTime_C f1, Force2DTime_C f2);
    // Batch step with zero body force: fills div_out[0..n_steps-1]
    void         stokes_mac_run_steps_c(void* handle, double t_start,
                                        int n_steps, double* div_out);
    const double* stokes_mac_get_p_c   (void* handle);
    const double* stokes_mac_get_u_c   (void* handle);
    const double* stokes_mac_get_v_c   (void* handle);
}
