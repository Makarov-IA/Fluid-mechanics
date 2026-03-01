#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <vector>

class StokesMac2D {
public:
    using ForceFn = double (*)(double, double, double);

    StokesMac2D(int nx, int ny, double lx, double ly, double nu, double dt);

    double step(double t, ForceFn f1, ForceFn f2);

    const double* p_data() const { return p_.data(); }
    const double* u_data() const { return u_.data(); }
    const double* v_data() const { return v_.data(); }

    int nx() const { return nx_; }
    int ny() const { return ny_; }

private:
    int nx_;
    int ny_;
    double lx_;
    double ly_;
    double nu_;
    double dt_;
    double dx_;
    double dy_;
    double dx2_;
    double dy2_;
    double u_lid_top_;

    // Storage layout:
    // p(i,j), i=0..Nx-1,   j=0..Ny-1   -> size Nx*Ny
    // u(i,j), i=0..Nx,     j=0..Ny-1   -> size (Nx+1)*Ny
    // v(i,j), i=0..Nx-1,   j=0..Ny     -> size Nx*(Ny+1)
    std::vector<double> p_;
    std::vector<double> u_;
    std::vector<double> v_;
    int nu_unknowns_;
    int nv_unknowns_;
    int np_unknowns_;
    int total_unknowns_;
    Eigen::SparseMatrix<double> system_mat_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> system_solver_;

    inline int p_idx(int i, int j) const { return j * nx_ + i; }
    inline int u_idx(int i, int j) const { return j * (nx_ + 1) + i; }
    inline int v_idx(int i, int j) const { return j * nx_ + i; }
    inline int u_unknown_idx(int i, int j) const { return j * (nx_ - 1) + (i - 1); }  // i=1..Nx-1, j=0..Ny-1
    inline int v_unknown_idx(int i, int j) const { return nu_unknowns_ + ((j - 1) * nx_ + i); }  // i=0..Nx-1, j=1..Ny-1
    inline int p_unknown_idx(int i, int j) const { return nu_unknowns_ + nv_unknowns_ + p_idx(i, j); }

    void apply_velocity_bc(std::vector<double>& u, std::vector<double>& v) const;
    void compute_divergence(const std::vector<double>& u, const std::vector<double>& v, std::vector<double>& div) const;
    void build_monolithic_system();
};

extern "C" {
    typedef double (*Force2DTime_C)(double, double, double);

    void* stokes_mac_create_c(int Nx, int Ny, double Lx, double Ly, double nu, double dt);
    void stokes_mac_free_c(void* handle);
    double stokes_mac_step_c(void* handle, double t, Force2DTime_C f1, Force2DTime_C f2);
    const double* stokes_mac_get_p_c(void* handle);
    const double* stokes_mac_get_u_c(void* handle);
    const double* stokes_mac_get_v_c(void* handle);
}
