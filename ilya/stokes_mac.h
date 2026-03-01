#pragma once

#include <vector>

class StokesMac2D {
public:
    using ForceFn = double (*)(double, double, double);

    StokesMac2D(int nx, int ny, double lx, double ly, double nu, double dt, int poisson_max_iter, double poisson_tol);

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
    int poisson_max_iter_;
    double poisson_tol_;
    double dx_;
    double dy_;
    double dx2_;
    double dy2_;

    // Storage layout:
    // p(i,j), i=0..Nx-1,   j=0..Ny-1   -> size Nx*Ny
    // u(i,j), i=0..Nx,     j=0..Ny-1   -> size (Nx+1)*Ny
    // v(i,j), i=0..Nx-1,   j=0..Ny     -> size Nx*(Ny+1)
    std::vector<double> p_;
    std::vector<double> u_;
    std::vector<double> v_;

    inline int p_idx(int i, int j) const { return j * nx_ + i; }
    inline int u_idx(int i, int j) const { return j * (nx_ + 1) + i; }
    inline int v_idx(int i, int j) const { return j * nx_ + i; }

    void apply_velocity_bc(std::vector<double>& u, std::vector<double>& v) const;
    void compute_laplacian_u(const std::vector<double>& u, std::vector<double>& lap_u) const;
    void compute_laplacian_v(const std::vector<double>& v, std::vector<double>& lap_v) const;
    void compute_divergence(const std::vector<double>& u, const std::vector<double>& v, std::vector<double>& div) const;
    void solve_poisson_jacobi(const std::vector<double>& rhs, std::vector<double>& p) const;
};

extern "C" {
    typedef double (*Force2DTime_C)(double, double, double);

    void* stokes_mac_create_c(
        int Nx, int Ny, double Lx, double Ly, double nu, double dt, int poisson_max_iter, double poisson_tol
    );
    void stokes_mac_free_c(void* handle);
    double stokes_mac_step_c(void* handle, double t, Force2DTime_C f1, Force2DTime_C f2);
    const double* stokes_mac_get_p_c(void* handle);
    const double* stokes_mac_get_u_c(void* handle);
    const double* stokes_mac_get_v_c(void* handle);
}
