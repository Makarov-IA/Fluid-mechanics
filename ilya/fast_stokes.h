#pragma once
// ---------------------------------------------------------------------------
// Fast solver for the monolithic MAC Stokes step system
//
//   [ H  G ] [u]   [r]        H = I/dt - nu*Lap   (velocity block)
//   [ D  0 ] [p] = [0]        D = -G^T            (divergence)
//
// It returns the same discrete solution as the direct LDL^T solve, up to
// round-off:
//   * H^-1 is exact: an orthonormal eigen-transform along x (sine basis for
//     u and v, applied as a dense matrix product -> dgemm on the AMX unit)
//     followed by exact tridiagonal (Thomas) solves along y.
//   * The pressure Schur complement  A = -D H^-1 G  is solved by CG with the
//     Cahouet-Chabard preconditioner  (1/dt)(-Lap_p)^+ + nu*I  (cosine basis
//     along x + tridiagonal along y).
//   * The CG start is a polynomial extrapolation of the previous pressures,
//     so a step usually needs only 1-2 iterations.
//   * CG stops when ||D u||_2 <= tol * max|u| / min(dx, dy).
//
// Requirements (same as the monolithic matrix): uniform grid, constant nu and
// dt, zero normal velocity on the walls.  macOS only (Accelerate + GCD).
// ---------------------------------------------------------------------------
#if defined(__APPLE__)
#define STOKES_HAS_FAST_SOLVER 1

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fast_stokes_detail {

enum class Basis { DST1, DST2, DCT2 };
enum class YBC { Dirichlet, Ghost, Neumann };  // end-row diagonal weights 2, 3, 1

// Orthonormal eigenvectors (Q[j*n+k] = q_k(j)) and eigenvalues of the 1-D
// second difference / h^2 with the boundary treatment implied by the basis.
inline void make_basis(Basis b, int n, double h, std::vector<double>& Q, std::vector<double>& lam) {
    Q.assign(static_cast<size_t>(n) * n, 0.0);
    lam.assign(n, 0.0);
    for (int k = 0; k < n; ++k) {
        double freq = 0.0;
        for (int j = 0; j < n; ++j) {
            double val = 0.0;
            switch (b) {
                case Basis::DST1: freq = M_PI * (k + 1) / (n + 1); val = std::sin(freq * (j + 1));   break;
                case Basis::DST2: freq = M_PI * (k + 1) / n;       val = std::sin(freq * (j + 0.5)); break;
                case Basis::DCT2: freq = M_PI * k / n;             val = std::cos(freq * (j + 0.5)); break;
            }
            Q[static_cast<size_t>(j) * n + k] = val;
        }
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += Q[static_cast<size_t>(j) * n + k] * Q[static_cast<size_t>(j) * n + k];
        s = 1.0 / std::sqrt(s);
        for (int j = 0; j < n; ++j) Q[static_cast<size_t>(j) * n + k] *= s;
        lam[k] = (2.0 - 2.0 * std::cos(freq)) / (h * h);
    }
}

// out = scale * Qx * T^-1 * (Qx^T in) + add_identity * in on a (ny x nx)
// row-major field, where for every x-mode k
//   T_k = (shift0 + coef*lam_x[k]) I + coef * Ly,
// Ly being the 1-D y second difference / dy^2 with the given end treatment.
struct SepSolver {
    int ny = 0, nx = 0;
    double off = 0.0, scale = 1.0, add_identity = 0.0;
    bool neumann_zero_mode = false;     // T_0 is singular -> pseudo-inverse K0
    std::vector<double> Qx, inv_m, cp, t1, t2;
    std::vector<double> K0, col_in, col_out;

    void init(Basis bx, int nx_, double dx, YBC ybc, int ny_, double dy,
              double shift0, double coef, double scale_, double add_identity_) {
        nx = nx_; ny = ny_; scale = scale_; add_identity = add_identity_;
        std::vector<double> lx;
        make_basis(bx, nx, dx, Qx, lx);
        const double dy2 = dy * dy;
        off = -coef / dy2;
        const double end_w = ybc == YBC::Dirichlet ? 2.0 : (ybc == YBC::Ghost ? 3.0 : 1.0);
        neumann_zero_mode = (ybc == YBC::Neumann && shift0 == 0.0);

        // Thomas factors, stored row-major so the sweeps vectorise over k.
        inv_m.assign(static_cast<size_t>(ny) * nx, 0.0);
        cp.assign(static_cast<size_t>(ny) * nx, 0.0);
        for (int k = 0; k < nx; ++k) {
            // k = 0 of the singular Neumann operator is regularised here and
            // overwritten by the pseudo-inverse in apply().
            const double shift = shift0 + coef * lx[k] + ((neumann_zero_mode && k == 0) ? 1.0 : 0.0);
            double cprev = 0.0;
            for (int j = 0; j < ny; ++j) {
                const double w = (j == 0 || j == ny - 1) ? end_w : 2.0;
                const double m = shift + coef * w / dy2 - (j > 0 ? off * cprev : 0.0);
                inv_m[static_cast<size_t>(j) * nx + k] = 1.0 / m;
                cprev = off / m;
                cp[static_cast<size_t>(j) * nx + k] = cprev;
            }
        }

        if (neumann_zero_mode) {
            std::vector<double> Qy, ly;
            make_basis(Basis::DCT2, ny, dy, Qy, ly);
            K0.assign(static_cast<size_t>(ny) * ny, 0.0);
            for (int a = 0; a < ny; ++a)
                for (int b = 0; b < ny; ++b) {
                    double s = 0.0;
                    for (int m = 1; m < ny; ++m)
                        s += Qy[static_cast<size_t>(a) * ny + m] * Qy[static_cast<size_t>(b) * ny + m]
                           / (coef * ly[m]);
                    K0[static_cast<size_t>(a) * ny + b] = s;
                }
            col_in.assign(ny, 0.0);
            col_out.assign(ny, 0.0);
        }
        t1.assign(static_cast<size_t>(ny) * nx, 0.0);
        t2.assign(static_cast<size_t>(ny) * nx, 0.0);
    }

    void apply(const double* in, double* out) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ny, nx, nx,
                    1.0, in, nx, Qx.data(), nx, 0.0, t1.data(), nx);
        if (neumann_zero_mode)
            for (int j = 0; j < ny; ++j) col_in[j] = t1[static_cast<size_t>(j) * nx];

        double* y = t2.data();
        const double* x = t1.data();
        for (int k = 0; k < nx; ++k) y[k] = x[k] * inv_m[k];
        for (int j = 1; j < ny; ++j) {
            const size_t r = static_cast<size_t>(j) * nx, rp = r - nx;
            for (int k = 0; k < nx; ++k) y[r + k] = (x[r + k] - off * y[rp + k]) * inv_m[r + k];
        }
        for (int j = ny - 2; j >= 0; --j) {
            const size_t r = static_cast<size_t>(j) * nx, rn = r + nx;
            for (int k = 0; k < nx; ++k) y[r + k] -= cp[r + k] * y[rn + k];
        }

        if (neumann_zero_mode) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, ny, ny, 1.0, K0.data(), ny,
                        col_in.data(), 1, 0.0, col_out.data(), 1);
            for (int j = 0; j < ny; ++j) y[static_cast<size_t>(j) * nx] = col_out[j];
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ny, nx, nx,
                    scale, t2.data(), nx, Qx.data(), nx, 0.0, out, nx);
        if (add_identity != 0.0) {
            const size_t n = static_cast<size_t>(ny) * nx;
            for (size_t k = 0; k < n; ++k) out[k] += add_identity * in[k];
        }
    }
};

}  // namespace fast_stokes_detail

class FastStokesSolver {
public:
    FastStokesSolver(int nx, int ny, double dx, double dy, double nu, double dt,
                     double tol, int extrapolation, bool parallel)
        : nx_(nx), ny_(ny), dx_(dx), dy_(dy), tol_(tol),
          extrapolation_(extrapolation), parallel_(parallel) {
        using namespace fast_stokes_detail;
        if (nx < 3 || ny < 3) throw std::invalid_argument("fast solver needs nx, ny >= 3");
        if (tol <= 0.0) throw std::invalid_argument("fast solver tol must be positive");
        if (extrapolation < 0 || extrapolation > 3)
            throw std::invalid_argument("fast solver extrapolation must be in 0..3");
        nu_n_ = (nx - 1) * ny;
        nv_n_ = nx * (ny - 1);
        np_n_ = nx * ny;
        // u: x Dirichlet at wall faces (DST-I), y no-slip ghost (DST-II type)
        Hu_.init(Basis::DST1, nx - 1, dx, YBC::Ghost, ny, dy, 1.0 / dt, nu, 1.0, 0.0);
        // v: x no-slip ghost (DST-II), y Dirichlet at wall faces (DST-I type)
        Hv_.init(Basis::DST2, nx, dx, YBC::Dirichlet, ny - 1, dy, 1.0 / dt, nu, 1.0, 0.0);
        // Cahouet-Chabard: (1/dt)(-Lap_p)^+ + nu I, Neumann pressure Laplacian
        P_.init(Basis::DCT2, nx, dx, YBC::Neumann, ny, dy, 0.0, 1.0, 1.0 / dt, nu);

        for (auto* v : {&wu_, &qu_, &gu_}) v->assign(nu_n_, 0.0);
        for (auto* v : {&wv_, &qv_, &gv_}) v->assign(nv_n_, 0.0);
        for (auto* v : {&p_, &h1_, &h2_, &h3_, &r_, &z_, &d_, &Ad_}) v->assign(np_n_, 0.0);
    }

    // Forget the pressure history (call after the state is overwritten).
    void reset() { history_ = 0; }

    long steps() const { return steps_; }
    long cg_iterations() const { return total_iters_; }
    int max_cg_iterations() const { return max_iters_; }

    // rhs and sol use the monolithic layout [u | v | p].  p_state is the
    // current physical pressure, used as the initial guess after reset().
    // negate_p stores -p in sol (for the LDL^T sign convention of the caller).
    int solve(const double* rhs, double* sol, const double* p_state, bool negate_p) {
        if (history_ == 0) {
            p_.assign(p_state, p_state + np_n_);
        } else {
            const int order = static_cast<int>(std::min<long>(history_ - 1, extrapolation_));
            for (int k = 0; k < np_n_; ++k) {
                const double pn = p_[k], p1 = h1_[k], p2 = h2_[k], p3 = h3_[k];
                double g = pn;
                if (order == 1)      g = 2.0 * pn - p1;
                else if (order == 2) g = 3.0 * pn - 3.0 * p1 + p2;
                else if (order == 3) g = 4.0 * pn - 6.0 * p1 + 4.0 * p2 - p3;
                h3_[k] = p2; h2_[k] = p1; h1_[k] = pn; p_[k] = g;
            }
        }
        remove_mean(p_.data());

        // u = H^-1 (rhs - G p0)
        par2([&] {
                 grad_u(p_.data(), gu_.data());
                 for (int k = 0; k < nu_n_; ++k) gu_[k] = rhs[k] - gu_[k];
                 Hu_.apply(gu_.data(), wu_.data());
             },
             [&] {
                 grad_v(p_.data(), gv_.data());
                 for (int k = 0; k < nv_n_; ++k) gv_[k] = rhs[nu_n_ + k] - gv_[k];
                 Hv_.apply(gv_.data(), wv_.data());
             });

        // Schur residual r = -D u
        div(wu_.data(), wv_.data(), r_.data());
        double rr = 0.0, umax = 0.0;
        for (int k = 0; k < np_n_; ++k) { r_[k] = -r_[k]; rr += r_[k] * r_[k]; }
        for (int k = 0; k < nu_n_; ++k) umax = std::max(umax, std::abs(wu_[k]));
        for (int k = 0; k < nv_n_; ++k) umax = std::max(umax, std::abs(wv_[k]));
        const double stop = tol_ * umax / std::min(dx_, dy_);
        const double stop2 = std::max(stop * stop, 1e-300);

        int it = 0;
        double rz = 0.0;
        while (rr > stop2 && it < kMaxIterations) {
            P_.apply(r_.data(), z_.data());
            double rz_new = 0.0;
            for (int k = 0; k < np_n_; ++k) rz_new += r_[k] * z_[k];
            if (it == 0) {
                d_ = z_;
            } else {
                const double beta = rz_new / rz;
                for (int k = 0; k < np_n_; ++k) d_[k] = z_[k] + beta * d_[k];
            }
            rz = rz_new;

            // q = H^-1 G d,  A d = -D q
            par2([&] { grad_u(d_.data(), gu_.data()); Hu_.apply(gu_.data(), qu_.data()); },
                 [&] { grad_v(d_.data(), gv_.data()); Hv_.apply(gv_.data(), qv_.data()); });
            div(qu_.data(), qv_.data(), Ad_.data());
            double dAd = 0.0;
            for (int k = 0; k < np_n_; ++k) dAd -= d_[k] * Ad_[k];
            const double alpha = rz / dAd;

            rr = 0.0;
            for (int k = 0; k < np_n_; ++k) {
                p_[k] += alpha * d_[k];
                r_[k] += alpha * Ad_[k];
                rr += r_[k] * r_[k];
            }
            for (int k = 0; k < nu_n_; ++k) wu_[k] -= alpha * qu_[k];
            for (int k = 0; k < nv_n_; ++k) wv_[k] -= alpha * qv_[k];
            ++it;
        }
        if (rr > stop2)
            throw std::runtime_error("Fast Stokes solver: CG did not converge");
        ++history_;

        std::copy(wu_.begin(), wu_.end(), sol);
        std::copy(wv_.begin(), wv_.end(), sol + nu_n_);
        const double p0 = p_[0];                       // gauge p(0,0) = 0
        const double sgn = negate_p ? -1.0 : 1.0;
        for (int k = 0; k < np_n_; ++k) sol[nu_n_ + nv_n_ + k] = sgn * (p_[k] - p0);

        ++steps_;
        total_iters_ += it;
        max_iters_ = std::max(max_iters_, it);
        return it;
    }

private:
    static constexpr int kMaxIterations = 200;

    int nx_, ny_, nu_n_, nv_n_, np_n_;
    double dx_, dy_, tol_;
    int extrapolation_;
    bool parallel_;
    fast_stokes_detail::SepSolver Hu_, Hv_, P_;
    std::vector<double> wu_, wv_, qu_, qv_, gu_, gv_;
    std::vector<double> p_, h1_, h2_, h3_, r_, z_, d_, Ad_;
    long history_ = 0;
    long steps_ = 0, total_iters_ = 0;
    int max_iters_ = 0;

    // Run the independent u and v branches on two cores.
    template <class A, class B>
    void par2(A&& a, B&& b) {
        if (!parallel_) { a(); b(); return; }
        struct Ctx { A* a; B* b; } ctx{&a, &b};
        dispatch_apply_f(2, DISPATCH_APPLY_AUTO, &ctx, [](void* c, size_t i) {
            auto* x = static_cast<Ctx*>(c);
            if (i == 0) (*x->a)(); else (*x->b)();
        });
    }

    void grad_u(const double* p, double* gu) const {
        const double idx = 1.0 / dx_;
        for (int j = 0; j < ny_; ++j) {
            const double* pr = p + j * nx_;
            double* g = gu + j * (nx_ - 1);
            for (int i = 0; i < nx_ - 1; ++i) g[i] = (pr[i + 1] - pr[i]) * idx;
        }
    }

    void grad_v(const double* p, double* gv) const {
        const double idy = 1.0 / dy_;
        for (int j = 1; j < ny_; ++j) {
            const double* pn = p + j * nx_;
            const double* ps = pn - nx_;
            double* g = gv + (j - 1) * nx_;
            for (int i = 0; i < nx_; ++i) g[i] = (pn[i] - ps[i]) * idy;
        }
    }

    // Cell divergence; wall-normal velocities are zero (not unknowns).
    void div(const double* u, const double* v, double* out) const {
        const double idx = 1.0 / dx_, idy = 1.0 / dy_;
        const int nxu = nx_ - 1;
        for (int j = 0; j < ny_; ++j) {
            const double* ur = u + j * nxu;
            const double* vn = (j + 1 <= ny_ - 1) ? v + j * nx_ : nullptr;
            const double* vs = (j >= 1) ? v + (j - 1) * nx_ : nullptr;
            double* o = out + j * nx_;
            o[0] = ur[0] * idx;
            for (int i = 1; i < nx_ - 1; ++i) o[i] = (ur[i] - ur[i - 1]) * idx;
            o[nx_ - 1] = -ur[nxu - 1] * idx;
            if (vn && vs)  for (int i = 0; i < nx_; ++i) o[i] += (vn[i] - vs[i]) * idy;
            else if (vn)   for (int i = 0; i < nx_; ++i) o[i] += vn[i] * idy;
            else           for (int i = 0; i < nx_; ++i) o[i] -= vs[i] * idy;
        }
    }

    void remove_mean(double* p) const {
        double s = 0.0;
        for (int k = 0; k < np_n_; ++k) s += p[k];
        s /= np_n_;
        for (int k = 0; k < np_n_; ++k) p[k] -= s;
    }
};

#endif  // __APPLE__
