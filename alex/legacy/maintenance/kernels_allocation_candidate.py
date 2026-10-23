"""Compiled finite differences and Thomas sweeps; no fastmath or changed equations.

Only independent lines in the two sweeps use prange. Stencils, residuals and
the sequence of pseudo-time steps stay serial. All arrays are float64, (nx, ny).
"""
import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def arakawa_at(p, w, i, j, inv_4dxdy):
    pe, pw, pn, ps = p[i+1, j], p[i-1, j], p[i, j+1], p[i, j-1]
    we, ww, wn, ws = w[i+1, j], w[i-1, j], w[i, j+1], w[i, j-1]
    j1 = (pe-pw)*(wn-ws) - (pn-ps)*(we-ww)
    j2 = (pe*(w[i+1, j+1]-w[i+1, j-1]) - pw*(w[i-1, j+1]-w[i-1, j-1])
          - pn*(w[i+1, j+1]-w[i-1, j+1]) + ps*(w[i+1, j-1]-w[i-1, j-1]))
    j3 = (wn*(p[i+1, j+1]-p[i-1, j+1]) - ws*(p[i+1, j-1]-p[i-1, j-1])
          - we*(p[i+1, j+1]-p[i+1, j-1]) + ww*(p[i-1, j+1]-p[i-1, j-1]))
    return (j1+j2+j3)*inv_4dxdy/3.0


@njit(cache=True, inline="always")
def laplacian_at(q, i, j, idx2, idy2):
    return (q[i-1, j]-2*q[i, j]+q[i+1, j])*idx2 + (q[i, j-1]-2*q[i, j]+q[i, j+1])*idy2


@njit(cache=True, inline="always")
def mixed_at(q, i, j, idx2, idy2):
    west = q[i-1, j-1]-2*q[i-1, j]+q[i-1, j+1]
    middle = q[i, j-1]-2*q[i, j]+q[i, j+1]
    east = q[i+1, j-1]-2*q[i+1, j]+q[i+1, j+1]
    return (west-2*middle+east)*idx2*idy2


@njit(cache=True)
def psi_rhs(p, w, dt, idx2, idy2, rhs):
    for i in range(1, p.shape[0]-1):
        for j in range(1, p.shape[1]-1):
            rhs[i, j] = p[i, j]+dt*w[i, j]+dt*dt*mixed_at(p, i, j, idx2, idy2)


@njit(cache=True)
def omega_rhs(p, w, source, dt, nu, idx2, idy2, inv_4dxdy, rhs):
    for i in range(1, p.shape[0]-1):
        for j in range(1, p.shape[1]-1):
            rhs[i, j] = (w[i, j] + dt*(arakawa_at(p, w, i, j, inv_4dxdy)+source[i, j])
                         + (dt*nu)**2*mixed_at(w, i, j, idx2, idy2))


@njit(cache=True)
def thom_boundary(p, w, dx, dy, bottom_u, top_u, left_v, right_v):
    for i in range(1, p.shape[0]-1):
        w[i, 0] = -2*p[i, 1]/dy**2 + 2*bottom_u/dy
        w[i, -1] = -2*p[i, -2]/dy**2 - 2*top_u/dy
    for j in range(1, p.shape[1]-1):
        w[0, j] = -2*p[1, j]/dx**2 - 2*left_v/dx
        w[-1, j] = -2*p[-2, j]/dx**2 + 2*right_v/dx
    # Preserve the C++ corner convention (stationary side wall wins for u/v).
    w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 0.0


def thomas_factors(size, coefficient):
    """Factor I - coefficient*second_difference once for all identical lines."""
    inverse = np.empty(size)
    upper = np.empty(size)
    diagonal = 1+2*coefficient
    for k in range(size):
        denominator = diagonal if k == 0 else diagonal+coefficient*upper[k-1]
        inverse[k] = 1.0/denominator
        upper[k] = -coefficient*inverse[k]
    if not np.isfinite(inverse).all() or not np.isfinite(upper).all():
        raise FloatingPointError("Invalid Thomas factorization; check grid, dt and Re")
    return inverse, upper


# Batched Thomas sweeps. The recurrence runs in the OUTER loop and the inner loop
# goes across independent lines, so consecutive points no longer wait on each
# other and the inner loop vectorises. Every point gets exactly the same
# arithmetic as a line-by-line sweep: results are bitwise identical.
# Lines are processed in fixed blocks; prange over blocks keeps SIMD inside
# each block and does not make the result depend on the thread count.
BLOCK = 32


@njit(cache=True, inline="always")
def _block_x(rhs, out, left, right, coefficient, inverse, upper, j0, j1):
    """Lines along i (one per j) for j in [j0, j1)."""
    nx = rhs.shape[0]
    for i in range(1, nx-1):
        inv = inverse[i-1]
        for j in range(j0, j1):
            value = rhs[i, j]
            if i == 1:
                value += coefficient*left[j]
            if i == nx-2:
                value += coefficient*right[j]
            previous = out[i-1, j] if i > 1 else 0.0
            out[i, j] = (value+coefficient*previous)*inv
    for i in range(nx-3, 0, -1):
        u = upper[i-1]
        for j in range(j0, j1):
            out[i, j] -= u*out[i+1, j]


@njit(cache=True, inline="always")
def _block_y(rhs, out, bottom, top, coefficient, inverse, upper, i0, i1):
    """Lines along j (one per i) for i in [i0, i1)."""
    ny = rhs.shape[1]
    for j in range(1, ny-1):
        inv = inverse[j-1]
        for i in range(i0, i1):
            value = rhs[i, j]
            if j == 1:
                value += coefficient*bottom[i]
            if j == ny-2:
                value += coefficient*top[i]
            previous = out[i, j-1] if j > 1 else 0.0
            out[i, j] = (value+coefficient*previous)*inv
    for j in range(ny-3, 0, -1):
        u = upper[j-1]
        for i in range(i0, i1):
            out[i, j] -= u*out[i, j+1]


# Separate Python functions give serial/parallel variants distinct disk caches.
@njit(cache=True)
def sweep_x_serial(rhs, out, left, right, coefficient, inverse, upper):
    _block_x(rhs, out, left, right, coefficient, inverse, upper, 1, rhs.shape[1]-1)


@njit(cache=True)
def sweep_y_serial(rhs, out, bottom, top, coefficient, inverse, upper):
    _block_y(rhs, out, bottom, top, coefficient, inverse, upper, 1, rhs.shape[0]-1)


@njit(cache=True, parallel=True)
def sweep_x_parallel(rhs, out, left, right, coefficient, inverse, upper):
    lines = rhs.shape[1]-2
    for b in prange((lines+BLOCK-1)//BLOCK):
        j0 = 1+b*BLOCK
        _block_x(rhs, out, left, right, coefficient, inverse, upper, j0, min(j0+BLOCK, lines+1))


@njit(cache=True, parallel=True)
def sweep_y_parallel(rhs, out, bottom, top, coefficient, inverse, upper):
    lines = rhs.shape[0]-2
    for b in prange((lines+BLOCK-1)//BLOCK):
        i0 = 1+b*BLOCK
        _block_y(rhs, out, bottom, top, coefficient, inverse, upper, i0, min(i0+BLOCK, lines+1))


@njit(cache=True)
def residuals_buffered(p, w, source, nu, idx2, idy2, inv_4dxdy, col_p, col_w, poison):
    """Max residuals, or (inf, inf) if any input or residual is non-finite.

    One running max through the whole grid is a serial dependency chain; per-column
    accumulators make the inner loop independent so it vectorises. max is exact
    and order-free, so the result is unchanged. x*0.0 is 0 for finite x and NaN
    for inf/NaN: the poison sums replace per-point isfinite branches. A NaN must
    never be read as convergence.
    """
    nx, ny = p.shape
    col_p.fill(0.0)
    col_w.fill(0.0)
    poison.fill(0.0)
    for i in range(nx):
        for j in range(ny):
            poison[j] += p[i, j]*0.0 + w[i, j]*0.0 + source[i, j]*0.0
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            a = laplacian_at(p, i, j, idx2, idy2)+w[i, j]
            b = nu*laplacian_at(w, i, j, idx2, idy2)+arakawa_at(p, w, i, j, inv_4dxdy)+source[i, j]
            poison[j] += a*0.0 + b*0.0
            col_p[j] = max(col_p[j], abs(a))
            col_w[j] = max(col_w[j], abs(b))
    if np.sum(poison) != 0.0:
        return math.inf, math.inf
    return np.max(col_p), np.max(col_w)


@njit(cache=True)
def residuals(p, w, source, nu, idx2, idy2, inv_4dxdy):
    """Standalone diagnostic; Solver reuses scratch arrays via residuals_buffered."""
    ny = p.shape[1]
    return residuals_buffered(p, w, source, nu, idx2, idy2, inv_4dxdy,
                             np.empty(ny), np.empty(ny), np.empty(ny))


@njit(cache=True)
def steady_residual(p, w, source, nu, idx2, idy2, inv_4dxdy, rp, rw):
    """Pointwise F_psi, F_omega on the interior: the same expressions residuals() takes
    the max of, so Newton drives to zero exactly what the solver reports."""
    for i in range(1, p.shape[0]-1):
        for j in range(1, p.shape[1]-1):
            rp[i, j] = laplacian_at(p, i, j, idx2, idy2)+w[i, j]
            rw[i, j] = nu*laplacian_at(w, i, j, idx2, idy2)+arakawa_at(p, w, i, j, inv_4dxdy)+source[i, j]


@njit(cache=True)
def steady_tangent(p, w, dp, dw, nu, idx2, idy2, inv_4dxdy, rp, rw):
    """Exact linearisation of steady_residual at (p, w) in direction (dp, dw).

    J_A is bilinear, so dJ_A = J_A(dp, w) + J_A(p, dw) with no truncation. The caller
    sets the walls: dp = 0 and dw from Thom's formula with zero wall velocities.
    """
    for i in range(1, p.shape[0]-1):
        for j in range(1, p.shape[1]-1):
            rp[i, j] = laplacian_at(dp, i, j, idx2, idy2)+dw[i, j]
            rw[i, j] = (nu*laplacian_at(dw, i, j, idx2, idy2)
                        + arakawa_at(dp, w, i, j, inv_4dxdy)+arakawa_at(p, dw, i, j, inv_4dxdy))
