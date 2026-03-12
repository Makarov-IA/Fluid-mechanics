# Monolithic MAC Navier-Stokes Solver (2D)

This folder contains a 2D incompressible Navier-Stokes solver on a staggered MAC grid.

The implementation is semi-implicit:
- viscosity and pressure are treated implicitly,
- convection is treated explicitly from the previous time layer,
- the linear system solved at each step is still a monolithic Stokes system.

Time discretization: first-order backward Euler for diffusion/pressure, explicit Euler for convection.
Space discretization: second-order central differences on a uniform MAC grid.

## Continuous equations

```text
du/dt + u * du/dx + v * du/dy - nu * Laplace(u) + dp/dx = f1
dv/dt + u * dv/dx + v * dv/dy - nu * Laplace(v) + dp/dy = f2
du/dx + dv/dy = 0
```

In vector form:

```text
U_t + (U · grad) U - nu * Laplace(U) + grad(p) = F
div(U) = 0
```

where

```text
U = (u, v)
F = (f1, f2)
```

## MAC grid layout

```text
p[i,j] : cell centers,        i=0..Nx-1, j=0..Ny-1, size Nx x Ny
u[i,j] : vertical faces,      i=0..Nx,   j=0..Ny-1, size (Nx+1) x Ny
v[i,j] : horizontal faces,    i=0..Nx-1, j=0..Ny,   size Nx x (Ny+1)
```

```text
dx = Lx / Nx
dy = Ly / Ny
```

## Boundary conditions (lid-driven cavity)

```text
Top wall:    u = U_lid = 1, v = 0
Other walls: u = 0,         v = 0
Pressure gauge: p[0,0] = 0
```

Ghost-node form used in the code:

```text
Bottom wall for u: u(i,-1)  = -u(i,0)
Top wall for u:    u(i,Ny)  = 2*U_lid - u(i,Ny-1)
Left wall for v:   v(-1,j)  = -v(0,j)
Right wall for v:  v(Nx,j)  = -v(Nx-1,j)
Horizontal walls:  v(i,0) = v(i,Ny) = 0
Vertical walls:    u(0,j) = u(Nx,j) = 0
```

## Time-discrete equations

At each time step the code solves for

```text
u^(n+1), v^(n+1), p^(n+1)
```

while the nonlinear term is evaluated from the previous layer

```text
u^n, v^n.
```

So the discrete equations are:

```text
(u^(n+1) - u^n)/dt + N_u(u^n, v^n) - nu * Laplace(u^(n+1)) + dp^(n+1)/dx = f1^(n+1)
(v^(n+1) - v^n)/dt + N_v(u^n, v^n) - nu * Laplace(v^(n+1)) + dp^(n+1)/dy = f2^(n+1)
div(U^(n+1)) = 0
```

with

```text
N_u = u * du/dx + v * du/dy
N_v = u * dv/dx + v * dv/dy
```

This is an IMEX scheme:
- implicit in diffusion and pressure,
- explicit in convection.

That is why the matrix does not depend on the current nonlinear iterate and can be factorized once.

## Monolithic linear system per step

Unknown vector:

```text
x = [ all u^(n+1), all v^(n+1), all p^(n+1) ]^T
```

Linear system:

```text
[ (1/dt)I - nu*L_u    0            Gx ] [u^(n+1)]   [ (1/dt)u^n - N_u^n + f1 ]
[ 0                   (1/dt)I-nu*L_v Gy ] [v^(n+1)] = [ (1/dt)v^n - N_v^n + f2 ]
[ Dx                  Dy           0  ] [p^(n+1)]   [ 0                      ]
```

where

```text
N_u^n = N_u(u^n, v^n)
N_v^n = N_v(u^n, v^n)
```

## Discrete equations on the MAC grid

Below,

```text
u(i,j) means velocity at vertical face (i+1/2, j)
v(i,j) means velocity at horizontal face (i, j+1/2)
p(i,j) means pressure at cell center
```

### 1) u-equation, i=1..Nx-1, j=0..Ny-1

```text
(u_new(i,j) - u_old(i,j))/dt
+ N_u_old(i,j)
- nu * (u_xx_new + u_yy_new)
+ (p_new(i,j) - p_new(i-1,j)) / dx
= f1(x_i, y_j, t_new)
```

or equivalently

```text
[(1/dt)I - nu*L_u] u_new + Gx p_new
= (1/dt) u_old - N_u_old + f1 + lid_term
```

Second-order central differences for diffusion:

```text
u_xx(i,j) = (u(i+1,j) - 2*u(i,j) + u(i-1,j)) / dx^2
```

Interior in y:

```text
u_yy(i,j) = (u(i,j+1) - 2*u(i,j) + u(i,j-1)) / dy^2
```

Bottom wall:

```text
u_yy(i,0) = (u(i,1) - 3*u(i,0)) / dy^2
```

Top wall:

```text
u_yy(i,Ny-1) = (u(i,Ny-2) - 3*u(i,Ny-1)) / dy^2 + 2*U_lid/dy^2
```

The lid contribution `2*nu*U_lid/dy^2` is moved to the right-hand side.

### 2) v-equation, i=0..Nx-1, j=1..Ny-1

```text
(v_new(i,j) - v_old(i,j))/dt
+ N_v_old(i,j)
- nu * (v_xx_new + v_yy_new)
+ (p_new(i,j) - p_new(i,j-1)) / dy
= f2(x_i, y_j, t_new)
```

Second-order central differences for diffusion:

Interior in x:

```text
v_xx(i,j) = (v(i+1,j) - 2*v(i,j) + v(i-1,j)) / dx^2
```

Left/right walls:

```text
v_xx(i,j) -> (v(neighbor,j) - 3*v(i,j)) / dx^2
```

In y:

```text
v_yy(i,j) = (v(i,j+1) - 2*v(i,j) + v(i,j-1)) / dy^2
```

### 3) Incompressibility at pressure cells

```text
(u_new(i+1,j) - u_new(i,j))/dx + (v_new(i,j+1) - v_new(i,j))/dy = 0
```

### 4) Pressure gauge

For one cell, continuity is replaced by

```text
p_new(0,0) = 0
```

to remove the constant-pressure null space.

## Discrete nonlinear term

The current implementation computes the convective term from the old layer using central differences.

### u-momentum convection

At the `u` location:

```text
N_u(i,j) = u(i,j) * du/dx(i,j) + v_at_u(i,j) * du/dy(i,j)
```

with

```text
du/dx(i,j) = (u(i+1,j) - u(i-1,j)) / (2*dx)
du/dy(i,j) = (u_bc(i,j+1) - u_bc(i,j-1)) / (2*dy)
```

and the interpolated cross-velocity

```text
v_at_u(i,j) =
0.25 * [ v(i-1,j) + v(i,j) + v(i-1,j+1) + v(i,j+1) ]
```

### v-momentum convection

At the `v` location:

```text
N_v(i,j) = u_at_v(i,j) * dv/dx(i,j) + v(i,j) * dv/dy(i,j)
```

with

```text
dv/dx(i,j) = (v_bc(i+1,j) - v_bc(i-1,j)) / (2*dx)
dv/dy(i,j) = (v(i,j+1) - v(i,j-1)) / (2*dy)
```

and the interpolated cross-velocity

```text
u_at_v(i,j) =
0.25 * [ u(i,j-1) + u(i+1,j-1) + u(i,j) + u(i+1,j) ]
```

## Why NaN can appear at large Reynolds number

Because convection is explicit, the method is CFL-limited. Roughly,

```text
dt * max(|u|) / dx << 1
dt * max(|v|) / dy << 1
```

must hold well enough.

If `nu` is very small and `dt` is too large, the explicit convective term can blow up and produce `NaN`.
For high-Re runs, either
- decrease `dt`,
- reduce resolution for experiments,
- or replace the central convective discretization by a more dissipative upwind flux.

## Global indexing of unknowns

```text
u unknowns first:
  i=1..Nx-1, j=0..Ny-1
  count nu_unknowns = (Nx-1)*Ny

then v unknowns:
  i=0..Nx-1, j=1..Ny-1
  count nv_unknowns = Nx*(Ny-1)

then p unknowns:
  i=0..Nx-1, j=0..Ny-1
  count np_unknowns = Nx*Ny
```

Helper mappings in code:

```text
u_unknown_idx(i,j)
v_unknown_idx(i,j)
p_unknown_idx(i,j)
```

## Output from Python script

`main.py`:
- runs time stepping through the C++ solver,
- creates `stokes_velocity_quiver.gif`,
- creates `stokes_streamlines.gif`,
- creates `stokes_pressure.gif`,
- creates `stokes_vorticity.gif`,
- creates `stokes_max_divergence.png`.

## What vorticity means

In 2D the scalar vorticity is

```text
omega = dv/dx - du/dy
```

It measures local rotation of the flow:
- large positive `omega` means local counterclockwise spin,
- large negative `omega` means local clockwise spin,
- `omega` close to zero means the flow there is mostly shear-free rotation-wise.
