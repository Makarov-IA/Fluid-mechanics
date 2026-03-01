# Monolithic MAC Stokes Solver (2D)

This folder contains a 2D incompressible Stokes solver on a staggered MAC grid.
Time discretization: first-order backward Euler.
Space discretization: second-order central differences.

## Continuous equations

```text
du/dt - nu * Laplace(u) + dp/dx = f1
dv/dt - nu * Laplace(v) + dp/dy = f2
du/dx + dv/dy = 0
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

## Monolithic linear system per time step

Unknown vector is:

```text
x = [ all u^(n+1), all v^(n+1), all p^(n+1) ]^T
```

System:

```text
[ (1/dt)I - nu*L_u    0            Gx ] [u^(n+1)]   [ (1/dt)u^n + f1 ]
[ 0                   (1/dt)I-nu*L_v Gy ] [v^(n+1)] = [ (1/dt)v^n + f2 ]
[ Dx                  Dy           0  ] [p^(n+1)]   [ 0               ]
```

## Discrete equations (all cases)

Below, `u(i,j)` means velocity at vertical face `(i+1/2, j)`,
`v(i,j)` means velocity at horizontal face `(i, j+1/2)`,
`p(i,j)` means pressure at cell center.

### 1) u-equation, i=1..Nx-1, j=0..Ny-1

```text
(u_new(i,j) - u_old(i,j))/dt
- nu * ( u_xx + u_yy )
+ ( p_new(i,j) - p_new(i-1,j) )/dx
= f1(x_i, y_j, t_new)
```

Interior in y (`1 <= j <= Ny-2`):

```text
u_xx = (u(i+1,j) - 2*u(i,j) + u(i-1,j)) / dx^2
u_yy = (u(i,j+1) - 2*u(i,j) + u(i,j-1)) / dy^2
```

Bottom wall (`j=0`, no-slip, ghost `u(i,-1) = -u(i,0)`):

```text
u_yy = (u(i,1) - 3*u(i,0)) / dy^2
```

Top lid (`j=Ny-1`, ghost `u(i,Ny) = 2*U_lid - u(i,Ny-1)`):

```text
u_yy = (u(i,Ny-2) - 3*u(i,Ny-1)) / dy^2 + 2*U_lid/dy^2
```

The `+ 2*nu*U_lid/dy^2` term is moved to RHS.

### 2) v-equation, i=0..Nx-1, j=1..Ny-1

```text
(v_new(i,j) - v_old(i,j))/dt
- nu * ( v_xx + v_yy )
+ ( p_new(i,j) - p_new(i,j-1) )/dy
= f2(x_i, y_j, t_new)
```

Interior in x (`1 <= i <= Nx-2`):

```text
v_xx = (v(i+1,j) - 2*v(i,j) + v(i-1,j)) / dx^2
```

Left/right walls (`i=0` or `i=Nx-1`, no-slip via ghost):

```text
v_xx -> (v(neighbor,j) - 3*v(i,j)) / dx^2
```

In y (`j=1..Ny-1`, with v(i,0)=v(i,Ny)=0):

```text
v_yy = (v(i,j+1) - 2*v(i,j) + v(i,j-1)) / dy^2
```

### 3) Incompressibility at p-cells, i=0..Nx-1, j=0..Ny-1

```text
(u_new(i+1,j) - u_new(i,j))/dx + (v_new(i,j+1) - v_new(i,j))/dy = 0
```

This is second-order on uniform MAC grids.

### 4) Pressure gauge

For one cell, continuity equation is replaced by:

```text
p_new(0,0) = 0
```

This removes pressure null-space (constant shift ambiguity).

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

These map grid indices `(i,j)` to global matrix row/column indices.

## Output from Python script

`main.py`:
- runs time stepping through C++ solver,
- creates `stokes_evolution.gif`,
- creates `stokes_max_divergence.png`.
