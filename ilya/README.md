# 2-D MAC Navier-Stokes Solver

This directory contains a 2-D incompressible Navier-Stokes solver on a
staggered MAC grid. The scheme is semi-implicit:

- viscosity and pressure are treated implicitly,
- convection is treated explicitly from the previous time layer,
- each time step solves one monolithic Stokes system.

The project has four user-facing modes:

- `simulation` — time-dependent run with plots, videos, and a fixed-time state export,
- `steady` — fixed-point Newton-GMRES solve that starts from the fixed-time state
  saved by `simulation`,
- `linearize` — matrix-free eigenmode solve for the operator linearized around
  `plots/steady/state_internal.pkl`,
- `projected-run` — simulation from the steady state with the forcing component
  along selected unstable eigenmodes removed.

## Workflow

1. Build the shared library:

   ```bash
   make compile
   ```

2. Run the unsteady simulation:

   ```bash
   make run
   ```

   This writes:

   - `plots/final_state/*` — final-time plots,
   - `plots/fixed_time_state/state.pkl` — cell-centred snapshot nearest to
     `output.fixed_time_state_t`,
   - `plots/fixed_time_state/state_internal.pkl` — exact MAC-state used by
     `steady`,
   - `plots/*.mp4` — videos,
   - `plots/stokes_velocity_change.png` only when
     `output.save_velocity_change_plot: true`.

3. Run the steady solver:

   ```bash
   make steady
   ```

   `steady` reads only `plots/fixed_time_state/state_internal.pkl` as its
   initial guess.
   The converged internal state is written to `plots/steady/state_internal.pkl`.

4. Linearize around the steady state and compute eigenvectors:

   ```bash
   make linearize
   ```

   This writes `plots/linearized/eigenpairs.pkl`.

5. Run the projected-forcing simulation:

   ```bash
   make projected-run
   ```

   This starts from `plots/steady/state_internal.pkl`, uses
   `plots/linearized/eigenpairs.pkl`, removes the configured unstable-mode
   projection from `[fu, fv]`, and writes outputs under `plots/projected_run`.
   It uses `projected_run.t_end`, `projected_run.n_steps`, and its own video
   settings. Tolerance-based early stop is always disabled for this mode.

## Configuration

All runtime parameters live in `config.yaml`.

- `domain`, `grid`, `physics`, `time`: geometry and discretisation
- `output.video_fps`, `output.video_speed`: video export settings
- `output.save_velocity_change_plot`: opt-in plot of
  `||U_n - U_{n-1}||_inf` versus time
- `output.fixed_time_state_t`: target time for the snapshot exported to
  `plots/fixed_time_state/*.pkl`
- `convergence.tol`: early stop for simulation mode
- `steady_solver.*`: Newton-GMRES parameters
- `linearization.*`: matrix-free eigenmode parameters
- `projected_run.*`: independent projected-run runtime settings and unstable-mode
  forcing projection parameters
- `boundary`, `forcing`: symbolic expressions evaluated with NumPy

## Numerics

The solver advances

```text
u_t + (u · ∇)u - νΔu + ∇p = f
∇·u = 0
```

with backward Euler for viscosity/pressure and explicit Euler for convection.
The MAC layout is

```text
p[i,j] : cell centres        size Nx × Ny
u[i,j] : vertical faces      size (Nx+1) × Ny
v[i,j] : horizontal faces    size Nx × (Ny+1)
```

The steady solver looks for a fixed point of one IMEX step:

```text
U* = Φ(U*)
```

and solves `G(U) = Φ(U) - U = 0` by damped Newton-GMRES.

The linearization mode uses the stationary Navier-Stokes residual with the time
derivatives set to zero:

```text
R(U, p) = [(u · ∇)u - νΔu + ∇p - f, ∇·u]
```

It computes eigenpairs of the velocity operator
`L = -P D_momentum R(U*)` by matrix-free finite differences, where `P` is the
MAC pressure projection enforcing `∇·u = 0`. The saved eigenvectors use the
internal MAC ordering `[u_vec, v_vec, p]`; pressure modes are recovered from the
projection solve.
