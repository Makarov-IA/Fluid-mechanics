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
- `linearize` — eigenmode solve for the operator linearized around
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

   - `plots/run/final_state/*` — final-time plots,
   - `plots/run/fixed_time_state/state.pkl` — cell-centred snapshot nearest to
     `run.fixed_time_state_t`,
   - `plots/run/fixed_time_state/state_internal.pkl` — exact MAC-state used by
     `steady`,
   - `plots/run/*.mp4` — videos,
   - `plots/run/stokes_velocity_change.png` only when
     `run.save_velocity_change_plot: true`.

3. Run the steady solver:

   ```bash
   make steady
   ```

   `steady` reads only `plots/run/fixed_time_state/state_internal.pkl` as its
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

- `domain`, `grid`, `physics`: geometry and viscosity
- `output_dir` (optional, top level): folder for all results. Without it each
  mode writes next to its input: `steady` with
  `initial_state_path: plots_fast/run/...` writes to `plots_fast/steady`,
  `linearize` / `projected-run` follow `linearization.state_path` /
  `projected_run.state_path`; `run` uses `plots`.
- `run.t_end`, `run.n_steps`: time interval and step count for `make run`
- `run.video_fps`, `run.video_speed`: video export settings
- `run.save_velocity_change_plot`: opt-in plot of
  `||U_n - U_{n-1}||_inf / Δt` versus time
- `run.fixed_time_state_t`: target time for the snapshot exported to
  `plots/run/fixed_time_state/*.pkl`
- `run.convergence_tol`: early stop for simulation mode
- `linear_solver.method`: how each time step's Stokes system is solved in
  `run` and `projected-run` — `direct` (sparse LDLᵀ, default) or `fast`
  (Uzawa-PCG in `fast_stokes.h`: exact sine-basis/tridiagonal velocity solves,
  CG on the pressure Schur complement; same solution up to round-off, macOS
  only). `fast_tol`, `fast_extrapolation`, `fast_parallel` tune it. `steady`
  always uses `direct`.
- `steady_solver.*`: Newton-GMRES parameters
- `linearization.*`: linearization and eigenmode-selection parameters
- `linearization.sigma`, `linearization.krylov_dim`: shift and Krylov size of the
  shift-invert eigensolver
- `projected_run.*`: independent projected-run runtime settings and the
  stabilisation: `method` (`forcing` | `feedback`), `feedback_alpha`,
  `real_threshold` (modes with `Re λ` above it are stabilised)
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

and solves `G(U) = Φ(U) - U = 0` by damped Newton-GMRES inside the C++ backend.

The linearization mode uses the stationary Navier-Stokes residual with the time
derivatives set to zero:

```text
R(U, p) = [(u · ∇)u - νΔu + ∇p - f, ∇·u]
```

The C++ backend analytically linearizes the momentum residual,
`J = -D_u R(U*)`, assembles it as a sparse matrix (exact coloured probing of
the stencil) and works with the velocity–pressure pencil

```text
[ J  -G ] [q]         [q]
[ D   0 ] [π] = λ ·   [0]        ⇔   L q = λ q,  L = P J,  ∇·q = 0
```

(`G` — pressure gradient, `D` — divergence with the gauge `p(0,0) = 0`).
Eigenvalues with the largest real part are found by shift-invert Arnoldi
around `linearization.sigma` (one sparse LU of `A − σB`), then every pair is
refined by inverse iteration, and the matching **adjoint (left) vector** `a`
(`aᵀL = λaᵀ`, normalised `aᵀq = 1`) is computed from the transposed LU.
`plots/linearized/eigenpairs.pkl` stores `eigenvalues`, `eigenvectors`
(`[u_vec, v_vec, p]`), `adjoint_vectors` (`[u_vec, v_vec]`) and the residuals
`‖[Jq − Gπ − λq; Dq]‖`.

### Stabilisation (`projected_run.method`)

`forcing` (open loop): the forcing is replaced once by `F − Proj(F)`, the
orthogonal projection onto the unstable modes removed.

`feedback`: let `u_s` be the steady state, `q_k`, `a_k` the unstable right and
adjoint modes and `Π = Q Wᵀ` the real oblique projector onto their span along
the stable eigenspace (`Q = [Re q, Im q]`, `W` from `a` with `WᵀQ = I`).
In every step the time derivative uses the velocity without the unstable part
of the deviation,

```text
u*ₙ = uⁿ − δₙ,        δₙ = α · Π (uⁿ − u_s),

(uⁿ⁺¹ − u*ₙ)/Δt + (uⁿ·∇)uⁿ − νΔuⁿ⁺¹ + ∇pⁿ⁺¹ = f,     ∇·uⁿ⁺¹ = 0,
```

i.e. the correction is moved to the right-hand side and convection keeps `uⁿ`:

```text
(uⁿ⁺¹ − uⁿ)/Δt + (uⁿ·∇)uⁿ − νΔuⁿ⁺¹ + ∇pⁿ⁺¹ = f − δₙ/Δt.
```

The matrix of the step is unchanged, so both linear solvers work.  `α ∈ (0, 2)`
(`projected_run.feedback_alpha`, default 1): `α = 1` removes the unstable
component of the deviation completely every step; `α/Δt` is the feedback
gain.  `u_s` stays a steady solution (`δ = 0` there).  The run saves
`stabilization_correction.png` with `‖δₙ‖∞` and `‖uⁿ − u_s‖∞` versus time.
`steady` is never controlled.

## Внутренние Задачи

- **Задача о кювете / cavity-flow**: set body forces to zero and prescribe wall
  velocities in `boundary.*`; `make run` produces the time evolution and
  fixed-time state.
- **Задача с произвольными правыми частями**: set symbolic `forcing.fu` and
  `forcing.fv` expressions in `config.yaml`; they are evaluated on MAC faces
  with NumPy.
- **Поиск нестационарных мод**: run `make steady`, then `make linearize` to find
  eigenmodes of the stationary Navier-Stokes operator; `make projected-run` can
  run from the steady state with selected unstable forcing components removed.
