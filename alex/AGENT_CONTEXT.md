# Agent Context

This file is for future coding agents working on `alex`. It records stable project decisions and current conventions. Keep it short and update it when project structure, run flow, or numerical assumptions change.

## Project Intent

`alex` is a coursework-oriented fluid mechanics project using the streamfunction-vorticity formulation.

There are two backends:

- `cpu`: the author's C++ solver. Preserve it as the main original implementation.
- `gpu`: a PyTorch/CUDA solver used for fast lid-driven cavity experiments.

The CPU and GPU solvers share the same physical variables and snapshot format, but they are not identical numerical algorithms.

## Current Structure

```text
alex/
├── RUN
├── README.md
├── AGENT_CONTEXT.md
├── cpu/
├── gpu/
├── common/
├── docs/
├── scripts/
│   ├── bash/
│   └── python/
├── pipeline/
├── results/
└── paper/
```

Important locations:

```text
alex/cpu/solver/solver.cpp
alex/cpu/forcing/omega_forcing.cpp
alex/cpu/configs/pulsation.cfg

alex/gpu/torch_cavity_solver.py
alex/gpu/configs/

alex/common/snapshot_io.py

alex/docs/math_model.md
alex/docs/solver_comparison.md
```

Only one general README should exist:

```text
alex/README.md
```

Do not reintroduce many scattered Markdown files unless the user explicitly asks for them.

## Run Configuration

The central run configuration is:

```text
alex/RUN
```

It defines:

```text
BACKEND
EXP_NAME
CONFIG_PATH
RESULTS_ROOT
BIN_DIR
FIG_DIR
VIDEO_DIR
TABLE_DIR
STATIONARY_DIR
NEWTON_DIR
STABILITY_DIR
DEVICE
DTYPE
```

Results are grouped by `EXP_NAME`, not by config filename.

## Results Layout

Use this structure for all new outputs:

```text
alex/results/binaries/<EXP_NAME>/
alex/results/figures/<EXP_NAME>/
alex/results/videos/<EXP_NAME>/
alex/results/tables/<EXP_NAME>/
alex/results/stationary/<EXP_NAME>/
alex/results/newton/<EXP_NAME>/
alex/results/linear_stability/<EXP_NAME>/
```

Binary snapshots:

```text
result_*.bin
```

Snapshot fields:

```text
psi
omega
u
v
```

Service tables remain CSV, for example:

```text
residual_history.csv
vortex_summary.csv
eigenvalues.csv
filtered_snapshots.csv
```

## Script Layout

Keep `alex/scripts` clean:

```text
alex/scripts/run.sh
alex/scripts/render.sh
alex/scripts/video.sh
alex/scripts/bash/
alex/scripts/python/
```

The top-level `alex/scripts/*.sh` files are thin entrypoints only.

Full bash workflows live in:

```text
alex/scripts/bash/
```

Python tools live in:

```text
alex/scripts/python/
```

`alex/pipeline` is a compatibility layer. Its scripts should call `alex/scripts/bash/*` instead of duplicating logic.

## Standard Commands

Main workflow:

```bash
bash alex/scripts/run.sh
bash alex/scripts/render.sh
bash alex/scripts/video.sh
```

Stationary/Newton/stability workflow:

```bash
bash alex/scripts/bash/find_stationary.sh
bash alex/scripts/bash/newton_from_csv.sh
bash alex/scripts/bash/linear_stability.sh
bash alex/scripts/bash/analyze_filtered.sh
bash alex/scripts/bash/filtered_video.sh
```

GPU batch workflow:

```bash
bash alex/gpu/run_all.sh
bash alex/gpu/render_all.sh
```

Parallel GPU run example:

```bash
RUN_PARALLEL=1 GPU_LIST=0,1,2 bash alex/gpu/run_all.sh
```

Build CPU backend:

```bash
make -C alex all
```

## Numerical Decisions

The mathematical formulation is documented in:

```text
alex/docs/math_model.md
```

The CPU/GPU algorithm comparison is documented in:

```text
alex/docs/solver_comparison.md
```

Important facts:

- The CPU backend uses the author's C++ factorized scheme with tridiagonal solves.
- The CPU backend uses `OmegaForcing` through `alex/cpu/forcing/omega_forcing.cpp`.
- In the CPU equations, the source term is effectively `S_omega = -OmegaForcing`.
- The GPU backend is for lid-driven cavity with `S_omega = 0`.
- The GPU backend solves the Poisson equation for `psi` via DST/FFT.
- The GPU backend advances `omega` explicitly in pseudo-time.
- Both backends can use central or Arakawa Jacobian through `use_arakawa`.
- Do not claim CPU and GPU are bitwise or method-identical.

Safe wording:

```text
The GPU backend is an accelerated implementation of the same streamfunction-vorticity formulation for the cavity benchmark, not a line-by-line port of the C++ factorized solver.
```

## Files To Preserve

Do not delete or rewrite the author's CPU solver casually:

```text
alex/cpu/solver/solver.cpp
alex/cpu/solver/solver.hpp
alex/cpu/solver/solver_utils.hpp
alex/cpu/forcing/omega_forcing.cpp
alex/cpu/forcing/omega_forcing.hpp
```

If refactoring around these files, keep behavior stable unless the user explicitly asks for numerical changes.

## Git And Generated Files

Heavy/generated raw outputs should generally remain ignored:

```text
alex/results/binaries/
alex/results/figures/*/frames/
alex/results/linear_stability/*/filtered_binaries/
```

Paper-ready files may be committed:

```text
alex/results/figures/**/*.png
alex/results/videos/**/*.mp4
alex/results/tables/**/*.csv
```

Before committing, check:

```bash
git status --short
git diff --cached --stat
```

## Verification Checklist

After structural changes, run:

```bash
make -C alex all
python3 -m py_compile \
  alex/common/snapshot_io.py \
  alex/gpu/torch_cavity_solver.py \
  alex/gpu/profile_cavity.py \
  alex/scripts/python/*.py
bash -n alex/scripts/*.sh alex/scripts/bash/*.sh alex/gpu/*.sh alex/pipeline/*.sh alex/pipeline/tools/*.sh
```

If `torch` is not installed locally, syntax checking by `py_compile` may fail on GPU files because they import `torch`. In that case, note the limitation and verify on the server.

## Open Design Note

If the user later wants the GPU solver to match the CPU solver more closely, add a second GPU scheme, for example:

```text
scheme=factorized
```

That scheme should reproduce:

```text
solvePsi
ApplyThomBoundary
solveOmega
computeResiduals
```

using batched tridiagonal solves on GPU. Keep the current DST cavity solver as a separate mode.
