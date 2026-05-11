# Torch CUDA cavity experiments, 601 x 601

This folder contains a GPU backend for the lid-driven cavity verification runs:

- `Re = 1000`
- `Re = 5000`
- `Re = 10000`
- grid `601 x 601`
- domain `1 x 1`
- top lid velocity `u = 1`
- all other walls fixed
- forcing is zero
- nonlinear term is the central Jacobian (`use_arakawa=false`)

The solver writes the same binary snapshot format as the C++ pipeline:

```text
result_*.bin
residual_history.csv
```

The existing plotting and stationary-detection scripts can read these snapshots.

## Server setup with existing conda environment

Activate your existing conda environment with `torch`, `numpy`, `matplotlib`,
`scipy`, `imageio`, `imageio-ffmpeg`, and `tqdm` already installed. Then, from
the repo root:

```bash
make -C alex all
```

If you need to install missing plotting packages inside that conda environment:

```bash
conda install numpy matplotlib scipy tqdm
pip install imageio imageio-ffmpeg
```

Check CUDA:

```bash
python3 - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Run all experiments

Sequential, on GPU 0:

```bash
bash alex/torch_cavity_601/run_all.sh
```

Parallel over GPUs:

```bash
RUN_PARALLEL=1 GPU_LIST=0,1,2 bash alex/torch_cavity_601/run_all.sh
```

Outputs go to:

```text
alex/torch_cavity_601/results/re1000
alex/torch_cavity_601/results/re5000
alex/torch_cavity_601/results/re10000
```

## Render plots and videos

After the runs:

```bash
bash alex/torch_cavity_601/render_all.sh
```

This creates per-Re:

```text
plots/frames/*.png
plots/videos/*.mp4
profiles/final_streamplot.png
profiles/final_profiles.png
profiles/vortex_summary.csv
```

## Quick smoke test

Before the long runs:

```bash
bash alex/torch_cavity_601/smoke_test.sh
```
