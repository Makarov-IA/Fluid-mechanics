# Alex Fluid Mechanics Project

Единая структура проекта:

```text
alex/
├── RUN
├── cpu/
├── gpu/
├── common/
├── scripts/
│   ├── bash/
│   └── python/
├── pipeline/
├── results/
└── paper/
```

## Backends

`alex/cpu` - авторский C++ solver:

```text
alex/cpu/solver/
alex/cpu/forcing/
alex/cpu/configs/pulsation.cfg
```

`alex/gpu` - PyTorch/CUDA solver для lid-driven cavity:

```text
alex/gpu/torch_cavity_solver.py
alex/gpu/configs/
```

Общий бинарный формат снапшотов находится в:

```text
alex/common/snapshot_io.py
```

## Docs

Математическая постановка:

```text
alex/docs/math_model.md
```

Сравнение CPU и GPU солверов:

```text
alex/docs/solver_comparison.md
```

## RUN

Все основные настройки задаются в одном файле:

```text
alex/RUN
```

Там выбираются:

```text
BACKEND
EXP_NAME
CONFIG_PATH
DEVICE
DTYPE
```

Все папки результатов строятся по `EXP_NAME`.

## Results

```text
alex/results/binaries/<EXP_NAME>/
alex/results/figures/<EXP_NAME>/
alex/results/videos/<EXP_NAME>/
alex/results/tables/<EXP_NAME>/
alex/results/stationary/<EXP_NAME>/
alex/results/newton/<EXP_NAME>/
alex/results/linear_stability/<EXP_NAME>/
```

Основные данные пишутся в бинарники:

```text
result_*.bin
```

В них хранятся:

```text
psi, omega, u, v
```

Служебные таблицы остаются CSV.

## Main Commands

Короткие entrypoints:

```bash
bash alex/scripts/run.sh
bash alex/scripts/render.sh
bash alex/scripts/video.sh
```

Полные bash-скрипты лежат здесь:

```text
alex/scripts/bash/
```

Python tools лежат здесь:

```text
alex/scripts/python/
```

## Stationary, Newton, Stability

```bash
bash alex/scripts/bash/find_stationary.sh
bash alex/scripts/bash/newton_from_csv.sh
bash alex/scripts/bash/linear_stability.sh
bash alex/scripts/bash/analyze_filtered.sh
bash alex/scripts/bash/filtered_video.sh
```

## GPU Batch

```bash
bash alex/gpu/run_all.sh
bash alex/gpu/render_all.sh
```

Параллельно по нескольким GPU:

```bash
RUN_PARALLEL=1 GPU_LIST=0,1,2 bash alex/gpu/run_all.sh
```

Smoke test:

```bash
bash alex/gpu/smoke_test.sh
```

## Compatibility

`alex/pipeline` оставлен как совместимый слой. Его скрипты вызывают новые скрипты из `alex/scripts/bash`.
