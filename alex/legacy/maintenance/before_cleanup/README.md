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

## Python: факторизованный CPU-солвер

`solver_python/` повторяет последовательность текущего C++:
`solvePsi → ApplyThomBoundary → solveOmega → computeResiduals`.
Аракава и факторизованная схема встроены; переключателей численного метода нет.
Это релаксация по **псевдовремени**, включая уравнение для `psi`, а не интегратор
физически нестационарного течения. DST/FFT из GPU-солвера здесь не используется.

Единственный рабочий конфиг: `solver_python/config.yaml`. Новый запуск не читает
`RUN`; все параметры, включая число потоков, находятся в YAML.
Команды из папки `alex`, в Python-окружении с установленными зависимостями:

```bash
python -m pip install -r solver_python/requirements.txt
python -m solver_python --check
python -m solver_python --benchmark 1000
python -m solver_python
```

Также можно вызвать `python /полный/путь/alex/solver_python/run.py` из любой папки.
`--check` проверяет конфиг и источник; `--benchmark` считает без записи результатов,
отдельно измеряя прогрев JIT и вычисления. Рабочий YAML сейчас содержит
демонстрационный массив из четырёх магнитов, Re=2400 и 10 000 шагов.

Простые bash-запуски находятся в `solver_python_run/`. Пути и аргументы заданы
внутри каждого скрипта; используется ровно `../.venv/bin/python` относительно
`alex`, без поиска другого Python, активации окружений и резервных вариантов.
Из папки `alex`:

```bash
bash solver_python_run/check.sh           # YAML, источник и диапазон магнитного поля
bash solver_python_run/run.sh             # Расчёт по единому YAML
bash solver_python_run/benchmark.sh       # Замер скорости, без записи снимков
bash solver_python_run/test.sh            # Все автоматические тесты
bash solver_python_run/verify_magnets.sh   # Проверка магнитов + 10 000 шагов
```

Скрипты также работают из любой другой папки при вызове по полному пути.
`verify_magnets.sh` использует магниты из рабочего YAML и требует `type: magnets`.
Он проверяет нормировку каждого диполя, суперпозицию, смену полярности,
аналитические производные против численных, знак и масштаб источника, отклик
солвера и чтение всех сохранённых снимков. Результаты каждого такого запуска
попадают в `results/binaries/magnets_check_<дата-время>/`, отчёт — `verification.json`.
В `forcing.npz` сохранены `Bz_T`, `source`, `xs`, `ys`; `xs, ys` безразмерные.
Этот контрольный запуск проверяет работоспособность, а не гарантирует стационар.

NumPy хранит поля `float64`, Numba компилирует циклы. Матрицы прогонок
факторизуются один раз. `execution.threads: 1` включает последовательные ядра;
значение больше 1 распараллеливает только независимые линии прогонок.
Направления прогонок и шаги идут последовательно. На малых сетках дополнительные
потоки могут замедлять расчёт; сравнивайте через `--benchmark`.

В `forcing` доступны ровно три режима:

- `type: none` — нулевой источник, например для чистой каверны.
- `type: custom`, `file: sources/harmonics.py` — файл Python с функцией
  `source(x, y, lx, ly)`. `x` имеет форму `(nx, 1)`, `y` — `(1, ny)`.
  Функция возвращает скаляр или массив, приводимый к `(nx, ny)` через broadcasting.
  Возвращать нужно **готовый безразмерный источник завихренности S**, который
  прибавляется в `omega_t = (1/Re) Δomega + J_A + S`. Это не сила `f_x`
  и не магнитное поле; дополнительного дифференцирования нет.
  Файл исполняется как пользовательский Python-код; пример с прежними
  шестью гармониками находится в `sources/harmonics.py`.
- `type: magnets` — любое число вертикальных точечных диполей
  `[x_m, y_m, d_m, B_axis_T]`; рабочий пример всех полей находится в том же YAML.
  `d` отсчитывается от точки диполя до дна, `plane_height_m` добавляет высоту
  расчётной плоскости. `B` — вклад отдельного магнита непосредственно над ним
  в этой плоскости. Ток задаётся плотностью `[jx, jy]` в А/м².
  Поле и его производные суммируются аналитически, источник переводится в
  безразмерный вид. Проверяется `Re = U*L/nu`. Примерные числа магнитов —
  демонстрационные, не параметры измеренной установки.

Источник вычисляется один раз. Относительные пути в YAML разрешаются от самого
YAML, а не от текущей рабочей папки. Неизвестные/повторные ключи, недопустимые
числа и нормальные скорости на непроницаемых стенках отклоняются.

Результаты: `results/binaries/<experiment>/result_*.bin` в общем формате
`ALEXBIN1` (`psi, omega, u, v`) и `residual_history.csv` с прежними колонками.
Шаг `0` — первый выполненный шаг, его время равно `dt`, как в C++.
Последний шаг сохраняется всегда. Дополнительно записываются конфиг, копия
пользовательского источника, `forcing.npz` и `summary.json` с типом времени.
Непустой каталог результатов не перезаписывается.

`time.mode` поддерживает `fixed_dt_steps`, `fixed_steps` (`dt=t_max/n_time_steps`)
и `till_converges`. В последнем случае `n_time_steps` — предел итераций;
при его исчерпании записывается `status: step_limit`, процесс завершается с кодом 2.
NaN/Inf считаются ошибкой, а не сходимостью.

Исправлены знаки конвекции и источника, знаки Тома для нижней/правой стенок
и границы промежуточной переменной `g=(I-dt/Re*Dyy)omega` в прогонках.
Исходный C++ не изменён. Общий бинарный читатель совместим; графические скрипты
не менялись. Старые Ньютон/анализ устойчивости пока не адаптированы к новому YAML,
источнику и исправленным знакам — их нельзя автоматически подключать к этим данным.

Проверки:

```bash
python -m pip install pytest
python -m pytest solver_python/tests -q
```

Тесты сравнивают шаг с независимыми плотными матричными решениями, проверяют
стационар, инварианты Аракавы, производные диполей, YAML, многопоточность и формат
снимков. При наличии компилятора C++ дополнительно сравнивают траекторию с
исправленной **временной копией** исходного C++-солвера; исходники не меняются.

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
