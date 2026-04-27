# Project Specification

## Назначение

Проект решает двумерные несжимаемые уравнения Навье-Стокса на равномерной
staggered MAC-сетке. Основная цель текущей версии - запуск нестационарной
симуляции, сохранение графиков/видео, поиск стационарного состояния методом
Newton-GMRES и линеаризация около найденного стационара.

В текущем проекте есть четыре пользовательских сценария:

- `simulation` - нестационарный расчет по времени.
- `steady` - поиск стационарного решения как неподвижной точки одного шага по времени.
- `linearize` - линеаризация около найденного steady-state и поиск собственных векторов.
- `projected-run` - прогон от steady-state с вычитанием forcing-проекции на выбранные неустойчивые моды.

В текущей версии нет:

- режима `mode` в `config.yaml`;
- нескольких вариантов начального приближения для steady;
- начального приближения по среднему, случайному полю, нулю или restart-файлу.

Выбор сценария делается командой:

```bash
make run
make steady
make linearize
make projected-run
```

или напрямую:

```bash
.venv/bin/python main.py --mode simulation
.venv/bin/python main.py --mode steady
.venv/bin/python main.py --mode linearize
.venv/bin/python main.py --mode projected-run
```

## Структура проекта

```text
ilya/
  Makefile                - пользовательские команды сборки и запуска
  compile.sh              - сборка C++ shared library
  config.yaml             - единственный runtime-конфиг
  main.py                 - entrypoint и orchestration workflow
  requirements.txt        - Python-зависимости
  stokes_mac.h            - C++ класс MAC-солвера и C ABI
  stokes_mac.cpp          - реализация C++ солвера
  solver/
    config.py             - SimConfig, Snapshot, выражения, BC/force arrays
    lib.py                - ctypes wrapper над solver.{dylib,so,dll}
  simulation/
    runner.py             - Python-обертка нестационарной симуляции
    steady.py             - Python-обертка C++ fixed-point Newton-GMRES
    linearized.py         - Python-обертка C++ dense linearization/eig
    projected_run.py      - подготовка projected-run forcing
  viz/
    levels.py             - общие color levels
    plots.py              - PNG/pickle графики и панели состояния
    video.py              - параллельный рендер MP4
```

Скомпилированная библиотека лежит в корне `ilya`:

```text
solver.dylib  - macOS
solver.so     - Linux
solver.dll    - Windows
```

## Зависимости

Python-зависимости:

```text
numpy
scipy
matplotlib
pillow
pyyaml
rich
imageio
imageio-ffmpeg
```

C++:

- C++17;
- Eigen headers из `../External_libs`;
- OpenMP опционален.

SciPy не участвует в основных численных solve/eig этапах; он остается только
для вспомогательной визуализации в `viz/plots.py`.

`compile.sh` выбирает расширение shared library по ОС. На macOS используется
`clang++`; OpenMP включается только если найден Homebrew `libomp`. На Linux и
Windows передается `-fopenmp`.

Команды:

```bash
make venv
make compile
make run
make steady
make linearize
make projected-run
make clean-plots
make clean-lib
make clean
```

`make clean-plots` удаляет `plots`. `make clean-lib` удаляет собранную shared
library. Эти команды не трогают исходники.

## Поддерживаемые постановки задач

1. **Задача о кювете / cavity-flow.** Задается геометрия, вязкость и граничные
   скорости в `boundary.*`; правая часть может быть нулевой. `make run` строит
   нестационарную траекторию, графики, видео и snapshot в заданный момент.
2. **Задача с произвольными правыми частями.** `forcing.fu` и `forcing.fv`
   задаются строковыми выражениями от `x`, `y`, `t` и вычисляются на MAC-face
   сетках через NumPy. Для `steady` и `linearize` forcing должен быть
   time-independent.
3. **Поиск нестационарных мод около стационара.** `make steady` ищет
   стационарное состояние, `make linearize` применяет оператор
   `L=-P D_momentum R(U0)` и находит его собственные моды через
   C++ dense eig на малых сетках или C++ Arnoldi на больших; `make projected-run` может стартовать из стационара и
   убрать из forcing компоненты по выбранным растущим модам.

## Конфигурация

Все runtime-параметры читаются из `config.yaml` через `SimConfig.from_yaml`.

Минимальная схема:

```yaml
domain:
  lx: 1.0
  ly: 0.5

grid:
  nx: 105
  ny: 105

physics:
  nu: 0.00384615384

run:
  t_end: 10.0
  n_steps: 100000
  video_fps: 30
  video_speed: 1
  save_velocity_change_plot: true
  fixed_time_state_t: 2.2
  convergence_tol: 1.0e-6

steady_solver:
  max_newton_iters: 12
  residual_tol: 1.0e-8
  krylov_tol: 1.0e-6
  krylov_maxiter: 200
  krylov_restart: 100
  jacobian_rdiff: 1.0e-4
  line_search: armijo
  min_step: 1.0e-3

linearization:
  state_path: plots/steady/state_internal.pkl
  n_eigs: 6
  which: LR

projected_run:
  t_end: 20.0
  n_steps: 100000
  video_fps: 10
  video_speed: 1
  save_velocity_change_plot: true
  state_path: plots/steady/state_internal.pkl
  eigenpairs_path: plots/linearized/eigenpairs.pkl
  real_threshold: 1.0e-8
  projection_rcond: 1.0e-12

boundary:
  u_top: "0.0"

forcing:
  fu: "0.0"
  fv: "0.0"
```

Поля:

- `domain.lx`, `domain.ly` - размеры прямоугольной области.
- `grid.nx`, `grid.ny` - число pressure-cells по `x` и `y`; оба значения должны быть больше 1.
- `physics.nu` - кинематическая вязкость, строго положительная.
- `run.t_end` - конечное время обычного `make run`.
- `run.n_steps` - число внутренних шагов по времени для обычного `make run`.
- `run.video_fps` - FPS итогового видео обычного `make run`.
- `run.video_speed` - во сколько раз ускорять физическое время на видео.
- `run.save_velocity_change_plot` - включает сохранение `plots/run/stokes_velocity_change.png`.
- `run.fixed_time_state_t` - время, ближайший snapshot к которому сохраняется в `plots/run/fixed_time_state/*.pkl`.
- `run.convergence_tol` - досрочная остановка simulation по `||U_n-U_{n-1}||_inf`; `0` отключает остановку.
- `steady_solver.*` - параметры Newton-GMRES для steady workflow.
- `linearization.*` - параметры линеаризации и выбора eigenmodes.
- `projected_run.*` - отдельные runtime-параметры projected-run и параметры вычитания forcing-проекции на выбранные eigenmodes.
- `boundary.*` - выражения для граничных условий.
- `forcing.fu`, `forcing.fv` - выражения правых частей на MAC-face сетках.

Производные величины:

```text
dt = t_end / n_steps
capture_fps = video_fps / video_speed
frame_every = max(1, round(1 / (capture_fps * dt)))
```

`frame_every` - сколько внутренних C++ шагов приходится на один сохраненный
snapshot. История `div_history` и `velocity_change_history` хранится для всех
внутренних шагов, а не только для snapshot-кадров.

### Выражения в config.yaml

`forcing` и `boundary` задаются строками, которые вычисляются через NumPy-aware
`eval` с отключенными builtins. Доступные имена:

```text
sin, cos, tan, exp, log, sqrt, abs, tanh, sinh, cosh, pi, e, np
```

Также в выражения передаются координаты `x`, `y`, а для forcing еще `t`.

Пример:

```yaml
forcing:
  fu: "sin(2*pi*x)*cos(4*pi*y)"
  fv: "0"
```

Важно: в simulation Python вычисляет force arrays один раз на начало batch
`t_start`, после чего C++ использует эти массивы постоянными для всех шагов
внутри этого batch. Если forcing зависит от `t`, он обновляется не на каждом
внутреннем шаге, а на каждом batch.

## Математическая модель

Решается система:

```text
u_t + (u · grad)u - nu Laplace(u) + grad(p) = f
div(u) = 0
```

Вектор скорости двумерный:

```text
u = (u, v)
```

Схема по времени IMEX:

- конвекция - явно, центральными разностями с предыдущего слоя;
- вязкость - неявно, backward Euler;
- давление - неявно;
- на каждом шаге решается один монолитный saddle-point Stokes-like system.

Система имеет вид:

```text
(1/dt - nu Laplace) u_new + dp_new/dx = (1/dt) u_old - N_u(u_old,v_old) + f_u + BC_u
(1/dt - nu Laplace) v_new + dp_new/dy = (1/dt) v_old - N_v(u_old,v_old) + f_v + BC_v
div(u_new, v_new) = 0
p_new[0,0] = 0
```

`p[0,0] = 0` - pressure gauge, убирающий константную неопределенность давления.

## MAC-сетка

Область равномерная:

```text
dx = lx / nx
dy = ly / ny
```

Расположение неизвестных:

```text
p[i,j] - cell centers,    i=0..nx-1, j=0..ny-1, size nx x ny
u[i,j] - vertical faces,  i=0..nx,   j=0..ny-1, size (nx+1) x ny
v[i,j] - horizontal faces,i=0..nx-1, j=0..ny,   size nx x (ny+1)
```

Координаты pressure/cell-centered output:

```text
xc[i] = (i + 0.5) * dx
yc[j] = (j + 0.5) * dy
```

Interior velocity unknowns:

```text
u_interior: i=1..nx-1, j=0..ny-1, size (nx-1)*ny
v_interior: i=0..nx-1, j=1..ny-1, size nx*(ny-1)
p_cells:    i=0..nx-1, j=0..ny-1, size nx*ny
```

Глобальный порядок неизвестных в C++ monolithic system:

```text
[ u_interior, v_interior, p_cells ]
```

Индексы:

```text
u_unknown_idx(i,j) = j*(nx-1) + (i-1)
v_unknown_idx(i,j) = nu_unknowns + (j-1)*nx + i
p_unknown_idx(i,j) = nu_unknowns + nv_unknowns + (j*nx + i)
```

Плоское хранение в C++ row-major, `j` first:

```text
p_idx(i,j) = j*nx + i
u_idx(i,j) = j*(nx+1) + i
v_idx(i,j) = j*nx + i
```

Python wrapper при чтении C++ полей делает reshape в `(ny,nx).T`, чтобы в Python
массивы имели форму `(nx, ny)` и индексировались как `[i,j]`.

## Граничные условия

Все BC по умолчанию нулевые.

Есть два типа boundary arrays.

Ghost-correction arrays:

```text
u_top   size nx-1, x=i*dx, y=ly, i=1..nx-1
u_bot   size nx-1, x=i*dx, y=0,  i=1..nx-1
v_left  size ny-1, x=0,  y=j*dy, j=1..ny-1
v_right size ny-1, x=lx, y=j*dy, j=1..ny-1
```

Они используются в ghost-node формулах для диффузии и конвекции около стенок.

Direct face Dirichlet arrays:

```text
u_left  size ny, x=0,  y=(j+0.5)*dy, j=0..ny-1
u_right size ny, x=lx, y=(j+0.5)*dy, j=0..ny-1
v_bot   size nx, x=(i+0.5)*dx, y=0,  i=0..nx-1
v_top   size nx, x=(i+0.5)*dx, y=ly, i=0..nx-1
```

Они напрямую записываются в boundary faces velocity field.

Ghost formulas:

```text
u(i,-1)  = 2*u_bot[i-1] - u(i,0)
u(i,ny)  = 2*u_top[i-1] - u(i,ny-1)
v(-1,j)  = 2*v_left[j-1] - v(0,j)
v(nx,j)  = 2*v_right[j-1] - v(nx-1,j)
```

Для corner/face значений используются текущие direct face BC, по умолчанию ноль.

## Дискретизация конвекции

Конвекция считается явно на старом поле.

Для `u` на interior vertical faces:

```text
N_u(i,j) = u(i,j) * du/dx + v_at_u * du/dy
du/dx = (u(i+1,j) - u(i-1,j)) / (2*dx)
du/dy = (u_ghost(i,j+1) - u_ghost(i,j-1)) / (2*dy)
v_at_u = 0.25 * (v(i-1,j) + v(i,j) + v(i-1,j+1) + v(i,j+1))
```

Для `v` на interior horizontal faces:

```text
N_v(i,j) = u_at_v * dv/dx + v(i,j) * dv/dy
dv/dx = (v_ghost(i+1,j) - v_ghost(i-1,j)) / (2*dx)
dv/dy = (v(i,j+1) - v(i,j-1)) / (2*dy)
u_at_v = 0.25 * (u(i,j-1) + u(i+1,j-1) + u(i,j) + u(i+1,j))
```

C++ параллелит вычисление advection, scatter и divergence через OpenMP, если
библиотека собрана с OpenMP.

## C++ solver

Основной класс:

```cpp
class StokesMac2D {
public:
    StokesMac2D(int nx, int ny, double lx, double ly, double nu, double dt);
    double step(double t, ForceFn f1, ForceFn f2);
    double step_with_force_arrays(double t, const double* fu, const double* fv);
    void run_steps(double t_start, int n_steps, double* div_out);
    void run_steps_diagnostics(double t_start, int n_steps, double* div_out, double* change_out);
    void run_steps_with_force(double t_start, int n_steps, const double* fu, const double* fv, double* div_out);
    void run_steps_with_force_diagnostics(double t_start, int n_steps, const double* fu, const double* fv, double* div_out, double* change_out);
    void set_bc_arrays(...);
    void set_state_arrays(const double* u_interior, const double* v_interior, const double* p_cells);
    void get_state_arrays(double* u_interior, double* v_interior, double* p_cells) const;
    const double* p_data() const;
    const double* u_data() const;
    const double* v_data() const;
};
```

При создании solver:

1. Проверяются `nx`, `ny`, `dt`, `lx`, `ly`, `nu`.
2. Выделяются поля `p_`, `u_`, `v_`.
3. Выделяются work buffers `adv_u_`, `adv_v_`, `rhs_`, `sol_`.
4. Создается sparse matrix монолитной системы.
5. Выполняется `Eigen::SparseLU::analyzePattern`.
6. Выполняется `Eigen::SparseLU::factorize`.
7. Boundary velocities инициализируются текущими BC.

Матрица постоянная для заданных `nx`, `ny`, `lx`, `ly`, `nu`, `dt`, поэтому
факторизуется один раз в конструкторе и переиспользуется на каждом шаге.

На каждом step:

1. Считается explicit advection из текущего velocity field.
2. Собирается RHS.
3. Решается `A*x = rhs`.
4. `sol_` scatter-ится обратно в `u_`, `v_`, `p_`.
5. Считается `last_velocity_change_ = max(||u_new-u_old||_inf, ||v_new-v_old||_inf)`.
6. Возвращается `max_divergence()`.

`max_divergence()` считает:

```text
max | (u(i+1,j)-u(i,j))/dx + (v(i,j+1)-v(i,j))/dy |
```

по всем pressure cells, кроме gauge cell `(0,0)`.

## C ABI

Python взаимодействует с C++ только через `extern "C"` functions:

```cpp
void* stokes_mac_create_c(int Nx, int Ny, double Lx, double Ly, double nu, double dt);
void  stokes_mac_free_c(void* handle);
double stokes_mac_step_c(void* handle, double t, Force2DTime_C f1, Force2DTime_C f2);
void  stokes_mac_run_steps_c(void* handle, double t_start, int n_steps, double* div_out);
void  stokes_mac_run_steps_diagnostics_c(void* handle, double t_start, int n_steps, double* div_out, double* change_out);
void  stokes_mac_set_bc_c(void* handle, const double* u_top, const double* u_bot, const double* v_left, const double* v_right, const double* u_left, const double* u_right, const double* v_bot, const double* v_top);
void  stokes_mac_run_steps_with_force_c(void* handle, double t_start, int n_steps, const double* fu, const double* fv, double* div_out);
void  stokes_mac_run_steps_with_force_diagnostics_c(void* handle, double t_start, int n_steps, const double* fu, const double* fv, double* div_out, double* change_out);
void  stokes_mac_set_state_c(void* handle, const double* u_interior, const double* v_interior, const double* p_cells);
void  stokes_mac_get_state_c(void* handle, double* u_interior, double* v_interior, double* p_cells);
const double* stokes_mac_get_p_c(void* handle);
const double* stokes_mac_get_u_c(void* handle);
const double* stokes_mac_get_v_c(void* handle);
```

Правило ownership:

- `create` возвращает opaque pointer.
- Python обязан вызвать `free`.
- `get_p/u/v` возвращают raw pointer на внутреннюю память solver; Python должен сразу скопировать данные.
- `set_state` требует `u_interior` и `v_interior`; `p_cells` может быть `NULL`, тогда pressure сбрасывается в ноль.

## Python wrapper

`solver/lib.py` содержит `StokesMACLib`.

Обязанности wrapper:

- найти `solver.dylib`, `solver.so` или `solver.dll`;
- загрузить библиотеку через `ctypes.CDLL`;
- задать `argtypes` и `restype` для C ABI;
- управлять lifetime handle;
- конвертировать NumPy arrays в `double*`;
- копировать C++ fields в NumPy arrays формы `(nx,ny)`, `(nx+1,ny)`, `(nx,ny+1)`;
- экспортировать `get_state()` как `(u_interior, v_interior, p_cells)`.

Python wrapper не реализует численную схему. Он только вызывает C++ solver.

## Simulation workflow

`main.py --mode simulation` выполняет:

1. Читает `config.yaml`.
2. Строит `xc`, `yc`.
3. Находит compiled solver library.
4. Вызывает `run_simulation`.
5. Находит final snapshot.
6. Находит snapshot, ближайший к `run.fixed_time_state_t`.
7. Считает общие color levels по всем snapshots.
8. Сохраняет static plots и pickle-состояния.
9. Рендерит MP4 videos.

`simulation/runner.py`:

1. Создает `StokesMACLib`.
2. Передает BC arrays в C++ solver.
3. Идет batch loop с шагом `frame_every`.
4. Для каждого batch:
   - вычисляет force arrays в Python, если forcing не нулевой;
   - вызывает C++ batch stepping с diagnostics;
   - добавляет per-step `div_history`;
   - добавляет per-step `velocity_change_history`;
   - получает поля `p,u,v`;
   - усредняет face velocities в cell-centered `uc,vc`;
   - считает vorticity;
   - сохраняет один `Snapshot`;
   - сохраняет exact `MacState` с внутренними `u_vec`, `v_vec`, `p`.
5. Если последний `velocity_change < run.convergence_tol`, simulation останавливается досрочно.

Cell-centered velocity:

```text
uc = 0.5 * (u[:-1,:] + u[1:,:])
vc = 0.5 * (v[:,:-1] + v[:,1:])
```

Vorticity:

```text
omega = d(vc)/dx - d(uc)/dy
```

через `np.gradient(vc, xc, axis=0) - np.gradient(uc, yc, axis=1)`.

`Snapshot` хранит:

```python
step: int
t: float
p: np.ndarray      # shape (nx,ny), float32
uc: np.ndarray     # shape (nx,ny), float32
vc: np.ndarray     # shape (nx,ny), float32
omega: np.ndarray  # shape (nx,ny), float32
```

`MacState` хранит точное внутреннее состояние C++ solver:

```python
u_vec: np.ndarray  # shape ((nx-1)*ny,), float64
v_vec: np.ndarray  # shape (nx*(ny-1),), float64
p: np.ndarray      # shape (nx*ny,), float64
```

## Simulation outputs

После `make run` создается отдельная папка `plots/run/`.

Основные файлы:

```text
plots/run/stokes_max_divergence.png
plots/run/stokes_velocity_change.png          # только если save_velocity_change_plot: true
plots/run/stokes_streamlines.mp4
plots/run/stokes_pressure.mp4
plots/run/stokes_vorticity.mp4
plots/run/final_state/streamlines.png
plots/run/final_state/pressure.png
plots/run/final_state/vorticity.png
plots/run/final_state/state.pkl
plots/run/fixed_time_state/streamlines.png
plots/run/fixed_time_state/pressure.png
plots/run/fixed_time_state/vorticity.png
plots/run/fixed_time_state/state.pkl
plots/run/fixed_time_state/state_internal.pkl
```

`plots/run/fixed_time_state/state_internal.pkl` - единственный файл, который steady
workflow использует как начальное приближение.

`state.pkl` - pickle-словарь для просмотра/анализа cell-centered состояния:

```text
x: np.ndarray, shape (nx,ny)
y: np.ndarray, shape (nx,ny)
u: np.ndarray, shape (nx,ny)
v: np.ndarray, shape (nx,ny)
p: np.ndarray, shape (nx,ny)
```

`x` и `y` строятся через `np.meshgrid(xc, yc, indexing="ij")`. `u`, `v`, `p`
соответствуют `uc[i,j]`, `vc[i,j]`, `p[i,j]` на cell centers.

`state_internal.pkl` - pickle-словарь exact MAC-state:

```text
nx, ny, lx, ly, nu, dt, step, t
u_vec: np.ndarray, shape ((nx-1)*ny,)
v_vec: np.ndarray, shape (nx*(ny-1),)
p: np.ndarray, shape (nx*ny,)
```

## Графики

`stokes_max_divergence.png`:

- строится по `div_history`;
- длинная история downsample-ится до примерно 5000 точек;
- если downsampling был, на графике пишется stride.

`stokes_velocity_change.png`:

- включается ключом `run.save_velocity_change_plot`;
- строится по всем внутренним solver steps без downsampling;
- величина: `||U_n-U_{n-1}||_inf / Δt`;
- шкала `y` логарифмическая;
- шаг major ticks по `x` равен `0.1` секунды;
- подписи `x` повернуты на 90 градусов;
- ширина графика выбирается примерно `3.5 cm` на `1` секунду физического времени, но не меньше `8` inches.

`steady_iterate_change.png`:

- создается только в steady workflow;
- показывает `||U^{k+1}-U^k||_inf` по принятым Newton updates;
- если Newton не принял ни одного шага, на графике выводится `No accepted Newton updates`.

## Визуализация полей

`viz/levels.py` считает общие levels для набора snapshots:

- speed levels: от `0` до максимального `sqrt(uc^2+vc^2)`, 25 levels;
- pressure levels: 2-й и 98-й percentiles по всем snapshots, 61 levels;
- vorticity levels: симметрично от `-omega_max` до `omega_max`, где `omega_max` - 98-й percentile `abs(omega)`, 61 levels.

`viz/plots.py` сохраняет три панели:

- `streamlines.png` - background `|u|` + streamplot;
- `pressure.png` - filled contours + contour lines;
- `vorticity.png` - filled contours + contour lines.

На все панели накладываются vortex markers:

- строится приближенная stream function `psi`;
- локальные maxima считаются counterclockwise vortex centers;
- локальные minima считаются clockwise vortex centers;
- near-boundary extrema отбрасываются margin-ом.

## Видео

`viz/video.py` рендерит три видео:

```text
stokes_streamlines.mp4
stokes_pressure.mp4
stokes_vorticity.mp4
```

Каждый тип видео рендерится отдельным process worker через
`ProcessPoolExecutor` со spawn context. Progress bars обновляются через
`multiprocessing.Manager().Queue()`.

Параметры writer:

```text
fps = cfg.video_fps
codec = libx264
crf = 20
macro_block_size = 1
```

Видео строятся по сохраненным snapshots, а не по каждому внутреннему C++ шагу.
Количество frames равно `len(snapshots)`.

## Steady workflow

`make steady` запускает `main.py --mode steady`.

Алгоритм steady не решает стационарный residual напрямую. Он ищет неподвижную
точку одного временного шага:

```text
U* = Phi(U*)
G(U) = Phi(U) - U = 0
```

где `Phi(U)` - один IMEX-шаг C++ solver при `t=0` с текущими time-independent
forcing и BC.

Нелинейная переменная steady:

```text
U = [u_interior, v_interior]
```

Pressure не входит в нелинейный state vector Newton. Pressure получается как
часть результата одного C++ шага.

### Проверка time independence

Steady mode запрещает time-dependent forcing и BC. Проверяются строки:

```text
forcing.fu
forcing.fv
boundary.u_top
boundary.u_bot
boundary.v_left
boundary.v_right
boundary.u_left
boundary.u_right
boundary.v_bot
boundary.v_top
```

Если в выражении найден отдельный token `t`, steady падает с ошибкой.

### Начальное приближение

Текущая версия поддерживает ровно один источник:

```text
plots/run/fixed_time_state/state_internal.pkl
```

Если файла нет, нужно сначала выполнить:

```bash
make run
```

Pickle содержит точные внутренние MAC unknowns:

```text
u_vec: interior u-faces in C++ unknown ordering
v_vec: interior v-faces in C++ unknown ordering
p: pressure cells in C++ storage ordering
```

При загрузке проверяется, что `nx`, `ny`, `lx`, `ly`, `nu`, `dt` в pickle
совпадают с текущим `config.yaml`, а длины `u_vec`, `v_vec`, `p` равны
`(nx-1)*ny`, `nx*(ny-1)` и `nx*ny`. Если файл отсутствует или параметры не
совпадают, steady падает с ошибкой; нужно заново выполнить `make run`.

### FixedPointMap

`FixedPointMap.evaluate(state)`:

1. Делит `state` на `u_vec`, `v_vec`.
2. Вызывает `solver.set_state(u_vec, v_vec)`.
3. Выполняет один шаг C++ solver:

   ```python
   solver.run_steps_with_force(0.0, 1, fu0, fv0)
   ```

4. Забирает `p_mac`, `u_mac`, `v_mac`.
5. Забирает `u_next`, `v_next`, `p_cells`.
6. Собирает `next_state = [u_next, v_next]`.
7. Возвращает:

   ```text
   residual = next_state - state
   residual_inf = max(abs(residual))
   max_div
   p/u/v fields
   ```

Если solver возвращает NaN или Inf, выбрасывается ошибка.

### Newton-GMRES

`solve_steady` в Python только загружает начальное приближение и вызывает C++
backend. Внутри C++ используется damped Newton:

```text
DG(U_k) delta_k = -G(U_k)
U_{k+1} = U_k + alpha * delta_k
```

Критерий сходимости:

```text
||G(U_k)||_inf < steady_solver.residual_tol
```

Максимум итераций:

```text
steady_solver.max_newton_iters
```

Jacobian-vector product считается конечной разностью:

```text
DG(U) v ~= (G(U + eps*v) - G(U)) / eps
eps = jacobian_rdiff * max(||U||_inf, 1) / ||v||_inf
```

Линейная система решается C++ restarted GMRES:

```text
relative tolerance = steady_solver.krylov_tol
restart = min(steady_solver.krylov_restart, state_size)
max Arnoldi iterations = steady_solver.krylov_maxiter
```

Если GMRES не достигает tolerance за заданное число итераций, C++ все равно
пробует использовать текущую найденную поправку `delta`, как и старая Python
реализация.

Line search:

- если `line_search: armijo`, начинается с `alpha=1`;
- пробуется `U + alpha*delta`;
- шаг принимается, если

  ```text
  ||G(U + alpha*delta)||_inf < (1 - 1e-4*alpha) * ||G(U)||_inf
  ```

- если не принято, `alpha` делится на 2;
- если `alpha < min_step`, Newton останавливается и сохраняет последний принятый state.

Если `line_search: none`, принимается полный шаг без Armijo.

### Steady outputs

После `make steady` создается:

```text
plots/steady/streamlines.png
plots/steady/pressure.png
plots/steady/vorticity.png
plots/steady/state.pkl
plots/steady/state_internal.pkl
plots/steady/steady_iterate_change.png
```

`state.pkl` имеет тот же pickle-формат `{x, y, u, v, p}`.
`state_internal.pkl` содержит exact MAC-state найденного steady state и является
правильным файлом для последующей линеаризации.

## Linearization workflow

`make linearize` запускает `main.py --mode linearize`.

Входной файл по умолчанию:

```text
plots/steady/state_internal.pkl
```

Его нужно получить перед этим через:

```bash
make steady
```

Линеаризация строится около найденного стационарного MAC-state
`U0 = [u0_vec, v0_vec, p0]`. Здесь используется чистый стационарный residual
Навье-Стокса:

```text
R(U, p) = [
  (u · grad)u - nu Laplace(u) + grad(p) - f,
  div(u)
]
```

Производные по времени в этом residual равны нулю. Для замыкания давления
первая строка divergence residual заменяется gauge condition:

```text
p[0,0] = 0
```

Код аналитически линеаризует momentum residual по velocity-возмущению
`w = (a,b)`:

```text
δR_u =
  u0 ∂a/∂x + v0 ∂a/∂y
+ a  ∂u0/∂x + b  ∂u0/∂y
- nu Δa

δR_v =
  u0 ∂b/∂x + v0 ∂b/∂y
+ a  ∂v0/∂x + b  ∂v0/∂y
- nu Δb
```

После этого строится velocity-оператор:

```text
L w = P (-δR_momentum)
```

`P` - MAC pressure projection. Она решает sparse saddle-system `[I G; D 0]`,
чтобы результат `L w` был divergence-free. Pressure не является динамической
переменной eigenproblem; `p_modes` восстанавливаются из этого projection solve.
Минус выбран в stability-sign convention: для возмущений можно читать
линейную динамику как `v_t = L v`.

Для поиска собственных пар C++ выбирает backend по размеру задачи.

На малых velocity-пространствах явно собирается dense-матрица `L`: оператор
применяется ко всем базисным векторам velocity-пространства, затем вызывается
`Eigen::EigenSolver`, то есть полный dense eigen solve внутри C++ backend.

На больших velocity-пространствах dense-матрица не собирается. Вместо этого
запускается matrix-free Arnoldi: C++ хранит только Krylov basis и малую
Hessenberg-матрицу, а `Eigen::EigenSolver` вызывается уже для этой малой
матрицы. Для текущей сетки `105x105` размер velocity-пространства равен
`21840`, поэтому используется Arnoldi, а не full dense eig.

Параметр `linearization.which: LR` используется после вычисления спектра
dense-оператора или Ritz-спектра Arnoldi, чтобы выбрать `n_eigs` modes с
наибольшей вещественной частью `lambda`.

После `make linearize` создается:

```text
plots/linearized/eigenpairs.pkl
```

Pickle содержит:

```text
operator
state_path
state_metadata
base_residual_inf
matvec_count
dense_operator_bytes
eig_message
eigenvalues
eigenvectors
u_modes
v_modes
p_modes
```

`eigenvectors` хранит столбцы в порядке `[u_vec, v_vec, p]`. Для удобства те же
моды сохранены как MAC-grid массивы:

```text
u_modes shape: (mode, nx-1, ny)
v_modes shape: (mode, nx, ny-1)
p_modes shape: (mode, nx, ny)
```

## Projected-Run workflow

`make projected-run` запускает `main.py --mode projected-run`.

Этот режим делает отдельный нестационарный прогон, но стартует не с нуля, а из:

```text
plots/steady/state_internal.pkl
```

Также он читает:

```text
plots/linearized/eigenpairs.pkl
```

Runtime-параметры у него свои, из блока `projected_run`:

```text
projected_run.t_end
projected_run.n_steps
projected_run.video_fps
projected_run.video_speed
projected_run.save_velocity_change_plot
```

Projected-run всегда отключает early stop по `run.convergence_tol` и идет до
собственного `projected_run.t_end`.

Алгоритм:

1. Берется forcing vector в velocity ordering:

   ```text
   F = [fu, fv]
   ```

2. Из eigenpairs выбираются моды, для которых:

   ```text
   Re(lambda) > projected_run.real_threshold
   ```

   `lambda` берется из `plots/linearized/eigenpairs.pkl` и относится к
   оператору `L = -P D_momentum R(U0)`, а не к map одного шага по времени.

3. Из velocity-компонент комплексных eigenvectors строится вещественное
   подпространство:

   ```text
   span(Re(q_k), Im(q_k))
   ```

   Pressure-компонента eigenvector в проекцию forcing не входит, потому что
   правая часть `F = [fu, fv]` живет только в velocity-пространстве.

4. Правая часть проектируется на это подпространство:

   ```text
   F_bad = Proj_unstable(F)
   ```

5. В simulation передается отфильтрованная правая часть:

   ```text
   F_filtered = F - F_bad
   ```

Для текущего forcing это делается на каждом batch. Если forcing зависит от `t`,
проекция пересчитывается для текущего batch-force.

После `make projected-run` создается:

```text
plots/projected_run/projection.pkl
plots/projected_run/stokes_max_divergence.png
plots/projected_run/stokes_velocity_change.png
plots/projected_run/stokes_streamlines.mp4
plots/projected_run/stokes_pressure.mp4
plots/projected_run/stokes_vorticity.mp4
plots/projected_run/final_state/state.pkl
plots/projected_run/final_state/state_internal.pkl
```

`projection.pkl` содержит выбранные индексы мод, eigenvalues, rank вещественного
базиса и нормы `force`, `removed`, `remaining`.

## Сходимость и диагностика

В simulation:

```text
velocity_change = ||U_n - U_{n-1}||_inf
velocity_change_rate = velocity_change / Δt
```

В истории хранится разность между соседними внутренними time steps, а график
`stokes_velocity_change` показывает `velocity_change_rate`. Малое значение на
этом графике не гарантирует, что Newton в steady сойдется. Причины:

- величина зависит от `dt`; при маленьком `dt` соседние слои могут быть близки;
- локальный провал на графике может быть поворотом/медленным участком траектории, а не стационарным состоянием;
- steady берет exact MAC state из `state_internal.pkl`, но малый `velocity_change`
  все равно не гарантирует попадание в Newton basin;
- Newton решает `G(U)=Phi(U)-U`, а не минимизирует сам график.

В steady:

```text
residual_inf = ||Phi(U) - U||_inf
```

Метод считается сошедшимся только если `residual_inf < residual_tol`.

При остановке steady важно смотреть:

- начальную `||G(U)||_inf`;
- уменьшалась ли невязка по Newton iterations;
- `GMRES info`;
- `alpha` line-search;
- причину остановки.

Типичная проблемная картина:

```text
GMRES info=maxiter
line search could not reduce ||G(U)||
```

Это означает, что внутренний линейный solve не дал достаточно хорошего
Newton-направления, и damping больше не может уменьшить fixed-point residual.

## Важные ограничения текущей реализации

1. Steady solver решает fixed-point residual одного time step, а не
   стационарный residual дискретизированных уравнений напрямую. Это рабочий
   подход для поиска неподвижной точки time-step map, но он может быть плохо
   обусловлен при маленьком `dt`.

2. Newton-GMRES без preconditioner. На больших сетках или сложных режимах GMRES
   может не сходиться за `krylov_maxiter`.

3. C++ matrix factorization использует SparseLU. Матрица факторизуется один раз,
   что быстро для многих шагов, но память и время факторизации растут с сеткой.

4. Time-dependent forcing в simulation обновляется по batch, а не по каждому
   внутреннему step.

5. Dense eig в `make linearize` хранит явную матрицу размера
   `N_velocity × N_velocity`, то есть минимум `8*N_velocity^2` байт только под
   оператор до рабочих массивов eig. На сетке 105×105 это уже несколько GiB.

6. `stokes_velocity_change.png` показывает уже нормированную по времени величину
   `||U_n-U_{n-1}||_inf / Δt`, а не абсолютное изменение за шаг.

7. Проект не содержит автоматических tests. Минимальная проверка после изменений:

   ```bash
   make compile
   make run
   make steady
   make linearize
   make projected-run
   ```

## Как пересобрать функционально такой же проект с нуля

Минимальный план реализации:

1. Создать C++ класс `StokesMac2D` с MAC fields `p,u,v`, постоянной sparse
   matrix и SparseLU factorization.
2. Реализовать MAC indexing и unknown ordering ровно как в этой спецификации.
3. Реализовать сборку монолитной IMEX системы:
   - u-momentum block;
   - v-momentum block;
   - incompressibility block;
   - pressure gauge `p[0,0]=0`.
4. Реализовать ghost-node BC и direct face BC.
5. Реализовать explicit central advection с bilinear interpolation между face
   grids.
6. Реализовать batch stepping и diagnostics `max_divergence`,
   `last_velocity_change`.
7. Экспортировать C ABI, совместимый с `solver/lib.py`.
8. Написать Python `SimConfig`, который читает `config.yaml`, вычисляет `dt`,
   `frame_every`, BC arrays и force arrays.
9. Написать `ctypes` wrapper, который:
   - находит shared library;
   - биндингует C ABI;
   - управляет handle;
   - возвращает NumPy copies of fields/state.
10. Написать simulation runner:
    - batch loop;
    - per-step histories;
    - per-batch snapshots;
    - early stop по `run.convergence_tol`.
11. Написать visualization:
    - divergence plot;
    - optional velocity-change plot;
    - final/fixed-time panels;
    - pickle export;
    - parallel video rendering.
12. Написать steady workflow:
    - загрузка `plots/run/fixed_time_state/state_internal.pkl`;
    - проверка grid metadata и длин внутренних массивов;
    - вызов C++ backend для `G(U)=Phi(U)-U`;
    - C++ finite-difference JVP;
    - C++ restarted GMRES;
    - C++ Armijo line-search;
    - steady plots and pickle state.
13. Написать linearization workflow:
    - загрузка `plots/steady/state_internal.pkl`;
    - передача state в C++ backend;
    - C++ сборка стационарного residual `R(U,p)` без производных по времени;
    - C++ аналитическая линеаризация momentum residual;
    - C++ exact dense eig на малых системах;
    - C++ matrix-free Arnoldi на больших системах;
    - C++ `Eigen::EigenSolver` для dense matrix или малой Hessenberg matrix;
    - pickle export eigenpairs в ordering `[u_vec, v_vec, p]`.
14. Написать projected-run workflow:
    - построение runtime-конфига из `projected_run.*`;
    - отключение early stop по `run.convergence_tol`;
    - загрузка `plots/steady/state_internal.pkl`;
    - загрузка `plots/linearized/eigenpairs.pkl`;
    - выбор eigenmodes по `Re(value) > threshold`;
    - вещественная projection basis из `Re/Im` комплексных мод;
    - вычитание `Proj(F)` из forcing;
    - simulation стартует из `U0`.
15. Добавить `Makefile`, `compile.sh`, `requirements.txt` и `README.md`.

Если нужно сохранить текущую функциональность один-в-один, нельзя добавлять
скрытые режимы в config и нельзя менять смысл `make run`, `make steady`,
`make linearize`, `make projected-run`.
