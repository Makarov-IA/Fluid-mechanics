# Handoff для нового агента: курсовая по папке `alex`

Этот файл создан как максимально подробная передача контекста после первичного
исследования проекта `alex`. Пользователь попросил не терять проделанную работу:
новый агент должен продолжить с этого места, а не начинать заново.

Рабочая директория:

```text
/Users/aasentsov/Desktop/fluid_mechanics/Fluid-mechanics
```

Задача пользователя в целом: изучить весь код в `alex`, статьи, формулы,
алгоритмы, структуры данных и численные эксперименты, затем написать связный
академический LaTeX-текст курсовой работы на русском языке и, после отдельной
команды пользователя, положить LaTeX и нужные графики в:

```text
alex/paper
alex/paper/images
```

ВАЖНО: пользователь пока НЕ дал команду начинать писать финальную курсовую.
Сначала нужно было составить карту проекта и дождаться уточнений.

## 1. Самое важное: обнаруженное расхождение

Есть принципиальный конфликт между пользовательским описанием и текущим
состоянием `alex`.

Пользователь в исходном запросе явно сказал:

- для задачи о каверне нужно писать про реализацию **без схемы Аракавы**;
- схему Аракавы можно упомянуть только в конце как альтернативу;
- основная схема для каверны должна быть стандартная центрально-разностная
  аппроксимация якобиана/конвективного члена.

Но текущий `HEAD` в папке `alex` использует схему Аракавы:

- C++ solver:
  `alex/solver/solver.cpp`, функция `Solver::arakawaJacobian`, строки около 39;
- Newton:
  `alex/scripts/newton_from_csv.py`, функция `arakawa_jacobian`, строки около 209;
- linear stability:
  `alex/scripts/linear_stability.py`, `LinearizedOmegaOperator.apply`, строки
  около 67--73.

То есть текущая рабочая версия не совпадает с фразой пользователя про
“без Аракавы”. Однако вариант без Аракавы найден в истории git:

- коммит `f933c43` (`new solver new life`);
- коммит `d731964` (`fixed solver + video creator`);
- в этих версиях `alex/solver/solver.cpp` содержит обычные центральные
  разности для конвективного члена и задачу каверны с движущейся верхней
  стенкой.

Пример центральной аппроксимации из `f933c43`/`d731964`:

```text
u = psi_y, v = -psi_x,
J = psi_y omega_x - psi_x omega_y,
```

в коде невязка вихря считалась как

```cpp
tmp = (1/Re) * Delta_h omega
    - psi_y * omega_x
    + psi_x * omega_y;
```

а в `solveOmega()` конвективные поправки входили в коэффициенты прогонки по
направлениям через стандартные центральные разности, без усреднения Аракавы.

Новый агент должен обязательно уточнить у пользователя:

1. Для раздела о каверне писать по старой версии из истории git
   (`f933c43`/`d731964`) или пользователь восстановит/даст актуальный код без
   Аракавы?
2. Текущий `HEAD` описывать как вторую задачу и стабилизацию?
3. Где лежат или как называются две статьи? В репозитории PDF/статей не найдено.

## 2. Состояние git и осторожность

Рабочее дерево уже было грязным до моих изменений. Нельзя откатывать чужие
изменения. На момент исследования `git status --short alex` показывал:

```text
 M alex/README.md
 M alex/eigenvalue_method.md
 M alex/linear_stability/videos/filtered_omega.mp4
 M alex/linear_stability/videos/filtered_psi.mp4
 M alex/linear_stability/videos/filtered_streamplot.mp4
 M alex/linear_stability_README.md
 D alex/scripts/analyze_filtered.sh
 D alex/scripts/filtered_video.sh
 D alex/scripts/find_stationary.sh
 D alex/scripts/gif.sh
 M alex/scripts/linear_stability.py
 D alex/scripts/linear_stability.sh
 D alex/scripts/make_gifs.py
 D alex/scripts/newton_from_csv.sh
 D alex/scripts/plot.sh
 D alex/scripts/run.sh
 D alex/scripts/streamplot_csv.sh
 D alex/scripts/video.sh
?? alex/pipeline/
```

Этот handoff добавляет:

```text
?? alex/paper/AGENT_HANDOFF.md
```

Не удалять и не восстанавливать старые shell-скрипты без явного запроса
пользователя. В текущей структуре вместо удалённых скриптов есть папка
`alex/pipeline/`.

## 3. Локальные статьи не найдены

Поиск файлов `*.pdf`, `*.djvu`, `*.bib`, `*.tex`, `*.md` показал, что в проекте
есть markdown-документация, но нет PDF-статей, относящихся к `alex`.

Найдены только:

```text
./README.md
./alex/README.md
./alex/eigenvalue_method.md
./alex/linear_stability_README.md
./alex/linearize.md
./alex/pipeline/README.md
./ilya/README.md
./ilya/project.md
```

PDF-файлы, которые нашлись, относятся к `.venv`/matplotlib и не являются
статьями. Поэтому нельзя утверждать “по статье написано”, пока пользователь не
даст сами статьи или ссылки. В итоговом тексте нужно либо получить статьи, либо
оставить аккуратные пометки о необходимости уточнения источников.

## 4. Общая структура `alex`

Файлы верхнего уровня:

```text
alex/main.cpp
alex/Makefile
alex/README.md
alex/linear_stability_README.md
alex/eigenvalue_method.md
alex/linearize.md
alex/requirements.txt
alex/configs/config.cfg
alex/utils/config.hpp
alex/forcing/omega_forcing.hpp
alex/solver/solver.hpp
alex/solver/solver.cpp
alex/solver/solver_utils.hpp
alex/scripts/find_stationary.py
alex/scripts/newton_from_csv.py
alex/scripts/linear_stability.py
alex/scripts/plot_fields.py
alex/scripts/streamplot_csv.py
alex/scripts/make_videos.py
alex/pipeline/*.sh
alex/data/results
alex/plots
alex/stationary_detection
alex/linear_stability
```

Каталоги с результатами большие:

```text
alex/data/results                         исходные CSV-снимки
alex/plots/frames                         исходные PNG-кадры
alex/plots/videos                         исходные MP4
alex/stationary_detection                 поиск почти стационара и Ньютон
alex/linear_stability                     спектр, моды, фильтрованные результаты
alex/linear_stability/filtered_results    CSV после вычитания неустойчивых мод
alex/linear_stability/plots/frames        PNG после фильтрации
alex/linear_stability/videos              MP4 после фильтрации
```

Размеры, которые были посчитаны:

```text
alex/data/results files: 1002, about 2135.67 MB
alex/plots/frames files: 3003, about 740.68 MB
alex/linear_stability/filtered_results files: 1001, about 1640.39 MB
alex/linear_stability/plots/frames files: 3003, about 767.6 MB
```

## 5. Текущая постановка в `HEAD`

Текущий конфиг:

```text
alex/configs/config.cfg
```

Содержимое ключевых параметров:

```text
mode=fixed_dt_steps
nx=201
ny=101
lx=1.0
ly=0.5
t_max=20.0
n_time_steps=2000000
dt=0.00001
Re=260
steady_tolerance=1e-10
save_every_step=1000
save_dir=data/results

bc.left.u=0.0
bc.left.v=0.0
bc.right.u=0.0
bc.right.v=0.0
bc.bottom.u=0.0
bc.bottom.v=0.0
bc.top.u=0.0
bc.top.v=0.0
```

Следовательно текущая задача в `HEAD` не является каверной с движущейся верхней
крышкой. Это прямоугольная область

```text
0 <= x <= 1,
0 <= y <= 0.5,
```

со всеми неподвижными стенками и внешним вихревым форсированием.

Форсирование задано в:

```text
alex/forcing/omega_forcing.hpp
```

Функция:

```cpp
OmegaForcing(double x, double y, double lx, double ly)
```

строит сумму синусоидальных мод с коэффициентами:

```text
a22 = -19 ly / (2 pi)
a42 = -6  ly / (2 pi)
a62 =  7  ly / (2 pi)
a24 =  14 ly / (4 pi)
a13 = a22 / 50
a31 = a22 / 50
```

и возвращает минус производную по `y` от набора мод. В Python та же формула
повторена в `newton_from_csv.py`, функция `omega_forcing`.

Математически текущая стационарная система в документации и коде:

```latex
\Delta \psi + \omega = 0,
\qquad
\frac{1}{Re}\Delta\omega - J(\psi,\omega) - F_\omega(x,y)=0.
```

В текущем коде используется соглашение:

```latex
u = \frac{\partial \psi}{\partial y},
\qquad
v = -\frac{\partial \psi}{\partial x},
\qquad
\omega = \frac{\partial v}{\partial x}
       - \frac{\partial u}{\partial y}
       = -\Delta \psi.
```

Отсюда уравнение Пуассона:

```latex
\Delta \psi = -\omega,
\quad \text{или} \quad \Delta\psi+\omega=0.
```

## 6. Старая постановка каверны без Аракавы

Старая версия из истории git, которую нужно использовать, если пользователь
подтвердит “каверна без Аракавы”:

```bash
git show f933c43:alex/solver/solver.cpp
git show f933c43:alex/solver/solver.hpp
git show f933c43:alex/configs/config.cfg
```

В этой версии:

```text
nx=101
ny=101
t_max=20
n_time_steps=100000
Re=5000
save_every_step=100
```

Область квадратная, шаги:

```cpp
dx_ = 1.0 / (nx_ - 1);
dy_ = 1.0 / (ny_ - 1);
```

Граничные скорости:

```text
левая стенка: u=0, v=0
правая стенка: u=0, v=0
нижняя стенка: u=0, v=0
верхняя стенка: u=1, v=0
```

Функция тока на стенках полагается нулевой:

```text
psi = 0 на границе
```

Формула Тома для граничной завихренности:

```text
omega(i,0)        = -2 psi(i,1) / dy^2
omega(i,ny-1)     = -2 psi(i,ny-2) / dy^2 - 2/dy
omega(0,j)        = -2 psi(1,j) / dx^2
omega(nx-1,j)     = -2 psi(nx-2,j) / dx^2
```

Член `-2/dy` на верхней стенке отвечает скорости движущейся крышки `U=1`.

В старом C++-солвере центральный якобиан/конвективный член появляется через:

```cpp
psi_y = (psi(i,j+1)-psi(i,j-1))/(2 dy)
psi_x = (psi(i+1,j)-psi(i-1,j))/(2 dx)
omega_x = (omega(i+1,j)-omega(i-1,j))/(2 dx)
omega_y = (omega(i,j+1)-omega(i,j-1))/(2 dy)
```

и

```text
J(psi,omega) = psi_y omega_x - psi_x omega_y.
```

В невязке:

```text
(1/Re) Delta_h omega - J_h(psi,omega)
```

где `J_h` -- стандартный центральный якобиан, не Аракава.

Текущий пользователь явно хочет, чтобы раздел про каверну был написан именно
про эту центральную схему, а Аракава была упомянута только как специальная
консервативная альтернатива.

## 7. Карта текущего C++-кода

### `alex/main.cpp`

Роль:

- читает путь конфига из argv или берёт `configs/config.cfg`;
- читает `Config`;
- если передан второй аргумент, переопределяет `save_dir`;
- создаёт `Solver solver(cfg)`;
- вызывает `solver.solve()`.

### `alex/utils/config.hpp`

Структуры:

```cpp
struct WallVelocity { double u, v; };
struct BoundaryConditions { left, right, bottom, top; };
struct Config {
    int nx, ny;
    double lx, ly;
    string mode;
    double t_max;
    int n_time_steps;
    double dt;
    int Re;
    double steady_tolerance;
    int save_every_step;
    string save_dir;
    BoundaryConditions bc;
    LogInfo log_info;
};
```

Поддерживаемые режимы:

```text
fixed_steps
fixed_dt_steps
till_converges
```

Проверки:

- `nx, ny > 1`;
- `lx, ly > 0`;
- для непроницаемых стенок:
  - `bc.left.u = 0`,
  - `bc.right.u = 0`,
  - `bc.bottom.v = 0`,
  - `bc.top.v = 0`;
- положительные времена, число шагов, `steady_tolerance`, `save_every_step`.

### `alex/solver/solver.hpp`

Главный класс:

```cpp
class Solver
```

Основные поля:

```text
cfg_
nx_, ny_
dx_, dy_, dt_
Re_
step_
residual_
psi_
omega_
u_
v_
f_
g_
```

`psi_`, `omega_`, `u_`, `v_` имеют тип `Eigen::MatrixXd` размера
`nx_ x ny_`.

### `alex/solver/solver.cpp`

Ключевые функции:

- `Solver::Solver`:
  задаёт размеры, шаги, `dt`, `Re`, обнуляет массивы.
- `arakawaJacobian(i,j)`:
  текущий дискретный якобиан Аракавы `J_A=(J1+J2+J3)/3`.
- `updateVelocities()`:
  на стенках задаёт скорости из конфига, внутри использует
  `u=(psi(i,j+1)-psi(i,j-1))/(2dy)`,
  `v=-(psi(i+1,j)-psi(i-1,j))/(2dx)`.
- `ApplyThomBoundary()`:
  граничная завихренность по формуле Тома с учётом скоростей стенок.
- `computeRHS(f_)`:
  правая часть для обновления `psi`. В текущем solver это не прямое решение
  стационарного Пуассона, а псевдовременная/факторизованная схема.
- `computeOmegaRHS(g_)`:
  правая часть для обновления `omega`; использует Аракаву и внешнее
  `OmegaForcing`.
- `solvePsi()`:
  решает по направлениям трёхдиагональные системы с помощью `Progonka`.
- `solveOmega()`:
  аналогично решает уравнение для `omega` через две прогонки.
- `computeResiduals()`:
  считает максимум по внутренним узлам для:
  `Delta_h psi + omega`
  и
  `(1/Re)Delta_h omega - J_h - F_omega`.
- `step()`:
  порядок:
  1. `solvePsi()`;
  2. `ApplyThomBoundary()`;
  3. `solveOmega()`;
  4. `computeResiduals()`.
- `solve()`:
  основной цикл, сохранение `residual_history.csv` и `result_*.csv`.
- `save()`:
  пишет CSV:
  `x,y,psi,omega,u,v`.

### `alex/solver/solver_utils.hpp`

Функция:

```cpp
Progonka(a,b,c,d,res)
```

Это метод прогонки для трёхдиагональной системы. В коде комментарии:

```text
c - diag
b - upper diag
a - lower diag
d - right side
```

Фактически используется прямой ход для `alpha`, `beta`, затем обратная
подстановка.

## 8. Математическая схема текущего solver-а

Текущая документация `alex/README.md` трактует C++ solver как
псевдовременное решение системы:

```latex
\psi_t = \Delta\psi + \omega,
```

```latex
\omega_t =
\frac{1}{Re}\Delta\omega
-J(\psi,\omega)
-F_\omega.
```

Стационарный предел:

```latex
\Delta\psi+\omega=0,
\qquad
\frac{1}{Re}\Delta\omega-J(\psi,\omega)-F_\omega=0.
```

Важное замечание для курсовой: в текущем C++ шаг для `psi` и `omega` реализован
через факторизованные одномерные прогонки. Это ближе к ADI/приближённой
факторизации, чем к классическому “на каждом шаге явно обновить вихрь и потом
итерационно решить Пуассон”. Для каверны без Аракавы из старой версии также
использовалась факторизация/прогонки.

Если писать строго по текущему `HEAD`, нельзя утверждать, что Пуассон решается
простыми итерациями; в C++ `solvePsi()` использует два прохода прогонки. В
Python для Ньютона и спектра обратный лапласиан реализован иначе: через
синус-преобразование в `DirichletLaplaceInverse`.

## 9. Python: поиск стационара

Файл:

```text
alex/scripts/find_stationary.py
```

Назначение: выбрать snapshot, наиболее близкий к стационарному, по разности
соседних сохранённых состояний.

Состояние:

```text
q_k = (psi_k, omega_k)
```

По умолчанию поле `state`, то есть используются обе компоненты `psi` и `omega`.

Метрика:

```latex
D_k =
\frac{\|q_k-q_{k-1}\|_2}{\|q_k\|_2}.
```

Скрипт:

- собирает `result_*.csv`;
- сортирует по номеру шага;
- для каждой пары соседних снимков считает `abs_l2`, `rel_l2`, `linf`;
- пишет `consecutive_difference_norm.csv`;
- строит `consecutive_difference_norm.png`;
- копирует лучший snapshot в `stationary_state.csv`;
- строит `stationary_streamplot.png`.

## 10. Python: метод Ньютона

Файл:

```text
alex/scripts/newton_from_csv.py
```

Назначение: уточнить стационарное решение из выбранного CSV.

Ключевые структуры:

```python
WallVelocity
BoundaryConditions
Problem(xs, ys, re, bc)
DirichletLaplaceInverse
```

Упаковка неизвестных:

```python
pack(psi, omega)
```

возвращает вектор:

```text
z = [psi_inner.ravel(), omega_inner.ravel()]
```

Только внутренние узлы входят в вектор неизвестных. Граничные значения `psi`
нулевые, граничные значения `omega` восстанавливаются формулой Тома.

Распаковка:

```python
unpack(z, problem)
```

создаёт полные массивы `psi`, `omega` размера `nx x ny`, заполняет внутренние
значения и применяет `apply_thom_boundary`.

Невязка:

```python
make_residual(problem)
```

возвращает:

```latex
F(z)=
\begin{pmatrix}
\Delta_h\psi+\omega\\
\frac1{Re}\Delta_h\omega-J_h(\psi,\omega)-F_\omega
\end{pmatrix}.
```

В текущем `HEAD` `J_h` -- Аракава.

Якобиан-векторное произведение:

```python
make_jacobian_matvec(problem, z)
```

линеаризует:

```latex
J(\psi+\delta\psi,\omega+\delta\omega)
\approx
J(\psi,\omega)
+J(\delta\psi,\omega)
+J(\psi,\delta\omega).
```

Метод Ньютона:

```latex
J(z_k)\delta z_k=-F(z_k),
\qquad
z_{k+1}=z_k+\alpha_k\delta z_k.
```

Линейная система решается самописным GMRES:

```python
gmres_solve(...)
```

Параметры из `alex/pipeline/04_newton.sh`:

```text
MAX_NEWTON=10
NEWTON_TOL=1e-8
LINEAR_TOL=1e-3
GMRES_RESTART=30
GMRES_MAX_ITER=160
FD_EPS=1e-9
LINE_SEARCH_STEPS=12
--jacobian exact
--preconditioner stokes
--verify-jv
```

Демпфирование: line search уменьшает `alpha` вдвое, пока `L2`-норма невязки
не уменьшится.

Критерии:

- `Linf` невязки для остановки Ньютона;
- `L2` используется в line search;
- GMRES контролирует относительную невязку.

Предобуславливатель:

```python
make_stokes_preconditioner(problem)
```

использует обратный дискретный лапласиан с нулевыми граничными условиями
`DirichletLaplaceInverse`. Обратный лапласиан реализован через собственные
синус-функции дискретного оператора.

## 11. Python: линеаризация и спектр

Файл:

```text
alex/scripts/linear_stability.py
```

Основной класс:

```python
LinearizedOmegaOperator(problem, psi0, omega0)
```

Оператор строится только для возмущения завихренности `eta`. Возмущение
функции тока `phi` восстанавливается из:

```latex
\Delta_h\phi=-\eta.
```

Метод:

```python
full_fields_from_eta(eta_vec)
```

создаёт полные поля `phi`, `eta`, решает дискретный Пуассон через
`DirichletLaplaceInverse`, затем применяет тангенциальную формулу Тома для
граничной завихренности возмущения.

Действие оператора:

```python
apply(eta_vec)
```

возвращает:

```latex
\mathcal A\eta =
\frac1{Re}\Delta_h\eta
-J_h(\phi,\omega_0)
-J_h(\psi_0,\eta).
```

В текущем `HEAD` `J_h` -- Аракава.

Собственная задача:

```latex
\mathcal A v=\lambda v.
```

Решается через:

```python
scipy.sparse.linalg.eigs(A, k=EIGS_COUNT, which="LR")
```

`which="LR"` означает largest real part, то есть ищутся собственные значения с
наибольшей действительной частью.

Матрица явно не собирается. Используется `LinearOperator`, который умеет только
умножать вектор на `A`.

После нахождения собственных значений:

- всё пишется в `alex/linear_stability/eigenvalues.csv`;
- неустойчивые значения с `Re(lambda)>UNSTABLE_TOL` пишутся в
  `unstable_eigenvalues.csv`;
- неустойчивые моды пишутся в `unstable_modes/eig_XXX.csv`.

Параметры из `alex/pipeline/05_linear_stability.sh`:

```text
EIGS_COUNT=30
EIGS_TOL=1e-8
EIGS_MAX_ITER=3000
UNSTABLE_TOL=1e-9
MAX_UNSTABLE_MODES=8
NO_PLOTS=true
```

## 12. Проекция и вычитание неустойчивых мод

В текущей реализации фильтрация делает не ручное “вычитание с амплитудой”, а
ортогональную проекцию на вещественное неустойчивое подпространство.

Для комплексной моды:

```latex
v_j=a_j+i b_j
```

строится вещественное подпространство:

```latex
\operatorname{span}(\operatorname{Re}v_j,\operatorname{Im}v_j).
```

В коде полное состояние для проекции:

```text
state = [psi_inner.ravel(), omega_inner.ravel()]
```

Функции:

```python
pack_real_state
unpack_real_state
orthonormalize_real_vectors
project_onto_basis
build_unstable_projection_basis
subtract_unstable_modes_from_snapshots
```

Ортонормировка: два прохода модифицированного Грама--Шмидта, обычное
евклидово скалярное произведение по внутренним узлам полного состояния:

```latex
\langle q_1,q_2\rangle =
\sum_{\text{inner}}(\psi_1\psi_2+\omega_1\omega_2).
```

Для каждого snapshot:

```latex
r(t_k)=q(t_k)-q_0,
```

```latex
r_{\mathrm{bad}}(t_k)
=
\sum_{\ell=1}^{m}
\langle r(t_k),e_\ell\rangle e_\ell,
```

```latex
q_{\mathrm{filtered}}(t_k)=q(t_k)-r_{\mathrm{bad}}(t_k).
```

После вычитания применяется `apply_thom_boundary`, затем CSV сохраняется в
`alex/linear_stability/filtered_results`.

Метаданные:

```text
alex/linear_stability/filtered_snapshots.csv
```

Поля:

```text
source_csv
filtered_csv
time
removed_norm
projection_norm
basis_size
coefficients
modes
plot_png
```

## 13. Скрипты pipeline

Главный слой запуска:

```text
alex/pipeline
```

Команды:

```bash
bash alex/pipeline/00_run_solver.sh
bash alex/pipeline/01_frames.sh
bash alex/pipeline/02_video.sh
bash alex/pipeline/03_find_stationary.sh
bash alex/pipeline/04_newton.sh
bash alex/pipeline/05_linear_stability.sh
bash alex/pipeline/06_analyze_filtered.sh
bash alex/pipeline/07_filtered_video.sh
```

Роли:

- `00_run_solver.sh`: собирает C++ и запускает solver;
- `01_frames.sh`: строит PNG-кадры `psi`, `omega`, `streamplot`;
- `02_video.sh`: собирает MP4;
- `03_find_stationary.sh`: ищет почти стационарный snapshot;
- `04_newton.sh`: уточняет стационар методом Ньютона;
- `05_linear_stability.sh`: считает спектр, моды, фильтрует snapshot-ы;
- `06_analyze_filtered.sh`: анализирует разности соседних фильтрованных кадров;
- `07_filtered_video.sh`: строит кадры и видео после фильтрации.

## 14. Визуализация

Файл:

```text
alex/scripts/plot_fields.py
```

Строит для каждого CSV:

```text
*_psi.png
*_omega.png
*_streamplot.png
```

`psi` и `omega` строятся как `contourf` плюс контуры и quiver-стрелки скорости.

`streamplot` строит фон по скорости `|u|`, линии тока через `streamplot`.

Если есть `residual_history.csv`, строится:

```text
residual_history.png
```

Но в текущем исходном `residual_history.csv` последняя строка повреждена:

```text
907000,9.07001,0.0767007,15.3976,1
```

То есть `max_residual` ошибочно равен `1`, а не `15.3976`; файл также содержит
только 908 строк, хотя `result_*.csv` есть до `1000000`. Для академического
текста осторожно: график невязок может быть неполным/повреждённым.

Другие визуализации:

```text
alex/scripts/streamplot_csv.py
```

строит streamplot для одного CSV.

```text
alex/scripts/make_videos.py
```

делает `psi.mp4`, `omega.mp4`, `streamplot.mp4`.

## 15. Численные результаты, уже найденные

### 15.1 Исходные CSV

В `alex/data/results` найдено:

```text
1001 result_*.csv
```

От `result_0.csv` до `result_1000000.csv` с шагом сохранения 1000.

Формат каждого CSV:

```text
x,y,psi,omega,u,v
```

Сетка:

```text
201 x 101 = 20301 узел
```

### 15.2 Кандидат в стационар

Файл:

```text
alex/stationary_detection/consecutive_difference_norm.csv
```

Количество строк:

```text
999
```

Минимум относительной разности:

```text
previous_step = 235000
step          = 236000
time          = 2.36001
abs_l2        = 1.27122138824192
rel_l2        = 0.00106281738545142
linf          = 0.0284325394224823
```

График:

```text
alex/stationary_detection/consecutive_difference_norm.png
```

На графике видно: сначала разность соседних кадров быстро падает почти с
единицы до `~10^{-3}`, затем после `t≈2.36` снова начинает расти. Это признак,
что траектория проходит близко к неустойчивому стационарному состоянию, но
потом уходит от него по неустойчивому направлению.

Кандидат:

```text
alex/stationary_detection/stationary_state.csv
alex/stationary_detection/stationary_streamplot.png
```

Визуально на streamplot кандидата видны четыре крупные вихревые ячейки в
прямоугольнике `1 x 0.5`, с симметрией относительно середины области и
максимумами скорости примерно вдоль центральных разделительных линий.

### 15.3 Ньютоновский стационар

Файл:

```text
alex/stationary_detection/newton_equilibrium.csv
alex/stationary_detection/newton_equilibrium_streamplot.png
```

Статистика по `newton_equilibrium.csv`:

```text
rows: 20301
psi min = -0.0696388698688714
psi max =  0.0622158790682918
psi mean = -0.0014565255800383283
||psi||_2 = 3.965286887060821

omega min = -25.0093018310438
omega max =  26.0354160314564
omega mean = -0.00032828596675759167
||omega||_2 = 1186.0701501557046

u min = -1.04309079997616
u max =  1.04309079997616
||u||_2 = 54.22985224230947

v min = -0.563416504390198
v max =  0.563416504390198
||v||_2 = 28.651300296287676
```

Визуально Ньютон уточняет почти стационарный четырёхвихревой режим:
структура становится более регулярной, сохраняется симметрия ячеек.

Отдельного сохранённого графика сходимости Ньютона нет. Если нужен для
курсовой, есть варианты:

1. найти лог запуска, если он где-то сохранился;
2. перезапустить `alex/pipeline/04_newton.sh` и сохранить stdout;
3. доработать `newton_from_csv.py`, чтобы писать историю Ньютона в CSV/PNG.

Не делать это без согласия пользователя, потому что расчёты/файлы большие и
рабочее дерево грязное.

### 15.4 Спектр

Файлы:

```text
alex/linear_stability/eigenvalues.csv
alex/linear_stability/unstable_eigenvalues.csv
```

Найдено 30 собственных пар. Неустойчивая комплексно-сопряжённая пара:

```text
lambda_0 = 0.21701100237718393 + 1.7868186726299133 i
residual = 2.1663851436818140e-12

lambda_1 = 0.21701100237710524 - 1.7868186726295903 i
residual = 1.9351634958919610e-12
```

Частота:

```text
imag(lambda)/(2 pi) ≈ ±0.284381024
```

Рост:

```text
Re(lambda) ≈ 0.217011 > 0
```

Остальные из найденных имеют отрицательную действительную часть, например:

```text
lambda_2 ≈ -0.40110933899361984
lambda_3 ≈ -1.3604255618640773 + 1.235388472367885 i
```

Для курсовой нужен график спектра на комплексной плоскости. Его пока не видел
готовым среди PNG; можно будет сгенерировать из `eigenvalues.csv` в
`alex/paper/images`.

### 15.5 Неустойчивые моды

Файлы:

```text
alex/linear_stability/unstable_modes/eig_000.csv
alex/linear_stability/unstable_modes/eig_001.csv
```

Каждый содержит:

```text
x,y,psi_real,psi_imag,omega_real,omega_imag,lambda_real,lambda_imag
```

`eig_001` является комплексно-сопряжённой парой к `eig_000`.

Для курсовой нужны графики:

- `Re psi` и/или `Im psi` неустойчивой моды;
- `Re omega` и/или `Im omega`;
- возможно амплитуда моды.

Готовых PNG для самих мод не найдено; нужно сгенерировать.

### 15.6 Фильтрация неустойчивого подпространства

Файл:

```text
alex/linear_stability/filtered_snapshots.csv
```

Количество строк:

```text
1001
```

Базис проекции:

```text
basis_size = 2
modes = eig_000_real; eig_000_imag
```

То есть взята одна комплексная мода и её вещественная/мнимая части.

Норма проекции:

```text
first projection_norm = 25.748697324244283
last  projection_norm = 436.6939047442143
min   projection_norm = 7.444119580244065
max   projection_norm = 447.60820252194736
```

Фильтрованные CSV:

```text
alex/linear_stability/filtered_results/result_*.csv
```

### 15.7 Анализ после фильтрации

Файл:

```text
alex/linear_stability/filtered_detection/consecutive_difference_norm.csv
```

Минимум:

```text
previous_step = 345000
step          = 346000
time          = 3.46001
abs_l2        = 0.13633223461849
rel_l2        = 0.000115441560998205
linf          = 0.00590895243690071
```

Последняя относительная разность:

```text
raw last rel_l2      ≈ 0.0054439367
filtered last rel_l2 ≈ 0.0019536694
```

То есть фильтрация существенно уменьшила поздний рост разности соседних
кадров, но не полностью устранила динамику. Это хорошо интерпретируется:
вычитается только компонента вдоль найденного линейного неустойчивого
подпространства, а не вся эволюция.

График:

```text
alex/linear_stability/filtered_detection/consecutive_difference_norm.png
```

На нём минимум глубже (`~1.15e-4`), а поздний рост более волнистый и слабый.

## 16. Примеры статистики полей

`alex/data/results/result_0.csv`:

```text
psi = 0 везде
u = v = 0
omega small, вызвана первым шагом форсирования
omega min ≈ -3.512e-4
omega max ≈  3.540e-4
```

`alex/data/results/result_1000000.csv`:

```text
psi min = -0.098351014514809
psi max =  0.02763848277148
||psi||_2 = 4.820698079188049

omega min = -20.93975973763541
omega max =  26.247831209725494
||omega||_2 = 1122.7374799124855

u min/max = ±0.965266202467983
v min/max = ±0.522825468271449
```

`alex/linear_stability/filtered_results/result_1000000.csv`:

```text
psi min = -0.0614331860618963
psi max =  0.0542661669180665
||psi||_2 = 3.5213174846030175

omega min = -19.8461290564974
omega max =  20.6317289484971
||omega||_2 = 1040.9207685389197

u min/max = ±0.920799255958571
v min/max = ±0.500079657153515
```

Интерпретация: после фильтрации поле ближе к симметричному стационарному
режиму, амплитуда отклонения в `psi`, `omega`, скоростях уменьшается, но
структура течения сохраняется.

## 17. Графики, которые уже есть и которые надо использовать

Уже существуют:

```text
alex/stationary_detection/consecutive_difference_norm.png
alex/stationary_detection/stationary_streamplot.png
alex/stationary_detection/newton_equilibrium_streamplot.png
alex/linear_stability/filtered_detection/consecutive_difference_norm.png
alex/linear_stability/filtered_detection/stationary_streamplot.png
alex/plots/frames/result_1000000_streamplot.png
alex/plots/frames/result_1000000_psi.png
alex/plots/frames/result_1000000_omega.png
alex/linear_stability/plots/frames/result_1000000_streamplot.png
alex/linear_stability/plots/frames/result_1000000_psi.png
alex/linear_stability/plots/frames/result_1000000_omega.png
```

Уже просмотренные изображения:

- `stationary_streamplot.png`: четыре вихревые ячейки, кандидат при
  `t=2.36001`.
- `newton_equilibrium_streamplot.png`: более регулярный четырёхвихревой
  стационар.
- `consecutive_difference_norm.png`: минимум около `t=2.36001`, затем рост.
- `filtered_detection/consecutive_difference_norm.png`: минимум около
  `t=3.46001`, поздний рост меньше.
- `plots/frames/result_1000000_streamplot.png`: исходная поздняя динамика,
  нарушенная симметрия.
- `linear_stability/plots/frames/result_1000000_streamplot.png`: после
  фильтрации более симметричная картина.

Нужно сгенерировать для курсовой:

1. график спектра `lambda` на комплексной плоскости;
2. графики неустойчивой моды `eig_000`: `Re/Im psi`, `Re/Im omega`;
3. график нормы проекции/removed_norm во времени из `filtered_snapshots.csv`;
4. возможно сравнение raw vs filtered `rel_l2` на одном графике;
5. если пользователь подтвердит каверну без Аракавы, нужны отдельные графики
   каверны, которых в текущем `HEAD` может не быть.

## 18. Требования пользователя к будущей курсовой

Писать на русском, академически, но живо. Не документация к коду, а
восстановление:

- математической постановки;
- численной схемы;
- алгоритмов;
- структур данных;
- результатов и их физической интерпретации.

Нельзя:

- вставлять большие листинги кода;
- поверхностно описывать “функция делает то-то”;
- делать Аракаву центральной частью раздела о каверне, если пользователь
  подтверждает вариант без Аракавы;
- оставлять графики без интерпретации;
- выдумывать параметры/результаты/статьи.

Нужно использовать аккуратный LaTeX:

```latex
\begin{equation}
...
\label{eq:...}
\end{equation}
```

для важных формул, `\cref`/`\ref` для ссылок, единые обозначения:

```text
\psi, \omega, u, v, Re, D_x, D_y, \Delta_h, J_h
```

Пользователь хочет структуру:

1. Введение
2. Постановка задачи о каверне
3. Численная схема для задачи о каверне
4. Реализация и хранение данных для задачи о каверне
5. Результаты вычислений для задачи о каверне
6. Постановка задачи из второй статьи
7. Численная схема для второй задачи
8. Реализация и результаты для второй задачи
9. Поиск стационарного решения
10. Метод Ньютона
11. Линеаризация и спектральный анализ
12. Неустойчивые моды и стабилизация решения
13. Численные эксперименты по стабилизации
14. Заключение

Но из-за найденного расхождения возможно нужно адаптировать:

- разделы 2--5: старая каверна без Аракавы;
- разделы 6--13: текущая форсированная задача, Ньютон, спектр, фильтрация.

## 19. Математический каркас для будущего текста

### 19.1 Каверна

Если пользователь подтверждает старую реализацию:

Область:

```latex
\Omega=(0,1)\times(0,1).
```

Стенки:

```latex
u=1,\ v=0 \quad \text{на } y=1,
```

```latex
u=v=0 \quad \text{на остальных стенках}.
```

Функция тока и завихренность:

```latex
u=\psi_y,\qquad v=-\psi_x,
```

```latex
\omega=v_x-u_y=-\Delta\psi.
```

Система:

```latex
\Delta\psi+\omega=0,
```

```latex
\omega_t + J(\psi,\omega)
=
\frac1{Re}\Delta\omega.
```

или в используемом виде:

```latex
\omega_t =
\frac1{Re}\Delta\omega-J(\psi,\omega).
```

Якобиан:

```latex
J(\psi,\omega)=\psi_y\omega_x-\psi_x\omega_y.
```

Центральная аппроксимация без Аракавы:

```latex
(D_x f)_{i,j}=\frac{f_{i+1,j}-f_{i-1,j}}{2h_x},
\qquad
(D_y f)_{i,j}=\frac{f_{i,j+1}-f_{i,j-1}}{2h_y}.
```

```latex
J_h(\psi,\omega)_{i,j}
=
(D_y\psi)_{i,j}(D_x\omega)_{i,j}
-
(D_x\psi)_{i,j}(D_y\omega)_{i,j}.
```

Лапласиан:

```latex
\Delta_h f_{i,j}
=
\frac{f_{i+1,j}-2f_{i,j}+f_{i-1,j}}{h_x^2}

\frac{f_{i,j+1}-2f_{i,j}+f_{i,j-1}}{h_y^2}.
```

Граничная завихренность по Тому:

```latex
\omega_{i,0}=-\frac{2\psi_{i,1}}{h_y^2},
```

```latex
\omega_{i,N_y}=-\frac{2\psi_{i,N_y-1}}{h_y^2}
-\frac{2U}{h_y}.
```

Аналогично для боковых стенок.

Точность:

- центральные разности: второй порядок по пространству на гладких решениях;
- временная/псевдовременная схема: осторожно описать как первый порядок по
  времени плюс факторизованное неявное обращение диффузионных операторов, если
  ориентироваться на код.

### 19.2 Текущая форсированная задача

Область:

```latex
\Omega=(0,L_x)\times(0,L_y),\quad L_x=1,\ L_y=0.5.
```

Все стенки неподвижны:

```latex
\psi|_{\partial\Omega}=0,
```

тангенциальные скорости стенок равны нулю.

Система:

```latex
\Delta\psi+\omega=0,
```

```latex
\omega_t=
\frac1{Re}\Delta\omega
-J(\psi,\omega)
-F_\omega(x,y).
```

Стационар:

```latex
F(U)=0,
\qquad
U=(\psi,\omega).
```

Форсирование:

```latex
F_\omega(x,y)
=-\sum_{(m,n)} \partial_y
\left[
a_{mn}\sin\frac{\pi m x}{L_x}
\sin\frac{\pi n y}{L_y}
\right]
```

Но формулу надо аккуратно переписать: в коде `mode_dy` фактически

```text
-a_mn (pi n / ly) sin(pi m x/lx) sin(pi n y/ly)
```

а затем возвращается `-forcing`. Нужно не ошибиться со знаком. Лучше вывести
по коду перед финальным текстом.

### 19.3 Ньютон

Состояние:

```latex
U=(\psi_{1,1},\ldots,\psi_{N_x-1,N_y-1},
\omega_{1,1},\ldots,\omega_{N_x-1,N_y-1})^T.
```

Нелинейная система:

```latex
F(U)=0.
```

Ньютон:

```latex
J(U^k)\delta U^k=-F(U^k),
```

```latex
U^{k+1}=U^k+\alpha_k\delta U^k.
```

Матрица Якоби явно не строится, используется действие на вектор. Линейная
система решается GMRES, предобуславливатель -- приближённый стоксов блок через
обратный дискретный лапласиан.

### 19.4 Линеаризация

Пусть:

```latex
\psi=\psi_0+\phi,\qquad \omega=\omega_0+\eta.
```

Линеаризованная связь:

```latex
\Delta\phi+\eta=0.
```

Линеаризованное уравнение:

```latex
\eta_t=
\frac1{Re}\Delta\eta
-J(\phi,\omega_0)
-J(\psi_0,\eta).
```

После исключения `phi=-\Delta^{-1}\eta`:

```latex
\eta_t=\mathcal A\eta,
```

```latex
\mathcal A\eta
=
\frac1{Re}\Delta\eta
-J(-\Delta^{-1}\eta,\omega_0)
-J(\psi_0,\eta).
```

Спектр:

```latex
\mathcal A v=\lambda v.
```

Если `Re lambda > 0`, мода растёт.

### 19.5 Стабилизация

Вещественный базис:

```latex
e_1,\ldots,e_m
```

получен ортонормировкой `Re/Im` неустойчивых комплексных мод.

Проекция:

```latex
r_{\mathrm{unst}}=
\sum_{\ell=1}^m
\langle U-U_*,e_\ell\rangle e_\ell.
```

Коррекция:

```latex
U_{\mathrm{corr}}=
U-r_{\mathrm{unst}}.
```

Интерпретация: удаляется не всё движение, а только компонента вдоль
линейного неустойчивого подпространства около найденного стационара.

## 20. Что новый агент должен сделать дальше

Не писать сразу курсовую, если пользователь ещё не сказал “начинай”. Сначала
попросить уточнения, если их ещё нет:

1. Где статьи?
2. Какая версия кода считается основной для каверны без Аракавы?
3. Текущая форсированная задача -- это “вторая статья”?
4. Нужно ли восстанавливать/перезапускать график сходимости Ньютона?

После команды “начинай”:

1. Создать `alex/paper/images`.
2. Скопировать/сгенерировать нужные PNG:
   - `stationary_detection/consecutive_difference_norm.png`;
   - `stationary_detection/stationary_streamplot.png`;
   - `stationary_detection/newton_equilibrium_streamplot.png`;
   - `linear_stability/filtered_detection/consecutive_difference_norm.png`;
   - raw vs filtered streamplot/psi/omega;
   - спектр;
   - моды;
   - projection/removed norm.
3. Написать LaTeX-текст в `alex/paper`, вероятно `paper.tex`.
4. Не выдумывать данные по каверне, если их нет в текущей папке. Если нужны
   графики каверны без Аракавы, либо восстановить старый код в отдельную
   рабочую область/ветку с согласия пользователя, либо оставить пометки
   “требуется вставить”.
5. Следить за тем, чтобы раздел о каверне не называл Аракаву основной схемой,
   если пользователь это подтвердит.

## 21. Краткий ответ, который уже был дан пользователю

Пользователю уже сообщено:

- локальных PDF/статей не найдено;
- текущий `HEAD` использует Аракаву;
- вариант без Аракавы найден в истории git;
- текущая рабочая задача -- форсированная прямоугольная область с `Re=260`;
- собрана карта файлов и результатов;
- перед финальным текстом нужны уточнения.

Новый агент должен держать это в голове и не противоречить этому без проверки.

