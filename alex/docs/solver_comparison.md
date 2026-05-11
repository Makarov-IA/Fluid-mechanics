# Отличие CPU И GPU Солверов

CPU и GPU backends используют одну физическую постановку в переменных `psi-omega`, но реализуют разные численные алгоритмы.

Коротко:

```text
CPU backend - авторская C++ факторизованная схема с прогонками.
GPU backend - PyTorch/CUDA реализация для cavity с DST-решением Пуассона.
```

## Общие Части

Оба backend'а используют:

```text
psi, omega, u, v
```

и одну связь:

```text
Delta psi = - omega.
```

Оба сохраняют одинаковый бинарный формат:

```text
result_*.bin
```

Внутри хранятся:

```text
psi,
omega,
u,
v.
```

Общий модуль чтения/записи:

```text
alex/common/snapshot_io.py
```

Оба используют:

```text
u = d psi / d y,
v = - d psi / d x.
```

Оба поддерживают выбор якобиана:

```text
use_arakawa=true
use_arakawa=false
```

## CPU Backend

Расположение:

```text
alex/cpu/
```

Основной solver:

```text
alex/cpu/solver/solver.cpp
```

Конфиг:

```text
alex/cpu/configs/pulsation.cfg
```

Правая часть:

```text
alex/cpu/forcing/omega_forcing.cpp
```

CPU backend решает задачу с объемным forcing:

```text
S_omega = - OmegaForcing(x, y, lx, ly).
```

### Схема Для Psi

В C++ solver функция тока обновляется факторизованной схемой.

Упрощенно:

```text
(I - dt delta_xx)(I - dt delta_yy) psi^{n+1}
= psi^n + dt omega^n + O(dt^2).
```

На каждом шаге решаются одномерные трехдиагональные системы:

```text
прогонка по x,
прогонка по y.
```

### Схема Для Omega

Завихренность также обновляется факторизованно:

```text
(I - dt/Re delta_xx)(I - dt/Re delta_yy) omega^{n+1}
= omega^n - dt J(psi^n, omega^n) - dt OmegaForcing + O(dt^2).
```

Диффузионный член обрабатывается неявно через прогонки.

### Особенность CPU Backend

Это основной авторский C++ код проекта. Он ближе к факторизованной схеме из курсовой и важен как самостоятельная реализация.

## GPU Backend

Расположение:

```text
alex/gpu/
```

Основной solver:

```text
alex/gpu/torch_cavity_solver.py
```

Конфиги:

```text
alex/gpu/configs/cavity_re1000.cfg
alex/gpu/configs/cavity_re5000.cfg
alex/gpu/configs/cavity_re10000.cfg
```

GPU backend используется для lid-driven cavity:

```text
S_omega = 0.
```

Движение задается верхней крышкой:

```text
u_top = 1,
v_top = 0.
```

### Решение Пуассона

На каждом шаге GPU backend решает:

```text
Delta_h psi = - omega
```

напрямую через дискретное синус-преобразование:

```text
DST -> деление на собственные значения Delta_h -> inverse DST.
```

Это возможно, потому что:

```text
psi = 0 on boundary.
```

Пятиточечный дискретный Лапласиан диагонализуется в синус-базисе:

```text
lambda_{k,l} = lambda_k^x + lambda_l^y,
```

где:

```text
lambda_k^x = -4 / dx^2 * sin^2(pi k / (2 (Nx - 1))),
lambda_l^y = -4 / dy^2 * sin^2(pi l / (2 (Ny - 1))).
```

DST считается через `torch.fft.fft` по нечетному продолжению массива.

### Шаг По Omega

Завихренность обновляется явно:

```text
omega^{n+1}
= omega^n + dt [ (1/Re) Delta_h omega^n - J_h(psi^n, omega^n) ].
```

После обновления применяются граничные условия Тома.

## Главное Отличие

CPU backend:

```text
факторизованная схема,
прогонки,
неявная обработка диффузии,
forcing для pulsation.
```

GPU backend:

```text
прямое DST-решение Пуассона,
явный шаг по omega,
CUDA-тензоры,
forcing = 0 для cavity.
```

Поэтому GPU solver не является побайтовой копией C++ solver. Это отдельная ускоренная реализация той же `psi-omega` постановки для cavity benchmark.

## Что Можно Сравнивать

Корректные величины для верификации:

```text
положение центра основного вихря,
значение psi в центре вихря,
профиль u(0.5, y),
профиль v(x, 0.5),
картина линий тока,
история невязки.
```

Не стоит ожидать совпадения всех значений поля точка-в-точку между CPU и GPU, потому что отличаются:

```text
временная схема,
решение уравнения Пуассона,
порядок операций,
CPU/GPU арифметика.
```

## Если Нужно Свести Разницу

Чтобы приблизить GPU backend к CPU backend, нужно реализовать в PyTorch второй режим:

```text
scheme=factorized
```

Он должен повторять:

```text
solvePsi,
ApplyThomBoundary,
solveOmega,
computeResiduals.
```

То есть заменить DST и явный шаг на батчевую GPU-версию прогонки. Это будет ближе к C++ методу, но обычно медленнее текущего DST-варианта.
