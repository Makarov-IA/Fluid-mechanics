# Математическая Постановка

В проекте используется двумерная постановка для несжимаемой вязкой жидкости в переменных функция тока-завихренность.

## Основные Переменные

Скорость задается через функцию тока:

```text
u = d psi / d y,
v = - d psi / d x.
```

Завихренность:

```text
omega = d v / d x - d u / d y.
```

Из этих определений следует связь:

```text
Delta psi = - omega.
```

Здесь:

```text
psi   - функция тока,
omega - завихренность,
u, v  - компоненты скорости,
Re    - число Рейнольдса.
```

## Уравнения

Нестационарное уравнение переноса завихренности записывается в виде:

```text
d omega / d t + J(psi, omega) = (1 / Re) Delta omega + S_omega.
```

Якобиан:

```text
J(psi, omega)
= psi_x omega_y - psi_y omega_x.
```

Итоговая система:

```text
Delta psi = - omega,

d omega / d t
= (1 / Re) Delta omega - J(psi, omega) + S_omega.
```

В стационарном случае:

```text
Delta psi + omega = 0,

(1 / Re) Delta omega - J(psi, omega) + S_omega = 0.
```

В C++ backend используется функция `OmegaForcing`. В коде она входит со знаком минус:

```text
S_omega = - OmegaForcing(x, y, lx, ly).
```

Поэтому стационарная невязка CPU backend имеет вид:

```text
R_omega = (1 / Re) Delta omega - J(psi, omega) - OmegaForcing.
```

Для GPU cavity:

```text
S_omega = 0.
```

## Граничные Условия

Для замкнутых стенок используется постоянная функция тока:

```text
psi = 0 on boundary.
```

Скорости на стенках задаются через конфиг:

```text
bc.left.u,   bc.left.v
bc.right.u,  bc.right.v
bc.bottom.u, bc.bottom.v
bc.top.u,    bc.top.v
```

Завихренность на стенке задается формулой Тома. Например, для горизонтальной стенки:

```text
omega_wall = - 2 psi_near / h^2 - 2 U_wall / h.
```

Для верхней стенки lid-driven cavity:

```text
u_top = 1,
v_top = 0.
```

Для задачи пульсации:

```text
u_wall = 0,
v_wall = 0.
```

## Две Физические Задачи

### CPU Pulsation

CPU backend решает задачу вынужденного течения в прямоугольной области.

Типичный конфиг:

```text
alex/cpu/configs/pulsation.cfg
```

Область:

```text
0 <= x <= lx,
0 <= y <= ly.
```

Правая часть задается функцией:

```text
OmegaForcing(x, y, lx, ly).
```

Она реализована в:

```text
alex/cpu/forcing/omega_forcing.cpp
```

В уравнении завихренности она входит как объемное воздействие со знаком, указанным выше:

```text
S_omega = - OmegaForcing.
```

### GPU Cavity

GPU backend решает lid-driven cavity.

Типичные конфиги:

```text
alex/gpu/configs/cavity_re1000.cfg
alex/gpu/configs/cavity_re5000.cfg
alex/gpu/configs/cavity_re10000.cfg
```

Область:

```text
0 <= x <= 1,
0 <= y <= 1.
```

Правая часть:

```text
S_omega = 0.
```

Движение создается движущейся верхней крышкой:

```text
u_top = 1,
v_top = 0.
```

## Дискретизация

Используется равномерная сетка:

```text
x_i = i dx,
y_j = j dy.
```

Пятиточечный оператор Лапласа:

```text
(Delta_h q)_{i,j}
= (q_{i-1,j} - 2 q_{i,j} + q_{i+1,j}) / dx^2
 + (q_{i,j-1} - 2 q_{i,j} + q_{i,j+1}) / dy^2.
```

Центральная аппроксимация якобиана:

```text
J_h(psi, omega)
= ((psi_{i+1,j} - psi_{i-1,j}) / (2 dx))
 * ((omega_{i,j+1} - omega_{i,j-1}) / (2 dy))
 - ((psi_{i,j+1} - psi_{i,j-1}) / (2 dy))
 * ((omega_{i+1,j} - omega_{i-1,j}) / (2 dx)).
```

Также поддерживается схема Аракавы:

```text
J_A = (J_1 + J_2 + J_3) / 3.
```

Переключатель:

```text
use_arakawa=true/false
```

## Невязки

Для контроля стационарности используются:

```text
R_psi = Delta_h psi + omega,

R_omega = (1 / Re) Delta_h omega - J_h(psi, omega) + S_omega.
```

В расчетах сохраняется:

```text
residual_history.csv
```

с величинами:

```text
psi_res,
omega_res,
max_residual.
```
