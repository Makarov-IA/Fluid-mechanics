# Task2: omega-psi solver

Решается система в квадрате `[0,1]x[0,1]`:

- `∂ω/∂t - νΔω = g(x,y,t)`
- `Δψ = ω`

То есть в коде используется эквивалентная форма:

- `ω_t = νΔω + g`
- `ω = Δψ`

Граничные условия скорости через функцию тока:
- на нижней/левой/правой границе: `u=0`, `v=0`
- на верхней: `u=1` (по умолчанию, задается `--lid`), `v=0`

Это соответствует вашим условиям: производные `psi` (компоненты скорости) нулевые на трех сторонах и `u=1` сверху.

## Build

```bash
cd alex/task2
make all
```

## Run

```bash
./task2 --Nx 81 --Ny 81 --nu 0.01 --lid 1 --forcing zero --tol 1e-6 --max-iter 50000 --output-dir results
```

Или через bash-скрипт:

```bash
bash scripts/run.sh --Nx 81 --Ny 81 --nu 0.01 --lid 1 --forcing zero --output-dir results
```

## Forcing g

- `--forcing zero` : `g = 0`
- `--forcing sin --forcing-amp A` : `g = A*sin(pi*x)*sin(pi*y)`
- `--forcing-omega W` добавляет множитель `cos(W*t)`

## Output files

В `--output-dir` сохраняются:
- `psi.txt`
- `omega.txt`
- `u.txt`
- `v.txt`
- `g.txt`
- `meta.txt`

## Plots

```bash
bash scripts/plot.sh --input-dir results --output-dir plots --field all --save --no-show
```

Доступные поля: `psi`, `omega`, `u`, `v`, `g`, `speed`.

## Tests

Автотесты:

```bash
bash scripts/test.sh
```

или

```bash
make test
```

Проверяются сценарии:
1. `g=0`, `lid=0` -> почти нулевое решение.
2. `g=sin`, `lid=0` -> ненулевое поле `omega`.
3. `g=0`, `lid=1` -> полный прогон, файлы и генерация графика.
