# Task2: implicit finite-difference solver

Решается система на `[0,1] x [0,1] x [0,1]`:

- `omega_t - nu * Delta(omega) = g(x,y,t)`
- `Delta(psi) = omega`

Схема:
- по времени: неявная Backward Euler (1-й порядок)
- по пространству: разностный лапласиан на 5-точечном шаблоне
- для `Delta(psi)=omega`: итерационный разностный решатель (Gauss-Seidel)

## Build

```bash
cd alex/task2
make all
```

## Run

```bash
./task2 --Nx 81 --Ny 81 --Nt 200 --nu 0.01 --lid 1 \
  --forcing sin --forcing-amp 0.5 --forcing-omega 6.283185307 \
  --save-every 20 --output-dir results
```

`t` всегда на отрезке `[0,1]`, `dt = 1 / Nt`.

## Output

В `results` сохраняются:
- `psi.txt`, `omega.txt` (финальный момент `t=1`)
- `meta.txt`
- `snapshots/index.csv`
- `snapshots/psi_stepXXXXX.txt`, `snapshots/omega_stepXXXXX.txt`

## Plot

Выбор графика через `plot.sh`:

```bash
bash scripts/plot.sh --field omega --graph contour --snapshot-index 2 --save --no-show
```

Доступно:
- `--field`: `psi` или `omega`
- `--graph`: `heatmap`, `surface`, `contour`, `all`
- `--snapshot-index`: индекс среза по времени (`-1` = финальный)
- `--list-snapshots`: показать доступные срезы

Графики `u`, `v`, `g` убраны.

## Tests

```bash
make test
```
