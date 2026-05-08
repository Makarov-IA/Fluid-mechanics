# Linear Stability Pipeline

Этот README описывает практический пайплайн:

1. взять стационар, уточненный Ньютоном;
2. построить линеаризованный оператор;
3. найти собственные значения и собственные векторы через SciPy;
4. выбрать моды с положительной действительной частью;
5. вычесть их из каждого расчетного CSV в правильный момент времени;
6. построить графики и видео существующими общими скриптами.

Подробная математика метода собственных значений лежит в:

```text
alex/eigenvalue_method.md
```

## Основная команда

```bash
bash alex/scripts/linear_stability.sh
```

Скрипт берет стационар:

```text
alex/stationary_detection/newton_equilibrium.csv
```

и исходные кадры:

```text
alex/data/results/result_*.csv
```

Результаты сохраняются в:

```text
alex/linear_stability
```

## SciPy

Теперь для спектра используется:

```python
scipy.sparse.linalg.eigs
```

Если SciPy не установлен:

```bash
python3 -m pip install -r alex/requirements.txt
```

Если в корне проекта есть `.venv`, то `linear_stability.sh` автоматически
использует `.venv/bin/python`. При желании интерпретатор можно задать явно:

```bash
PYTHON_BIN=/path/to/python bash alex/scripts/linear_stability.sh
```

## Что именно решается

После линеаризации около стационара получается задача:

$$
\mathcal A v = \lambda v.
$$

Здесь \(v\) - собственная функция для возмущения вихря, а
\(\lambda\) - собственное значение.

Если

$$
\operatorname{Re}\lambda > 0,
$$

то мода растет во времени и считается неустойчивой.

## Почему матрица не собирается явно

Для сетки \(201\times101\) число неизвестных для вихря:

$$
n=(201-2)(101-2)=19701.
$$

Полная плотная матрица такого размера слишком тяжелая. Поэтому код создает
`LinearOperator`: объект, который для SciPy выглядит как матрица, но на самом
деле умеет только считать произведение:

$$
v \mapsto \mathcal A v.
$$

Это позволяет искать несколько важных собственных пар без хранения всей
матрицы.

## Настройки в `linear_stability.sh`

Количество собственных пар с максимальной действительной частью:

```bash
EIGS_COUNT=30
```

Точность ARPACK:

```bash
EIGS_TOL="1e-8"
```

Максимальное число итераций:

```bash
EIGS_MAX_ITER=3000
```

Если собственные значения уже найдены, а нужно только заново получить
`filtered_results` по сохраненным `unstable_modes/eig_*.csv`, можно не
запускать SciPy повторно:

```bash
FILTER_ONLY=true bash alex/scripts/linear_stability.sh
```

Это полезно, если прошлый запуск остановился уже на этапе фильтрации.

Порог неустойчивости:

```bash
UNSTABLE_TOL="1e-9"
```

Мода считается неустойчивой, если:

$$
\operatorname{Re}\lambda > \texttt{UNSTABLE\_TOL}.
$$

Амплитуда вычитания мод:

```bash
MODE_AMPLITUDE="1e-3"
```

## Выходные файлы

Все найденные собственные значения:

```text
alex/linear_stability/eigenvalues.csv
```

Только неустойчивые собственные значения:

```text
alex/linear_stability/unstable_eigenvalues.csv
```

Собственные функции неустойчивых мод:

```text
alex/linear_stability/unstable_modes/eig_XXX.csv
```

Отфильтрованные CSV:

```text
alex/linear_stability/filtered_results/result_*.csv
```

Индекс отфильтрованных файлов:

```text
alex/linear_stability/filtered_snapshots.csv
```

В нем есть:

```text
source_csv,filtered_csv,time,removed_norm,mode_amplitude,modes,plot_png
```

Если `removed_norm = 0`, значит из этого кадра ничего не вычиталось.

## Как вычитаются моды

Собственная функция зависит от времени:

$$
v_j(t)=e^{\lambda_j t}v_j.
$$

Поэтому для кадра в момент \(t_k\) код вычитает:

$$
\alpha
\operatorname{Re}
\left(
e^{\lambda_j t_k}v_j
\right).
$$

Если неустойчивых мод несколько:

$$
q_{\text{filtered}}(t_k)
=
q(t_k)
-
\alpha
\sum_{\operatorname{Re}\lambda_j>0}
\operatorname{Re}
\left(
e^{\lambda_j t_k}v_j
\right).
$$

## График после фильтрации

Чтобы построить график разности соседних кадров для отфильтрованных CSV:

```bash
bash alex/scripts/analyze_filtered.sh
```

Результат:

```text
alex/linear_stability/filtered_detection/consecutive_difference_norm.png
```

## Видео после фильтрации

Используется тот же общий рендерер, что и для обычных результатов:

```bash
bash alex/scripts/filtered_video.sh
```

Он внутри вызывает:

```bash
bash alex/scripts/plot.sh \
  alex/linear_stability/filtered_results \
  alex/linear_stability/plots \
  alex/linear_stability/filtered_snapshots.csv
```

а затем общий сборщик видео:

```text
alex/scripts/make_videos.py
```

Видео сохраняются в:

```text
alex/linear_stability/videos
```

## Полная последовательность

После завершения `run.sh`:

```bash
bash alex/scripts/find_stationary.sh
bash alex/scripts/newton_from_csv.sh
bash alex/scripts/linear_stability.sh
bash alex/scripts/analyze_filtered.sh
bash alex/scripts/filtered_video.sh
```

Если хочется нарисовать обычные исходные кадры и видео:

```bash
bash alex/scripts/plot.sh
bash alex/scripts/video.sh
```
