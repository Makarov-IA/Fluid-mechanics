# Alex pipeline

Эта папка - слой запуска. Python-файлы в `alex/scripts` считаются внутренней
реализацией; обычно их не нужно запускать руками.

## Если решение уже посчитано

```bash
bash alex/pipeline/01_frames.sh
bash alex/pipeline/02_video.sh

bash alex/pipeline/03_find_stationary.sh
bash alex/pipeline/04_newton.sh
bash alex/pipeline/05_linear_stability.sh
bash alex/pipeline/06_analyze_filtered.sh
bash alex/pipeline/07_filtered_video.sh
```

## Если считать с нуля

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

## Опционально

```bash
bash alex/pipeline/tools/streamplot_csv.sh
```
