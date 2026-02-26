#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TASK2_ROOT="$SCRIPT_DIR/.."

cd "$TASK2_ROOT" || exit 1

if command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
else
  echo "Python not found. Install Python 3 with matplotlib and numpy."
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: bash scripts/plot.sh --field omega --graph contour --snapshot-index 3 --save --no-show"
  echo "Fields: psi | omega"
  echo "Graphs: heatmap | surface | contour | all"
  echo "Use --list-snapshots to inspect available time snapshots"
fi

"${PYTHON_CMD[@]}" ./plot_solution.py "$@"
