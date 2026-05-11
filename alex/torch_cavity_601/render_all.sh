#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/alex/torch_cavity_601"

CASES=(
  "re1000:$EXP_DIR/configs/cavity_re1000.cfg:$EXP_DIR/results/re1000"
  "re5000:$EXP_DIR/configs/cavity_re5000.cfg:$EXP_DIR/results/re5000"
  "re10000:$EXP_DIR/configs/cavity_re10000.cfg:$EXP_DIR/results/re10000"
)

for item in "${CASES[@]}"; do
  IFS=':' read -r name config results <<< "$item"
  analysis="$EXP_DIR/analysis/$name"
  if [[ ! -d "$results" ]]; then
    echo "[render-all] missing results directory, skipped: $results" >&2
    continue
  fi
  bash "$EXP_DIR/render_one.sh" "$results" "$config" "$analysis"
done
