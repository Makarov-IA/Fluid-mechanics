#!/usr/bin/env bash
set -euo pipefail

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ALEX_DIR/scripts/bash/run.sh" "$@"
