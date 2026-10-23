#!/usr/bin/env bash

ALEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ALEX_DIR/../.venv/bin/python"
EXPERIMENT="pulsation_re260"
CONFIG="$ALEX_DIR/configs/$EXPERIMENT/$EXPERIMENT.yaml"
