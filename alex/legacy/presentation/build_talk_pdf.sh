#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error talk_10min.tex
elif command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode -halt-on-error talk_10min.tex
    pdflatex -interaction=nonstopmode -halt-on-error talk_10min.tex
else
    echo "No LaTeX engine found. Install latexmk/pdflatex or upload talk_10min.tex to Overleaf." >&2
    exit 1
fi
