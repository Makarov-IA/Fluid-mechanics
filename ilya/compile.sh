#!/usr/bin/env bash
set -euo pipefail

g++ -O3 -std=c++17 -shared \
  -I . \
  -I ../External_libs \
  solver.cpp \
  grid.cpp \
  -static -static-libgcc -static-libstdc++ \
  -o solver.dll

echo "Built: solver.dll"
