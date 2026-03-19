#!/usr/bin/env bash
set -euo pipefail

# Определение ОС
OS="$(uname -s)"

# Настройки по умолчанию
CXX="g++"
EXT=""
STATIC_FLAGS=""
OMP_FLAGS=""

if [[ "$OS" == "Darwin" ]]; then
    echo "Detected: macOS"
    CXX="clang++"
    EXT="dylib"
    STATIC_FLAGS=""
    # OpenMP on macOS requires libomp (brew install libomp)
    LIBOMP="$(brew --prefix libomp 2>/dev/null || true)"
    if [[ -d "$LIBOMP" ]]; then
        OMP_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP}/include -L${LIBOMP}/lib -lomp"
        echo "OpenMP: enabled (libomp at ${LIBOMP})"
    else
        echo "OpenMP: disabled (run 'brew install libomp' to enable)"
    fi
elif [[ "$OS" == "MINGW64"* ]] || [[ "$OS" == "MSYS"* ]] || [[ "$OS" == "Windows"* ]]; then
    echo "Detected: Windows"
    CXX="g++"
    EXT="dll"
    STATIC_FLAGS="-static -static-libgcc -static-libstdc++"
    OMP_FLAGS="-fopenmp"
else
    echo "Detected: Linux"
    CXX="g++"
    EXT="so"
    STATIC_FLAGS=""
    OMP_FLAGS="-fopenmp"
fi

OUTPUT_NAME="solver.${EXT}"

echo "Building ${OUTPUT_NAME} with ${CXX}..."

# Компиляция
${CXX} -O3 -std=c++17 -shared -fPIC \
  -I . \
  -I ../External_libs \
  ${OMP_FLAGS} \
  stokes_mac.cpp \
  ${STATIC_FLAGS} \
  -o "${OUTPUT_NAME}"

echo "Built: ${OUTPUT_NAME}"