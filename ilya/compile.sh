#!/usr/bin/env bash
set -euo pipefail

# Определение ОС
OS="$(uname -s)"

# Настройки по умолчанию
CXX="g++"
EXT=""
STATIC_FLAGS=""

if [[ "$OS" == "Darwin" ]]; then
    # macOS
    echo "Detected: macOS"
    CXX="clang++"
    EXT="dylib"
    # На macOS не рекомендуется использовать -static вместе с -shared
    STATIC_FLAGS=""
elif [[ "$OS" == "MINGW64"* ]] || [[ "$OS" == "MSYS"* ]] || [[ "$OS" == "Windows"* ]]; then
    # Windows (Git Bash / MinGW)
    echo "Detected: Windows"
    CXX="g++"
    EXT="dll"
    # На Windows статическая линковка рантаймов полезна для переносимости
    STATIC_FLAGS="-static -static-libgcc -static-libstdc++"
else
    # Linux (на всякий случай)
    echo "Detected: Linux"
    CXX="g++"
    EXT="so"
    STATIC_FLAGS=""
fi

OUTPUT_NAME="solver.${EXT}"

echo "Building ${OUTPUT_NAME} with ${CXX}..."

# Компиляция
${CXX} -O3 -std=c++17 -shared -fPIC \
  -I . \
  -I ../External_libs \
  stokes_mac.cpp \
  ${STATIC_FLAGS} \
  -o "${OUTPUT_NAME}"

echo "Built: ${OUTPUT_NAME}"