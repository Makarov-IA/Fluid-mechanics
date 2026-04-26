#!/usr/bin/env bash
set -euo pipefail

os="$(uname -s)"
cxx="g++"
ext=""
omp_flags=()
static_flags=()

case "$os" in
    Darwin)
        echo "Detected: macOS"
        cxx="clang++"
        ext="dylib"
        libomp="$(brew --prefix libomp 2>/dev/null || true)"
        if [[ -d "$libomp" ]]; then
            omp_flags=(
                -Xpreprocessor
                -fopenmp
                "-I${libomp}/include"
                "-L${libomp}/lib"
                -lomp
            )
            echo "OpenMP: enabled (libomp at ${libomp})"
        else
            echo "OpenMP: disabled (run 'brew install libomp' to enable)"
        fi
        ;;
    MINGW64*|MSYS*|Windows*)
        echo "Detected: Windows"
        ext="dll"
        omp_flags=(-fopenmp)
        static_flags=(-static -static-libgcc -static-libstdc++)
        ;;
    *)
        echo "Detected: Linux"
        ext="so"
        omp_flags=(-fopenmp)
        ;;
esac

output_name="solver.${ext}"
echo "Building ${output_name} with ${cxx}..."

cmd=(
    "${cxx}"
    -O3
    -std=c++17
    -shared
    -fPIC
    -I .
    -I ../External_libs
)

if ((${#omp_flags[@]})); then
    cmd+=("${omp_flags[@]}")
fi

cmd+=(stokes_mac.cpp)

if ((${#static_flags[@]})); then
    cmd+=("${static_flags[@]}")
fi

cmd+=(-o "${output_name}")

"${cmd[@]}"

echo "Built: ${output_name}"
