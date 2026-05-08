#!/usr/bin/env bash
set -euo pipefail

os="$(uname -s)"
cxx="g++"
ext=""
omp_flags=()
static_flags=()
solver_flags=()
solver_libs=()
solver_backend="${LINEAR_SOLVER:-auto}"

umfpack_include_dir=""
umfpack_lib_dir=""

find_umfpack() {
    local candidates=()
    local prefix include_dir lib_dir

    if command -v brew >/dev/null 2>&1; then
        candidates+=("$(brew --prefix suite-sparse 2>/dev/null || true)")
        candidates+=("$(brew --prefix suitesparse 2>/dev/null || true)")
    fi

    candidates+=(
        /opt/homebrew/opt/suite-sparse
        /opt/homebrew
        /usr/local
        /usr
    )

    for prefix in "${candidates[@]}"; do
        [[ -n "$prefix" ]] || continue
        for include_dir in "$prefix/include" "$prefix/include/suitesparse"; do
            [[ -f "$include_dir/umfpack.h" ]] || continue
            for lib_dir in "$prefix/lib" "$prefix/lib64"; do
                if [[ -f "$lib_dir/libumfpack.dylib" ||
                      -f "$lib_dir/libumfpack.so" ||
                      -f "$lib_dir/libumfpack.a" ]]; then
                    umfpack_include_dir="$include_dir"
                    umfpack_lib_dir="$lib_dir"
                    return 0
                fi
            done
        done
    done

    return 1
}

enable_umfpack() {
    solver_flags=(
        -DUSE_UMFPACK
        "-I${umfpack_include_dir}"
        "-L${umfpack_lib_dir}"
    )
    solver_libs=(
        -lumfpack
        -lamd
        -lcholmod
        -lcolamd
        -lcamd
        -lccolamd
        -lsuitesparseconfig
    )
    echo "Linear solver: UMFPACK (${umfpack_lib_dir})"
}

enable_accelerate_qr() {
    solver_flags=(-DUSE_ACCELERATE_QR)
    solver_libs=(-framework Accelerate)
    echo "Linear solver: Apple Accelerate QR"
}

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

case "$solver_backend" in
    auto)
        if find_umfpack; then
            enable_umfpack
        else
            echo "Linear solver: Eigen SparseLU"
        fi
        ;;
    umfpack)
        if find_umfpack; then
            enable_umfpack
        else
            echo "UMFPACK was requested, but SuiteSparse was not found" >&2
            exit 1
        fi
        ;;
    accelerate|accelerate_qr)
        if [[ "$os" != "Darwin" ]]; then
            echo "Apple Accelerate QR is available only on macOS" >&2
            exit 1
        fi
        enable_accelerate_qr
        ;;
    sparselu|eigen)
        echo "Linear solver: Eigen SparseLU"
        ;;
    *)
        echo "Unknown LINEAR_SOLVER='${solver_backend}'" >&2
        echo "Use one of: auto, umfpack, accelerate, sparselu" >&2
        exit 1
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

if ((${#solver_flags[@]})); then
    cmd+=("${solver_flags[@]}")
fi

cmd+=(stokes_mac.cpp)

if ((${#solver_libs[@]})); then
    cmd+=("${solver_libs[@]}")
fi

if ((${#static_flags[@]})); then
    cmd+=("${static_flags[@]}")
fi

cmd+=(-o "${output_name}")

"${cmd[@]}"

echo "Built: ${output_name}"
