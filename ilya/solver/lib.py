"""Python wrapper around the compiled C++ Stokes MAC solver (ctypes)."""

from __future__ import annotations

import ctypes as ct
import platform
from pathlib import Path

import numpy as np


def find_solver_lib(directory: Path) -> Path:
    """Return the path to the compiled solver shared library for the current OS."""
    ext_map = {"Darwin": ".dylib", "Windows": ".dll", "Linux": ".so"}
    ext = ext_map.get(platform.system())
    if ext is None:
        raise RuntimeError(f"Unsupported OS: {platform.system()}")

    exact = directory / f"solver{ext}"
    if exact.exists():
        return exact

    candidates = [
        f for f in directory.iterdir()
        if f.name.startswith("solver") and f.suffix == ext
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Solver library not found in {directory}. "
            f"Expected '{exact.name}'. Run compile.sh first."
        )
    lib = candidates[0]
    print(f"[warning] Using alternative library name: {lib.name}")
    return lib


class StokesMACLib:
    """
    Python wrapper around the compiled C++ Stokes MAC solver.

    Grid conventions (see stokes_mac.h):
        p[i, j]  shape (Nx,   Ny  )  — pressure at cell centres
        u[i, j]  shape (Nx+1, Ny  )  — x-velocity at vertical faces
        v[i, j]  shape (Nx,   Ny+1)  — y-velocity at horizontal faces
    """

    def __init__(
        self,
        lib_path: Path,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        nu: float,
        dt: float,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self._handle: ct.c_void_p | None = None

        try:
            self._dll = ct.CDLL(str(lib_path), mode=ct.RTLD_GLOBAL)
        except AttributeError:
            self._dll = ct.CDLL(str(lib_path))

        self._bind_c_api()

        self._handle = self._dll.stokes_mac_create_c(nx, ny, lx, ly, nu, dt)
        if not self._handle:
            raise RuntimeError("stokes_mac_create_c returned NULL")

    def __enter__(self) -> "StokesMACLib":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def set_bc_arrays(self, bcs: dict[str, np.ndarray]) -> None:
        """Push boundary-condition arrays to the C++ solver."""

        def _ptr(arr: np.ndarray) -> ct.POINTER(ct.c_double):
            return arr.astype(np.float64, copy=False).ctypes.data_as(
                ct.POINTER(ct.c_double)
            )

        self._dll.stokes_mac_set_bc_c(
            self._handle,
            _ptr(bcs["u_top"]),
            _ptr(bcs["u_bot"]),
            _ptr(bcs["v_left"]),
            _ptr(bcs["v_right"]),
            _ptr(bcs["u_left"]),
            _ptr(bcs["u_right"]),
            _ptr(bcs["v_bot"]),
            _ptr(bcs["v_top"]),
        )

    def run_steps(self, t_start: float, n_steps: int) -> np.ndarray:
        """Run n_steps with zero body force. Returns max|div u| per step."""
        div_out = np.empty(n_steps, dtype=np.float64)
        self._dll.stokes_mac_run_steps_c(
            self._handle,
            ct.c_double(t_start),
            ct.c_int(n_steps),
            div_out.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        return div_out

    def run_steps_with_force(
        self,
        t_start: float,
        n_steps: int,
        fu: np.ndarray | None,
        fv: np.ndarray | None,
    ) -> np.ndarray:
        """Run n_steps with constant pre-evaluated force arrays. Returns max|div u| per step."""
        div_out = np.empty(n_steps, dtype=np.float64)

        def _ptr_or_null(arr):
            if arr is None:
                return ct.cast(None, ct.POINTER(ct.c_double))
            return arr.ctypes.data_as(ct.POINTER(ct.c_double))

        self._dll.stokes_mac_run_steps_with_force_c(
            self._handle,
            ct.c_double(t_start),
            ct.c_int(n_steps),
            _ptr_or_null(fu),
            _ptr_or_null(fv),
            div_out.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        return div_out

    def get_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p, u, v) arrays on the MAC grid (fresh float64 copies)."""
        p = self._copy_field(
            self._dll.stokes_mac_get_p_c(self._handle), (self.nx, self.ny)
        )
        u = self._copy_field(
            self._dll.stokes_mac_get_u_c(self._handle), (self.nx + 1, self.ny)
        )
        v = self._copy_field(
            self._dll.stokes_mac_get_v_c(self._handle), (self.nx, self.ny + 1)
        )
        return p, u, v

    def close(self) -> None:
        if self._handle is not None:
            self._dll.stokes_mac_free_c(self._handle)
            self._handle = None

    def _bind_c_api(self) -> None:
        dll = self._dll
        dll.stokes_mac_create_c.argtypes = [
            ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ]
        dll.stokes_mac_create_c.restype = ct.c_void_p
        dll.stokes_mac_free_c.argtypes = [ct.c_void_p]
        dll.stokes_mac_free_c.restype = None
        dll.stokes_mac_run_steps_c.argtypes = [
            ct.c_void_p, ct.c_double, ct.c_int, ct.POINTER(ct.c_double),
        ]
        dll.stokes_mac_run_steps_c.restype = None

        _dbl_p = ct.POINTER(ct.c_double)
        dll.stokes_mac_set_bc_c.argtypes = [
            ct.c_void_p,
            _dbl_p, _dbl_p, _dbl_p, _dbl_p,  # u_top, u_bot, v_left, v_right
            _dbl_p, _dbl_p, _dbl_p, _dbl_p,  # u_left, u_right, v_bot, v_top
        ]
        dll.stokes_mac_set_bc_c.restype = None

        dll.stokes_mac_run_steps_with_force_c.argtypes = [
            ct.c_void_p, ct.c_double, ct.c_int,
            _dbl_p, _dbl_p,  # fu, fv (may be NULL)
            _dbl_p,          # div_out
        ]
        dll.stokes_mac_run_steps_with_force_c.restype = None

        for name in ("stokes_mac_get_p_c", "stokes_mac_get_u_c", "stokes_mac_get_v_c"):
            fn = getattr(dll, name)
            fn.argtypes = [ct.c_void_p]
            fn.restype = ct.POINTER(ct.c_double)

    def _copy_field(
        self, ptr: ct.POINTER(ct.c_double), shape_xy: tuple[int, int]
    ) -> np.ndarray:
        """Copy a C row-major array into a (Nx, Ny) numpy array (x-first)."""
        nx, ny = shape_xy
        return np.ctypeslib.as_array(ptr, shape=(nx * ny,)).copy().reshape(ny, nx).T
