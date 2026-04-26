"""ctypes wrapper around the compiled MAC Navier-Stokes solver."""

from __future__ import annotations

import ctypes as ct
import platform
from pathlib import Path

import numpy as np


def find_solver_lib(directory: Path) -> Path:
    """Return the platform-specific shared library path."""
    ext_map = {"Darwin": ".dylib", "Windows": ".dll", "Linux": ".so"}
    ext = ext_map.get(platform.system())
    if ext is None:
        raise RuntimeError(f"Unsupported OS: {platform.system()}")

    exact = directory / f"solver{ext}"
    if exact.exists():
        return exact

    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.name.startswith("solver") and path.suffix == ext
    )
    if not candidates:
        raise FileNotFoundError(
            f"Solver library not found in {directory}. "
            f"Expected '{exact.name}'. Run compile.sh first."
        )

    lib = candidates[0]
    print(f"[warning] Using alternative library name: {lib.name}")
    return lib


class StokesMACLib:
    """Python wrapper around the compiled C++ MAC solver."""

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
        self._nu = (nx - 1) * ny
        self._nv = nx * (ny - 1)
        self._np = nx * ny
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

    def run_steps_diagnostics(
        self,
        t_start: float,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run n_steps with zero body force and return (div, ||ΔU||∞) per step."""
        div_out = np.empty(n_steps, dtype=np.float64)
        change_out = np.empty(n_steps, dtype=np.float64)
        self._dll.stokes_mac_run_steps_diagnostics_c(
            self._handle,
            ct.c_double(t_start),
            ct.c_int(n_steps),
            div_out.ctypes.data_as(ct.POINTER(ct.c_double)),
            change_out.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        return div_out, change_out

    def run_steps_with_force(
        self,
        t_start: float,
        n_steps: int,
        fu: np.ndarray | None,
        fv: np.ndarray | None,
    ) -> np.ndarray:
        """Run n_steps with constant pre-evaluated force arrays."""
        div_out = np.empty(n_steps, dtype=np.float64)

        def _ptr_or_null(arr: np.ndarray | None) -> ct.POINTER(ct.c_double):
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

    def run_steps_with_force_diagnostics(
        self,
        t_start: float,
        n_steps: int,
        fu: np.ndarray | None,
        fv: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run n_steps with constant force arrays and return (div, ||ΔU||∞) per step."""
        div_out = np.empty(n_steps, dtype=np.float64)
        change_out = np.empty(n_steps, dtype=np.float64)

        def _ptr_or_null(arr: np.ndarray | None) -> ct.POINTER(ct.c_double):
            if arr is None:
                return ct.cast(None, ct.POINTER(ct.c_double))
            return arr.ctypes.data_as(ct.POINTER(ct.c_double))

        self._dll.stokes_mac_run_steps_with_force_diagnostics_c(
            self._handle,
            ct.c_double(t_start),
            ct.c_int(n_steps),
            _ptr_or_null(fu),
            _ptr_or_null(fv),
            div_out.ctypes.data_as(ct.POINTER(ct.c_double)),
            change_out.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        return div_out, change_out

    def get_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p, u, v) arrays on the MAC grid."""
        p = self._copy_field(self._dll.stokes_mac_get_p_c(self._handle), (self.nx, self.ny))
        u = self._copy_field(
            self._dll.stokes_mac_get_u_c(self._handle),
            (self.nx + 1, self.ny),
        )
        v = self._copy_field(
            self._dll.stokes_mac_get_v_c(self._handle),
            (self.nx, self.ny + 1),
        )
        return p, u, v

    def set_state(
        self,
        u_vec: np.ndarray,
        v_vec: np.ndarray,
        p_vec: np.ndarray | None = None,
    ) -> None:
        """Overwrite the current solver state from interior unknown arrays."""
        u_arr = np.ascontiguousarray(u_vec, dtype=np.float64)
        v_arr = np.ascontiguousarray(v_vec, dtype=np.float64)
        if u_arr.shape != (self._nu,):
            raise ValueError(f"u_vec must have shape {(self._nu,)}, got {u_arr.shape}")
        if v_arr.shape != (self._nv,):
            raise ValueError(f"v_vec must have shape {(self._nv,)}, got {v_arr.shape}")

        double_ptr = ct.POINTER(ct.c_double)
        if p_vec is not None:
            p_arr = np.ascontiguousarray(p_vec, dtype=np.float64)
            if p_arr.shape != (self._np,):
                raise ValueError(f"p_vec must have shape {(self._np,)}, got {p_arr.shape}")
            p_ptr = p_arr.ctypes.data_as(double_ptr)
        else:
            p_ptr = ct.cast(None, double_ptr)

        self._dll.stokes_mac_set_state_c(
            self._handle,
            u_arr.ctypes.data_as(double_ptr),
            v_arr.ctypes.data_as(double_ptr),
            p_ptr,
        )

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (u_interior, v_interior, p_cells) in solver unknown ordering."""
        u_arr = np.empty(self._nu, dtype=np.float64)
        v_arr = np.empty(self._nv, dtype=np.float64)
        p_arr = np.empty(self._np, dtype=np.float64)
        double_ptr = ct.POINTER(ct.c_double)
        self._dll.stokes_mac_get_state_c(
            self._handle,
            u_arr.ctypes.data_as(double_ptr),
            v_arr.ctypes.data_as(double_ptr),
            p_arr.ctypes.data_as(double_ptr),
        )
        return u_arr, v_arr, p_arr

    @property
    def nu_u(self) -> int:
        return self._nu

    @property
    def nv_u(self) -> int:
        return self._nv

    @property
    def np_u(self) -> int:
        return self._np

    def close(self) -> None:
        if self._handle is not None:
            self._dll.stokes_mac_free_c(self._handle)
            self._handle = None

    def _bind_c_api(self) -> None:
        dll = self._dll
        dll.stokes_mac_create_c.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
        ]
        dll.stokes_mac_create_c.restype = ct.c_void_p

        dll.stokes_mac_free_c.argtypes = [ct.c_void_p]
        dll.stokes_mac_free_c.restype = None

        dll.stokes_mac_run_steps_c.argtypes = [
            ct.c_void_p,
            ct.c_double,
            ct.c_int,
            ct.POINTER(ct.c_double),
        ]
        dll.stokes_mac_run_steps_c.restype = None

        dll.stokes_mac_run_steps_diagnostics_c.argtypes = [
            ct.c_void_p,
            ct.c_double,
            ct.c_int,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        dll.stokes_mac_run_steps_diagnostics_c.restype = None

        double_ptr = ct.POINTER(ct.c_double)
        dll.stokes_mac_set_bc_c.argtypes = [
            ct.c_void_p,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
        ]
        dll.stokes_mac_set_bc_c.restype = None

        dll.stokes_mac_run_steps_with_force_c.argtypes = [
            ct.c_void_p,
            ct.c_double,
            ct.c_int,
            double_ptr,
            double_ptr,
            double_ptr,
        ]
        dll.stokes_mac_run_steps_with_force_c.restype = None

        dll.stokes_mac_run_steps_with_force_diagnostics_c.argtypes = [
            ct.c_void_p,
            ct.c_double,
            ct.c_int,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
        ]
        dll.stokes_mac_run_steps_with_force_diagnostics_c.restype = None

        dll.stokes_mac_set_state_c.argtypes = [
            ct.c_void_p,
            double_ptr,
            double_ptr,
            double_ptr,
        ]
        dll.stokes_mac_set_state_c.restype = None

        dll.stokes_mac_get_state_c.argtypes = [
            ct.c_void_p,
            double_ptr,
            double_ptr,
            double_ptr,
        ]
        dll.stokes_mac_get_state_c.restype = None

        for name in ("stokes_mac_get_p_c", "stokes_mac_get_u_c", "stokes_mac_get_v_c"):
            fn = getattr(dll, name)
            fn.argtypes = [ct.c_void_p]
            fn.restype = ct.POINTER(ct.c_double)

    def _copy_field(
        self,
        ptr: ct.POINTER(ct.c_double),
        shape_xy: tuple[int, int],
    ) -> np.ndarray:
        """Copy a C row-major array into a (Nx, Ny) NumPy array (x-first)."""
        nx, ny = shape_xy
        return np.ctypeslib.as_array(ptr, shape=(nx * ny,)).copy().reshape(ny, nx).T
