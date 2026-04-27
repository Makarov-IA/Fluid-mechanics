"""ctypes wrapper around the compiled MAC Navier-Stokes solver."""

from __future__ import annotations

import ctypes as ct
import platform
from pathlib import Path

import numpy as np

DoublePtr = ct.POINTER(ct.c_double)


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
        arrays = [
            self._double_array(bcs["u_top"]),
            self._double_array(bcs["u_bot"]),
            self._double_array(bcs["v_left"]),
            self._double_array(bcs["v_right"]),
            self._double_array(bcs["u_left"]),
            self._double_array(bcs["u_right"]),
            self._double_array(bcs["v_bot"]),
            self._double_array(bcs["v_top"]),
        ]
        ptrs = [self._double_ptr(arr) for arr in arrays]
        self._dll.stokes_mac_set_bc_c(
            self._handle,
            *ptrs,
        )

    def run_steps(self, t_start: float, n_steps: int) -> np.ndarray:
        """Run n_steps with zero body force. Returns max|div u| per step."""
        div_out = np.empty(n_steps, dtype=np.float64)
        self._dll.stokes_mac_run_steps_c(
            self._handle,
            ct.c_double(t_start),
            ct.c_int(n_steps),
            self._double_ptr(div_out),
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
            self._double_ptr(div_out),
            self._double_ptr(change_out),
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
        fu_arr, fv_arr = self._force_arrays(fu, fv)

        self._dll.stokes_mac_run_steps_with_force_c(
            self._handle,
            ct.c_double(t_start),
            ct.c_int(n_steps),
            self._maybe_double_ptr(fu_arr),
            self._maybe_double_ptr(fv_arr),
            self._double_ptr(div_out),
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
        fu_arr, fv_arr = self._force_arrays(fu, fv)

        self._dll.stokes_mac_run_steps_with_force_diagnostics_c(
            self._handle,
            ct.c_double(t_start),
            ct.c_int(n_steps),
            self._maybe_double_ptr(fu_arr),
            self._maybe_double_ptr(fv_arr),
            self._double_ptr(div_out),
            self._double_ptr(change_out),
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
        u_arr = self._double_array(u_vec)
        v_arr = self._double_array(v_vec)
        if u_arr.shape != (self._nu,):
            raise ValueError(f"u_vec must have shape {(self._nu,)}, got {u_arr.shape}")
        if v_arr.shape != (self._nv,):
            raise ValueError(f"v_vec must have shape {(self._nv,)}, got {v_arr.shape}")

        if p_vec is not None:
            p_arr = self._double_array(p_vec)
            if p_arr.shape != (self._np,):
                raise ValueError(f"p_vec must have shape {(self._np,)}, got {p_arr.shape}")
            p_ptr = self._double_ptr(p_arr)
        else:
            p_ptr = self._maybe_double_ptr(None)

        self._dll.stokes_mac_set_state_c(
            self._handle,
            self._double_ptr(u_arr),
            self._double_ptr(v_arr),
            p_ptr,
        )

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (u_interior, v_interior, p_cells) in solver unknown ordering."""
        u_arr = np.empty(self._nu, dtype=np.float64)
        v_arr = np.empty(self._nv, dtype=np.float64)
        p_arr = np.empty(self._np, dtype=np.float64)
        self._dll.stokes_mac_get_state_c(
            self._handle,
            self._double_ptr(u_arr),
            self._double_ptr(v_arr),
            self._double_ptr(p_arr),
        )
        return u_arr, v_arr, p_arr

    def solve_linearized_eig(
        self,
        n_eigs: int,
        which: str,
        fu: np.ndarray | None,
        fv: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float, int, int]:
        """Run C++ dense linearization and Eigen eigensolver."""
        eig_real = np.empty(n_eigs, dtype=np.float64)
        eig_imag = np.empty(n_eigs, dtype=np.float64)
        full_size = self._nu + self._nv + self._np
        vec_real = np.empty((n_eigs, full_size), dtype=np.float64)
        vec_imag = np.empty((n_eigs, full_size), dtype=np.float64)
        base_residual_inf = ct.c_double()
        matvec_count = ct.c_longlong()
        dense_operator_bytes = ct.c_longlong()
        fu_arr, fv_arr = self._force_arrays(fu, fv)

        status = self._dll.stokes_mac_linearized_eig_c(
            self._handle,
            ct.c_int(n_eigs),
            which.encode("ascii"),
            self._maybe_double_ptr(fu_arr),
            self._maybe_double_ptr(fv_arr),
            self._double_ptr(eig_real),
            self._double_ptr(eig_imag),
            self._double_ptr(vec_real),
            self._double_ptr(vec_imag),
            ct.byref(base_residual_inf),
            ct.byref(matvec_count),
            ct.byref(dense_operator_bytes),
        )
        if status != 0:
            raise RuntimeError(f"C++ linearized eigensolve failed with status {status}")

        eigenvalues = eig_real + 1j * eig_imag
        eigenvectors = (vec_real + 1j * vec_imag).T
        return (
            eigenvalues,
            eigenvectors,
            float(base_residual_inf.value),
            int(matvec_count.value),
            int(dense_operator_bytes.value),
        )

    def solve_steady_newton(
        self,
        fu: np.ndarray | None,
        fv: np.ndarray | None,
        max_newton_iters: int,
        residual_tol: float,
        krylov_tol: float,
        krylov_maxiter: int,
        krylov_restart: int,
        jacobian_rdiff: float,
        line_search: str,
        min_step: float,
    ) -> tuple[int, float, float, bool, int, np.ndarray]:
        """Run the fixed-point Newton-GMRES steady solve entirely in C++."""
        changes = np.empty(max_newton_iters, dtype=np.float64)
        newton_iters = ct.c_int()
        residual_inf = ct.c_double()
        max_div = ct.c_double()
        converged = ct.c_int()
        stop_code = ct.c_int()
        change_count = ct.c_int()
        fu_arr, fv_arr = self._force_arrays(fu, fv)

        status = self._dll.stokes_mac_solve_steady_c(
            self._handle,
            ct.c_int(max_newton_iters),
            ct.c_double(residual_tol),
            ct.c_double(krylov_tol),
            ct.c_int(krylov_maxiter),
            ct.c_int(krylov_restart),
            ct.c_double(jacobian_rdiff),
            line_search.encode("ascii"),
            ct.c_double(min_step),
            self._maybe_double_ptr(fu_arr),
            self._maybe_double_ptr(fv_arr),
            ct.byref(newton_iters),
            ct.byref(residual_inf),
            ct.byref(max_div),
            ct.byref(converged),
            ct.byref(stop_code),
            self._double_ptr(changes),
            ct.byref(change_count),
        )
        if status != 0:
            raise RuntimeError(f"C++ steady Newton-GMRES failed with status {status}")

        used_changes = changes[: max(0, int(change_count.value))].copy()
        return (
            int(newton_iters.value),
            float(residual_inf.value),
            float(max_div.value),
            bool(converged.value),
            int(stop_code.value),
            used_changes,
        )

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
        double_ptr = DoublePtr
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
            double_ptr,
        ]
        dll.stokes_mac_run_steps_c.restype = None

        dll.stokes_mac_run_steps_diagnostics_c.argtypes = [
            ct.c_void_p,
            ct.c_double,
            ct.c_int,
            double_ptr,
            double_ptr,
        ]
        dll.stokes_mac_run_steps_diagnostics_c.restype = None

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
            fn.restype = double_ptr

        dll.stokes_mac_linearized_eig_c.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_char_p,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
            ct.POINTER(ct.c_longlong),
            ct.POINTER(ct.c_longlong),
        ]
        dll.stokes_mac_linearized_eig_c.restype = ct.c_int

        dll.stokes_mac_solve_steady_c.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_char_p,
            ct.c_double,
            double_ptr,
            double_ptr,
            ct.POINTER(ct.c_int),
            double_ptr,
            double_ptr,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int),
            double_ptr,
            ct.POINTER(ct.c_int),
        ]
        dll.stokes_mac_solve_steady_c.restype = ct.c_int

    def _copy_field(self, ptr: DoublePtr, shape_xy: tuple[int, int]) -> np.ndarray:
        """Copy a C row-major array into a (Nx, Ny) NumPy array (x-first)."""
        nx, ny = shape_xy
        return np.ctypeslib.as_array(ptr, shape=(nx * ny,)).copy().reshape(ny, nx).T

    @staticmethod
    def _double_array(arr: np.ndarray) -> np.ndarray:
        """Return arr as a contiguous float64 NumPy array."""
        return np.ascontiguousarray(arr, dtype=np.float64)

    @staticmethod
    def _double_ptr(arr: np.ndarray) -> DoublePtr:
        """Return a ctypes double pointer for a contiguous float64 NumPy array."""
        return arr.ctypes.data_as(DoublePtr)

    @staticmethod
    def _maybe_double_ptr(arr: np.ndarray | None) -> DoublePtr:
        """Return NULL for None, otherwise a ctypes double pointer."""
        if arr is None:
            return ct.cast(None, DoublePtr)
        return arr.ctypes.data_as(DoublePtr)

    @staticmethod
    def _force_arrays(
        fu: np.ndarray | None,
        fv: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Keep optional force arrays contiguous and alive across the C call."""
        fu_arr = None if fu is None else StokesMACLib._double_array(fu)
        fv_arr = None if fv is None else StokesMACLib._double_array(fv)
        return fu_arr, fv_arr
