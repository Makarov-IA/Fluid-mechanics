import ctypes as ct
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


Func2D = ct.CFUNCTYPE(ct.c_double, ct.c_double, ct.c_double)


class SolverLib:
    def __init__(self, lib_path: str | Path):
        self.lib_path = str(lib_path)
        self.lib = ct.CDLL(self.lib_path)

        self.lib.solve_dirichlet_c.argtypes = [
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            Func2D,
            Func2D,
        ]
        self.lib.solve_dirichlet_c.restype = ct.POINTER(ct.c_double)

        self.lib.solver_free.argtypes = [ct.POINTER(ct.c_double)]
        self.lib.solver_free.restype = None

    def solve_dirichlet(self, spx, spy, nx, ny, hx, hy, f, g):
        f_cb = Func2D(f)
        g_cb = Func2D(g)

        ptr = self.lib.solve_dirichlet_c(spx, spy, nx, ny, hx, hy, f_cb, g_cb)
        if not ptr:
            raise RuntimeError("solve_dirichlet_c returned null pointer")

        try:
            size = nx * ny
            if np is not None:
                arr = np.ctypeslib.as_array(ptr, shape=(size,))
                return arr.copy().reshape((ny, nx))

            flat = [ptr[i] for i in range(size)]
            return [flat[row * nx : (row + 1) * nx] for row in range(ny)]
        finally:
            self.lib.solver_free(ptr)


if __name__ == "__main__":
    dll_path = Path(__file__).with_name("solver.dll")
    solver = SolverLib(dll_path)

    nx, ny = 80, 80
    lx, ly = 1.0, 1.0
    hx, hy = lx / nx, ly / ny

    if np is None:
        print("numpy is required for exact-solution comparison")
        raise SystemExit(1)
    if plt is None:
        print("matplotlib is required for plotting")
        raise SystemExit(1)

    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(x, y)

    def u_exact_1(xv, yv):
        return np.sin(np.pi * xv) * np.sin(np.pi * yv)

    def f_1(xv, yv):
        return 2.0 * (np.pi**2) * np.sin(np.pi * xv) * np.sin(np.pi * yv)

    def u_exact_2(xv, yv):
        return xv * xv + yv * yv

    def f_2(xv, yv):
        return -4.0

    def u_exact_3(xv, yv):
        return np.exp(xv + yv)

    def f_3(xv, yv):
        return -2.0 * np.exp(xv + yv)

    test_cases = [
        ("sin(pi x) sin(pi y)", f_1, u_exact_1),
        ("x^2 + y^2", f_2, u_exact_2),
        ("exp(x + y)", f_3, u_exact_3),
    ]

    for idx, (name, f_func, u_exact_func) in enumerate(test_cases, start=1):
        def f_cb(xv, yv):
            return float(f_func(xv, yv))

        def g_cb(xv, yv):
            return float(u_exact_func(xv, yv))

        u_num = solver.solve_dirichlet(0.0, 0.0, nx, ny, hx, hy, f_cb, g_cb)
        u_true = u_exact_func(X, Y)

        max_abs_err = np.max(np.abs(u_num - u_true))
        print(f"[{idx}] {name}: max|error| = {max_abs_err:.6e}")

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        ax1.plot_surface(X, Y, u_num, cmap="viridis", edgecolor="none")
        ax1.set_title(f"Computed: {name}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("u")

        ax2.plot_surface(X, Y, u_true, cmap="viridis", edgecolor="none")
        ax2.set_title(f"Exact: {name}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("u")

        fig.suptitle(f"Poisson with Dirichlet BC, case {idx}")
        plt.tight_layout()
        plt.savefig(f"plots/comparison_case_{idx}.png", dpi=200, bbox_inches="tight")

    plt.show()
