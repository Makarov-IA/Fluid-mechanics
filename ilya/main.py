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
            ct.c_double, ct.c_double,
            ct.c_int, ct.c_int,
            ct.c_double, ct.c_double,
            Func2D, Func2D,
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
            return [flat[row * nx:(row + 1) * nx] for row in range(ny)]
        finally:
            self.lib.solver_free(ptr)


if __name__ == "__main__":
    # Пример:
    #   Windows: компилируем solver.dll из C++ файлов.

    dll_path = Path(__file__).with_name("solver.dll")
    solver = SolverLib(dll_path)

    def f(x, y):
        return 1.0

    def g(x, y):
        return 0.0

    nx, ny = 100, 100
    lx, ly = 1.0, 1.0
    hx, hy = lx / nx, ly / ny

    u = solver.solve_dirichlet(0.0, 0.0, nx, ny, hx, hy, f, g)
    if np is not None:
        print(u.shape, float(u.min()), float(u.max()))
    else:
        print(len(u), len(u[0]), min(map(min, u)), max(map(max, u)))

    if np is not None and plt is not None:
        x = np.linspace(0.0, lx, nx)
        y = np.linspace(0.0, ly, ny)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, u, cmap="viridis", edgecolor="none")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x, y)")
        ax.set_title("Poisson solution")
        plt.tight_layout()
        plt.show()
    elif plt is None:
        print("matplotlib is not installed, skip 3D plot")
