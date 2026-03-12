import ctypes as ct
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# --- Кроссплатформенное определение имени библиотеки ---
def get_solver_lib_path() -> Path:
    """
    Определяет правильное имя файла библиотеки в зависимости от ОС.
    Ищет в той же директории, что и скрипт.
    """
    base_name = "solver"
    script_dir = Path(__file__).parent

    system = platform.system()

    if system == "Windows":
        ext = ".dll"
    elif system == "Darwin":  # macOS
        ext = ".dylib"
    elif system == "Linux":
        ext = ".so"
    else:
        raise RuntimeError(f"Неподдерживаемая ОС: {system}")

    lib_path = script_dir / f"{base_name}{ext}"

    if not lib_path.exists():
        # Если точное имя не найдено, пробуем найти любой файл с префиксом 'solver'
        # Это полезно, если на macOS файл называется libsolver.dylib
        candidates = list(script_dir.glob(f"{base_name}*{ext}"))
        if not candidates:
            # Последняя попытка: любой файл solver.* в папке
            candidates = [
                f
                for f in script_dir.iterdir()
                if f.name.startswith("solver") and f.suffix in [".dll", ".dylib", ".so"]
            ]

        if not candidates:
            raise FileNotFoundError(
                f"Библиотека решателя не найдена в {script_dir}. "
                f"Ожидалось имя: {lib_path.name}"
            )
        lib_path = candidates[0]
        print(f"Warning: Using alternative library name: {lib_path.name}")

    return lib_path


# ---------------------------------------------------------

Force3D = ct.CFUNCTYPE(ct.c_double, ct.c_double, ct.c_double, ct.c_double)


class StokesMACLib:
    """
    C++ Stokes MAC solver wrapper.

    Array layouts from C++:
    - p[i, j], shape (Nx, Ny), i along x, j along y
    - u[i, j], shape (Nx+1, Ny), vertical faces
    - v[i, j], shape (Nx, Ny+1), horizontal faces
    """

    def __init__(
        self,
        dll_path: Path,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        nu: float,
        dt: float,
    ):
        self.nx = nx
        self.ny = ny
        self.handle = None

        # Кроссплатформенная загрузка библиотеки
        # На macOS иногда требуется указать режим загрузки RTLD_GLOBAL
        try:
            self.dll = ct.CDLL(str(dll_path), mode=ct.RTLD_GLOBAL)
        except AttributeError:
            # Fallback для старых версий Python/Windows где RTLD_GLOBAL может отсутствовать
            self.dll = ct.CDLL(str(dll_path))

        self.dll.stokes_mac_create_c.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.c_double,
        ]
        self.dll.stokes_mac_create_c.restype = ct.c_void_p

        self.dll.stokes_mac_free_c.argtypes = [ct.c_void_p]
        self.dll.stokes_mac_free_c.restype = None

        self.dll.stokes_mac_step_c.argtypes = [
            ct.c_void_p,
            ct.c_double,
            Force3D,
            Force3D,
        ]
        self.dll.stokes_mac_step_c.restype = ct.c_double

        self.dll.stokes_mac_get_p_c.argtypes = [ct.c_void_p]
        self.dll.stokes_mac_get_p_c.restype = ct.POINTER(ct.c_double)

        self.dll.stokes_mac_get_u_c.argtypes = [ct.c_void_p]
        self.dll.stokes_mac_get_u_c.restype = ct.POINTER(ct.c_double)

        self.dll.stokes_mac_get_v_c.argtypes = [ct.c_void_p]
        self.dll.stokes_mac_get_v_c.restype = ct.POINTER(ct.c_double)

        self.handle = self.dll.stokes_mac_create_c(nx, ny, lx, ly, nu, dt)
        if not self.handle:
            raise RuntimeError("stokes_mac_create_c failed")

    def close(self) -> None:
        if self.handle:
            self.dll.stokes_mac_free_c(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def step(self, t: float, f1, f2) -> float:
        f1_cb = Force3D(f1)
        f2_cb = Force3D(f2)
        return self.dll.stokes_mac_step_c(self.handle, t, f1_cb, f2_cb)

    def get_fields(self):
        p_ptr = self.dll.stokes_mac_get_p_c(self.handle)
        u_ptr = self.dll.stokes_mac_get_u_c(self.handle)
        v_ptr = self.dll.stokes_mac_get_v_c(self.handle)

        p = (
            np.ctypeslib.as_array(p_ptr, shape=(self.nx * self.ny,))
            .copy()
            .reshape((self.ny, self.nx))
            .T
        )
        u = (
            np.ctypeslib.as_array(u_ptr, shape=((self.nx + 1) * self.ny,))
            .copy()
            .reshape((self.ny, self.nx + 1))
            .T
        )
        v = (
            np.ctypeslib.as_array(v_ptr, shape=(self.nx * (self.ny + 1),))
            .copy()
            .reshape((self.ny + 1, self.nx))
            .T
        )
        return p, u, v


if __name__ == "__main__":

    # Re ~= U_lid * L / nu = 20 for U_lid = 1, L = 1.
    nx, ny, n_steps = 201, 201, 20000
    lx, ly, lt = 1.0, 1.0, 2.0
    nu = 1 / 1000
    dt = lt / n_steps  # 1e-4
    frame_every = 200
    gif_fps = 12

    # --- ИСПОЛЬЗОВАНИЕ КРОССПЛАТФОРМЕННОГО ПУТИ ---
    solver_path = get_solver_lib_path()
    print(f"Loading solver library: {solver_path.name}")

    solver = StokesMACLib(solver_path, nx, ny, lx, ly, nu, dt)
    # -----------------------------------------------

    def f1(x, y, t):
        return 0.0

    def f2(x, y, t):
        return 0.0

    out_dir = Path(__file__).resolve().with_name("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    xc = (np.arange(nx) + 0.5) * (lx / nx)
    yc = (np.arange(ny) + 0.5) * (ly / ny)
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

    frame_sets = {
        "quiver": [],
        "streamlines": [],
        "pressure": [],
        "vorticity": [],
    }
    snapshots = []
    div_history = []
    t_history = []

    def save_frame(fig, kind: str, step_idx: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_path = out_dir / f"{kind}_{step_idx:05d}.png"
        fig.savefig(frame_path, dpi=130)
        plt.close(fig)
        frame_sets[kind].append(frame_path)

    def style_axes(ax, title: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, lx)
        ax.set_ylim(0.0, ly)

    def build_quiver_frame(
        uc,
        vc,
        step_idx,
        t_val,
        speed_levels,
        quiver_scale,
        arrow_factor,
    ):
        fig, ax = plt.subplots(figsize=(6.2, 6.0))

        skip = 6
        speed = np.sqrt(uc * uc + vc * vc)
        bg = ax.contourf(Xc, Yc, speed, levels=speed_levels, cmap="viridis")
        ax.quiver(
            Xc[::skip, ::skip],
            Yc[::skip, ::skip],
            arrow_factor * uc[::skip, ::skip],
            arrow_factor * vc[::skip, ::skip],
            color="#111111",
            pivot="mid",
            width=0.0035,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.5,
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
        )
        fig.colorbar(bg, ax=ax, label="|u|")
        style_axes(ax, f"Velocity field, t={t_val:.3f}")
        fig.tight_layout()
        save_frame(fig, "quiver", step_idx)

    def build_streamlines_frame(
        uc,
        vc,
        step_idx,
        t_val,
        speed_levels,
    ):
        fig, ax = plt.subplots(figsize=(6.2, 6.0))

        speed = np.sqrt(uc * uc + vc * vc)
        bg = ax.contourf(Xc, Yc, speed, levels=speed_levels, cmap="viridis")
        ax.streamplot(
            xc,
            yc,
            uc.T,
            vc.T,
            color="white",
            linewidth=0.8,
            density=1.5,
            arrowsize=0.9,
        )
        fig.colorbar(bg, ax=ax, label="|u|")
        style_axes(ax, f"Velocity field and streamlines, t={t_val:.3f}")
        fig.tight_layout()
        save_frame(fig, "streamlines", step_idx)

    def build_pressure_frame(p, step_idx, t_val, p_levels):
        fig, ax = plt.subplots(figsize=(6.2, 6.0))

        cp = ax.contourf(Xc, Yc, p, levels=p_levels, cmap="coolwarm")
        ax.contour(
            Xc, Yc, p, levels=p_levels, colors="black", linewidths=0.25, alpha=0.7
        )
        fig.colorbar(cp, ax=ax, label="p")
        style_axes(ax, f"Pressure contour, t={t_val:.3f}")
        fig.tight_layout()
        save_frame(fig, "pressure", step_idx)

    def build_vorticity_frame(omega, step_idx, t_val, omega_levels):
        fig, ax = plt.subplots(figsize=(6.2, 6.0))

        cw = ax.contourf(Xc, Yc, omega, levels=omega_levels, cmap="coolwarm")
        ax.contour(
            Xc,
            Yc,
            omega,
            levels=omega_levels,
            colors="black",
            linewidths=0.25,
            alpha=0.7,
        )
        fig.colorbar(cw, ax=ax, label="omega")
        style_axes(ax, f"Vorticity contour, t={t_val:.3f}")
        fig.tight_layout()
        save_frame(fig, "vorticity", step_idx)

    def save_gif(frame_paths, gif_path):
        out_dir.mkdir(parents=True, exist_ok=True)
        gif_images = [Image.open(frame).copy() for frame in frame_paths]
        if gif_images:
            gif_images[0].save(
                gif_path,
                save_all=True,
                append_images=gif_images[1:],
                duration=int(1000 / gif_fps),
                loop=0,
            )
        for frame in frame_paths:
            frame.unlink(missing_ok=True)

    for n in range(1, n_steps + 1):
        t_now = n * dt
        div_inf = solver.step(t_now, f1, f2)
        t_history.append(t_now)
        div_history.append(div_inf)
        if n % frame_every == 0 or n == 1 or n == n_steps:
            p_step, u_step, v_step = solver.get_fields()
            uc_step = 0.5 * (u_step[:-1, :] + u_step[1:, :])
            vc_step = 0.5 * (v_step[:, :-1] + v_step[:, 1:])
            omega_step = np.gradient(vc_step, xc, axis=0) - np.gradient(
                uc_step, yc, axis=1
            )
            snapshots.append(
                (
                    n,
                    t_now,
                    p_step.copy(),
                    uc_step.copy(),
                    vc_step.copy(),
                    omega_step.copy(),
                )
            )
        if n % 50 == 0:
            print(f"step={n:4d}, max|div|={div_inf:.4e}")

    p, u, v = solver.get_fields()
    solver.close()

    # Cell-centered velocity for plotting.
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    speed_max = max(
        np.max(np.sqrt(su * su + sv * sv)) for _, _, _, su, sv, _ in snapshots
    )
    speed_max = max(speed_max, 1e-12)
    speed_levels = np.linspace(0.0, speed_max, 25)
    p_all = np.concatenate([sp.ravel() for _, _, sp, _, _, _ in snapshots])
    p_lo, p_hi = np.percentile(p_all, [2.0, 98.0])
    if p_hi <= p_lo:
        p_lo, p_hi = np.min(p_all), np.max(p_all)
    p_levels = np.linspace(p_lo, p_hi, 61)
    omega_all = np.concatenate([so.ravel() for _, _, _, _, _, so in snapshots])
    omega_abs = np.percentile(np.abs(omega_all), 98.0)
    omega_abs = max(omega_abs, 1e-12)
    omega_levels = np.linspace(-omega_abs, omega_abs, 61)
    quiver_scale = 1.0
    target_arrow_len = 0.1 * min(lx, ly)
    arrow_factor = target_arrow_len / speed_max

    for step_idx, t_val, p_snap, uc_snap, vc_snap, omega_snap in snapshots:
        build_quiver_frame(
            uc_snap,
            vc_snap,
            step_idx,
            t_val,
            speed_levels,
            quiver_scale,
            arrow_factor,
        )
        build_streamlines_frame(
            uc_snap,
            vc_snap,
            step_idx,
            t_val,
            speed_levels,
        )
        build_pressure_frame(
            p_snap,
            step_idx,
            t_val,
            p_levels,
        )
        build_vorticity_frame(
            omega_snap,
            step_idx,
            t_val,
            omega_levels,
        )

    fig_div, ax_div = plt.subplots(figsize=(7.5, 4.8))
    ax_div.plot(t_history, div_history, color="#0d47a1", linewidth=1.6)
    ax_div.set_title("Max divergence vs time")
    ax_div.set_xlabel("t")
    ax_div.set_ylabel("max |div u|")
    ax_div.grid(True, alpha=0.35)
    fig_div.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    div_path = out_dir / "stokes_max_divergence.png"
    fig_div.savefig(div_path, dpi=180)
    plt.close(fig_div)

    quiver_gif_path = out_dir / "stokes_velocity_quiver.gif"
    streamlines_gif_path = out_dir / "stokes_streamlines.gif"
    pressure_gif_path = out_dir / "stokes_pressure.gif"
    vorticity_gif_path = out_dir / "stokes_vorticity.gif"

    save_gif(frame_sets["quiver"], quiver_gif_path)
    save_gif(frame_sets["streamlines"], streamlines_gif_path)
    save_gif(frame_sets["pressure"], pressure_gif_path)
    save_gif(frame_sets["vorticity"], vorticity_gif_path)

    print(f"Saved: {quiver_gif_path}")
    print(f"Saved: {streamlines_gif_path}")
    print(f"Saved: {pressure_gif_path}")
    print(f"Saved: {vorticity_gif_path}")
    print(f"Saved: {div_path}")
