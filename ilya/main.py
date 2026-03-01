import ctypes as ct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


Force3D = ct.CFUNCTYPE(ct.c_double, ct.c_double, ct.c_double, ct.c_double)


class StokesMACLib:
    """
    C++ Stokes MAC solver wrapper.

    Array layouts from C++:
    - p[i, j], shape (Nx, Ny), i along x, j along y
    - u[i, j], shape (Nx+1, Ny), vertical faces
    - v[i, j], shape (Nx, Ny+1), horizontal faces
    """

    def __init__(self, dll_path: Path, nx: int, ny: int, lx: float, ly: float, nu: float, dt: float):
        self.nx = nx
        self.ny = ny
        self.handle = None
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

        self.dll.stokes_mac_step_c.argtypes = [ct.c_void_p, ct.c_double, Force3D, Force3D]
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

        p = np.ctypeslib.as_array(p_ptr, shape=(self.nx * self.ny,)).copy().reshape((self.ny, self.nx)).T
        u = np.ctypeslib.as_array(u_ptr, shape=((self.nx + 1) * self.ny,)).copy().reshape((self.ny, self.nx + 1)).T
        v = np.ctypeslib.as_array(v_ptr, shape=(self.nx * (self.ny + 1),)).copy().reshape((self.ny + 1, self.nx)).T
        return p, u, v


if __name__ == "__main__":
    nx, ny, n_steps = 64, 64, 2000
    lx, ly, lt = 1.0, 1.0, 0.04
    nu = 5.002
    dt = lt/n_steps
    frame_every = 5
    gif_fps = 12

    solver = StokesMACLib(Path(__file__).with_name("solver.dll"), nx, ny, lx, ly, nu, dt)

    def f1(x, y, t):
        return 0.0

    def f2(x, y, t):
        return 0.0

    out_dir = Path(__file__).with_name("plots")
    out_dir.mkdir(exist_ok=True)

    xc = (np.arange(nx) + 0.5) * (lx / nx)
    yc = (np.arange(ny) + 0.5) * (ly / ny)
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

    frames = []
    snapshots = []
    div_history = []
    t_history = []

    def build_frame(p, uc, vc, step_idx, t_val, speed_levels, p_levels, quiver_scale, arrow_factor):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

        skip = 6
        speed = np.sqrt(uc * uc + vc * vc)
        bg = ax1.contourf(Xc, Yc, speed, levels=speed_levels, cmap="viridis")
        ax1.quiver(
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
        fig.colorbar(bg, ax=ax1, label="|u|")
        ax1.set_title(f"Velocity field, t={t_val:.3f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        ax1.set_xlim(0.0, lx)
        ax1.set_ylim(0.0, ly)

        cp = ax2.contourf(Xc, Yc, p, levels=p_levels, cmap="coolwarm")
        ax2.contour(Xc, Yc, p, levels=p_levels, colors="black", linewidths=0.25, alpha=0.7)
        fig.colorbar(cp, ax=ax2, label="p")
        ax2.set_title(f"Pressure contour, t={t_val:.3f}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_aspect("equal")
        ax2.set_xlim(0.0, lx)
        ax2.set_ylim(0.0, ly)

        fig.tight_layout()
        frame_path = out_dir / f"frame_{step_idx:05d}.png"
        fig.savefig(frame_path, dpi=130)
        plt.close(fig)
        frames.append(frame_path)

    for n in range(1, n_steps + 1):
        t_now = n * dt
        div_inf = solver.step(t_now, f1, f2)
        t_history.append(t_now)
        div_history.append(div_inf)
        if n % frame_every == 0 or n == 1 or n == n_steps:
            p_step, u_step, v_step = solver.get_fields()
            uc_step = 0.5 * (u_step[:-1, :] + u_step[1:, :])
            vc_step = 0.5 * (v_step[:, :-1] + v_step[:, 1:])
            snapshots.append((n, t_now, p_step.copy(), uc_step.copy(), vc_step.copy()))
        if n % 50 == 0:
            print(f"step={n:4d}, max|div|={div_inf:.4e}")

    p, u, v = solver.get_fields()
    solver.close()

    # Cell-centered velocity for plotting.
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    speed = np.sqrt(uc * uc + vc * vc)

    speed_max = max(np.max(np.sqrt(su * su + sv * sv)) for _, _, _, su, sv in snapshots)
    speed_max = max(speed_max, 1e-12)
    speed_levels = np.linspace(0.0, speed_max, 25)
    p_all = np.concatenate([sp.ravel() for _, _, sp, _, _ in snapshots])
    p_lo, p_hi = np.percentile(p_all, [2.0, 98.0])
    if p_hi <= p_lo:
        p_lo, p_hi = np.min(p_all), np.max(p_all)
    p_levels = np.linspace(p_lo, p_hi, 61)
    quiver_scale = 1.0
    target_arrow_len = 0.1 * min(lx, ly)
    arrow_factor = target_arrow_len / speed_max

    for step_idx, t_val, p_snap, uc_snap, vc_snap in snapshots:
        build_frame(p_snap, uc_snap, vc_snap, step_idx, t_val, speed_levels, p_levels, quiver_scale, arrow_factor)

    fig_div, ax_div = plt.subplots(figsize=(7.5, 4.8))
    ax_div.plot(t_history, div_history, color="#0d47a1", linewidth=1.6)
    ax_div.set_title("Max divergence vs time")
    ax_div.set_xlabel("t")
    ax_div.set_ylabel("max |div u|")
    ax_div.grid(True, alpha=0.35)
    fig_div.tight_layout()
    div_path = out_dir / "stokes_max_divergence.png"
    fig_div.savefig(div_path, dpi=180)
    plt.close(fig_div)

    gif_path = out_dir / "stokes_evolution.gif"
    gif_images = [Image.open(frame).copy() for frame in frames]
    if gif_images:
        gif_images[0].save(
            gif_path,
            save_all=True,
            append_images=gif_images[1:],
            duration=int(1000 / gif_fps),
            loop=0,
        )
    for frame in frames:
        frame.unlink(missing_ok=True)

    print(f"Saved: {gif_path}")
    print(f"Saved: {div_path}")
