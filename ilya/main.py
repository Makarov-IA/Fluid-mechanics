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
            ct.c_int,
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

        self.handle = self.dll.stokes_mac_create_c(nx, ny, lx, ly, nu, dt, 7000, 1e-8)
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
    nx, ny = 64, 64
    lx, ly = 1.0, 1.0
    nu = 0.01
    dt = 1e-3
    n_steps = 200
    frame_every = 5
    gif_fps = 12

    solver = StokesMACLib(Path(__file__).with_name("solver.dll"), nx, ny, lx, ly, nu, dt)

    def f1(x, y, t):
        return float(np.sin(np.pi * x) * np.sin(np.pi * y))

    def f2(x, y, t):
        return 0.0

    out_dir = Path(__file__).with_name("plots")
    out_dir.mkdir(exist_ok=True)

    xc = (np.arange(nx) + 0.5) * (lx / nx)
    yc = (np.arange(ny) + 0.5) * (ly / ny)
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

    frames = []
    snapshots = []

    def build_frame(p, uc, vc, step_idx, t_val, speed_levels, p_levels, quiver_scale):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

        skip = 6
        speed = np.sqrt(uc * uc + vc * vc)
        bg = ax1.contourf(Xc, Yc, speed, levels=speed_levels, cmap="viridis")
        ax1.quiver(
            Xc[::skip, ::skip],
            Yc[::skip, ::skip],
            uc[::skip, ::skip],
            vc[::skip, ::skip],
            color="white",
            pivot="mid",
            width=0.005,
            headwidth=4.5,
            headlength=6.0,
            headaxislength=5.5,
            scale=quiver_scale,
        )
        fig.colorbar(bg, ax=ax1, label="|u|")
        ax1.set_title(f"Velocity field, step={step_idx}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        ax1.set_xlim(0.0, lx)
        ax1.set_ylim(0.0, ly)

        cp = ax2.contourf(Xc, Yc, p, levels=p_levels, cmap="coolwarm")
        ax2.contour(Xc, Yc, p, levels=p_levels[::2], colors="black", linewidths=0.35, alpha=0.6)
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
    p_abs_max = max(np.max(np.abs(sp)) for _, _, sp, _, _ in snapshots)
    speed_max = max(speed_max, 1e-12)
    p_abs_max = max(p_abs_max, 1e-12)
    speed_levels = np.linspace(0.0, speed_max, 25)
    p_levels = np.linspace(-p_abs_max, p_abs_max, 25)
    quiver_scale = 20.0 / speed_max

    for step_idx, t_val, p_snap, uc_snap, vc_snap in snapshots:
        build_frame(p_snap, uc_snap, vc_snap, step_idx, t_val, speed_levels, p_levels, quiver_scale)

    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
    skip = 4
    im2 = ax2.contourf(Xc, Yc, speed, levels=24, cmap="viridis")
    ax2.quiver(Xc[::skip, ::skip], Yc[::skip, ::skip], uc[::skip, ::skip], vc[::skip, ::skip], color="white")
    fig2.colorbar(im2, ax=ax2, label="|u|")
    ax2.set_title("Vector field (u, v)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    fig2.tight_layout()
    fig2.savefig(out_dir / "stokes_vector_field.png", dpi=180)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7.5, 5.5))
    ip = ax3.contourf(Xc, Yc, p, levels=24, cmap="coolwarm")
    ax3.contour(Xc, Yc, p, levels=16, colors="black", linewidths=0.35, alpha=0.6)
    fig3.colorbar(ip, ax=ax3, label="p")
    ax3.set_title("Pressure contour plot")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect("equal")
    fig3.tight_layout()
    fig3.savefig(out_dir / "stokes_pressure_contour.png", dpi=180)
    plt.close(fig3)

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

    print(f"Saved: {out_dir / 'stokes_vector_field.png'}")
    print(f"Saved: {out_dir / 'stokes_pressure_contour.png'}")
    print(f"Saved: {gif_path}")
