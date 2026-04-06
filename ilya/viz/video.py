"""Video rendering in parallel worker processes."""

from __future__ import annotations

import os
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from solver.config import SimConfig, Snapshot
from viz.plots import (
    draw_streamlines,
    draw_pressure,
    draw_vorticity,
    style_axes,
    fig_to_pil,
)

matplotlib.use("Agg")

console = Console()

_VIDEO_SPECS = [
    ("streamlines", "stokes_streamlines.mp4"),
    ("pressure",    "stokes_pressure.mp4"),
    ("vorticity",   "stokes_vorticity.mp4"),
]


def _video_worker(task: dict) -> tuple[str, str]:
    """Render every frame for one video type and write an MP4 to disk."""
    kind: str                 = task["kind"]
    queue                     = task["queue"]
    lx, ly                    = task["lx"], task["ly"]
    xc: np.ndarray            = task["xc"]
    yc: np.ndarray            = task["yc"]
    Xc: np.ndarray            = task["Xc"]
    Yc: np.ndarray            = task["Yc"]
    snapshots: list[Snapshot] = task["snapshots"]
    fps: float                = task["fps"]

    video_path = Path(task["video_path"])
    video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(video_path),
        fps=fps,
        codec="libx264",
        output_params=["-crf", "20"],
        macro_block_size=1,
    )

    with writer:
        for snap in snapshots:
            fig, ax = plt.subplots(figsize=(6.2, 6.0))

            if kind == "streamlines":
                draw_streamlines(ax, fig, snap, xc, yc, Xc, Yc, task["speed_levels"])
                title = f"Streamlines, t={snap.t:.3f}"
            elif kind == "pressure":
                draw_pressure(ax, fig, snap, Xc, Yc, task["p_levels"])
                title = f"Pressure, t={snap.t:.3f}"
            elif kind == "vorticity":
                draw_vorticity(ax, fig, snap, Xc, Yc, task["omega_levels"])
                title = f"Vorticity, t={snap.t:.3f}"
            else:
                raise ValueError(f"Unknown video kind: {kind!r}")

            style_axes(ax, title, lx, ly)
            fig.tight_layout()
            frame = np.array(fig_to_pil(fig, dpi=110).convert("RGB"))
            writer.append_data(frame)
            queue.put(kind)

    return kind, str(video_path)


def render_videos(
    snapshots: list[Snapshot],
    cfg: SimConfig,
    out_dir: Path,
    xc: np.ndarray,
    yc: np.ndarray,
    sl: np.ndarray,
    pl: np.ndarray,
    ol: np.ndarray,
    speed_max: float,
) -> dict[str, Path]:
    """Render all videos in parallel worker processes with per-video progress bars."""
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    fps = cfg.video_fps
    n_frames = len(snapshots)

    manager = mp.Manager()
    queue = manager.Queue()

    common = {
        "queue": queue, "lx": cfg.lx, "ly": cfg.ly,
        "xc": xc, "yc": yc, "Xc": Xc, "Yc": Yc,
        "fps": fps, "snapshots": snapshots,
    }
    tasks = [
        {**common, "kind": "streamlines",
         "video_path": str(out_dir / "stokes_streamlines.mp4"), "speed_levels": sl},
        {**common, "kind": "pressure",
         "video_path": str(out_dir / "stokes_pressure.mp4"),    "p_levels": pl},
        {**common, "kind": "vorticity",
         "video_path": str(out_dir / "stokes_vorticity.mp4"),   "omega_levels": ol},
    ]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description:<14}"),
        BarColumn(bar_width=38),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    results: dict[str, Path] = {}

    with progress:
        bars = {kind: progress.add_task(kind, total=n_frames) for kind, _ in _VIDEO_SPECS}

        def _listener() -> None:
            received = 0
            total = n_frames * len(tasks)
            while received < total:
                kind = queue.get()
                progress.update(bars[kind], advance=1)
                received += 1

        listener = threading.Thread(target=_listener, daemon=True)
        listener.start()

        n_workers = min(len(tasks), os.cpu_count() or 1)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
        ) as pool:
            for kind, path in pool.map(_video_worker, tasks):
                results[kind] = Path(path)

        listener.join()

    manager.shutdown()
    return results
