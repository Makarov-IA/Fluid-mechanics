"""Video rendering in parallel worker processes.

Every video is split into chunks of consecutive frames; all chunks of all
videos share one process pool (streamline chunks, the slowest, go first).
Each chunk is encoded to its own MP4 segment and the segments are joined
losslessly with ffmpeg's concat demuxer (stream copy, no re-encoding).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio
import imageio_ffmpeg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
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
from viz.plots import _field_colorbar, style_axes

matplotlib.use("Agg")

console = Console()

_VIDEO_SPECS = [
    ("streamlines", "stokes_streamlines.mp4"),
    ("pressure", "stokes_pressure.mp4"),
    ("vorticity", "stokes_vorticity.mp4"),
]

# Relative per-frame cost, used to size chunks (streamplot dominates).
_FRAME_COST = {"streamlines": 4.5, "pressure": 1.0, "vorticity": 1.0}


def _slim_snapshot(snap: Snapshot, kind: str) -> Snapshot:
    """Keep only the fields a video kind draws, to cut inter-process traffic."""
    return Snapshot(
        step=snap.step,
        t=snap.t,
        p=snap.p if kind == "pressure" else None,
        uc=snap.uc if kind == "streamlines" else None,
        vc=snap.vc if kind == "streamlines" else None,
        omega=snap.omega if kind == "vorticity" else None,
    )


class _FrameRenderer:
    """One reusable figure per worker process.

    The filled field is an ``imshow`` quantised to the same colour levels as the
    static ``contourf`` plots (so the colour bands match), the colour bar and the
    layout are built once, and only the per-frame artists (contour lines,
    streamlines, title) are redrawn.
    """

    _STYLE = {
        "streamlines": ("viridis", "|u|", "Streamlines"),
        "pressure": ("coolwarm", "p", "Pressure"),
        "vorticity": ("coolwarm", "ω", "Vorticity"),
    }

    def __init__(self, kind: str, levels: np.ndarray, lx: float, ly: float,
                 xc: np.ndarray, yc: np.ndarray) -> None:
        if kind not in self._STYLE:
            raise ValueError(f"Unknown video kind: {kind!r}")
        self.kind, self.levels, self.lx, self.ly = kind, levels, lx, ly
        self.xc, self.yc = xc, yc
        self.x_grid, self.y_grid = np.meshgrid(xc, yc, indexing="ij")
        cmap, label, self.title = self._STYLE[kind]

        self.fig, self.ax = plt.subplots(figsize=(6.2, 6.0))
        self.fig.set_dpi(110)
        # Values outside the levels stay unfilled, as with contourf.
        colormap = plt.get_cmap(cmap).copy()
        colormap.set_under((0.0, 0.0, 0.0, 0.0))
        colormap.set_over((0.0, 0.0, 0.0, 0.0))
        norm = BoundaryNorm(levels, colormap.N)
        self.image = self.ax.imshow(
            np.zeros((len(yc), len(xc))),
            origin="lower",
            extent=(0.0, lx, 0.0, ly),
            cmap=colormap,
            norm=norm,
            interpolation="bilinear",
            aspect="equal",
        )
        _field_colorbar(self.fig, self.ax, self.image, self.x_grid, self.y_grid, label)
        self.fig.axes[-1].minorticks_off()  # no tick per colour level
        style_axes(self.ax, f"{self.title}, t=0.000", lx, ly)
        self.fig.tight_layout()
        self._base = self._children()

    def _children(self) -> set[int]:
        ax = self.ax
        return {id(a) for a in (*ax.collections, *ax.patches, *ax.lines)}

    def _clear_frame_artists(self) -> None:
        ax = self.ax
        for artist in (*ax.collections, *ax.patches, *ax.lines):
            if id(artist) not in self._base:
                artist.remove()

    def render(self, snap: Snapshot) -> np.ndarray:
        self._clear_frame_artists()
        ax = self.ax
        if self.kind == "streamlines":
            self.image.set_data(np.hypot(snap.uc, snap.vc).T)
            ax.streamplot(
                self.xc, self.yc, snap.uc.T, snap.vc.T,
                color="white", linewidth=0.8, density=1.5, arrowsize=0.9,
            )
        else:
            field = snap.p if self.kind == "pressure" else snap.omega
            self.image.set_data(field.T)
            ax.contour(
                self.x_grid, self.y_grid, field, levels=self.levels,
                colors="black", linewidths=0.25, alpha=0.7,
            )
        ax.set_xlim(0.0, self.lx)
        ax.set_ylim(0.0, self.ly)
        ax.set_title(f"{self.title}, t={snap.t:.3f}", fontsize=10)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def _video_worker(task: dict) -> tuple[str, int, str]:
    """Render one chunk of frames for one video type into an MP4 segment."""
    kind: str = task["kind"]
    queue = task["queue"]
    renderer = _FrameRenderer(
        kind, task["levels"], task["lx"], task["ly"], task["xc"], task["yc"]
    )

    writer = imageio.get_writer(
        task["segment_path"],
        fps=task["fps"],
        codec="libx264",
        output_params=["-crf", "20"],
        macro_block_size=1,
    )
    with writer:
        for snap in task["snapshots"]:
            writer.append_data(renderer.render(snap))
            queue.put(kind)
    plt.close(renderer.fig)
    return kind, task["chunk"], task["segment_path"]


def _concat_segments(segments: list[str], video_path: Path, work_dir: Path) -> None:
    """Join MP4 segments without re-encoding."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if len(segments) == 1:
        os.replace(segments[0], video_path)
        return
    list_path = work_dir / f"{video_path.stem}_segments.txt"
    list_path.write_text("".join(f"file '{seg}'\n" for seg in segments))
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(video_path),
        ],
        check=True,
    )


def render_videos(
    snapshots: list[Snapshot],
    cfg: SimConfig,
    out_dir: Path,
    xc: np.ndarray,
    yc: np.ndarray,
    speed_levels: np.ndarray,
    p_levels: np.ndarray,
    omega_levels: np.ndarray,
) -> dict[str, Path]:
    """Render all videos in parallel worker processes with per-video progress bars."""
    n_frames = len(snapshots)
    if n_frames == 0:
        return {}
    n_workers = max(1, os.cpu_count() or 1)
    levels = {"streamlines": speed_levels, "pressure": p_levels, "vorticity": omega_levels}

    # Chunk sizes proportional to the frame cost: about three chunks per worker.
    total_cost = n_frames * sum(_FRAME_COST.values())
    target_cost = max(total_cost / (3 * n_workers), 1.0)

    manager = mp.Manager()
    queue = manager.Queue()

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

    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".video_segments_") as tmp, progress:
        work_dir = Path(tmp)
        tasks = []
        for kind, _ in _VIDEO_SPECS:  # streamlines first: the longest chunks start first
            chunk_len = max(1, int(round(target_cost / _FRAME_COST[kind])))
            slim = [_slim_snapshot(s, kind) for s in snapshots]
            for chunk, start in enumerate(range(0, n_frames, chunk_len)):
                tasks.append(
                    {
                        "kind": kind,
                        "chunk": chunk,
                        "queue": queue,
                        "lx": cfg.lx,
                        "ly": cfg.ly,
                        "xc": xc,
                        "yc": yc,
                        "fps": cfg.video_fps,
                        "levels": levels[kind],
                        "snapshots": slim[start : start + chunk_len],
                        "segment_path": str(work_dir / f"{kind}_{chunk:04d}.mp4"),
                    }
                )

        bars = {kind: progress.add_task(kind, total=n_frames) for kind, _ in _VIDEO_SPECS}

        def _listener() -> None:
            received = 0
            total = n_frames * len(_VIDEO_SPECS)
            while received < total:
                kind = queue.get()
                progress.update(bars[kind], advance=1)
                received += 1

        listener = threading.Thread(target=_listener, daemon=True)
        listener.start()

        segments: dict[str, list[tuple[int, str]]] = {kind: [] for kind, _ in _VIDEO_SPECS}
        with ProcessPoolExecutor(
            max_workers=min(n_workers, len(tasks)),
            mp_context=mp.get_context("spawn"),
        ) as pool:
            for kind, chunk, path in pool.map(_video_worker, tasks):
                segments[kind].append((chunk, path))

        listener.join()

        for kind, filename in _VIDEO_SPECS:
            ordered = [path for _, path in sorted(segments[kind])]
            video_path = out_dir / filename
            _concat_segments(ordered, video_path, work_dir)
            results[kind] = video_path

    manager.shutdown()
    return results
