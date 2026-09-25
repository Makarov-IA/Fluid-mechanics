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
    draw_pressure,
    draw_streamlines,
    draw_vorticity,
    fig_to_rgb,
    style_axes,
)

matplotlib.use("Agg")

console = Console()

_VIDEO_SPECS = [
    ("streamlines", "stokes_streamlines.mp4"),
    ("pressure", "stokes_pressure.mp4"),
    ("vorticity", "stokes_vorticity.mp4"),
]

# Relative per-frame cost, used to size chunks (streamplot dominates).
_FRAME_COST = {"streamlines": 3.0, "pressure": 1.0, "vorticity": 1.0}


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


def _video_worker(task: dict) -> tuple[str, int, str]:
    """Render one chunk of frames for one video type into an MP4 segment."""
    kind: str = task["kind"]
    queue = task["queue"]
    lx = task["lx"]
    ly = task["ly"]
    xc: np.ndarray = task["xc"]
    yc: np.ndarray = task["yc"]
    x_grid, y_grid = np.meshgrid(xc, yc, indexing="ij")
    snapshots: list[Snapshot] = task["snapshots"]

    writer = imageio.get_writer(
        task["segment_path"],
        fps=task["fps"],
        codec="libx264",
        output_params=["-crf", "20"],
        macro_block_size=1,
    )

    with writer:
        for snap in snapshots:
            fig, ax = plt.subplots(figsize=(6.2, 6.0))

            if kind == "streamlines":
                draw_streamlines(ax, fig, snap, xc, yc, x_grid, y_grid, task["levels"])
                title = f"Streamlines, t={snap.t:.3f}"
            elif kind == "pressure":
                draw_pressure(ax, fig, snap, x_grid, y_grid, task["levels"])
                title = f"Pressure, t={snap.t:.3f}"
            elif kind == "vorticity":
                draw_vorticity(ax, fig, snap, x_grid, y_grid, task["levels"])
                title = f"Vorticity, t={snap.t:.3f}"
            else:
                raise ValueError(f"Unknown video kind: {kind!r}")

            style_axes(ax, title, lx, ly)
            fig.tight_layout()
            writer.append_data(fig_to_rgb(fig, dpi=110))
            queue.put(kind)

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
