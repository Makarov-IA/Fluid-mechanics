#!/usr/bin/env python3
"""Render the selected experiment using the Python pipeline's binary readers.

Output lives under results/<YAML stem>/. Frames show pseudo-time evolution.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver_python.config import DEFAULT_CONFIG, ConfigError, load_config

ALEX = Path(__file__).resolve().parents[1]


def render(config_path: Path, fps: float, video: bool, time_offset: float = 0.,
           workers: int | None = None, scale_file: Path | None = None) -> dict:
    cfg = load_config(config_path)
    binaries = cfg.output_dir
    snapshots = sorted(binaries.glob("result_*.bin"))
    if not snapshots:
        raise FileNotFoundError(f"No result_*.bin in {binaries}. Run the solver first.")
    frames = cfg.result_dir/"figures"/"frames"
    frames.mkdir(parents=True, exist_ok=True)
    print(f"[render] {cfg.experiment}: {len(snapshots)} snapshots -> {frames}", flush=True)
    command = [sys.executable, str(ALEX/"solver_python/plot_fields.py"),
               str(binaries), str(frames), "--time-offset", str(time_offset)]
    if workers is not None:
        command += ["--workers", str(workers)]
    if scale_file is not None:
        command += ["--scale-file", str(scale_file.resolve())]
    subprocess.run(command, check=True)
    result = {"snapshots": len(snapshots), "frames": str(frames)}
    if video:
        videos = cfg.result_dir/"videos"
        videos.mkdir(parents=True, exist_ok=True)
        print(f"[render] video -> {videos}", flush=True)
        subprocess.run([sys.executable, str(ALEX/"solver_python/make_videos.py"),
                        str(frames), str(videos), "--fps", str(fps)], check=True)
        result["videos"] = str(videos)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", action="store_true", help="also assemble mp4 from the frames")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--time-offset", type=float, default=0.)
    parser.add_argument("--workers", type=int, help="parallel rendering processes, independent of solver threads")
    parser.add_argument("--scale-file", type=Path, help="shared plotting limits JSON for comparing experiments")
    args = parser.parse_args(argv)
    try:
        render(args.config, args.fps, args.video, args.time_offset, args.workers, args.scale_file)
        return 0
    except (ConfigError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[render] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
