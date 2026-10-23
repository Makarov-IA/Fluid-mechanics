#!/usr/bin/env python3
"""Draw the snapshots of the ONE working YAML, reusing the project's own plotters.

Paths come from the YAML, not from RUN: this backend has a single config.
Frames and video are produced by scripts/python, so there is one plotting code
path for every backend. The frame sequence is PSEUDO-time relaxation towards the
steady state, not a physical unsteady flow.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver_python.config import ConfigError, load_config

ALEX = Path(__file__).resolve().parents[1]


def render(config_path: Path, fps: float, video: bool) -> dict:
    cfg = load_config(config_path)
    binaries = cfg.output_dir
    snapshots = sorted(binaries.glob("result_*.bin"))
    if not snapshots:
        raise FileNotFoundError(f"No result_*.bin in {binaries}. Run the solver first.")
    root = Path(cfg.resolved["output"]["root"])
    frames = root/"figures"/cfg.experiment/"frames"
    frames.mkdir(parents=True, exist_ok=True)
    print(f"[render] {cfg.experiment}: {len(snapshots)} snapshots -> {frames}", flush=True)
    subprocess.run([sys.executable, str(ALEX/"scripts/python/plot_fields.py"),
                    str(binaries), str(frames)], check=True)
    result = {"snapshots": len(snapshots), "frames": str(frames)}
    if video:
        videos = root/"videos"/cfg.experiment
        videos.mkdir(parents=True, exist_ok=True)
        print(f"[render] video -> {videos}", flush=True)
        subprocess.run([sys.executable, str(ALEX/"scripts/python/make_videos.py"),
                        str(frames), str(videos), "--fps", str(fps)], check=True)
        result["videos"] = str(videos)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--video", action="store_true", help="also assemble mp4 from the frames")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        render(args.config, args.fps, args.video)
        return 0
    except (ConfigError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[render] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
