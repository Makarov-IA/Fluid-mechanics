import argparse
import glob
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg

FRAME_KINDS = ("psi", "omega", "streamplot")
GIF_NAMES = {
    "psi": "psi.gif",
    "omega": "omega.gif",
    "streamplot": "streamplot.gif",
}


def step_sort_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)(?!.*\d)", base)
    if match is None:
        return (-1, base)
    return (int(match.group(1)), base)


def collect_frame_paths(frames_dir, kind):
    pattern = os.path.join(frames_dir, f"*_{kind}.png")
    return sorted(glob.glob(pattern), key=step_sort_key)


def write_concat_file(frame_paths):
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        for path in frame_paths:
            escaped = path.replace("\\", "/").replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
        return handle.name
    finally:
        handle.close()


def make_gif(frame_paths, gif_path, duration_ms):
    fps = 1000.0 / max(duration_ms, 1)
    concat_path = write_concat_file(frame_paths)
    palette_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-vf",
                f"fps={fps},scale=iw:ih:flags=lanczos,palettegen=stats_mode=single",
                palette_path,
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-i",
                palette_path,
                "-lavfi",
                f"fps={fps},scale=iw:ih:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                gif_path,
            ],
            check=True,
        )
        return True
    finally:
        if os.path.exists(concat_path):
            os.remove(concat_path)
        if os.path.exists(palette_path):
            os.remove(palette_path)


def render_kind(kind, frames_dir, gifs_dir, duration_ms):
    frame_paths = collect_frame_paths(frames_dir, kind)
    gif_path = os.path.join(gifs_dir, GIF_NAMES[kind])
    if len(frame_paths) < 2:
        return kind, gif_path, False
    make_gif(frame_paths, gif_path, duration_ms)
    return kind, gif_path, True


def main():
    parser = argparse.ArgumentParser(description="Build GIFs from existing frame PNGs")
    parser.add_argument("frames_dir")
    parser.add_argument("gifs_dir")
    parser.add_argument("--duration", type=int, default=120, help="Frame duration in ms")
    args = parser.parse_args()

    frames_dir = os.path.abspath(os.path.normpath(args.frames_dir))
    gifs_dir = os.path.abspath(os.path.normpath(args.gifs_dir))
    os.makedirs(gifs_dir, exist_ok=True)

    workers = min(len(FRAME_KINDS), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(render_kind, kind, frames_dir, gifs_dir, args.duration)
            for kind in FRAME_KINDS
        ]
        for future in futures:
            kind, gif_path, created = future.result()
            if created:
                print(f"[gif] {kind}: {gif_path}")
            else:
                print(f"[gif] {kind}: not enough frames, skipped")


if __name__ == "__main__":
    main()
