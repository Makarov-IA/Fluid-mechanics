import argparse
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor

import imageio.v2 as imageio


FRAME_KINDS = ("psi", "omega", "streamplot")
VIDEO_NAMES = {
    "psi": "psi.mp4",
    "omega": "omega.mp4",
    "streamplot": "streamplot.mp4",
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


def make_video(frame_paths, video_path, fps):
    with imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="error",
    ) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    return True


def render_kind(kind, frames_dir, videos_dir, fps):
    frame_paths = collect_frame_paths(frames_dir, kind)
    video_path = os.path.join(videos_dir, VIDEO_NAMES[kind])
    if len(frame_paths) < 2:
        return kind, video_path, False
    make_video(frame_paths, video_path, fps)
    return kind, video_path, True


def main():
    parser = argparse.ArgumentParser(description="Build MP4 videos from existing frame PNGs")
    parser.add_argument("frames_dir")
    parser.add_argument("videos_dir")
    parser.add_argument("--fps", type=float, default=20.0, help="Frames per second")
    args = parser.parse_args()

    frames_dir = os.path.abspath(os.path.normpath(args.frames_dir))
    videos_dir = os.path.abspath(os.path.normpath(args.videos_dir))
    os.makedirs(videos_dir, exist_ok=True)

    workers = min(len(FRAME_KINDS), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(render_kind, kind, frames_dir, videos_dir, args.fps)
            for kind in FRAME_KINDS
        ]
        for future in futures:
            kind, video_path, created = future.result()
            if created:
                print(f"[video] {kind}: {video_path}")
            else:
                print(f"[video] {kind}: not enough frames, skipped")


if __name__ == "__main__":
    main()
