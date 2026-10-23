"""Saved-snapshot diagnostics for positive corner vortices, not particle tracks.

An extremum must lie in a corner window. Its core is the connected component
above half the positive psi maximum, not an inferred separatrix. Missing local
maxima mean 'not detected in this window', not proof of vortex extinction.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, maximum_filter

from .config import grid, load_config
from .forcing import build_forcing
from .snapshot_io import read_snapshot


CORNERS = ('bottom_left', 'bottom_right', 'top_left', 'top_right')
TITLES = ('Нижний левый', 'Нижний правый', 'Верхний левый', 'Верхний правый')


def corner_measurements(state):
    p = state.psi
    dx, dy = state.xs[1]-state.xs[0], state.ys[1]-state.ys[0]
    x, y = state.xs[:, None], state.ys[None, :]
    lx, ly = state.xs[-1], state.ys[-1]
    windows = ((x <= .2*lx) & (y <= .25*ly),
               (x >= .8*lx) & (y <= .25*ly),
               (x <= .2*lx) & (y >= .75*ly),
               (x >= .8*lx) & (y >= .75*ly))
    maxima = (p == maximum_filter(p, size=3)) & (p > 1e-12)
    maxima[[0, -1], :] = False
    maxima[:, [0, -1]] = False
    # Reject flat plateaus and saddle points on a discretized field.
    hxx = (p[:-2, 1:-1]-2*p[1:-1, 1:-1]+p[2:, 1:-1])/dx**2
    hyy = (p[1:-1, :-2]-2*p[1:-1, 1:-1]+p[1:-1, 2:])/dy**2
    hxy = (p[2:, 2:]-p[:-2, 2:]-p[2:, :-2]+p[:-2, :-2])/(4*dx*dy)
    maxima[1:-1, 1:-1] &= (hxx < 0) & (hyy < 0) & (hxx*hyy-hxy*hxy > 0)
    speed = np.hypot(state.u, state.v)
    rows = []
    for name, window in zip(CORNERS, windows):
        candidates = np.argwhere(maxima & window)
        row = dict(step=state.step, time=state.time, corner=name,
                   detected=False, localized_core=False, x=None, y=None,
                   psi_peak=None, core_area=None, core_circulation=None,
                   core_max_speed=None, candidate_count=len(candidates))
        if len(candidates):
            i, j = max(candidates, key=lambda ij: p[tuple(ij)])
            components, _ = label(p > .5*p[i, j])
            core = components == components[i, j]
            row.update(detected=True, localized_core=bool(np.all(window[core])),
                       x=float(x[i, 0]), y=float(y[0, j]), psi_peak=float(p[i, j]),
                       core_area=float(core.sum()*dx*dy),
                       core_circulation=float(state.omega[core].sum()*dx*dy),
                       core_max_speed=float(speed[core].max()))
        rows.append(row)
    return rows


def read_history(cfg):
    files = sorted(cfg.output_dir.glob('result_*.bin'), key=lambda f: int(f.stem.split('_')[-1]))
    if not files:
        raise FileNotFoundError(cfg.output_dir)
    rows = []
    for file in files:
        rows.extend(corner_measurements(read_snapshot(file)))
    return rows


def magnetic_preview(cfg):
    xs, ys = grid(cfg)
    base, _ = build_forcing(cfg, xs, ys)
    source, bz = build_forcing(replace(cfg, forcing=cfg.magnetic_actuation), xs, ys)
    out = cfg.result_dir/'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), layout='constrained')
    fields = (bz*1e3, source, base, base+source)
    titles = ('Bz магнитов, мТл', 'Добавочный источник S_mag',
              'Исходный источник S_base', 'S_base + S_mag')
    for ax, field, title in zip(axes.flat, fields, titles):
        limit = max(float(np.max(np.abs(field))), 1e-12)
        im = ax.pcolormesh(xs, ys, field.T, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='auto')
        fig.colorbar(im, ax=ax, shrink=.8)
        ax.set(title=title, xlabel='x', ylabel='y', aspect='equal')
    length = cfg.magnetic_actuation['scales']['length_m']
    for mx, my, _, field in cfg.magnetic_actuation['magnets']:
        axes[0, 0].plot(mx/length, my/length, 'ko', ms=5, clip_on=False)
        axes[0, 0].annotate('+' if field > 0 else '−', (mx/length, my/length),
                            xytext=(6, 0), textcoords='offset points', clip_on=False)
    fig.suptitle('Четыре постоянных диполя; воздействие активно с первого шага')
    fig.savefig(out/'corner_magnetic_source.png', dpi=160)
    plt.close(fig)
    report = dict(source_min=float(source.min()), source_max=float(source.max()),
                  source_relative_l2=float(np.linalg.norm(source[1:-1, 1:-1]) /
                                           np.linalg.norm(base[1:-1, 1:-1])),
                  magnetic_field_range_T=[float(bz.min()), float(bz.max())],
                  geometry='hypothetical point dipoles; not hardware specifications')
    (out/'corner_magnetic_source.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def analyze(cfg, reference=None):
    rows = read_history(cfg)
    refs = read_history(reference) if reference is not None else []
    table_dir = cfg.result_dir/'tables'
    out = cfg.result_dir/'figures'
    table_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in (('corner_vortex_history.csv', rows), ('corner_vortex_reference.csv', refs)):
        if data:
            with (table_dir/filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    def curve_label(config):
        if config is None:
            return ''
        if config.magnetic_actuation is None:
            return 'Без магнитной добавки'
        current = config.magnetic_actuation['current_density_A_m2']
        return f'Магниты: j=({current[0]:g}, {current[1]:g}) А/м²'
    for ax, corner, title in zip(axes.flat, CORNERS, TITLES):
        for data, name, color in ((refs, curve_label(reference), '0.45'),
                                  (rows, curve_label(cfg), '#176eb3')):
            if not data:
                continue
            selected = [r for r in data if r['corner'] == corner]
            values = [r['psi_peak'] if r['localized_core'] else np.nan for r in selected]
            ax.plot([r['time'] for r in selected], values, label=name, color=color)
        ax.set(title=title, xlabel='Псевдовремя τ', ylabel='ψ в центре углового вихря', yscale='log')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Локальные положительные максимумы ψ; пропуск = локальное ядро не обнаружено')
    fig.savefig(out/'corner_vortex_history.png', dpi=160)
    plt.close(fig)
    common_time = min(rows[-1]['time'], refs[-1]['time']) if refs else None
    report = dict(experiment=cfg.experiment, snapshots=len(rows)//4,
                  time_end=rows[-1]['time'], final=rows[-4:],
                  detection='Positive elliptic psi maxima in corner windows: 20% width, 25% height',
                  core='Connected psi > 0.5*psi_peak; diagnostic contour, not separatrix',
                  missing='Not detected or not localized in this window; not proof of extinction',
                  reference_experiment=reference.experiment if reference else None,
                  comparison_end=common_time, late_peak_change={})
    for corner in CORNERS:
        candidates = [r for r in rows if r['corner'] == corner]
        early = min(candidates, key=lambda r: abs(r['time']-(rows[-1]['time']-3)))
        late = candidates[-1]
        if early['localized_core'] and late['localized_core']:
            report['late_peak_change'][corner] = dict(
                from_time=early['time'], to_time=late['time'],
                relative_change=late['psi_peak']/early['psi_peak']-1)
    if refs:
        near = min(set(r['time'] for r in rows), key=lambda t: abs(t-common_time))
        report['at_common_time'] = [r for r in rows if r['time'] == near]
        report['reference_at_common_time'] = refs[-4:]
    (out/'corner_vortex_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def render_corner_frame(arguments):
    file, output, speed_limit, *labels = arguments
    experiment_label = f', {labels[0]}' if labels else ''
    state = read_snapshot(file)
    rows = corner_measurements(state)
    x, y = state.xs, state.ys
    lx, ly = x[-1], y[-1]
    limits = ((0, .2*lx, 0, .25*ly), (.8*lx, lx, 0, .25*ly),
              (0, .2*lx, .75*ly, ly), (.8*lx, lx, .75*ly, ly))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, row, title, (x0, x1, y0, y1) in zip(axes.flat, rows, TITLES, limits):
        ix = np.flatnonzero((x >= x0) & (x <= x1))
        iy = np.flatnonzero((y >= y0) & (y <= y1))
        xx, yy = x[ix], y[iy]
        u = state.u[np.ix_(ix, iy)].T
        v = state.v[np.ix_(ix, iy)].T
        p = state.psi[np.ix_(ix, iy)].T
        im = ax.pcolormesh(xx, yy, np.hypot(u, v), shading='auto',
                          cmap='viridis', vmin=0, vmax=speed_limit)
        if np.any(u) or np.any(v):
            ax.streamplot(xx, yy, u, v, color='white', density=.8, linewidth=.6, arrowsize=.7)
        detail = 'локальный максимум не найден'
        if row['detected']:
            ax.plot(row['x'], row['y'], '+', color='#ffe35b', ms=9, mew=1.5)
            ax.contour(xx, yy, p, levels=[.5*row['psi_peak']], colors=['#ffe35b'], linewidths=.9)
            detail = f"ψ_max={row['psi_peak']:.3e}"
            if not row['localized_core']:
                detail += '; ядро выходит из окна'
        ax.set(title=f'{title}\n{detail}', xlabel='x', ylabel='y',
               xlim=(x0, x1), ylim=(y0, y1), aspect='equal')
    fig.colorbar(im, ax=axes, label='|u|; единая шкала всех кадров', shrink=.85)
    fig.suptitle(f'Угловые вихри{experiment_label}: τ={state.time:.5f} (псевдовремя)\n'
                 'Жёлтый контур: ψ = 0.5 ψ_max; линии тока не являются траекториями')
    fig.savefig(output, dpi=110)
    plt.close(fig)
    return str(output)


def corner_video(cfg, workers=4, fps=20.):
    from .make_videos import make_video
    files = sorted(cfg.output_dir.glob('result_*.bin'), key=lambda f: int(f.stem.split('_')[-1]))
    limit = 1e-12
    for file in files:
        s = read_snapshot(file)
        x, y = s.xs[:, None], s.ys[None, :]
        mask = ((x <= .2*x[-1]) | (x >= .8*x[-1])) & ((y <= .25*y[0, -1]) | (y >= .75*y[0, -1]))
        limit = max(limit, float(np.hypot(s.u, s.v)[mask].max()))
    frames = cfg.result_dir/'figures'/'corner_frames'
    frames.mkdir(parents=True, exist_ok=True)
    current = cfg.magnetic_actuation['current_density_A_m2'] if cfg.magnetic_actuation else [0., 0.]
    name = f'j=({current[0]:g}, {current[1]:g}) А/м²'
    arguments = [(f, frames/f'{f.stem}_corners.png', limit, name) for f in files]
    paths = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, path in enumerate(pool.map(render_corner_frame, arguments, chunksize=4), 1):
            paths.append(path)
            if i % 100 == 0 or i == len(files):
                print(f'[corners] {i}/{len(files)} frames', flush=True)
    videos = cfg.result_dir/'videos'
    videos.mkdir(parents=True, exist_ok=True)
    make_video(paths, str(videos/'corners.mp4'), fps)
    (cfg.result_dir/'figures'/'corner_render_scale.json').write_text(json.dumps(
        dict(speed_limits=[0., limit], frames=len(files), fps=fps,
             first_time=read_snapshot(files[0]).time, last_time=read_snapshot(files[-1]).time), indent=2)+'\n')
    print(f'[corners] video: {videos / "corners.mp4"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--fps', type=float, default=20.)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.magnetic_actuation:
        print(json.dumps(magnetic_preview(cfg), indent=2))
    if not args.preview_only:
        reference = load_config(args.reference) if args.reference else None
        report = analyze(cfg, reference)
        print(json.dumps(dict(time_end=report['time_end'], final=report['final']), indent=2))
        if args.video:
            corner_video(cfg, args.workers, args.fps)


if __name__ == '__main__':
    main()
