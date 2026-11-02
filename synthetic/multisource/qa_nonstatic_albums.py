#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.qa_nonstatic_albums [--albums-dir synthetic/multisource/pila_stacks_v2/albums_nonstatic]
#                  [--out-dir figures/multisource_k2_nonstatic/data_qa] [--no-figures]
# Description: Independent QA gate for the non-static K=2 albums (make_nonstatic_albums.py). NO inversion is submitted
#              until every album passes and the user has seen the figures. Deliberately does NOT reuse the generator's
#              forward model or geometry: albums are read through PILA's own loader (load_insar_mintpy, the call PILA's
#              configs make), and truth is re-rendered with the numpy Mogi of synthetic/forward_numpy.py on the loader's
#              coordinates and LOS unit vectors. Checks per album:
#                1 forward     numpy re-render of the per-date truth (then referenced) = noise-free twin seen by PILA
#                2 intent      dV(t) follows the declared profile; N1 second source exactly 0 before tau 0.5; N2 second
#                              source dV < 0 on every date; N3 second source back below 5 % of its peak at the end;
#                              N4 first source moves 4 km and shallows as declared; static = identical dates
#                3 peaks       each source's max own peak |LOS| = 4 x sigma_ref (+-0.1 %), recomputed here
#                4 loader      shape, dates, units (mm), all-valid mask, finite, LOS geometry as declared,
#                              loaded cube = npz cube x 1000
#                5 noise       album - twin = scale x the SAME noise for every album of a given noise scale (paired);
#                              noise RMS / SNR = truth labels
#                6 bookkeeping file name <-> truth fields, dates = E1 acq dates and increasing, layout draw rules,
#                              reference-pixel leak < 5 %
#              Figures (PNG): per-album QA sheet (truth time series + noise-free / noisy maps at 4 dates) and an
#              overview of the random layouts. Exits non-zero if any album fails.
# Date:        2026-10-26

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from synthetic.forward_numpy import mogi

STACKS = 'synthetic/multisource/pila_stacks_v2'
EXPECTED_SCENARIOS = {'S3_static': ('constant', 'constant'), 'N1_staggered': ('linear', 'sigmoid'),
                      'N2_opposite': ('linear', 'deflate_linear'), 'N3_pulse': ('linear', 'pulse'),
                      'N4_moving': ('linear', 'linear')}
G1_EXPECTED = [(-6.0, 0.0, 4.0), (6.0, 0.0, 5.0)]
G1_SCALES = {0.0, 0.125, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0}
R_SCALES = {0.0, 0.18, 0.25, 0.5, 1.0}
FORWARD_TOL_MM, NOISE_TOL_MM = 1e-3, 1e-3


def load_like_pila(npz_path):
    """Load an album exactly as the K=2 PILA configs do (point path, multilook 1, asc track)."""
    return load_insar_mintpy(os.path.abspath(npz_path), 'asc', None, 0.0, 0.0, multilook=1, coh_valid_frac=0.5,
                             verbose=False)


def numpy_render_mm(data, truth):
    """Independent re-render [T, N] (mm) of the per-date truth on PILA's points, then referenced to the ref pixel."""
    x_m = data.xE_pts.astype(np.float64) * 1e3
    y_m = data.yN_pts.astype(np.float64) * 1e3
    los = [data.losE_pts.astype(np.float64), data.losN_pts.astype(np.float64), data.losU_pts.astype(np.float64)]
    n_dates = len(truth['dates'])
    total = np.zeros((n_dates, x_m.size))
    per_source = []
    for src in truth['sources_per_date']:
        field = np.zeros_like(total)
        for t in range(n_dates):
            uE, uN, uU = mogi(x_m, y_m, src['xcen_km'][t] * 1e3, src['ycen_km'][t] * 1e3, src['depth_km'][t] * 1e3,
                              src['dV_m3'][t])
            field[t] = los[0] * uE + los[1] * uN + los[2] * uU
        per_source.append(field)
        total += field
    return total, per_source


def profile_expected(name, tau):
    """Expected shape of dV(t), up to a positive amplitude (written independently of the generator)."""
    if name == 'constant':
        return np.ones_like(tau)
    if name == 'linear':
        return tau
    if name == 'staggered':
        return np.maximum(0.0, 2.0 * tau - 1.0)
    if name == 'sigmoid':   # as synthetic/trajectories.profile_sigmoid, written out here independently
        z = lambda t: 1.0 / (1.0 + np.exp(-8.0 * (2.0 * t - 1.0)))
        return (z(tau) - z(0.0)) / (z(1.0) - z(0.0))
    if name == 'deflate_linear':
        return -tau
    if name == 'pulse':
        return np.exp(-((tau - 0.5) ** 2) / (2 * 0.15 ** 2)) - np.exp(-(0.25) / (2 * 0.15 ** 2))
    raise ValueError(name)


def check_album(name, albums_dir, e1_meta, sigma_ref_mm, noise_by_scale, layout_rows):
    """Run checks 1-6 on one album; returns (row dict of check -> bool / value, data, truth, cubes for plotting)."""
    row = {'album': name}
    truth = json.load(open(os.path.join(albums_dir, f'{name}_truth.json')))
    data = load_like_pila(os.path.join(albums_dir, f'{name}.npz'))
    twin = load_like_pila(os.path.join(albums_dir, f'{name}_nonoise.npz'))
    obs_mm = data.los_points_mm.astype(np.float64)
    sig_mm = twin.los_points_mm.astype(np.float64)
    tau = np.array(truth['tau'])
    ref = truth['ref_pixel_flat']

    # --- 6 bookkeeping (names, dates) ---
    match = re.fullmatch(r'(G1|R(\d))_(S3_static|N1_staggered|N2_opposite|N3_pulse|N4_moving)_n(\d{4})', name)
    scale = int(match.group(4)) / 1000 if match else None
    row['name_matches_truth'] = bool(match and truth['scenario'] == match.group(3)
                                     and truth['layout_set'] == ('G1' if match.group(1) == 'G1' else 'R')
                                     and abs(truth['noise_scale'] - scale) < 1e-9
                                     and (truth['draw'] == (int(match.group(2)) if match.group(2) else None)))
    row['scale_in_design'] = scale in (G1_SCALES if truth['layout_set'] == 'G1' else R_SCALES)
    iso = [f'{d[:4]}-{d[4:6]}-{d[6:]}' for d in e1_meta['acq_dates']]
    row['dates_ok'] = (data.dates == iso and twin.dates == iso and truth['dates'] == e1_meta['acq_dates']
                       and all(a < b for a, b in zip(iso, iso[1:])) and truth['ref_date'] == e1_meta['ref_date'])
    days = np.array([(np.datetime64(d) - np.datetime64(f"{truth['ref_date'][:4]}-{truth['ref_date'][4:6]}-{truth['ref_date'][6:]}"))
                     .astype(int) for d in iso], dtype=float)
    row['tau_ok'] = bool(np.allclose(tau, days / days[-1]) and tau[0] > 0)

    # --- 4 loader ---
    row['loader_shape_ok'] = bool(obs_mm.shape == (30, 128 * 128) and data.n_points == 128 * 128 and data.mask_d.all())
    row['finite'] = bool(np.isfinite(obs_mm).all() and np.isfinite(sig_mm).all())
    with np.load(os.path.join(albums_dir, f'{name}.npz')) as f:
        raw_mm = f['los_asc_m'].astype(np.float64) * 1e3
        inc, head = float(f['incidence_deg']), float(f['heading_asc_deg'])
    # The reference-pixel index is in npz order, so the loader must NOT reorder rows (it flips only if y_m increases)
    loaded_img = data.los_mm.astype(np.float64)
    row['loader_equals_npz'] = bool(np.allclose(loaded_img, raw_mm, atol=1e-4)
                                    and np.allclose(obs_mm, raw_mm.reshape(raw_mm.shape[0], -1), atol=1e-4))
    inc_r, az_r = np.deg2rad(inc), np.deg2rad(90.0 - head)
    expected_los = (-np.sin(inc_r) * np.sin(az_r), np.sin(inc_r) * np.cos(az_r), np.cos(inc_r))   # MintPy enu2los
    row['los_vector_ok'] = bool(np.allclose(data.losE_pts, expected_los[0], atol=1e-5)
                                and np.allclose(data.losN_pts, expected_los[1], atol=1e-5)
                                and np.allclose(data.losU_pts, expected_los[2], atol=1e-5))
    row['ascending_geometry'] = bool(0.7 < expected_los[2] < 0.9 and expected_los[0] < 0)   # asc: LOS points west-up

    # --- 1 forward (independent numpy Mogi on PILA's points) ---
    render_mm, per_source = numpy_render_mm(data, truth)
    render_ref_mm = render_mm - render_mm[:, [ref]]
    row['forward_max_err_mm'] = float(np.abs(render_ref_mm - sig_mm).max())
    row['forward_ok'] = row['forward_max_err_mm'] < FORWARD_TOL_MM

    # --- 3 peaks + 2 intent ---
    target_mm = truth['peak_ratio'] * sigma_ref_mm
    intent_ok, peak_ok = True, True
    profiles = EXPECTED_SCENARIOS[truth['scenario']]
    row['profiles_ok'] = tuple(truth['profiles']) == profiles
    for index, (src, field) in enumerate(zip(truth['sources_per_date'], per_source)):
        own_peak = np.abs(field).max(axis=1)
        peak_ok &= bool(abs(own_peak.max() / target_mm - 1) < 1e-3 and np.allclose(own_peak, src['own_peak_mm'], rtol=1e-4, atol=1e-6))
        dV = np.array(src['dV_m3'])
        shape = profile_expected(profiles[index], tau)
        amp = dV[np.argmax(np.abs(shape))] / shape[np.argmax(np.abs(shape))]
        intent_ok &= bool(amp > 0 and np.allclose(dV, amp * shape, rtol=1e-9, atol=1e-6 * abs(amp)))
        if profiles[index] == 'staggered':
            intent_ok &= bool(np.all(dV[tau <= 0.5] == 0.0) and np.all(dV[tau > 0.5] > 0))
        if profiles[index] == 'sigmoid':   # monotonic onset, quiet in the first quarter, half-way at tau 0.5
            intent_ok &= bool(np.all(np.diff(dV) > 0) and np.all(own_peak[tau < 0.25] < 0.05 * own_peak.max()))
        if profiles[index] == 'deflate_linear':
            intent_ok &= bool(np.all(dV < 0))
        if profiles[index] == 'pulse':
            intent_ok &= bool(own_peak[-1] < 0.05 * own_peak.max() and own_peak[0] < 0.2 * own_peak.max())
        x, y, depth = (np.array(src[k]) for k in ('xcen_km', 'ycen_km', 'depth_km'))
        moving = truth['scenario'] == 'N4_moving' and index == 0
        if moving:
            # extrapolate the linear-in-tau track to tau = 0 and 1 (the declared start and end)
            at = lambda v, t_target: v[0] + (t_target - tau[0]) * (v[-1] - v[0]) / (tau[-1] - tau[0])
            start, end = [at(v, 0.0) for v in (x, y, depth)], [at(v, 1.0) for v in (x, y, depth)]
            linear = all(np.allclose(v, at(v, 0.0) + tau * (at(v, 1.0) - at(v, 0.0)), atol=1e-9) for v in (x, y, depth))
            intent_ok &= bool(linear and abs(np.hypot(end[0] - start[0], end[1] - start[1]) - 4.0) < 1e-6
                              and abs(end[2] - max(0.6 * start[2], 1.5)) < 1e-6)
            if truth['layout_set'] == 'G1':
                intent_ok &= bool(np.allclose(start + end, [-8.0, 0.0, 5.0, -4.0, 0.0, 3.0], atol=1e-9))
            else:
                intent_ok &= bool(2.0 <= start[2] <= 8.0)
        else:
            intent_ok &= bool(np.ptp(x) == 0 and np.ptp(y) == 0 and np.ptp(depth) == 0)
            if truth['layout_set'] == 'G1':
                intent_ok &= bool(np.allclose([x[0], y[0], depth[0]], G1_EXPECTED[index]))
        if truth['layout_set'] == 'R':
            intent_ok &= bool(np.all(np.abs(x) <= 10 + 1e-9) and np.all(np.abs(y) <= 10 + 1e-9)
                              and (moving or 2.0 <= depth[0] <= 8.0))
        # reference-pixel leak of the unit field (independent: from the numpy field)
        leak = np.max(np.abs(field[:, ref]) / np.maximum(own_peak, 1e-30) * (own_peak > 0))
        row[f'ref_leak_src{index}'] = float(leak)
    if truth['scenario'] == 'S3_static':
        intent_ok &= bool(np.abs(sig_mm - sig_mm[0]).max() < 1e-4)
    srcs = truth['sources_per_date']
    sep = np.hypot(np.array(srcs[0]['xcen_km']) - srcs[1]['xcen_km'], np.array(srcs[0]['ycen_km']) - srcs[1]['ycen_km'])
    row['min_separation_km'] = float(sep.min())
    if truth['layout_set'] == 'R':
        intent_ok &= bool(3.0 <= sep.min() and sep.max() <= 15.0)
    row['intent_ok'], row['peaks_ok'] = bool(intent_ok), bool(peak_ok)
    row['ref_leak_ok'] = max(row['ref_leak_src0'], row['ref_leak_src1']) < 0.05

    # --- 5 noise (paired across scenarios at the same scale) ---
    noise_mm = obs_mm - sig_mm
    if scale is not None:
        key = round(scale, 3)
        if key in noise_by_scale:
            row['noise_paired'] = bool(np.abs(noise_mm - noise_by_scale[key]).max() < NOISE_TOL_MM)
        else:
            noise_by_scale[key] = noise_mm
            row['noise_paired'] = True
        # Anchor to the source: noise = scale x (E1 noise referenced to the pixel), E1 loaded here independently
        e1_ref = noise_by_scale['E1_referenced']
        row['noise_equals_E1'] = bool(np.abs(noise_mm - scale * e1_ref).max() < NOISE_TOL_MM)
        rms = float(np.sqrt(np.mean(noise_mm ** 2)))
        peak = float(np.abs(sig_mm).max())
        row['noise_rms_mm'], row['album_peak_mm'] = rms, peak
        row['snr'] = np.inf if rms == 0 else peak / rms
        row['noise_label_ok'] = bool(abs(rms - truth['noise_rms_mm']) < 1e-3 and abs(peak - truth['album_peak_mm']) < 1e-3
                                     and ((truth['snr_peak_over_rms'] is None and scale == 0)
                                          or (scale > 0 and abs(row['snr'] / truth['snr_peak_over_rms'] - 1) < 1e-3)))
    checks = [k for k, v in row.items() if isinstance(v, bool)]
    row['PASS'] = all(row[k] for k in checks)
    row['failed_checks'] = ';'.join(k for k in checks if not row[k])
    if truth['layout_set'] == 'R':
        layout_rows.append(dict(album=name, scenario=truth['scenario'],
                                x0=srcs[0]['xcen_km'], y0=srcs[0]['ycen_km'], d0=srcs[0]['depth_km'],
                                x1=srcs[1]['xcen_km'][0], y1=srcs[1]['ycen_km'][0], d1=srcs[1]['depth_km'][0],
                                sep_km=row['min_separation_km'], pascal_start=truth['separation_dshallow_start']))
    return row, data, truth, obs_mm, sig_mm


def qa_sheet(path, name, data, truth, obs_mm, sig_mm):
    """Truth time series (x, y, depth, dV, own peak per source) + noise-free / noisy maps at 4 dates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dates = np.array(data.dates, dtype='datetime64[D]')
    tau = np.array(truth['tau'])
    picks = sorted({0, int(np.argmin(abs(tau - 0.25))), int(np.argmin(abs(tau - 0.5))), len(tau) - 1})
    shape = data.los_mm.shape[1:]
    extent = [data.xE_grid.min(), data.xE_grid.max(), data.yN_grid.min(), data.yN_grid.max()]
    fig = plt.figure(figsize=(18, 11))
    grid_spec = fig.add_gridspec(3, 5)
    colors = ['tab:blue', 'tab:red']
    for col, (key, label) in enumerate([('xcen_km', 'East (km)'), ('ycen_km', 'North (km)'), ('depth_km', 'Depth (km)'),
                                        ('dV_m3', 'dV (m$^3$)'), ('own_peak_mm', 'Own peak |LOS| (mm)')]):
        ax = fig.add_subplot(grid_spec[0, col])
        for index, src in enumerate(truth['sources_per_date']):
            ax.plot(dates, src[key], 'o-', ms=3, color=colors[index], label=f"source {index} ({src['profile']})")
        if key == 'own_peak_mm':
            ax.axhline(truth['peak_ratio'] * truth['sigma_ref_mm'], color='k', ls='--', lw=0.8, label='4 x sigma_ref')
            if truth['noise_rms_mm'] > 0:
                ax.axhline(truth['noise_rms_mm'], color='gray', ls=':', label='noise RMS')
        ax.set_xlabel('Date'); ax.set_ylabel(label); ax.tick_params(axis='x', rotation=45)
        if col == 0:
            ax.legend(fontsize=7)
        if key == 'own_peak_mm':
            ax.legend(fontsize=7)
    vmax = max(np.abs(sig_mm).max(), 1e-6)
    for row_index, (cube, label) in enumerate([(sig_mm, 'noise-free'), (obs_mm, 'album (what PILA sees)')], start=1):
        for col, t in enumerate(picks):
            ax = fig.add_subplot(grid_spec[row_index, col])
            image = ax.imshow(cube[t].reshape(shape), cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent, origin='upper')
            for index, src in enumerate(truth['sources_per_date']):
                ax.plot(src['xcen_km'][t], src['ycen_km'][t], marker='x' if src['dV_m3'][t] != 0 else 'o', mfc='none',
                        color=colors[index], ms=9, mew=2)
            ax.set_title(f"{label} {data.dates[t]} (tau {tau[t]:.2f})", fontsize=8)
            ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
        ax = fig.add_subplot(grid_spec[row_index, 4])
        plt.colorbar(image, ax=ax, fraction=0.8, label='LOS (mm, + toward satellite)'); ax.axis('off')
    snr = truth['snr_peak_over_rms']
    fig.suptitle(f"{name}: {truth['scenario']} ({truth['layout_set']}), noise x{truth['noise_scale']}, "
                 f"SNR {'inf' if snr is None else f'{snr:.1f}'}  [x = active source, o = dV 0]")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def layout_overview(path, layouts):
    """Random layouts: positions (lines = N4 moves), depths, separation histogram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for layout in layouts:
        axes[0].plot(layout['x0'], layout['y0'], '-', color='tab:blue', lw=1)
        axes[0].plot(layout['x0'][0], layout['y0'][0], 'o', color='tab:blue', ms=4)
        axes[0].plot(layout['x1'], layout['y1'], 's', color='tab:red', ms=4)
        axes[0].plot([layout['x0'][0], layout['x1']], [layout['y0'][0], layout['y1']], color='gray', lw=0.4)
    axes[0].add_patch(plt.Rectangle((-10, -10), 20, 20, fill=False, ls='--', color='k'))
    axes[0].set_xlabel('East (km)'); axes[0].set_ylabel('North (km)'); axes[0].set_aspect('equal')
    axes[0].set_title('Random layouts: source 0 (blue, lines = N4 moves), source 1 (red)')
    axes[1].hist([[l['d0'][0] for l in layouts], [l['d1'] for l in layouts]], bins=12, label=['source 0 (start)', 'source 1'])
    axes[1].set_xlabel('Depth (km)'); axes[1].set_ylabel('Albums'); axes[1].legend()
    axes[2].hist([l['sep_km'] for l in layouts], bins=12, color='gray')
    axes[2].set_xlabel('Minimum horizontal separation (km)'); axes[2].set_ylabel('Albums')
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--albums-dir', default=f'{STACKS}/albums_nonstatic_v2')
    parser.add_argument('--out-dir', default='figures/multisource_k2_nonstatic/data_qa_v2')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    t0 = time.time()
    if not os.path.isdir(args.albums_dir):
        raise FileNotFoundError(args.albums_dir)
    os.makedirs(os.path.join(args.out_dir, 'sheets'), exist_ok=True)

    print("[1/3] Listing albums and loading E1 metadata...")
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.albums_dir, '*.npz'))
                   if not p.endswith('_nonoise.npz'))
    e1_meta = json.load(open(f'{STACKS}/noise_sets/E1.json'))
    sigma_ref_mm = json.load(open(f'{STACKS}/noise_sets/noise_summary.json'))['sigma_ref_mm']
    print(f"  {len(names)} albums; expected 90 (40 G1 + 50 R)")
    with np.load(f'{STACKS}/noise_sets/E1.npz') as f:
        e1_noise_mm = f['noise_mm'].astype(np.float64).reshape(len(e1_meta['acq_dates']), -1)
    print(f"  Loaded E1 noise: shape={e1_noise_mm.shape}, dtype={e1_noise_mm.dtype}")

    print("[2/3] Checking every album...")
    rows, layouts = [], []
    noise_by_scale = {'E1_referenced': e1_noise_mm - e1_noise_mm[:, [e1_meta['ref_pixel_flat']]]}
    for step, name in enumerate(names, start=1):
        row, data, truth, obs_mm, sig_mm = check_album(name, args.albums_dir, e1_meta, sigma_ref_mm, noise_by_scale, layouts)
        rows.append(row)
        print(f"  ({step}/{len(names)}) {name}: {'PASS' if row['PASS'] else 'FAIL ' + row['failed_checks']} "
              f"(forward err {row['forward_max_err_mm']:.1e} mm)")
        if not args.no_figures:
            qa_sheet(os.path.join(args.out_dir, 'sheets', f'{name}.png'), name, data, truth, obs_mm, sig_mm)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, 'qa_summary.csv'), index=False, float_format='%.6g')
    design_ok = (len(names) == 90 and sum(n.startswith('G1_') for n in names) == 40
                 and sum(n.startswith('R') for n in names) == 50)

    print("[3/3] Layout overview...")
    if layouts and not args.no_figures:
        layout_overview(os.path.join(args.out_dir, 'random_layouts_overview.png'), layouts)
    n_pass = int(table['PASS'].sum())
    print(f"  {n_pass}/{len(table)} albums PASS; design count ok {design_ok}; "
          f"max forward err {table['forward_max_err_mm'].max():.2e} mm ({time.time() - t0:.1f}s)")
    print(f"Done. Output saved to {args.out_dir}/")
    if n_pass != len(table) or not design_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
