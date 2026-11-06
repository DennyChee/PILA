#!/usr/bin/env python
# Usage:       python plot_k2_nonstatic_recon_maps.py [--out-dir figures/multisource_k2_nonstatic/param_recovery]
#                  [--compare-dir figures/multisource_k2_nonstatic] [--pila-root saved/multisource_k2_nonstatic]
#                  [--album G1_N1_staggered_n0250] [--n-dates 6]
#              (also called automatically at the end of compare_k2_nonstatic.py)
# Description: Input / modelled / residual LOS maps for EVERY album of the non-static K=2 campaign, one figure per
#              album in its scenario folder (<album>_recon_maps.png). Columns = evenly spaced dates; rows = the input
#              PILA and classical both see, then for each method its modelled field and the residual (input - model):
#              Input = the album LOS (checked equal, to 1e-3 mm, to what every PILA run stored as its input).
#                * PILA best-of-N (slot_resid, slot_resid_dvl1u300): model = PHYSICS-ONLY sum of the two source fields
#                  (xP_src_mm of the basin_best fit); the low-rank residual term is NOT in the model map (it can absorb
#                  signal), but the residual-panel title also quotes the RMS of input - (physics + low-rank) (xPB_mm);
#                * classical per-date OLS: model = its two Mogi fields + its constant offset, rebuilt from the saved
#                  geometry with the same linear solve (classical_k2_floor.MogiK2.solve_linear); the rebuilt dV is
#                  checked against the saved dV (the offset was not saved).
#              All panels of one figure (residuals included) share ONE symmetric colour scale, so the size of the
#              misfit reads directly against the deformation. + = true centres (black), o = method centres.
#              Best-of-N seeds come from compare_k2_nonstatic.py's bestofN.csv (truth-free pick: lowest album misfit).
#              Units: LOS mm (+ toward satellite), km.
# Date:        2026-10-27

import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import classical_k2_floor as floor
from compare_k2_nonstatic import ALBUMS_DIR, ARMS, CLASSICAL_ROOT, check_dates, find_models_dir
from synthetic.multisource.make_pila_stack import load_ta69_geometry
from synthetic.multisource.nonstatic_scoring import truth_arrays

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METHOD_STYLE = {'slot_resid': ('tab:orange', 'PILA slot_resid'),
                'slot_resid_dvl1u300': ('tab:purple', 'PILA + dV-L1(u) 300'),
                'slot_resid_curv5000': ('tab:blue', 'PILA + curv 5000'),
                'slot_resid_curv15000_dvl1u300': ('tab:red', 'PILA + curv 15000 + dV-L1(u) 300'),
                'classical': ('#1e8449', 'classical per-date')}
INPUT_ATOL_MM = 1e-3                                  # PILA's stored input must equal the album to 0.001 mm
DV_CHECK_RTOL = 1e-4                                  # rebuilt classical dV must match the saved dV to 0.01 %


def pila_fit(pila_root, reused, album, arm, seed, truth):
    """basin_best fit arrays of one PILA run: params [T,2,4], physics [T,ny,nx] mm, physics+low-rank [T,ny,nx] mm."""
    if arm == 'slot_resid' and album in reused:
        run_root = os.path.join(PROJECT_ROOT, reused[album][str(seed)])
    else:
        run_root = os.path.join(PROJECT_ROOT, pila_root, f'k2ns_{album}_{arm}_seed{seed}')
    models = find_models_dir(run_root)
    if models is None:
        raise FileNotFoundError(f'{album} {arm} seed {seed}: no models dir under {run_root}')
    with np.load(os.path.join(models, 'basin_best', 'fit_K2.npz')) as fit:
        check_dates(fit['dates'], truth)             # guard: fit and truth on the same dates in the same order
        return {'params': fit['params_phys'].astype(np.float64),
                'model_mm': fit['xP_src_mm'].astype(np.float64).sum(axis=1),     # physics only, both sources
                'model_lowrank_mm': fit['xPB_mm'].astype(np.float64),            # physics + low-rank residual
                'data_mm': fit['data_mm'].astype(np.float64),
                'x_km': fit['xE_km'].astype(np.float64), 'y_km': fit['yN_km'].astype(np.float64)}  # the fit's own grid


def classical_fit(mogi_model, album, data_mm):
    """Classical per-date params [T,2,4] and model maps [T,ny,nx] mm (Mogi pair + offset), rebuilt and dV-checked."""
    params = np.load(os.path.join(PROJECT_ROOT, CLASSICAL_ROOT, album, 'per_date_params.npy')).astype(np.float64)
    if params.shape[0] != data_mm.shape[0]:
        raise ValueError(f'{album}: classical has {params.shape[0]} dates, data {data_mm.shape[0]}')
    flat = data_mm.reshape(data_mm.shape[0], -1)
    model = np.full_like(flat, np.nan)
    for t in range(flat.shape[0]):
        if not np.isfinite(params[t]).all():
            continue                                  # non-finite classical date: blank map (scored as failure)
        if not np.isfinite(flat[t]).all():
            raise ValueError(f'{album} date {t}: masked/NaN pixels -- solve_linear needs the coherent-pixel subset')
        coeffs, residual = mogi_model.solve_linear(params[t, :, :3].reshape(-1), flat[t])
        # sanity: same linear solve as the fit itself -> must reproduce the saved dV
        if not np.allclose(coeffs[:2], params[t, :, 3], rtol=DV_CHECK_RTOL, atol=1.0):
            raise RuntimeError(f'{album} date {t}: rebuilt classical dV {coeffs[:2]} != saved {params[t, :, 3]}')
        model[t] = residual + flat[t]                 # residual = model - data
    return params, model.reshape(data_mm.shape)


def rms_mm(field_mm):
    """RMS over finite pixels (mm)."""
    return float(np.sqrt(np.nanmean(field_mm ** 2)))


def plot_album(album, truth, x_km, y_km, data_mm, methods, n_dates, out_png):
    """Input | model | residual maps per method, columns = evenly spaced dates, one shared colour scale."""
    n_epoch = data_mm.shape[0]
    dates = np.unique(np.linspace(0, n_epoch - 1, n_dates).round().astype(int))
    truth_params, _ = truth_arrays(truth)
    # one symmetric scale for every panel (residuals too): the 99.9th percentile of |input| on the shown dates
    vmax_mm = max(float(np.nanpercentile(np.abs(data_mm[dates]), 99.9)), 1.0)
    rows = [('input (what PILA and classical see)', None, 'input')]
    for key, method in methods.items():
        rows += [(f'{METHOD_STYLE[key][1]}: model', key, 'model'), (f'{METHOD_STYLE[key][1]}: residual', key, 'residual')]

    fig, axes = plt.subplots(len(rows), len(dates), figsize=(2.55 * len(dates) + 3.6, 2.45 * len(rows)),
                             squeeze=False, gridspec_kw={'hspace': 0.55, 'wspace': 0.12, 'top': 0.93, 'bottom': 0.04})
    for col, t in enumerate(dates):
        for row, (row_name, key, kind) in enumerate(rows):
            axis = axes[row, col]
            if kind == 'input':
                field, title = data_mm[t], f'input  RMS {rms_mm(data_mm[t]):.1f} mm'
            elif kind == 'model':
                field = methods[key]['model_mm'][t]
                title = f'model  RMS {rms_mm(field):.1f} mm'
            else:
                field = data_mm[t] - methods[key]['model_mm'][t]
                title = f'residual  RMS {rms_mm(field):.1f} mm'
                if 'model_lowrank_mm' in methods[key]:
                    title += f'\n(with low-rank: {rms_mm(data_mm[t] - methods[key]["model_lowrank_mm"][t]):.1f} mm)'
            # flag a panel the shared scale saturates (e.g. a misplaced source stronger than the input)
            if np.nanpercentile(np.abs(field), 99.9) > vmax_mm:
                title += f'\nSATURATED (peak {np.nanmax(np.abs(field)):.0f} mm)'
            image = axis.pcolormesh(x_km, y_km, field, cmap='RdBu_r', vmin=-vmax_mm, vmax=vmax_mm, shading='auto',
                                    rasterized=True)
            axis.set_aspect('equal')
            axis.set_title(f"{truth['dates'][t]}  {title}" if row == 0 else title, fontsize=7)
            axis.tick_params(labelsize=6)
            # true centres '+' on every panel; the method's own centres 'o' on its model + residual panels
            axis.plot(truth_params[t, :, 0], truth_params[t, :, 1], '+', color='k', ms=8, mew=1.4, ls='none')
            if key is not None:
                centres = methods[key]['params'][t]
                axis.plot(centres[:, 0], centres[:, 1], 'o', mfc='none', mec=METHOD_STYLE[key][0], ms=7, mew=1.4,
                          ls='none')
            if col == 0:
                axis.set_ylabel(f'{row_name}\nNorth (km)', fontsize=7)
            if row == len(rows) - 1:
                axis.set_xlabel('East (km)', fontsize=7)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.5, pad=0.015, location='right')
    colorbar.set_label('LOS displacement (mm, + toward satellite)', fontsize=9)
    # legend outside the panels, top right above the colourbar
    handles = [Line2D([], [], marker='+', color='k', ls='none', ms=8, mew=1.4, label='true source centre')]
    handles += [Line2D([], [], marker='o', mfc='none', mec=METHOD_STYLE[key][0], ls='none', ms=7, mew=1.4,
                       label=f'{METHOD_STYLE[key][1]} centre') for key in methods]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.995, 0.93), fontsize=8, framealpha=0.95)
    snr = truth['snr_peak_over_rms']
    seeds = ', '.join(f"{METHOD_STYLE[k][1]} seed {m['seed']}" for k, m in methods.items() if 'seed' in m)
    fig.suptitle(f"{album}   [{truth['scenario']}]   {'clean (no noise)' if snr is None else f'album SNR {snr:.1f}'}\n"
                 f'input / model / residual (input - model) per date; ONE colour scale for all panels (+-{vmax_mm:.0f} mm). '
                 f'PILA model = physics only (best-of-N: {seeds}); classical model = 2 Mogi + offset',
                 fontsize=10, y=0.975)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='figures/multisource_k2_nonstatic/param_recovery')
    parser.add_argument('--compare-dir', default='figures/multisource_k2_nonstatic')
    parser.add_argument('--jobs-dir', default='jobs/multisource_k2_nonstatic')
    parser.add_argument('--pila-root', default='saved/multisource_k2_nonstatic')
    parser.add_argument('--album', help='draw only this album')
    parser.add_argument('--n-dates', type=int, default=6, help='evenly spaced dates (columns) per figure')
    args = parser.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)

    print('[1/3] Loading truth, best-of-N picks and the album geometry...')
    truths = {}
    for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, ALBUMS_DIR, '*_truth.json'))):
        if '_nonoise' not in path:
            with open(path) as handle:
                truth = json.load(handle)
            truths[truth['album']] = truth
    best_path = os.path.join(PROJECT_ROOT, args.compare_dir, 'bestofN.csv')
    if not os.path.exists(best_path):
        raise FileNotFoundError(f'{best_path}: run compare_k2_nonstatic.py first')
    best = pd.read_csv(best_path)
    reused_path = os.path.join(PROJECT_ROOT, args.jobs_dir, 'reused_runs.json')
    reused = {}
    if os.path.exists(reused_path):
        with open(reused_path) as handle:
            reused = json.load(handle)
    geometry = load_ta69_geometry()
    mogi_model = floor.MogiK2(geometry)
    albums = [args.album] if args.album else sorted(truths)
    print(f'  {len(truths)} albums; drawing {len(albums)}')

    print('[2/3] Input / model / residual maps...')
    t0 = time.time()
    for album in albums:
        truth = truths[album]
        methods = {}
        album_path = os.path.join(PROJECT_ROOT, ALBUMS_DIR, f'{album}.npz')
        with np.load(album_path) as album_file:
            album_mm = album_file['los_asc_m'].astype(np.float64) * 1e3         # input shown = the album itself
        data_mm = album_mm
        for arm in ARMS:
            pick = best[(best['album'] == album) & (best['arm'] == arm)]
            if not len(pick):
                print(f'  MISSING best-of-N for {album} {arm}')
                continue
            seed = int(pick['best_seed'].iloc[0])
            fit = pila_fit(args.pila_root, reused, album, arm, seed, truth)
            # every arm must have seen the album's data; tolerance because the G1_S3_static n0250/n0500 slot_resid
            # runs are REUSED from the earlier K=2 study (albums_snr), equal to these albums to ~7e-6 mm
            max_diff_mm = float(np.nanmax(np.abs(fit['data_mm'] - album_mm)))
            if max_diff_mm > INPUT_ATOL_MM:
                raise RuntimeError(f'{album} {arm}: PILA input differs from the album by {max_diff_mm:.2e} mm')
            methods[arm] = dict(fit, seed=seed)
        if not methods:
            print(f'  {album}: no PILA fit, skipped')
            continue
        # plot on the fit's own grid; it must be the grid the classical forward model uses (same pixel order)
        x_km, y_km = methods[ARMS[0]]['x_km'], methods[ARMS[0]]['y_km']
        grid = geometry['grid']
        if not (np.allclose(np.asarray(grid['xE_m']).reshape(x_km.shape) / 1e3, x_km, atol=1e-3)
                and np.allclose(np.asarray(grid['yN_m']).reshape(y_km.shape) / 1e3, y_km, atol=1e-3)):
            raise RuntimeError(f'{album}: fit grid != TA69 geometry grid (classical model would be misplaced)')
        if os.path.exists(os.path.join(PROJECT_ROOT, CLASSICAL_ROOT, album, 'per_date_params.npy')):
            params, model_mm = classical_fit(mogi_model, album, data_mm)
            methods['classical'] = {'params': params, 'model_mm': model_mm}
        scenario_dir = os.path.join(out_dir, truth['scenario'])
        os.makedirs(scenario_dir, exist_ok=True)
        plot_album(album, truth, x_km, y_km, data_mm, methods, args.n_dates,
                   os.path.join(scenario_dir, f'{album}_recon_maps.png'))
    print(f'  {len(albums)} figures in {time.time() - t0:.1f}s')

    print('[3/3] Checks passed: every PILA input == album (1e-3 mm), fit grid == TA69 grid, classical dV rebuilt to the saved values.')
    print(f'Done. Output saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
