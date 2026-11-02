#!/usr/bin/env python
# Usage:       python plot_k2_nonstatic_param_recovery.py [--out-dir figures/multisource_k2_nonstatic/param_recovery]
#                  [--compare-dir figures/multisource_k2_nonstatic] [--pila-root saved/multisource_k2_nonstatic]
#                  [--album G1_N1_staggered_n0250]
#              (also called automatically at the end of compare_k2_nonstatic.py)
# Description: Parameter-recovery / truth / method-comparison figures for EVERY album of the non-static K=2 campaign,
#              in the same layout as plot_vbica_vs_pila_fixed.py: per true source, every parameter vs epoch
#              (truth black; PILA best-of-N slot_resid orange; PILA best-of-N slot_resid_dvl1u300 purple; classical
#              per-date OLS green; x = pinned on the search box) + a source-location-error panel, grey shading on the
#              dates where that source is ACTIVE (scorer rule), footer with the scorer's metrics. One folder per
#              scenario, plus one cross-noise summary figure per scenario and a CSV.
#              Reads compare_k2_nonstatic.py outputs (bestofN.csv, classical.csv) for the best-of-N seed picks
#              (truth-free: lowest deterministic album misfit) and the metrics, so it never re-scores anything.
#              Fitted slots are matched to the true sources per date (Hungarian on horizontal position, same rule
#              as the scorer), so a slot-label swap between dates is NOT shown as an error.
#              Each album's data-QA sheet (deformation maps) is copied alongside as <album>_data_qa.png.
#              Units: km (position, depth, location error), m^3 (dV).
# Date:        2026-10-27

import argparse
import glob
import json
import os
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from compare_k2_nonstatic import ALBUMS_DIR, ARMS, CLASSICAL_ROOT, SCENARIOS, check_dates, find_models_dir, matched_fit
from synthetic.multisource.nonstatic_scoring import LOC_TOL_KM, active_threshold_mm, truth_arrays

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Data-QA sheets of the v2 albums (truth trajectories + noise-free / noisy LOS maps), copied next to each recovery
# figure as a visual guide to the deformation being inverted (qa_nonstatic_albums.py)
QA_SHEETS_DIR = 'figures/multisource_k2_nonstatic/data_qa_v2/sheets'
COLOR_TRUTH, COLOR_PEREP = 'k', '#1e8449'                                     # same colours as the vbICA figures
ARM_STYLE = {'slot_resid': ('tab:orange', 'PILA slot_resid'),
             'slot_resid_dvl1u300': ('tab:purple', 'PILA slot_resid + dV-L1(u) 300'),
             'slot_resid_curv5000': ('tab:blue', 'PILA slot_resid + curv 5000'),
             'slot_resid_curv15000_dvl1u300': ('tab:red', 'PILA slot_resid + curv 15000 + dV-L1(u) 300')}
PARAM_PANELS = [(0, 'xcen', 'km'), (1, 'ycen', 'km'), (2, 'depth', 'km'), (3, 'dV', 'm$^3$')]
# Search box shared by PILA's prior (configs/mogi_paras_symdV.json) and the classical solver (classical_k2_perdate.py)
BOX = {0: (-15.0, 15.0), 1: (-15.0, 15.0), 2: (0.5, 12.0)}
PIN_FRAC = 0.01                                                               # within 1 % of box width = pinned


LOC_FLOOR_KM = 0.01                                                           # log-axis floor for location error


def clip_to_truth(axis, truth_values, series, src, index):
    """Zoom dV panel on the truth range; draw off-scale fitted values as triangles on the panel edge.

    Without this one classical outlier (~1e10 m^3 on a near-noise date) flattens truth to a line at zero.
    """
    span = truth_values.max() - truth_values.min()
    pad = max(0.5 * span, 0.5 * np.abs(truth_values).max(), 1.0)
    low, high = truth_values.min() - pad, truth_values.max() + pad
    axis.set_ylim(low, high)
    epochs = np.arange(len(truth_values))
    for _, color, matched, _ in series:
        values = matched[:, src, index]
        for side, marker, mask in ((high, '^', values > high), (low, 'v', values < low)):
            if mask.any():
                axis.plot(epochs[mask], np.full(mask.sum(), side), marker, color=color, ms=6, ls='none',
                          clip_on=False)


def pinned_mask(matched):
    """[T, 2] True where any of x / y / depth of a matched slot sits on the search-box edge."""
    mask = np.zeros(matched.shape[:2], dtype=bool)
    for index, (low, high) in BOX.items():
        tol = PIN_FRAC * (high - low)
        values = matched[:, :, index]
        mask |= (np.abs(values - low) < tol) | (np.abs(values - high) < tol)
    return mask


def load_pila_fit(pila_root, reused, album, arm, seed, truth):
    """basin_best per-date params [T, 2, 4] (km, km, km, m^3) of one PILA run."""
    if arm == 'slot_resid' and album in reused:
        run_root = os.path.join(PROJECT_ROOT, reused[album][str(seed)])
    else:
        run_root = os.path.join(PROJECT_ROOT, pila_root, f'k2ns_{album}_{arm}_seed{seed}')
    models = find_models_dir(run_root)
    if models is None:
        raise FileNotFoundError(f'{album} {arm} seed {seed}: no models dir under {run_root}')
    with np.load(os.path.join(models, 'basin_best', 'fit_K2.npz')) as fit:
        check_dates(fit['dates'], truth)                 # guard: fit and truth on the same dates in the same order
        return fit['params_phys'].astype(np.float64)


def loc_err_km(matched, truth_params):
    """[T, 2] horizontal location error (km) of each matched slot."""
    return np.hypot(matched[:, :, 0] - truth_params[:, :, 0], matched[:, :, 1] - truth_params[:, :, 1])


def metric_text(row):
    """One-line scorer summary of one method on one album."""
    if row is None:
        return 'missing'
    seed = f" (seed {int(row['best_seed'])}, {int(row['n_run_success'])}/{int(row['n_runs'])} runs pass)" \
        if 'best_seed' in row and pd.notna(row.get('best_seed')) else ''
    return (f"{int(row['n_success'])}/{int(row['n_active'])} active source-dates correct "
            f"({100 * row['success_frac']:.0f} %, run {'PASS' if row['run_success'] else 'FAIL'}){seed}; "
            f"median loc err {row['median_loc_err_km']:.2f} km, |depth err| {row['median_abs_depth_err_pct']:.0f} %, "
            f"|dV err| {row['median_abs_dV_err_pct']:.0f} %; false detections {int(row['n_false_detection'])}")


def plot_album(album, truth, series, metrics, out_png):
    """Truth vs best-of-N PILA (both arms) vs classical per date for both true sources of one album."""
    truth_params, peaks_mm = truth_arrays(truth)
    active = peaks_mm >= active_threshold_mm(truth)                           # [T, 2] scorer's ACTIVE rule
    epochs = np.arange(truth_params.shape[0])
    n_cols = len(PARAM_PANELS) + 1
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'legend.fontsize': 8, 'lines.linewidth': 1.7})
    fig, axes = plt.subplots(2, n_cols, figsize=(4.6 * n_cols + 4.5, 7.6), squeeze=False)   # +4.5 in: legend

    for src in range(2):
        # --- Parameter panels: truth vs every method ---
        for col, (index, name, unit) in enumerate(PARAM_PANELS):
            axis = axes[src][col]
            for label, color, matched, pinned in series:
                axis.plot(epochs, matched[:, src, index], '-', color=color, label=label)
                if pinned[:, src].any():
                    axis.plot(epochs[pinned[:, src]], matched[pinned[:, src], src, index], 'x', color=color, ms=5,
                              ls='none', label=f'{label}: pinned on box' if col == 0 else None)
            # truth last + on top + dashed: an exact fit (e.g. classical on clean albums) must not hide it
            axis.plot(epochs, truth_params[:, src, index], '--', color=COLOR_TRUTH, lw=2.2, zorder=10, label='truth')
            if index == 3:
                axis.axhline(0.0, color='0.6', lw=0.8)
                clip_to_truth(axis, truth_params[:, src, index], series, src, index)
            axis.set_title(f'source {src}: {name}')
            axis.set_ylabel(f'{name} ({unit})')
        # --- Location-error panel (tolerance line = scorer's success limit) ---
        axis = axes[src][n_cols - 1]
        for label, color, matched, _ in series:
            # floor so an exact fit (clean albums, ~1e-9 km) stays on the log axis (drawn just above the bottom edge)
            axis.plot(epochs, np.maximum(loc_err_km(matched, truth_params)[:, src], LOC_FLOOR_KM), '-', color=color,
                      label=label)
        axis.axhline(LOC_TOL_KM, color='0.45', ls='-.', label=f'success limit {LOC_TOL_KM} km')
        axis.set_yscale('log')                                                # 0.01-30 km spans 3+ decades
        axis.set_ylim(0.5 * LOC_FLOOR_KM, 50.0)
        axis.set_title(f'source {src}: location error')
        axis.set_ylabel('location error (km)')
        for col in range(n_cols):
            axis = axes[src][col]
            limits = axis.get_ylim()
            axis.fill_between(epochs, *limits, where=active[:, src], color='0.88', zorder=0,
                              label='source active (detectable)' if col == 0 else None)
            axis.set_ylim(*limits)                                            # shading must not re-autoscale
            axis.set_xlabel('epoch index')

    # --- Sanity check: on every parameter panel the truth line exists, is finite and lies inside the y-limits ---
    problems = []
    for src in range(2):
        for col, (index, name, _) in enumerate(PARAM_PANELS):
            axis = axes[src][col]
            truth_lines = [line for line in axis.get_lines() if line.get_label() == 'truth']
            low, high = axis.get_ylim()
            values = truth_params[:, src, index]
            if len(truth_lines) != 1 or not np.isfinite(values).all() or values.min() < low or values.max() > high:
                problems.append(f'source {src} {name}')
    if problems:
        raise RuntimeError(f'{album}: truth line missing or outside axis on: {", ".join(problems)}')

    # --- One legend for the whole figure, outside the panels (no panel is covered) ---
    handles, labels = {}, []
    for axis in axes.ravel():
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in handles:
                handles[label] = handle
                labels.append(label)
    if any(line.get_marker() in ('^', 'v') for axis in axes.ravel() for line in axis.get_lines()):
        handles['fitted dV off scale (triangle on panel edge)'] = Line2D([], [], marker='^', ls='none', color='0.4')
        labels.append('fitted dV off scale (triangle on panel edge)')
    fig.legend([handles[label] for label in labels], labels, loc='center left', bbox_to_anchor=(0.835, 0.55),
               fontsize=9, framealpha=0.95)

    snr = truth['snr_peak_over_rms']
    snr_text = 'clean (no noise)' if snr is None else f'album SNR {snr:.1f}'
    fig.suptitle(f"{album}   [K=2 Mogi, {truth['scenario']}]   {snr_text}, noise x{truth['noise_scale']} E1\n"
                 'truth vs PILA best-of-N (lowest album misfit, truth-free pick) vs classical per-date OLS; '
                 'fitted slots matched to true sources per date', fontsize=12)
    footer = '\n'.join(f"{name}: {metric_text(metrics.get(key))}" for key, name in
                       [(arm, ARM_STYLE[arm][1]) for arm in ARMS] + [('classical', 'classical per-date')])
    footer += ('\nSuccess per active source-date = loc err < 1.5 km, |depth err| < 50 %, |dV err| < 50 %; '
               'run PASS = >= 80 % of active source-dates. Grey = dates the source is detectable (own peak >= '
               f'{active_threshold_mm(truth):.1f} mm).')
    fig.text(0.5, 0.01, footer, ha='center', va='bottom', fontsize=9, color='0.15')
    fig.tight_layout(rect=[0, 0.11, 0.83, 0.92])                          # right 17 % kept for the legend
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_scenario_summary(table, scenario, out_png):
    """Per scenario: success fraction and median parameter errors vs album SNR, G1 layout, every method."""
    rows = table[(table['scenario'] == scenario) & (table['layout_set'] == 'G1')].copy()
    rows['snr_plot'] = rows['album_snr'].fillna(60.0)                         # clean album plotted at SNR 60
    # floors keep the exact classical fit on clean albums (~1e-8) from squashing the log axes
    panels = [('success_frac', 'Fraction of active\nsource-dates correct', None),
              ('median_loc_err_km', 'Median location\nerror (km)', LOC_FLOOR_KM),
              ('median_abs_depth_err_pct', 'Median |depth error| (%)', 0.1),
              ('median_abs_dV_err_pct', 'Median |dV error| (%)', 0.1)]
    fig, axes = plt.subplots(1, len(panels), figsize=(23, 4.4))                # +3 in: legend outside
    for axis, (column, ylabel, floor) in zip(axes, panels):
        log_y = floor is not None
        for method, color, label in [(arm, *ARM_STYLE[arm]) for arm in ARMS] + [('classical', COLOR_PEREP, 'classical per-date')]:
            subset = rows[rows['method'] == method].sort_values('snr_plot')
            values = subset[column].clip(lower=floor) if log_y else subset[column]
            axis.plot(subset['snr_plot'], values, 'o-', color=color, label=label)
        if log_y:
            axis.axhline(floor, color='0.7', lw=0.8, ls=':')                  # values at the floor = exact fit
        if column == 'success_frac':
            axis.axhline(0.8, color='0.45', ls=':', label='run-success bar')
            axis.set_ylim(-0.05, 1.05)
        if log_y:
            axis.set_yscale('log')
        axis.set_xscale('log')
        axis.set_xlabel('Album SNR (peak / noise RMS; clean at 60)')
        axis.set_ylabel(ylabel)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='center left', bbox_to_anchor=(0.865, 0.5), fontsize=9)
    fig.suptitle(f'G1 {scenario}: parameter recovery vs SNR (PILA = best-of-N picks)')
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='figures/multisource_k2_nonstatic/param_recovery')
    parser.add_argument('--compare-dir', default='figures/multisource_k2_nonstatic',
                        help='where compare_k2_nonstatic.py wrote bestofN.csv / classical.csv')
    parser.add_argument('--jobs-dir', default='jobs/multisource_k2_nonstatic')
    parser.add_argument('--pila-root', default='saved/multisource_k2_nonstatic')
    parser.add_argument('--album', help='draw only this album')
    args = parser.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    compare_dir = os.path.join(PROJECT_ROOT, args.compare_dir)

    print('[1/4] Loading truth and compare_k2_nonstatic.py tables...')
    truths = {}
    for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, ALBUMS_DIR, '*_truth.json'))):
        if '_nonoise' not in path:
            with open(path) as handle:
                truth = json.load(handle)
            truths[truth['album']] = truth
    for name in ('bestofN.csv', 'classical.csv'):
        if not os.path.exists(os.path.join(compare_dir, name)):
            raise FileNotFoundError(f'{name} not in {compare_dir}: run compare_k2_nonstatic.py first')
    best = pd.read_csv(os.path.join(compare_dir, 'bestofN.csv'))
    classical = pd.read_csv(os.path.join(compare_dir, 'classical.csv'))
    reused_path = os.path.join(PROJECT_ROOT, args.jobs_dir, 'reused_runs.json')
    reused = {}
    if os.path.exists(reused_path):
        with open(reused_path) as handle:
            reused = json.load(handle)
    albums = [args.album] if args.album else sorted(truths)
    print(f'  {len(truths)} albums; best-of-N {best.shape}, classical {classical.shape}; drawing {len(albums)}')

    print('[2/4] Per-album parameter-recovery figures...')
    t0 = time.time()
    summary_rows = []
    for album in albums:
        truth = truths[album]
        truth_params, _ = truth_arrays(truth)
        series, metrics = [], {}
        for arm in ARMS:
            pick = best[(best['album'] == album) & (best['arm'] == arm)]
            if not len(pick):
                print(f'  MISSING best-of-N for {album} {arm} (skipped in figure)')
                continue
            row = pick.iloc[0]
            matched = matched_fit(load_pila_fit(args.pila_root, reused, album, arm, int(row["best_seed"]), truth),
                                  truth)
            color, label = ARM_STYLE[arm]
            series.append((f"{label} (seed {int(row['best_seed'])})", color, matched, pinned_mask(matched)))
            metrics[arm] = row
        classical_path = os.path.join(PROJECT_ROOT, CLASSICAL_ROOT, album, 'per_date_params.npy')
        crow = classical[classical['album'] == album]
        if os.path.exists(classical_path) and len(crow):
            matched = matched_fit(np.load(classical_path).astype(np.float64), truth)
            series.append(('classical per-date', COLOR_PEREP, matched, pinned_mask(matched)))
            metrics['classical'] = crow.iloc[0]
        scenario_dir = os.path.join(out_dir, truth['scenario'])
        os.makedirs(scenario_dir, exist_ok=True)
        plot_album(album, truth, series, metrics, os.path.join(scenario_dir, f'{album}_param_recovery.png'))
        qa_sheet = os.path.join(PROJECT_ROOT, QA_SHEETS_DIR, f'{album}.png')
        if os.path.exists(qa_sheet):
            shutil.copy2(qa_sheet, os.path.join(scenario_dir, f'{album}_data_qa.png'))   # copy: QA sheet stays put
        else:
            print(f'  MISSING data-QA sheet {qa_sheet}')
        for method, row in metrics.items():
            summary_rows.append({'album': album, 'scenario': truth['scenario'], 'layout_set': truth['layout_set'],
                                 'noise_scale': truth['noise_scale'], 'album_snr': truth['snr_peak_over_rms'],
                                 'method': method, **{k: row[k] for k in ('success_frac', 'run_success', 'n_active',
                                 'median_loc_err_km', 'median_abs_depth_err_pct', 'median_abs_dV_err_pct',
                                 'n_false_detection')}})
    print(f'  {len(albums)} figures in {time.time() - t0:.1f}s')

    print('[3/4] Summary table...')
    table = pd.DataFrame(summary_rows)
    if not args.album:
        table.to_csv(os.path.join(out_dir, 'param_recovery_summary.csv'), index=False, float_format='%.4g')

    print('[4/4] Per-scenario summary figures (G1 layout)...')
    if not args.album:
        for scenario in SCENARIOS:
            plot_scenario_summary(table, scenario, os.path.join(out_dir, scenario, f'G1_{scenario}_summary.png'))
    print(f'Done. Output saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
