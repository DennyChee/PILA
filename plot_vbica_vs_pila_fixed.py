#!/usr/bin/env python
# Usage:       python plot_vbica_vs_pila_fixed.py [--tag marapi_mogi_move_loc] [--cell curv15000_dvl1u300_pathfix]
#                  [--out-dir figures/vbica_vs_pila_fixed]
# Description: PILA-vs-vbICA scenario figures redrawn with the DECODER-PATH-FIXED PILA runs (2026-10-22), with the
#              original PILA run overlaid so the change in parameter recovery is visible. Per cube: every truth
#              parameter vs epoch (truth black; vbICA -> one static SA+MCMC inversion red dashed + 95 % band;
#              classical SA+MCMC per epoch on the raw cube green; PILA FIXED orange solid; PILA ORIGINAL orange
#              dashed, faint) + the source-location-error panel. Plus one cross-cube summary figure and CSV.
#              Same layout/metrics as pila_bench/make_scenario_comparison.py, whose loaders are imported READ-ONLY;
#              nothing under the vbICA tree is written.
# Background:  the original campaign configs left arch.args.insar.timeseries on marapi_mogi_rising_combined.h5, so
#              for 11/12 cubes the PILA decoder standardised its output with the wrong mean/std (amplitude ~2x on
#              move_loc*). The *_pathfix cell is the identical recipe (curv15000 + dvl1u300, same box) with that fixed.
# Date:        2026-10-22

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = '/eos-rs/INSAR_processing/denny/matlab/vbICA/Code_denny/pila_bench'
RESULTS_DIR = os.path.join(BENCH_DIR, 'results')                 # read only
CLASSIC_ROOT = os.path.join(BENCH_DIR, 'classic_out')            # read only
ORIGINAL_CELL = 'curv15000_dvl1u300'

sys.path.insert(0, BENCH_DIR)
from score_vs_truth import read_mat73, load_truth, MODEL_PARAM_MAP, CLASSIC_ORDER, LOC_KEYS, AMP_KEY  # noqa: E402
from make_scenario_comparison import load_perepoch_series, perepoch_column, as_float                  # noqa: E402

COLOR_TRUTH, COLOR_VBICA, COLOR_PILA, COLOR_PEREP = 'k', 'tab:red', 'tab:orange', '#1e8449'


def pila_series(tag, cell):
    """Per-epoch PILA parameters (physical units: m, m^3, deg) of one campaign cell, or None if absent."""
    path = os.path.join(PROJECT_ROOT, 'saved', f'combined_curvdvl1u_{tag}', cell, 'uspace_series.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


def pila_scores(series, truth, model, strong):
    """Mean location error (km), depth nMAE (%), median amplitude ratio over strong epochs."""
    names = list(truth['param_names'])
    tru = np.asarray(truth['params_per_epoch'], dtype=float)
    kx, ky = LOC_KEYS[model]
    loc_err_km = np.hypot(series[kx].to_numpy(float) - tru[:, names.index(kx)],
                          series[ky].to_numpy(float) - tru[:, names.index(ky)]) / 1000.0
    depth_key = 'd' if 'd' in names else 'depth'
    true_depth = tru[strong, names.index(depth_key)]
    depth_nmae_pct = 100 * np.mean(np.abs(series[depth_key].to_numpy(float)[strong] - true_depth)) / np.mean(np.abs(true_depth))
    amp = AMP_KEY[model]
    amp_ratio = np.median(series[amp].to_numpy(float)[strong] / tru[strong, names.index(amp)])
    return {'loc_err_km': float(np.mean(loc_err_km[strong])), 'depth_nmae_pct': float(depth_nmae_pct),
            'amp_ratio': float(amp_ratio)}, loc_err_km


def plot_scenario(tag, h2h_row, cell, out_dir):
    """Truth / vbICA / per-epoch classical / PILA fixed (+ PILA original) for one cube. Returns score dict or None."""
    n_comp = int(h2h_row['vbica_N'])
    stem = f'{tag}_N{n_comp:02d}_wh'
    dvt_path = os.path.join(CLASSIC_ROOT, tag, stem, stem + '_dvt.mat')
    if not os.path.exists(dvt_path):
        print(f"  MISSING (skipped): {dvt_path}")
        return None
    vbica = read_mat73(dvt_path)
    truth = load_truth(tag)
    model = str(vbica['model']).strip()
    tru = np.asarray(truth['params_per_epoch'], dtype=float)
    tru_names, tru_units = list(truth['param_names']), list(truth['param_units'])
    rms_mm = np.asarray(truth['signal_rms_mm_per_epoch'], dtype=float)
    peak = int(truth['peak_epoch_index'])
    strong = rms_mm >= 0.5 * rms_mm.max()
    epochs = np.arange(len(rms_mm))

    # vbICA: one static geometry + amplitude change trace (centred data -> change since epoch 0)
    post_med = np.atleast_1d(np.asarray(vbica['post_med'], dtype=float)).ravel()
    post_lo = np.atleast_1d(np.asarray(vbica['post_lo'], dtype=float)).ravel()
    post_hi = np.atleast_1d(np.asarray(vbica['post_hi'], dtype=float)).ravel()
    vb_pinned = bool(np.any(np.atleast_1d(np.asarray(vbica['pinned'])).ravel().astype(bool)))
    k_best, ic_corr = int(vbica['k_best']), float(vbica['best_abs_corr'])
    order, pmap, amp_key = CLASSIC_ORDER[model], MODEL_PARAM_MAP[model], AMP_KEY[model]
    amp_route_a = np.atleast_1d(np.asarray(vbica['amp_t_route_a'], dtype=float)).ravel()
    amp_route_a = amp_route_a - amp_route_a[0]

    fixed, original = pila_series(tag, cell), pila_series(tag, ORIGINAL_CELL)
    if fixed is None:
        raise FileNotFoundError(f'no PILA run for {tag} cell {cell}')
    score_fixed, loc_fixed_km = pila_scores(fixed, truth, model, strong)
    score_orig, loc_orig_km = pila_scores(original, truth, model, strong) if original is not None else ({}, None)
    perep = load_perepoch_series(tag, stem, len(rms_mm))

    panel_names = [pmap[c] for c in order if pmap[c] in tru_names]
    n_panels, n_cols = len(panel_names) + 1, 3
    n_rows = int(np.ceil(n_panels / n_cols))
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'legend.fontsize': 8, 'lines.linewidth': 1.7})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.4 * n_rows), squeeze=False)

    for index, tname in enumerate(panel_names):
        axis = axes[index // n_cols][index % n_cols]
        j = tru_names.index(tname)
        if tname == amp_key:
            # amplitude comparable only as change since epoch 0 (vbICA works on centred data)
            axis.plot(epochs, tru[:, j] - tru[0, j], '-', color=COLOR_TRUTH, label='truth')
            axis.plot(epochs, amp_route_a, '--', color=COLOR_VBICA, label=f'vbICA IC{k_best} -> {model}')
            if original is not None:
                values = original[tname].to_numpy(float)
                axis.plot(epochs, values - values[0], '--', color=COLOR_PILA, alpha=0.55, lw=1.2,
                          label='PILA original (wrong decoder std)')
            values = fixed[tname].to_numpy(float)
            axis.plot(epochs, values - values[0], '-', color=COLOR_PILA, label='PILA fixed (per epoch)')
            route_c = perepoch_column(perep, order, pmap, tname) if perep is not None else None
            if route_c is not None and np.isfinite(route_c).any():
                axis.plot(epochs, route_c - route_c[np.flatnonzero(np.isfinite(route_c))[0]], '-', color=COLOR_PEREP,
                          label='classical per-epoch, raw cube')
            axis.set_ylabel(f'change in {tname} ({tru_units[j]})')
            axis.set_title(f'{tname} -- change since epoch 0')
        else:
            axis.plot(epochs, tru[:, j], '-', color=COLOR_TRUTH, label='truth')
            i_par = order.index(next(c for c in order if pmap[c] == tname))
            axis.axhline(post_med[i_par], ls='--', color=COLOR_VBICA, label=f'vbICA IC{k_best} (static)')
            axis.axhspan(post_lo[i_par], post_hi[i_par], color=COLOR_VBICA, alpha=0.12, label='vbICA 95% posterior')
            if original is not None:
                axis.plot(epochs, original[tname].to_numpy(float), '--', color=COLOR_PILA, alpha=0.55, lw=1.2,
                          label='PILA original (wrong decoder std)')
            axis.plot(epochs, fixed[tname].to_numpy(float), '-', color=COLOR_PILA, label='PILA fixed (per epoch)')
            route_c = perepoch_column(perep, order, pmap, tname) if perep is not None else None
            if route_c is not None:
                axis.plot(epochs, route_c, '-', color=COLOR_PEREP, label='classical per-epoch, raw cube')
                if perep['pinned'].any():
                    axis.plot(epochs[perep['pinned']], route_c[perep['pinned']], 'x', color=COLOR_PEREP, ms=5,
                              ls='none', label='per-epoch: pinned on bound')
            axis.set_ylabel(f'{tname} ({tru_units[j]})')
            axis.set_title(tname)
        axis.fill_between(epochs, *axis.get_ylim(), where=strong, color='0.88', zorder=0,
                          label='strong-signal epochs' if index == 0 else None)
        axis.axvline(peak, color='0.4', lw=0.8, ls=':')
        axis.set_xlabel('epoch index')
        if index == 0:
            axis.legend(loc='best', framealpha=0.9)

    # --- Location-error panel ---
    axis = axes[(n_panels - 1) // n_cols][(n_panels - 1) % n_cols]
    kx, ky = LOC_KEYS[model]
    jx, jy = tru_names.index(kx), tru_names.index(ky)
    vb_err_km = np.hypot(post_med[order.index('xoff_m')] - tru[:, jx], post_med[order.index('yoff_m')] - tru[:, jy]) / 1000.0
    axis.plot(epochs, vb_err_km, '--', color=COLOR_VBICA, label=f'vbICA IC{k_best} (static)')
    if loc_orig_km is not None:
        axis.plot(epochs, loc_orig_km, '--', color=COLOR_PILA, alpha=0.55, lw=1.2, label='PILA original')
    axis.plot(epochs, loc_fixed_km, '-', color=COLOR_PILA, label='PILA fixed')
    if perep is not None:
        pe_x = perepoch_column(perep, order, pmap, kx, code='xoff_m')
        pe_y = perepoch_column(perep, order, pmap, ky, code='yoff_m')
        if pe_x is not None and pe_y is not None:
            axis.plot(epochs, np.hypot(pe_x - tru[:, jx], pe_y - tru[:, jy]) / 1000.0, '-', color=COLOR_PEREP,
                      label='classical per-epoch, raw cube')
    floor = as_float(h2h_row.get('static_floor_km'))
    if floor is not None:
        axis.axhline(floor, color='0.45', ls='-.', label=f'static floor = {floor:.3f} km')
    axis.fill_between(epochs, *axis.get_ylim(), where=strong, color='0.88', zorder=0)
    axis.axvline(peak, color='0.4', lw=0.8, ls=':')
    axis.set_title('source location error')
    axis.set_xlabel('epoch index')
    axis.set_ylabel('location error (km)')
    axis.legend(loc='best', framealpha=0.9)
    for k in range(n_panels, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis('off')

    fig.suptitle(f'{tag}   [{model}]\nvbICA N={n_comp}, whitening ON -- inverted IC{k_best} (|corr| {ic_corr:.3f})   vs   '
                 f'PILA FIXED ({cell}; original run dashed)', fontsize=12)
    vb_loc = as_float(h2h_row.get('vbica_loc_err_km'))
    pe_loc = as_float(h2h_row.get('perepoch_loc_err_km'))
    footer = (f"loc err (mean, strong epochs): vbICA {vb_loc:.3f} km | static floor {floor:.3f} km | "
              f"classical per-epoch {pe_loc:.3f} km | PILA original {score_orig.get('loc_err_km', np.nan):.3f} km -> "
              f"PILA fixed {score_fixed['loc_err_km']:.3f} km"
              f"\nPILA amplitude/true (median): original {score_orig.get('amp_ratio', np.nan):.2f} -> fixed "
              f"{score_fixed['amp_ratio']:.2f}   |   PILA depth error: original {score_orig.get('depth_nmae_pct', np.nan):.1f} % "
              f"-> fixed {score_fixed['depth_nmae_pct']:.1f} %"
              "\nBoth untuned: PILA = curv15000 + dvl1u300 (same recipe; fixed = decoder standardised with THIS cube's "
              "mean/std), vbICA = N5, whitening ON. vbICA is static by construction: compare it to the floor, PILA to zero.")
    if vb_pinned:
        footer += '\nWARNING: a vbICA parameter is PINNED ON A SEARCH BOUND -- not a result.'
    fig.text(0.5, 0.012, footer, ha='center', va='bottom', fontsize=9, color='0.15')
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    out_png = os.path.join(out_dir, f'{tag}_vbICA_IC{k_best}_of_N{n_comp:02d}_vs_PILA_fixed.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Wrote {os.path.relpath(out_png, PROJECT_ROOT)}")
    return {'tag': tag, 'model': model, 'vbica_loc_err_km': vb_loc, 'static_floor_km': floor,
            'perepoch_loc_err_km': pe_loc,
            **{f'pila_original_{k}': v for k, v in score_orig.items()}, **{f'pila_fixed_{k}': v for k, v in score_fixed.items()}}


def plot_summary(table, out_png):
    """Cross-cube bars: location error (all routes), PILA amplitude ratio and depth error (original vs fixed)."""
    short = [tag.replace('marapi_', '').replace('_combined', '') for tag in table['tag']]
    positions = np.arange(len(table))
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    axis = axes[0]
    width = 0.2
    for offset, (column, label, colour, hatch) in enumerate((
            ('vbica_loc_err_km', 'vbICA (static)', COLOR_VBICA, None),
            ('perepoch_loc_err_km', 'classical per-epoch', COLOR_PEREP, None),
            ('pila_original_loc_err_km', 'PILA original', COLOR_PILA, '//'),
            ('pila_fixed_loc_err_km', 'PILA fixed', COLOR_PILA, None))):
        axis.bar(positions + (offset - 1.5) * width, table[column], width=width, color=colour, hatch=hatch,
                 alpha=0.5 if hatch else 1.0, edgecolor='k' if hatch else colour, label=label)
    axis.scatter(positions, table['static_floor_km'].clip(lower=1e-3), marker='_', s=500, color='k', zorder=3,
                 label='static floor')
    axis.set_yscale('log')
    axis.set_ylabel('Location error (km)\n(mean, strong epochs)')
    axis.legend(ncol=5, fontsize=9)
    for axis, key, ylabel in ((axes[1], 'amp_ratio', 'PILA amplitude / true\n(median, strong epochs)'),
                              (axes[2], 'depth_nmae_pct', 'PILA depth error (%)\n(mean, strong epochs)')):
        axis.bar(positions - 0.17, table[f'pila_original_{key}'], width=0.34, color=COLOR_PILA, alpha=0.5, hatch='//',
                 edgecolor='k', label='PILA original')
        axis.bar(positions + 0.17, table[f'pila_fixed_{key}'], width=0.34, color=COLOR_PILA, label='PILA fixed')
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=9)
    axes[1].axhline(1.0, color='k', lw=1)
    axes[2].set_yscale('log')
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(short, rotation=35, ha='right')
    fig.suptitle('PILA vs vbICA with the decoder-path-fixed PILA runs (PILA original hatched)')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', help='build only this cube')
    parser.add_argument('--cell', default='curv15000_dvl1u300_pathfix')
    parser.add_argument('--out-dir', default='figures/vbica_vs_pila_fixed')
    args = parser.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('[1/3] Reading the vbICA head-to-head table (read only)...')
    h2h = pd.read_csv(os.path.join(RESULTS_DIR, 'headtohead.csv'))
    rows = h2h[h2h['tag'] == args.tag] if args.tag else h2h
    print(f'  Loaded: {h2h.shape}; building {len(rows)} cube(s)')

    print(f'[2/3] Scenario figures (PILA cell {args.cell})...')
    scores = [s for s in (plot_scenario(row['tag'], row, args.cell, out_dir) for _, row in rows.iterrows()) if s]
    table = pd.DataFrame(scores)
    table.to_csv(os.path.join(out_dir, 'vbica_vs_pila_fixed_summary.csv'), index=False, float_format='%.4g')

    print('[3/3] Cross-cube summary figure...')
    if len(table) > 1:
        plot_summary(table, os.path.join(out_dir, 'vbica_vs_pila_fixed_summary.png'))
    print(table[['tag', 'vbica_loc_err_km', 'static_floor_km', 'perepoch_loc_err_km', 'pila_original_loc_err_km',
                 'pila_fixed_loc_err_km', 'pila_original_amp_ratio', 'pila_fixed_amp_ratio']].round(3).to_string(index=False))
    print(f"Done. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
