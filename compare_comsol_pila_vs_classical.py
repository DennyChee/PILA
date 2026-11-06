#!/usr/bin/env python
# Usage:       conda activate pila
#              python compare_comsol_pila_vs_classical.py [--saved-root saved/comsol_wk6]
#                  [--classic classic_out_comsol] [--exported exported_comsol]
#                  [--truth data/comsol/wk6_chambers_inversion_20260928/metadata/wk6_truth.csv]
#                  [--outdir figures/comsol_wk6/classical] [--overwrite]
# Description: Compare PILA with the classical SA + Bayesian MCMC Mogi inversion
#              (iclassic_insar.m) on the COMSOL wk6 catalogue, both against the COMSOL truth.
#              One long-format CSV row per (chamber, surface, track, method), a grouped summary
#              CSV, and PNG figures: errors vs elongation, horizontal offset by track (incl. the
#              joint asc+desc classical fit), PILA vs MCMC 1:1, physics-only R^2, runtime.
# Date:        2026-10-02
#
# Scientific conventions (READ):
#   * Truth / error definitions are collect_comsol_results.add_truth_and_errors (reused):
#     depth truth = centroid depth, dV truth = cavity dV, horizontal truth = (0, 0);
#     error sign + = recovered too deep / too large.
#   * Classical estimate = MCMC posterior median (post_med). SA (m_best_sa) is only the
#     MCMC warm start (pilot: under-converged, R^2 0.97) and is kept as a diagnostic row.
#   * The maps are noise-free and the likelihood sigma is std(LOS): MCMC 2.5-97.5 % bounds
#     are written to the CSV but are NOT noise-based uncertainties (and the joint pooled
#     sigma makes joint widths incomparable to single-track ones). Compare point estimates.
#   * Physics-only R^2 / RMSE: every parameter set is run through PILA's own Mogi forward
#     model on the exact exported points (PILA's residual branch excluded) -> like-for-like.
#   * Because the true source is NOT a point source, a converged MCMC fit is the best-fit
#     Mogi: (MCMC - truth) = Mogi model bias, (PILA - MCMC) = PILA inversion error.
#   * Runtime: PILA = per-chamber training train_sec (compute_usage.json; NOT amortised:
#     one network per scene here); classical = SA + MCMC wall time. Both CPU.
#   * Figures are PNG only (user preference).

import argparse
import glob
import json
import os
import re
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from collect_comsol_results import add_truth_and_errors, collect_runs
from compare_pila_vs_classical import evaluate_los_mm, r2_rmse

# Methods in fixed categorical order (reference palette slots 1-3) + marker as 2nd encoding.
METHOD_STYLE = {
    'PILA':       dict(color='#2a78d6', marker='o', label='PILA (single track)', offset=-0.18),
    'MCMC':       dict(color='#eb6834', marker='s', label='Classical MCMC (single track)', offset=0.0),
    'MCMC_joint': dict(color='#1baf7a', marker='D', label='Classical MCMC (joint asc+desc)', offset=0.18),
}
TRACK_STYLE = {'asc': dict(color='#2a78d6', marker='^', label='asc', offset=-0.18),
               'desc': dict(color='#eb6834', marker='v', label='desc', offset=0.0),
               'joint': dict(color='#1baf7a', marker='D', label='joint asc+desc', offset=0.18)}
RESULT_NAME_PATTERN = re.compile(r'mogi_(comsol_wk6_cid(\d+)_(flat|cone)_(asc|desc|joint)_mogi)_\d{8}_result')
RAIL_FRACTION = 0.01          # a value within 1 % of the box width of a bound counts as railed
INK, MUTED, GRID = '#1f1f1e', '#52514e', '#e5e5e2'


def pila_runtime_sec(params_file):
    """PILA training wall time (s) from the run's models/compute_usage.json, NaN if absent."""
    run_dir = os.path.dirname(os.path.dirname(params_file))          # .../<ts>/figures/x -> <ts>
    usage_path = os.path.join(run_dir, 'models', 'compute_usage.json')
    if not os.path.exists(usage_path):
        return np.nan
    with open(usage_path) as fh:
        return float(json.load(fh).get('train_sec', np.nan))


def load_classical_rows(classic_root):
    """
    One row per classical result .mat and estimator (MCMC posterior median, SA best fit).

    Output DataFrame columns: cid, surface, track, config, method, xcen_km, ycen_km, d_km,
    dV_m3, ci_lo_* / ci_hi_* (MCMC only), railed, runtime_s, result_file.
    """
    rows = []
    # Keep ONE result per config (newest by mtime, the collect_comsol_results rule): a rerun
    # left beside an old file would otherwise duplicate scenes and bias every median.
    newest_by_config = {}
    for result_file in glob.glob(os.path.join(classic_root, '*', '*', 'mogi_*_result.mat')):
        config_dir = os.path.dirname(result_file)
        if (config_dir not in newest_by_config
                or os.path.getmtime(result_file) > os.path.getmtime(newest_by_config[config_dir])):
            newest_by_config[config_dir] = result_file
    for result_file in sorted(newest_by_config.values()):
        match = RESULT_NAME_PATTERN.fullmatch(os.path.splitext(os.path.basename(result_file))[0])
        if match is None:
            print(f"  skipping unexpected file {result_file}")
            continue
        config, cid, surface, track = match[1], int(match[2]), match[3], match[4]
        # iclassic saves -v7.3 (HDF5): h5py, 1-D vectors reshape back into param order
        # (xoff_m, yoff_m, depth_m, volume_m3).
        with h5py.File(result_file, 'r') as result:
            post_med = np.array(result['post_med']).reshape(-1)
            post_lo = np.array(result['post_lo']).reshape(-1)
            post_hi = np.array(result['post_hi']).reshape(-1)
            m_best_sa = np.array(result['m_best_sa']).reshape(-1)
            bounds = np.array(result['bounds']).T                     # back to [2 x n_par]
            runtime_sa_s = float(np.array(result['runtime_sa_sec']).squeeze())
            runtime_total_s = float(np.array(result['runtime_total_sec']).squeeze())
        box_width = bounds[1] - bounds[0]
        for method, vector, runtime_s in (('MCMC', post_med, runtime_total_s),
                                          ('SA', m_best_sa, runtime_sa_s)):
            railed = bool(np.any((vector - bounds[0] < RAIL_FRACTION * box_width)
                                 | (bounds[1] - vector < RAIL_FRACTION * box_width)))
            row = dict(cid=cid, surface=surface, track=track, config=config,
                       method=method if track != 'joint' else f'{method}_joint',
                       xcen_km=vector[0] / 1000.0, ycen_km=vector[1] / 1000.0,
                       d_km=vector[2] / 1000.0, dV_m3=vector[3], railed=railed,
                       runtime_s=runtime_s, result_file=result_file)
            if method == 'MCMC':
                # Written for completeness only -- see the header: NOT noise-based intervals.
                row.update(ci_lo_d_km=post_lo[2] / 1000.0, ci_hi_d_km=post_hi[2] / 1000.0,
                           ci_lo_dV_m3=post_lo[3], ci_hi_dV_m3=post_hi[3])
            rows.append(row)
    return pd.DataFrame(rows)


def add_physics_fit(table, exported_root):
    """Physics-only R^2 / RMSE (mm) of every row's Mogi parameters on its exported points."""
    r2_values, rmse_values = [], []
    cached_path, points = None, None                     # one export in RAM at a time
    for row in table.itertuples():
        sub_dir = 'joint' if row.track == 'joint' else 'single'
        export_path = os.path.join(exported_root, sub_dir, f'{row.config}.mat')
        if export_path != cached_path:
            if not os.path.exists(export_path):
                raise FileNotFoundError(f"{export_path} (run export_comsol_points.py)")
            points = {key: value.ravel() for key, value in loadmat(export_path).items()
                      if key in ('x_east_m', 'y_north_m', 'los_obs_m', 'losE', 'losN', 'losU')}
            cached_path = export_path
        params_si = dict(xcen=row.xcen_km * 1000.0, ycen=row.ycen_km * 1000.0,
                         d=row.d_km * 1000.0, dV=row.dV_m3)
        predicted_mm = evaluate_los_mm('mogi', params_si, points['x_east_m'], points['y_north_m'],
                                       points['losE'], points['losN'], points['losU'])
        r2, rmse_mm = r2_rmse(predicted_mm, points['los_obs_m'] * 1000.0)
        r2_values.append(r2)
        rmse_values.append(rmse_mm)
    table = table.copy()
    table['r2_phys'] = r2_values
    table['rmse_mm'] = rmse_values
    return table


def style_axis(ax, ylabel, xlabel=None, zero_line=False):
    """Recessive grid/axes shared by every panel."""
    if zero_line:
        ax.axhline(0.0, color=MUTED, lw=1.0, zorder=0)
    ax.set_ylabel(ylabel, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def strip_by_aspect(ax, table, column, groups, style_map, group_column, aspect_values,
                    factor=1.0):
    """Jittered points + median line per group, x = elongation A (categorical).

    aspect_values is the GLOBAL sorted A list, so every panel (and every subset, even one
    missing some A) shares the same category positions and ticks."""
    position = {aspect: index for index, aspect in enumerate(aspect_values)}
    rng = np.random.default_rng(0)                        # fixed jitter -> reproducible figure
    for group in groups:
        offset = style_map[group]['offset']               # fixed per group: same x in every panel
        subset = table[table[group_column] == group]
        if subset.empty:
            continue
        style = style_map[group]
        x_positions = subset['A'].map(position) + offset + rng.uniform(-0.05, 0.05, len(subset))
        ax.scatter(x_positions, subset[column] * factor, s=30, color=style['color'],
                   marker=style['marker'], alpha=0.75, edgecolor='white', linewidth=0.8,
                   label=style['label'], zorder=2)
        medians = subset.groupby('A')[column].median() * factor
        ax.plot([position[a] + offset for a in medians.index], medians.values,
                color=style['color'], lw=2.0, zorder=3)
    ax.set_xticks(range(len(aspect_values)))
    ax.set_xticklabels([f'{a:g}' for a in aspect_values])


def plot_errors_vs_elongation(table, out_png):
    """Rows = surface; cols = depth error %, dV error %, horizontal offset (m); per method."""
    methods = ['PILA', 'MCMC', 'MCMC_joint']
    panels = [('depth_err_pct', 'Depth error vs centroid (%)', True),
              ('dV_err_pct', 'dV error vs cavity dV (%)', True),
              ('offset_m', 'Horizontal offset from axis (m)', False)]
    aspect_values = sorted(table['A'].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    for row_index, surface in enumerate(('flat', 'cone')):
        subset = table[table['surface'] == surface]
        for col_index, (column, ylabel, zero_line) in enumerate(panels):
            ax = axes[row_index, col_index]
            strip_by_aspect(ax, subset, column, methods, METHOD_STYLE, 'method', aspect_values)
            style_axis(ax, ylabel, 'Chamber elongation A (categorical)' if row_index else None,
                       zero_line)
            ax.set_title(f'{surface} surface', color=INK, fontsize=10, loc='left')
    axes[0, 0].legend(frameon=False, loc='upper left', fontsize=8)
    fig.suptitle('Mogi inversion of COMSOL wk6 chambers: PILA vs classical MCMC vs truth '
                 '(points = chambers x track, line = median)', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_horizontal_by_track(table, out_png):
    """xcen / ycen (km) vs A coloured by track: tests the single-track LOS tradeoff."""
    aspect_values = sorted(table['A'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey='row')
    for col_index, (method_label, methods) in enumerate((('PILA', ['PILA']),
                                                         ('Classical MCMC', ['MCMC', 'MCMC_joint']))):
        subset = table[table['method'].isin(methods)]
        for row_index, column in enumerate(('xcen_km', 'ycen_km')):
            ax = axes[row_index, col_index]
            tracks = [track for track in ('asc', 'desc', 'joint') if (subset['track'] == track).any()]
            strip_by_aspect(ax, subset, column, tracks, TRACK_STYLE, 'track', aspect_values)
            style_axis(ax, f"{'East' if column == 'xcen_km' else 'North'} source offset (km)",
                       'Chamber elongation A (categorical)' if row_index else None, zero_line=True)
            ax.set_title(method_label, color=INK, fontsize=10, loc='left')
            ax.legend(frameon=False, loc='best', fontsize=8)
    fig.suptitle('Horizontal source position by track (truth = 0): does the joint fit remove '
                 'the single-track shift?', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_pila_vs_mcmc(table, out_png):
    """1:1 PILA vs MCMC (same scene) for depth (km) and dV (10^6 m^3)."""
    keys = ['cid', 'surface', 'track']
    pila = table[table['method'] == 'PILA'].set_index(keys)
    mcmc = table[table['method'] == 'MCMC'].set_index(keys)
    paired = pila.join(mcmc, lsuffix='_pila', rsuffix='_mcmc', how='inner')
    print(f"  1:1 plot: {len(paired)} paired scenes (PILA {len(pila)}, MCMC {len(mcmc)})")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (column, factor, unit) in zip(axes, (('d_km', 1.0, 'depth (km)'),
                                                 ('dV_m3', 1e-6, 'dV (10$^6$ m$^3$)'))):
        x_values = paired[f'{column}_mcmc'] * factor
        y_values = paired[f'{column}_pila'] * factor
        lo, hi = min(x_values.min(), y_values.min()), max(x_values.max(), y_values.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1.0, ls='--',
                zorder=0, label='1:1')
        aspect_values = sorted(paired['A_pila'].unique())
        # Ordinal blue ramp (one hue, light -> dark) for elongation A.
        ramp = ['#86b6ef', '#3987e5', '#256abf', '#104281']
        if len(aspect_values) > len(ramp):
            raise ValueError(f"{len(aspect_values)} elongation classes > {len(ramp)} ramp steps")
        for aspect, color in zip(aspect_values, ramp):
            sel = paired['A_pila'] == aspect
            ax.scatter(x_values[sel], y_values[sel], s=30, color=color, edgecolor='white',
                       linewidth=0.8, label=f'A = {aspect:g}', zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect('equal')
        ax.set_xlabel(f'Classical MCMC {unit}', color=INK)
        ax.set_ylabel(f'PILA {unit}', color=INK)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc='upper left', fontsize=8)
    fig.suptitle('Does PILA land on the best-fit Mogi? (one point per chamber x surface x track)',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_fit_and_runtime(table, out_png):
    """Physics-only R^2 (1 - R^2, log) and runtime (s, log) per method."""
    methods = [m for m in ('PILA', 'MCMC', 'MCMC_joint', 'SA', 'SA_joint') if (table['method'] == m).any()]
    style_map = dict(METHOD_STYLE, SA=dict(color='#52514e', marker='x', label='SA warm start'),
                     SA_joint=dict(color='#8a8984', marker='+', label='SA warm start (joint)'))
    # Joint rows: R^2 over pooled 2N asc+desc points and wall time for 2N points.
    tick_labels = {m: style_map[m]['label'] + (' [2N pts]' if m.endswith('_joint') else '')
                   for m in methods}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (column, ylabel, transform) in zip(axes, (
            ('r2_phys', 'Physics-only misfit 1 - R$^2$ (log)', lambda v: np.clip(1 - v, 1e-8, None)),
            ('runtime_s', 'Wall time per scene (s, log)', lambda v: v))):
        for index, method in enumerate(methods):
            values = transform(table.loc[table['method'] == method, column].to_numpy(dtype=float))
            style = style_map[method]
            ax.scatter(index + rng.uniform(-0.15, 0.15, values.size), values, s=24,
                       color=style['color'], marker=style['marker'], alpha=0.7, zorder=2)
            ax.plot([index - 0.3, index + 0.3], [np.nanmedian(values)] * 2, color=INK, lw=2.0, zorder=3)
        ax.set_yscale('log')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([tick_labels[m] for m in methods], rotation=20, ha='right', fontsize=8)
        style_axis(ax, ylabel)
    fig.suptitle('Fit quality and cost (black bar = median; PILA time = per-chamber training, '
                 'not amortised)', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='COMSOL wk6: PILA vs classical MCMC vs truth')
    parser.add_argument('--saved-root', default='saved/comsol_wk6')
    parser.add_argument('--classic', default='classic_out_comsol')
    parser.add_argument('--exported', default='exported_comsol')
    parser.add_argument('--truth', default='data/comsol/wk6_chambers_inversion_20260928/metadata/wk6_truth.csv')
    parser.add_argument('--outdir', default='figures/comsol_wk6/classical')
    parser.add_argument('--overwrite', action='store_true', help='allow replacing existing outputs')
    args = parser.parse_args()
    t0 = time.time()

    # --- [1/5] Inputs (refuse to overwrite before doing any work) ---
    results_csv = os.path.join(args.outdir, 'comsol_pila_vs_classical.csv')
    summary_csv = os.path.join(args.outdir, 'comsol_pila_vs_classical_summary.csv')
    figure_paths = [os.path.join(args.outdir, name) for name in (
        'errors_vs_elongation.png', 'horizontal_offset_by_track.png',
        'pila_vs_mcmc_1to1.png', 'fit_and_runtime.png')]
    existing = [path for path in [results_csv, summary_csv] + figure_paths if os.path.exists(path)]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs exist (pass --overwrite to replace): {existing}")
    for path in (args.truth, args.saved_root, args.classic, args.exported):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    print(f"[1/5] Loading truth {args.truth} ...")
    truth = pd.read_csv(args.truth)
    print(f"  Loaded: {truth.shape[0]} truth rows")

    # --- [2/5] PILA rows (both tracks, straight from the run dirs) ---
    print(f"[2/5] Collecting PILA runs under {args.saved_root} ...")
    pila = collect_runs(args.saved_root)
    pila['method'] = 'PILA'
    pila['config'] = [f'comsol_wk6_cid{c}_{s}_{t}_mogi' for c, s, t in
                      zip(pila['cid'], pila['surface'], pila['track'])]
    pila['runtime_s'] = [pila_runtime_sec(path) for path in pila['params_file']]
    print(f"  {len(pila)} PILA runs, railed: {int(pila['railed'].sum())}")

    # --- [3/5] Classical rows ---
    print(f"[3/5] Loading classical results under {args.classic} ...")
    classical = load_classical_rows(args.classic)
    if classical.empty:
        raise RuntimeError(f"no classical results under {args.classic}")
    print(f"  {len(classical)} rows: {classical.groupby('method').size().to_dict()}, "
          f"railed: {int(classical['railed'].sum())}")

    # --- [4/5] Truth, errors, physics-only fit; tables ---
    print("[4/5] Joining truth, computing errors and physics-only R^2 ...")
    combined = pd.concat([pila, classical], ignore_index=True)
    duplicated = combined.duplicated(['cid', 'surface', 'track', 'method'])
    if duplicated.any():
        raise ValueError(f"duplicate scene rows: {combined.loc[duplicated, 'config'].tolist()}")
    print(f"  PILA rows missing a runtime: {int(pila['runtime_s'].isna().sum())}")
    table = add_truth_and_errors(combined, truth)
    t_fit = time.time()
    table = add_physics_fit(table, args.exported)
    print(f"  physics fit for {len(table)} rows in {time.time() - t_fit:.1f}s")

    # Elongated chambers are not point sources, so even the best Mogi leaves R^2 ~0.97-0.99
    # (PILA pilot sweep): an absolute 0.99 cut would flag model inadequacy, not failure.
    # Gross failure = R^2 < 0.9; the convergence measure is the gap to the best Mogi fit
    # found by ANY method on the same scene (0 = this method found the best Mogi).
    table['low_fit_flag'] = table['r2_phys'] < 0.9
    table['r2_gap_vs_best'] = (table.groupby(['cid', 'surface', 'track'])['r2_phys']
                               .transform('max') - table['r2_phys'])
    # Joint scenes hold only MCMC_joint/SA_joint (no PILA joint run), and their R^2 is over
    # the pooled 2N asc+desc points -- not comparable to single-track R^2. No gap for them.
    table.loc[table['track'] == 'joint', 'r2_gap_vs_best'] = np.nan
    os.makedirs(args.outdir, exist_ok=True)
    table.drop(columns=['params_file', 'result_file'], errors='ignore').to_csv(
        results_csv, index=False, float_format='%.6g')
    summary = (table.groupby(['method', 'surface', 'A', 'track'])
               .agg(n=('cid', 'size'),
                    depth_err_pct_median=('depth_err_pct', 'median'),
                    depth_err_pct_min=('depth_err_pct', 'min'),
                    depth_err_pct_max=('depth_err_pct', 'max'),
                    dV_err_pct_median=('dV_err_pct', 'median'),
                    xcen_km_median=('xcen_km', 'median'),
                    offset_m_median=('offset_m', 'median'),
                    r2_phys_median=('r2_phys', 'median'),
                    r2_gap_vs_best_median=('r2_gap_vs_best', 'median'),
                    runtime_s_median=('runtime_s', 'median'),
                    n_railed=('railed', 'sum'),
                    n_low_fit=('low_fit_flag', 'sum'))
               .round(4).reset_index())
    summary.to_csv(summary_csv, index=False)
    print(f"  Wrote {results_csv} ({len(table)} rows) and {summary_csv}")
    print(summary[summary['method'].isin(['PILA', 'MCMC', 'MCMC_joint'])].to_string(index=False))
    flagged = table[table['low_fit_flag'] | table['railed']]
    if not flagged.empty:
        print(f"  FLAGGED ({len(flagged)} rows: R^2 < 0.9 or railed):")
        print(flagged[['method', 'cid', 'surface', 'track', 'r2_phys', 'railed']].to_string(index=False))

    # --- [5/5] Figures (SA rows excluded from the error figures: warm start only) ---
    print(f"[5/5] Writing figures to {args.outdir} ...")
    main_methods = table[table['method'].isin(['PILA', 'MCMC', 'MCMC_joint'])]
    plot_errors_vs_elongation(main_methods, figure_paths[0])
    plot_horizontal_by_track(main_methods, figure_paths[1])
    plot_pila_vs_mcmc(main_methods, figure_paths[2])
    plot_fit_and_runtime(table, figure_paths[3])
    print(f"Done in {time.time() - t0:.1f}s. Output saved to {args.outdir}/")


if __name__ == '__main__':
    main()
