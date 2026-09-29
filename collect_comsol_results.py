#!/usr/bin/env python
# Usage:       python collect_comsol_results.py [--saved-root saved/comsol_wk6]
#                  [--truth data/comsol/wk6_chambers_inversion_20260928/metadata/wk6_truth.csv]
#                  [--outdir figures/comsol_wk6] [--overwrite]
# Description: Collect the per-chamber PILA Mogi inversions of the COMSOL wk6 catalogue
#              (generate_comsol_jobs.py) and compare them with the COMSOL truth: one CSV row
#              per run (recovered, truth, errors, chamber shape, topography), a grouped summary
#              CSV, and figures of depth / dV / horizontal-offset error vs chamber elongation.
# Date:        2026-09-28
#
# Scientific conventions (READ):
#   * Depth truth = centroid depth (-z_centroid_m), NOT the top (2 km) or design d_m: a point
#     source is best compared with the chamber's volume centroid.
#   * dV truth = dV_cavity_m3 (COMSOL cavity volume change). Mogi dV is the point-source
#     strength; for a finite chamber the two differ even with a perfect fit (McTigue-type terms).
#   * Horizontal truth = chamber axis at (0, 0) km in the npz local frame.
#   * Vertical datum: depths are from the flat z = 0 datum for BOTH surfaces. The COMSOL cone
#     maps place displacement by radius only (surface elevation ignored), so the cone-vs-flat
#     depth gap measures the effect of the cone on the displacement field under Mogi.
#   * Figures are PNG only (user preference: no PDF export).
#   * Recovered values are read from each run's figures/*_params.txt "prior bounds" block
#     (xcen/ycen/d in km, dV in m^3), written by plot_insar_results.py at the end of training.
#   * Error sign: + = PILA too deep / too large.

import argparse
import glob
import os
import re
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference categorical palette, slots 1-2 (validated pair); markers give a second encoding.
TOPO_STYLE = {'flat': dict(color='#2a78d6', marker='o', label='Flat surface'),
              'cone': dict(color='#eb6834', marker='^', label='Cone (1 km high)')}
RUN_NAME_PATTERN = re.compile(r'comsol_wk6_cid(\d+)_(flat|cone)_(asc|desc)_mogi')
BOUND_LINE_PATTERN = re.compile(r'^\s+(\w+)\s+=\s+([-+\d.eE]+)\s+\[', re.M)


def newest_params_file(run_dir):
    """Newest figures/*_params.txt under one run directory (runs are timestamp sub-dirs)."""
    # ASSUMPTION: the newest params file (by mtime) is the most recent, completed attempt.
    candidates = glob.glob(os.path.join(run_dir, '*', '*', 'figures', '*_params.txt'))
    return max(candidates, key=os.path.getmtime) if candidates else None


def parse_params_file(params_path):
    """
    Read the prior-bounds block of a params.txt.

    Output: dict {xcen_km, ycen_km, d_km, dV_m3, railed(bool)}.
    """
    with open(params_path) as fh:
        text = fh.read()
    values = {name: float(value) for name, value in BOUND_LINE_PATTERN.findall(text)}
    missing = {'xcen', 'ycen', 'd', 'dV'} - set(values)
    if missing:
        raise ValueError(f"{params_path}: missing {sorted(missing)} in the prior-bounds block")
    prior_check = text.split('# prior check:')[-1]
    return dict(xcen_km=values['xcen'], ycen_km=values['ycen'], d_km=values['d'],
                dV_m3=values['dV'], railed='RAILED' in prior_check)


def collect_runs(saved_root):
    """One row per chamber run found under saved_root (newest params file per run)."""
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(saved_root, 'comsol_wk6_cid*'))):
        match = RUN_NAME_PATTERN.fullmatch(os.path.basename(run_dir))
        if match is None:
            print(f"  skipping non-matching dir: {run_dir}")
            continue
        params_path = newest_params_file(run_dir)
        if params_path is None:
            print(f"  WARNING: no params.txt in {run_dir} (unfinished run?) -- skipped")
            continue
        row = dict(cid=int(match[1]), surface=match[2], track=match[3], params_file=params_path)
        row.update(parse_params_file(params_path))
        rows.append(row)
    return pd.DataFrame(rows)


def add_truth_and_errors(runs, truth):
    """Join the COMSOL truth on (cid, surface) and compute errors (depth %, dV %, offset m)."""
    truth_cols = ['cid', 'surface', 'kind', 'A', 'amp', 'draw', 'elong',
                  'z_centroid_m', 'height_m', 'dV_cavity_m3', 'w_max_m']
    table = runs.merge(truth[truth_cols], on=['cid', 'surface'], how='left', validate='one_to_one')
    if table['z_centroid_m'].isna().any():
        bad = table.loc[table['z_centroid_m'].isna(), ['cid', 'surface']].values.tolist()
        raise ValueError(f"runs without a truth row: {bad}")
    table['depth_true_km'] = -table['z_centroid_m'] / 1000.0          # centroid depth, + down
    # Percent errors divide by the truth: refuse non-positive truth instead of emitting inf/NaN.
    for column in ('depth_true_km', 'dV_cavity_m3'):
        if (table[column] <= 0).any():
            raise ValueError(f"non-positive truth in {column}; percent errors undefined")
    table['depth_err_km'] = table['d_km'] - table['depth_true_km']
    table['depth_err_pct'] = 100.0 * table['depth_err_km'] / table['depth_true_km']
    table['dV_err_pct'] = 100.0 * (table['dV_m3'] / table['dV_cavity_m3'] - 1.0)
    table['offset_m'] = 1000.0 * np.hypot(table['xcen_km'], table['ycen_km'])  # axis at (0,0)
    return table.sort_values(['surface', 'A', 'amp', 'cid']).reset_index(drop=True)


def plot_errors_vs_elongation(table, out_stem):
    """Three panels: depth error (%), dV error (%), horizontal offset (m) vs elongation A."""
    panels = [('depth_err_pct', 'Depth error vs centroid (%)', True),
              ('dV_err_pct', 'dV error vs cavity dV (%)', True),
              ('offset_m', 'Horizontal offset from axis (m)', False)]
    aspect_values = sorted(table['A'].unique())
    position = {aspect: index for index, aspect in enumerate(aspect_values)}
    rng = np.random.default_rng(0)          # fixed jitter so reruns give identical figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (column, ylabel, zero_line) in zip(axes, panels):
        if zero_line:
            ax.axhline(0.0, color='#52514e', lw=1.0, zorder=0)
        for offset, (topo, style) in zip((-0.17, 0.17), TOPO_STYLE.items()):
            subset = table[table['surface'] == topo]
            if subset.empty:
                continue
            x_positions = subset['A'].map(position) + offset + rng.uniform(-0.06, 0.06, len(subset))
            ax.scatter(x_positions, subset[column], s=36, color=style['color'],
                       marker=style['marker'], alpha=0.75, edgecolor='white', linewidth=0.8,
                       label=style['label'], zorder=2)
            # Group median per elongation: the line a reader should follow.
            medians = subset.groupby('A')[column].median()
            ax.plot([position[a] + offset for a in medians.index], medians.values,
                    color=style['color'], lw=2.0, zorder=3)
        ax.set_xticks(range(len(aspect_values)))
        ax.set_xticklabels([f'{a:g}' for a in aspect_values])
        ax.set_xlabel('Chamber elongation A (height / width; categorical spacing)')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#e5e5e2', lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('PILA Mogi inversion of COMSOL wk6 chambers: error vs elongation '
                 '(points = chambers, line = median)', fontsize=12)
    plt.tight_layout()
    fig.savefig(f'{out_stem}.png', dpi=150)
    plt.close(fig)


def plot_recovered_vs_true(table, out_stem):
    """Two panels: recovered vs true centroid depth (km) and dV (10^6 m^3), with 1:1 lines."""
    panels = [('depth_true_km', 'd_km', 1.0, 'True centroid depth (km)', 'PILA Mogi depth (km)'),
              ('dV_cavity_m3', 'dV_m3', 1e-6, 'True cavity dV (10$^6$ m$^3$)', 'PILA Mogi dV (10$^6$ m$^3$)')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (true_col, rec_col, factor, xlabel, ylabel) in zip(axes, panels):
        lo = min(table[true_col].min(), table[rec_col].min()) * factor
        hi = max(table[true_col].max(), table[rec_col].max()) * factor
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#52514e', lw=1.0,
                ls='--', zorder=0, label='1:1')
        for topo, style in TOPO_STYLE.items():
            subset = table[table['surface'] == topo]
            ax.scatter(subset[true_col] * factor, subset[rec_col] * factor, s=36,
                       color=style['color'], marker=style['marker'], alpha=0.8,
                       edgecolor='white', linewidth=0.8, label=style['label'], zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect('equal')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color='#e5e5e2', lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('PILA Mogi vs COMSOL truth (one point per chamber)', fontsize=12)
    plt.tight_layout()
    fig.savefig(f'{out_stem}.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Collect COMSOL wk6 PILA inversions vs truth')
    parser.add_argument('--saved-root', default='saved/comsol_wk6')
    parser.add_argument('--truth', default='data/comsol/wk6_chambers_inversion_20260928/metadata/wk6_truth.csv')
    parser.add_argument('--outdir', default='figures/comsol_wk6')
    parser.add_argument('--overwrite', action='store_true', help='allow replacing existing outputs')
    args = parser.parse_args()
    t0 = time.time()

    # --- [1/4] Load truth ---
    if not os.path.exists(args.truth):
        raise FileNotFoundError(args.truth)
    print(f"[1/4] Loading truth from {args.truth} ...")
    truth = pd.read_csv(args.truth)
    print(f"  Loaded: {truth.shape[0]} truth rows, {truth.shape[1]} columns")

    # --- [2/4] Collect runs ---
    if not os.path.isdir(args.saved_root):
        raise FileNotFoundError(args.saved_root)
    print(f"[2/4] Collecting runs under {args.saved_root} ...")
    runs = collect_runs(args.saved_root)
    if runs.empty:
        raise RuntimeError(f"no finished runs found under {args.saved_root}")
    print(f"  Found {len(runs)} runs ({runs['surface'].value_counts().to_dict()}), "
          f"railed: {int(runs['railed'].sum())}")

    # --- [3/4] Join truth + errors, write tables ---
    print("[3/4] Joining truth and computing errors ...")
    table = add_truth_and_errors(runs, truth)
    os.makedirs(args.outdir, exist_ok=True)
    results_csv = os.path.join(args.outdir, 'comsol_wk6_results.csv')
    summary_csv = os.path.join(args.outdir, 'comsol_wk6_summary.csv')
    figure_stems = [os.path.join(args.outdir, 'comsol_wk6_errors_vs_elongation'),
                    os.path.join(args.outdir, 'comsol_wk6_recovered_vs_true')]
    outputs = [results_csv, summary_csv] + [f'{stem}.png' for stem in figure_stems]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs exist (pass --overwrite to replace): {existing}")
    table.to_csv(results_csv, index=False, float_format='%.6g')
    summary = (table.groupby(['surface', 'A'])
               .agg(n=('cid', 'size'),
                    depth_err_pct_median=('depth_err_pct', 'median'),
                    depth_err_pct_min=('depth_err_pct', 'min'),
                    depth_err_pct_max=('depth_err_pct', 'max'),
                    dV_err_pct_median=('dV_err_pct', 'median'),
                    offset_m_median=('offset_m', 'median'))
               .round(2).reset_index())
    summary.to_csv(summary_csv, index=False)
    print(f"  Wrote {results_csv} ({len(table)} rows) and {summary_csv}")
    print(summary.to_string(index=False))

    # --- [4/4] Figures ---
    print(f"[4/4] Writing figures to {args.outdir} ...")
    plot_errors_vs_elongation(table, figure_stems[0])
    plot_recovered_vs_true(table, figure_stems[1])
    print(f"Done in {time.time() - t0:.1f}s. Output saved to {args.outdir}/")


if __name__ == '__main__':
    main()
