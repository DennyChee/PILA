#!/usr/bin/env python
# Usage:       python compare_comsol_topo.py [--flat-root saved/comsol_wk6]
#                  [--topo-root saved/comsol_wk6_topo] [--outdir figures/comsol_wk6_topo] [--overwrite]
# Description: Does PILA's Williams & Wadge topography correction remove the cone bias on the
#              COMSOL wk6 catalogue? Compares, per track and chamber family A, the depth / dV /
#              offset errors of three cases: flat surface + flat model (reference), cone + flat
#              model (existing runs, saved/comsol_wk6) and cone + topography model (new runs,
#              saved/comsol_wk6_topo). Writes a per-chamber paired CSV, a summary CSV and a PNG.
# Date:        2026-10-12
#
# Scientific conventions (READ):
#   * Truth, errors and run parsing are reused from collect_comsol_results.py: depth truth =
#     centroid depth -z_centroid_m below the COMSOL base (z = 0); error sign + = too deep.
#   * Topography runs use h_ref_m = 0, so their depth is below the same base datum and compares
#     directly with the truth and with the flat-model runs.
#   * Success criterion: cone + topography error moves from the cone + flat-model value toward
#     the flat-surface value for the same chambers. The remaining gap is what the first-order
#     correction cannot capture (flat free surface, ~20 deg slope effects).
#   * Figures are PNG only (user preference).

import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from collect_comsol_results import add_truth_and_errors, collect_runs

TRUTH_CSV = 'data/comsol/wk6_chambers_inversion_20260928/metadata/wk6_truth.csv'
# Three cases, in plotting order: (label, colour, marker).
CASES = [('flat surface, flat model', '#2a78d6', 'o'),
         ('cone, flat model', '#eb6834', '^'),
         ('cone, topography model', '#1f9e5a', 's')]
METRICS = [('depth_err_pct', 'Depth error vs centroid (%)'),
           ('dV_err_pct', 'dV error vs cavity (%)'),
           ('offset_m', 'Horizontal offset (m)')]


def load_case(saved_root, truth, surface, case_label):
    """Runs of one surface under saved_root, joined with the truth, tagged with a case label."""
    if not os.path.isdir(saved_root):
        raise FileNotFoundError(saved_root)
    runs = collect_runs(saved_root)
    if runs.empty:
        raise RuntimeError(f"no finished runs under {saved_root}")
    runs = runs[runs['surface'] == surface]
    table = add_truth_and_errors(runs, truth)
    table['case'] = case_label
    print(f"  {case_label}: {len(table)} runs "
          f"({table.groupby('track').size().to_dict()}), railed {int(table['railed'].sum())}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flat-root', default='saved/comsol_wk6')
    parser.add_argument('--topo-root', default='saved/comsol_wk6_topo')
    parser.add_argument('--outdir', default='figures/comsol_wk6_topo')
    parser.add_argument('--overwrite', action='store_true', help='allow replacing existing outputs')
    args = parser.parse_args()
    t0 = time.time()

    # --- [1/4] Load truth and the three cases ---
    print(f"[1/4] Loading truth ({TRUTH_CSV}) and runs ...")
    if not os.path.exists(TRUTH_CSV):
        raise FileNotFoundError(TRUTH_CSV)
    truth = pd.read_csv(TRUTH_CSV)
    table = pd.concat([load_case(args.flat_root, truth, 'flat', CASES[0][0]),
                       load_case(args.flat_root, truth, 'cone', CASES[1][0]),
                       load_case(args.topo_root, truth, 'cone', CASES[2][0])], ignore_index=True)

    # --- [2/4] Per-chamber pairing (only chambers present in all three cases) ---
    print("[2/4] Pairing chambers across cases ...")
    paired = table.pivot_table(index=['track', 'cid', 'A'], columns='case',
                               values=['depth_err_pct', 'dV_err_pct', 'offset_m'])
    paired = paired.dropna()
    paired.columns = [f'{metric}|{case}' for metric, case in paired.columns]
    paired = paired.reset_index()
    # Fraction of the cone bias removed per chamber: 1 = topo run matches the flat surface,
    # 0 = no change from the cone flat-model run.
    cone_bias = paired[f'depth_err_pct|{CASES[1][0]}'] - paired[f'depth_err_pct|{CASES[0][0]}']
    topo_bias = paired[f'depth_err_pct|{CASES[2][0]}'] - paired[f'depth_err_pct|{CASES[0][0]}']
    paired['cone_depth_bias_removed_frac'] = 1.0 - topo_bias / cone_bias
    print(f"  {len(paired)} chamber-track pairs present in all three cases")

    # --- [3/4] Tables ---
    print("[3/4] Writing tables ...")
    os.makedirs(args.outdir, exist_ok=True)
    paired_csv = os.path.join(args.outdir, 'topo_vs_flat_paired.csv')
    summary_csv = os.path.join(args.outdir, 'topo_vs_flat_summary.csv')
    figure_png = os.path.join(args.outdir, 'topo_vs_flat_errors.png')
    existing = [path for path in (paired_csv, summary_csv, figure_png) if os.path.exists(path)]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs exist (pass --overwrite to replace): {existing}")
    paired.to_csv(paired_csv, index=False, float_format='%.4g')
    summary = (table.groupby(['track', 'A', 'case'], sort=False)
               .agg(n=('cid', 'size'), depth_err_pct_median=('depth_err_pct', 'median'),
                    dV_err_pct_median=('dV_err_pct', 'median'), offset_m_median=('offset_m', 'median'))
               .round(2).reset_index().sort_values(['track', 'A']))
    summary.to_csv(summary_csv, index=False)
    print(summary.to_string(index=False))
    bias_removed = paired.groupby(['track', 'A'])['cone_depth_bias_removed_frac'].median().round(2)
    print("  Median fraction of cone depth bias removed by topography (1 = all):")
    print(bias_removed.to_string())

    # --- [4/4] Figure: error per family A, one row per metric, one column per track ---
    print("[4/4] Plotting ...")
    tracks = sorted(table['track'].unique())
    fig, axes = plt.subplots(len(METRICS), len(tracks), figsize=(5.0 * len(tracks), 9.0),
                             squeeze=False, sharex=True)
    for column_index, track in enumerate(tracks):
        for row_index, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row_index, column_index]
            for case_index, (case_label, colour, marker) in enumerate(CASES):
                subset = table[(table['track'] == track) & (table['case'] == case_label)]
                # Small horizontal dodge so the three cases do not overplot at each A.
                x_positions = subset['A'] + (case_index - 1) * 0.25
                ax.scatter(x_positions, subset[metric], color=colour, marker=marker, s=18,
                           alpha=0.6, label=case_label if (row_index, column_index) == (0, 0) else None)
                medians = subset.groupby('A')[metric].median()
                ax.plot(medians.index + (case_index - 1) * 0.25, medians.values, color=colour, lw=1.5)
            if metric != 'offset_m':
                ax.axhline(0.0, color='0.5', lw=0.8, ls='--')
            ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(f'{track} track')
            if row_index == len(METRICS) - 1:
                ax.set_xlabel('Chamber shape-noise family A (dimensionless)')
    fig.legend(loc='upper center', ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_png, dpi=150)
    plt.close(fig)
    print(f"  figure: {figure_png}")
    print(f"Done in {time.time() - t0:.1f}s. Output saved to {args.outdir}/")


if __name__ == '__main__':
    main()
