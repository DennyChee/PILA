#!/usr/bin/env python
# Usage:       python compare_comsol_topo_classical.py [--outdir figures/comsol_wk6_topo/classical] [--overwrite]
# Description: Is PILA's horizontal source offset under the topography correction also present
#              in the classical inversion? On the COMSOL wk6 cone (asc + desc) compares four
#              Mogi inversions against the truth: classical flat (iclassic_insar.m, existing),
#              classical + topography (iokada_agung_residuals.m 'varying_depth'), PILA flat
#              (existing) and PILA + topography. Writes a long CSV, a summary CSV and a PNG.
#              Joint asc+desc runs (track 'joint', 2026-10-12) are included when present.
# Date:        2026-10-12
#
# Scientific conventions (READ):
#   * Truth / error definitions are collect_comsol_results.add_truth_and_errors: depth truth =
#     centroid depth below the COMSOL base (-z_centroid_m), dV truth = cavity dV, horizontal
#     truth = chamber axis (0, 0); + error = too deep / too large.
#   * Classical estimate = MCMC posterior median (post_med), as compare_comsol_pila_vs_classical.
#   * Classical flat and classical topo come from two codes (iclassic_insar vs iokada) with the
#     same points, LOS vectors, box, sampler profile ('june'), NBay and seed; iclassic was
#     rebuilt from iokada's sampler, so the code change itself should not move the answer.
#   * All depths are below the h = 0 datum (COMSOL base); topography runs use h_ref = 0.
#   * Joint classical-topo runs: desc pixels were shifted 0.094 m east only so iokada's binning
#     keeps one pixel per bin (build_comsol_iokada_shims.py); their file tag is 'asc'.
#   * Figures are PNG only (user preference).

import argparse
import glob
import os
import re
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_comsol_results import add_truth_and_errors, collect_runs

TRUTH_CSV = 'data/comsol/wk6_chambers_inversion_20260928/metadata/wk6_truth.csv'
CLASSICAL_FLAT_GLOB = 'classic_out_comsol/cone_{track}/*/mogi_*_result.mat'
CLASSICAL_TOPO_GLOB = 'classic_out_comsol_topo/varying_depth/*_cone_{track}_*/mogi_los_*_20000101_result.mat'
PILA_FLAT_ROOT = 'saved/comsol_wk6'
PILA_TOPO_ROOT = 'saved/comsol_wk6_topo'
NAME_PATTERN = re.compile(r'cid(\d+)_(flat|cone)_(asc|desc|joint)')
TRACKS = ('asc', 'desc', 'joint')
# (method label, colour, marker) in plotting order: classical pair, then PILA pair.
METHODS = [('classical, flat', '#7f7f7f', 'o'), ('classical, topography', '#2a78d6', 's'),
           ('PILA, flat', '#eb6834', '^'), ('PILA, topography', '#1f9e5a', 'D')]


def read_classical(result_glob, method):
    """MCMC posterior medians of classical result mats -> rows (x/y in km, d in km, dV in m^3)."""
    rows = []
    for track in TRACKS:
        for result_path in sorted(glob.glob(result_glob.format(track=track))):
            match = NAME_PATTERN.search(result_path)
            with h5py.File(result_path, 'r') as fh:        # -v7.3 mats are HDF5
                post_med = np.array(fh['post_med']).ravel()
            rows.append(dict(cid=int(match[1]), surface=match[2], track=match[3], method=method,
                             xcen_km=post_med[0] / 1e3, ycen_km=post_med[1] / 1e3,
                             d_km=post_med[2] / 1e3, dV_m3=post_med[3], source=result_path))
    return pd.DataFrame(rows)


def read_pila(saved_root, method):
    """PILA runs (cone only) via the existing collector."""
    runs = collect_runs(saved_root)
    runs = runs[runs['surface'] == 'cone'].copy()
    runs['method'] = method
    return runs.rename(columns={'params_file': 'source'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='figures/comsol_wk6_topo/classical')
    parser.add_argument('--overwrite', action='store_true', help='allow replacing existing outputs')
    args = parser.parse_args()
    t0 = time.time()

    # --- [1/4] Load the four methods ---
    print("[1/4] Loading results ...")
    if not os.path.exists(TRUTH_CSV):
        raise FileNotFoundError(TRUTH_CSV)
    truth = pd.read_csv(TRUTH_CSV)
    tables = [read_classical(CLASSICAL_FLAT_GLOB, METHODS[0][0]),
              read_classical(CLASSICAL_TOPO_GLOB, METHODS[1][0]),
              read_pila(PILA_FLAT_ROOT, METHODS[2][0]),
              read_pila(PILA_TOPO_ROOT, METHODS[3][0])]
    for table_part in tables:
        if table_part.empty:
            raise RuntimeError("a method has no results yet")
        print(f"  {table_part['method'].iloc[0]}: {len(table_part)} runs "
              f"{table_part.groupby('track').size().to_dict()}")
    columns = ['cid', 'surface', 'track', 'method', 'xcen_km', 'ycen_km', 'd_km', 'dV_m3', 'source']
    table = add_truth_and_errors(pd.concat([t[columns] for t in tables], ignore_index=True), truth)
    table['x_m'] = table['xcen_km'] * 1000.0
    table['y_m'] = table['ycen_km'] * 1000.0

    # --- [2/4] Tables ---
    print("[2/4] Writing tables ...")
    os.makedirs(args.outdir, exist_ok=True)
    long_csv = os.path.join(args.outdir, 'topo_classical_vs_pila_long.csv')
    summary_csv = os.path.join(args.outdir, 'topo_classical_vs_pila_summary.csv')
    figure_png = os.path.join(args.outdir, 'topo_classical_vs_pila_offsets.png')
    existing = [path for path in (long_csv, summary_csv, figure_png) if os.path.exists(path)]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs exist (pass --overwrite to replace): {existing}")
    table.to_csv(long_csv, index=False, float_format='%.6g')
    method_order = {label: index for index, (label, _, _) in enumerate(METHODS)}
    summary = (table.groupby(['track', 'A', 'method'])
               .agg(n=('cid', 'size'), depth_err_pct=('depth_err_pct', 'median'),
                    dV_err_pct=('dV_err_pct', 'median'), x_m=('x_m', 'median'),
                    y_m=('y_m', 'median'), offset_m=('offset_m', 'median'))
               .round(1).reset_index())
    summary['_order'] = summary['method'].map(method_order)
    summary = summary.sort_values(['track', 'A', '_order']).drop(columns='_order')
    summary.to_csv(summary_csv, index=False)
    print(summary.to_string(index=False))

    # Per-chamber agreement of the two topography solutions (same scene, two codes).
    paired = table.pivot_table(index=['track', 'cid'], columns='method', values=['x_m', 'd_km'])
    dx_m = (paired[('x_m', METHODS[3][0])] - paired[('x_m', METHODS[1][0])]).abs()
    dd_m = 1000.0 * (paired[('d_km', METHODS[3][0])] - paired[('d_km', METHODS[1][0])]).abs()
    print(f"  PILA-topo vs classical-topo, per chamber: median |dx| = {dx_m.median():.0f} m, "
          f"median |d depth| = {dd_m.median():.0f} m")

    # --- [3/4] Figure: x-offset and depth error per family A, by track ---
    print("[3/4] Plotting ...")
    panels = [('x_m', 'East offset of source (m)'), ('depth_err_pct', 'Depth error vs centroid (%)')]
    tracks_present = [track for track in TRACKS if (table['track'] == track).any()]
    fig, axes = plt.subplots(len(panels), len(tracks_present), figsize=(5.0 * len(tracks_present), 7.0),
                             squeeze=False, sharex=True)
    for column_index, track in enumerate(tracks_present):
        for row_index, (metric, ylabel) in enumerate(panels):
            ax = axes[row_index, column_index]
            for method_index, (label, colour, marker) in enumerate(METHODS):
                subset = table[(table['track'] == track) & (table['method'] == label)]
                dodge = (method_index - 1.5) * 0.18          # keep the 4 methods side by side
                ax.scatter(subset['A'] + dodge, subset[metric], color=colour, marker=marker, s=16,
                           alpha=0.6, label=label if (row_index, column_index) == (0, 0) else None)
                medians = subset.groupby('A')[metric].median()
                ax.plot(medians.index + dodge, medians.values, color=colour, lw=1.3)
            ax.axhline(0.0, color='0.5', lw=0.8, ls='--')
            ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title('Cone, joint asc + desc' if track == 'joint' else f'Cone, {track} track')
            if row_index == len(panels) - 1:
                ax.set_xlabel('Chamber shape-noise family A (dimensionless)')
    fig.legend(loc='upper center', ncol=4, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_png, dpi=150)
    plt.close(fig)
    print(f"  figure: {figure_png}")

    # --- [4/4] Done ---
    print(f"[4/4] Done in {time.time() - t0:.1f}s. Output saved to {args.outdir}/")


if __name__ == '__main__':
    main()
