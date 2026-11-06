#!/usr/bin/env python3
# Usage:       python compare_prior_variants.py \
#                  --truth  synthetic/cubes/marapi_mogi_rising_lineardV/marapi_mogi_rising_lineardV_truth.json \
#                  --series "Original (no KL)=saved/original_marapi_mogi_rising_lineardV/run/uspace_series.csv" \
#                           "N(0,I) KL=saved/baseline_marapi_mogi_rising_lineardV/run/uspace_series.csv" \
#                           "Temporal prior=saved/temporalprior_marapi_mogi_rising_lineardV/run/uspace_series.csv" \
#                  --out    saved/compare_lineardV_3way
# Description: Overlay the per-epoch recovered source parameters from several PILA prior
#              variants (e.g. original no-KL / static N(0,I) KL / causal temporal prior)
#              against the synthetic ground-truth trajectory, on shared axes, plus a
#              per-parameter fluctuation-RMSE-vs-truth table. This is the A/B(/C) figure
#              that answers "did the temporal prior actually help vs the baselines?".
# Date:        2026-07-09
#
# Scientific notes:
#   * Each --series is LABEL=path/to/uspace_series.csv (from dump_uspace_series.py), whose
#     physical columns (xcen,ycen,d,dV) are compared to truth params_per_epoch.
#   * Fluctuation RMSE = sqrt(mean((recovered - truth)^2)) per parameter, reported in
#     native units and as % of the truth RMS (matches plot_iterprior_vs_truth wording).
#   * Grey bands: epochs where TRUTH |dV| < dv_eps_frac*peak (default 5%) -> source
#     effectively absent, geometry unidentifiable. Shaded identically for all variants so
#     the comparison is fair; RMSE is reported both all-epoch and excluding these.
#   * Units: xcen/ycen/d in metres, dV in m^3. LOS convention positive = toward satellite.

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless HPC
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'legend.framealpha': 0.9, 'figure.titlesize': 16,
    'axes.linewidth': 1.1, 'lines.linewidth': 1.8,
})

# Distinct colours for up to 4 variants (truth is always black).
VARIANT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']


def parse_series_arg(items):
    """Parse ['LABEL=path', ...] into an ordered list of (label, path)."""
    out = []
    for it in items:
        if '=' not in it:
            raise ValueError(f"--series entry must be LABEL=path, got: {it!r}")
        label, path = it.split('=', 1)
        out.append((label.strip(), path.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description='Overlay recovered params from prior variants vs truth.')
    ap.add_argument('--truth', required=True, help='ground-truth sidecar JSON')
    ap.add_argument('--series', required=True, nargs='+',
                    help='one or more LABEL=uspace_series.csv entries (order = legend order)')
    ap.add_argument('--out', default='compare_prior_variants', help='output figure stem')
    ap.add_argument('--dv-eps-frac', type=float, default=0.05,
                    help='shade + separately report epochs with truth |dV| < this fraction of peak')
    args = ap.parse_args()

    # --- Load truth ---
    print(f"[1/4] Loading truth: {args.truth}")
    with open(args.truth) as fh:
        truth_meta = json.load(fh)
    param_names = truth_meta['param_names']                        # ['xcen','ycen','d','dV']
    param_units = truth_meta.get('param_units', [''] * len(param_names))
    truth = np.asarray(truth_meta['params_per_epoch'], dtype=float)  # (n_epoch, n_param)
    n_epoch_truth, n_param = truth.shape
    print(f"  Truth: {n_epoch_truth} epochs, params={param_names}")

    # --- Load each recovered series ---
    print("[2/4] Loading recovered series...")
    variants = parse_series_arg(args.series)
    recs = []  # (label, rec_array aligned to n_epoch)
    n_epoch = n_epoch_truth
    for label, path in variants:
        if not os.path.exists(path):
            raise FileNotFoundError(f"series not found for '{label}': {path}")
        df = pd.read_csv(path)
        rec = df[param_names].values
        n = min(len(rec), n_epoch_truth)
        if len(rec) != n_epoch_truth:
            print(f"  WARNING [{label}]: {len(rec)} rows vs {n_epoch_truth} truth epochs; using first {n}.")
        recs.append((label, rec[:n]))
        n_epoch = min(n_epoch, n)
    truth = truth[:n_epoch]

    # --- Truth-based dV gating (same shading for every variant) ---
    dv_gated = np.zeros(n_epoch, dtype=bool)
    if 'dV' in param_names:
        dv_col = param_names.index('dV')
        peak = float(np.nanmax(np.abs(truth[:, dv_col])))
        if peak > 0:
            dv_gated = np.abs(truth[:, dv_col]) < args.dv_eps_frac * peak

    # --- Fluctuation RMSE table (all epochs, and excluding gated) ---
    print("[3/4] Computing fluctuation RMSE vs truth...")
    keep = ~dv_gated
    header = f"{'parameter':<8} " + " ".join(f"{lab:>22}" for lab, _ in recs)
    lines = [header, "-" * len(header)]
    rmse_rows = {}
    for j, pname in enumerate(param_names):
        cells = []
        for lab, rec in recs:
            e_all = rec[:n_epoch, j] - truth[:, j]
            rmse_all = float(np.sqrt(np.nanmean(e_all ** 2)))
            rmse_keep = float(np.sqrt(np.nanmean((e_all[keep]) ** 2))) if keep.any() else np.nan
            tr_rms = float(np.sqrt(np.nanmean(truth[:, j] ** 2))) or 1.0
            cells.append(f"{rmse_all:.3g} ({100*rmse_all/tr_rms:.0f}%)|{rmse_keep:.3g}")
            rmse_rows[(pname, lab)] = (rmse_all, rmse_keep)
        lines.append(f"{pname:<8} " + " ".join(f"{c:>22}" for c in cells))
    table_str = "\n".join(lines)
    print("  RMSE-all (%oftruthRMS) | RMSE-excl-gated:")
    print(table_str)

    # --- Plot ---
    print("[4/4] Plotting overlay...")
    ncol = 2
    nrow = int(np.ceil(n_param / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 3.3 * nrow), squeeze=False)
    epochs = np.arange(n_epoch)
    for j, pname in enumerate(param_names):
        ax = axes[j // ncol][j % ncol]
        # grey bands for gated epochs
        in_band = False
        for e in range(n_epoch):
            if dv_gated[e] and not in_band:
                start = e; in_band = True
            if (not dv_gated[e] or e == n_epoch - 1) and in_band:
                ax.axvspan(start - 0.5, e - 0.5, color='0.85', zorder=0)
                in_band = False
        ax.plot(epochs, truth[:, j], color='black', lw=2.4, label='truth', zorder=5)
        for k, (lab, rec) in enumerate(recs):
            ax.plot(epochs, rec[:n_epoch, j], '--', marker='.', ms=3,
                    color=VARIANT_COLORS[k % len(VARIANT_COLORS)], label=lab, alpha=0.85)
        ax.set_title(pname)
        ax.set_xlabel('Epoch index')
        unit = param_units[j] if j < len(param_units) else ''
        ax.set_ylabel(f'{pname} ({unit})' if unit else pname)
    # legend + RMSE table in any spare slot (or below)
    for j in range(n_param, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{truth_meta.get('name', 'synthetic')}: parameter recovery — prior-variant comparison "
                 f"[fluct. RMSE vs truth; grey = truth |dV|<{100*args.dv_eps_frac:.0f}% peak]")
    fig.text(0.5, -0.06,
             "RMSE per param (all epochs, % of truth RMS | excl. grey): \n" + table_str,
             ha='center', va='top', family='monospace', fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(args.out + '.pdf', bbox_inches='tight')
    print(f"Done. Wrote {args.out}.png / .pdf")


if __name__ == '__main__':
    main()
