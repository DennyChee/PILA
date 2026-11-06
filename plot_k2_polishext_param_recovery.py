#!/usr/bin/env python
# Usage:       python plot_k2_polishext_param_recovery.py [--seeds 1,3,5,8,9,13,4,7,15] [--fig-dir figures/multisource_k2_polishext]
# Description: Late-rescue test, parameter recovery at the end of the original run (epoch 400) vs after the 1000-epoch
#              polish extension (epoch 1400), laid out like plot_k2_smoothing_param_recovery.py / plot_vbica_vs_pila_fixed.py:
#              one panel per parameter with both checkpoints overlaid (truth black, epoch 400 orange, epoch 1400 blue).
#              Rows = west / east source; columns = east, north, depth, dV, location error. Fits come from the dumps that
#              extend_k2_polish.py score wrote (epoch 400 = k2ext init, epoch 1400 = k2ext2 train); footer scores from the
#              matching extension_scores.csv rows. One PNG per seed. Truth used only for matching slots and scoring.
# Date:        2026-10-26

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_k2_smoothing_param_recovery import PARAMS, SOURCE_NAMES, TRUTH, COLOR_TRUTH, COLOR_PLAIN

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT = 'saved/multisource_k2_polishext'
LATE_SEEDS = [1, 3, 5, 8, 9, 13]       # separated at epoch >= 266 in the original run
COLOR_EXTENDED = 'tab:blue'
# (legend label, colour, epoch, tag whose extension_scores.csv holds that epoch, fit path glob relative to the tag dir)
CHECKPOINTS = [
    ('epoch 400 (original run end)', COLOR_PLAIN, 400, 'k2ext_slot_resid_seed{seed}', 'init/fit_K2_epoch400.npz'),
    ('epoch 1400 (+1000 polish epochs)', COLOR_EXTENDED, 1400, 'k2ext2_slot_resid_seed{seed}',
     'train/*/*/models/fit_K2_epoch1400.npz'),
]


def load_matched_fit(fit_glob, truth):
    """Load one dumped fit and reorder its slots to truth (Hungarian on median location, as compare_k2_keepbest).

    Inputs: absolute glob of one fit_K2_epoch<N>.npz, truth [2 x (x km, y km, depth km, dV m3)].
    Output: (dates datetime64 [n_epochs], params [n_epochs, 2 sources (truth order), 4] in km / m3).
    """
    from scipy.optimize import linear_sum_assignment
    paths = glob.glob(fit_glob)
    if len(paths) != 1:
        raise FileNotFoundError(f'{fit_glob}: expected 1 fit, found {len(paths)}')
    with np.load(paths[0]) as fit:
        params = fit['params_phys']
        dates = fit['dates'].astype('datetime64[D]')                    # stored as ISO 'YYYY-MM-DD' strings
    median_params = np.median(params, axis=0)
    truth_idx, slot_idx = linear_sum_assignment(
        np.linalg.norm(truth[:, None, :2] - median_params[None, :, :2], axis=-1))
    matched = np.empty_like(params)
    matched[:, truth_idx, :] = params[:, slot_idx, :]
    return dates, matched


def footer_line(label, scores_csv, epoch, params):
    """One footer line: scores of that checkpoint (extension_scores.csv) + per-source RMS 2nd difference."""
    scores = pd.read_csv(scores_csv)
    row = scores[scores.epoch == epoch]
    if len(row) != 1:
        raise ValueError(f'{scores_csv}: expected 1 row for epoch {epoch}, found {len(row)}')
    row = row.iloc[0]
    verdict = 'SUCCESS' if row.success else 'FAIL'
    curvature = [f'{name.split()[0]} depth {np.sqrt(np.mean(np.diff(params[:, src, 2], 2) ** 2)):.2f} km / '
                 f'dV {np.sqrt(np.mean(np.diff(params[:, src, 3], 2) ** 2)) * 1e-6:.2f}e6 m3'
                 for src, name in enumerate(SOURCE_NAMES)]
    return (f'{label}: {verdict} (worst loc {row.loc_max_km:.2f} km, depth {row.depth_pct_max:.0f} %, '
            f'dV {row.dV_pct_max:.0f} %, total misfit {row.misfit:.0f}, physics-only misfit {row.phys_misfit:.0f}, '
            f'physics R2 {row.physics_r2:.3f}) | RMS 2nd diff: {"; ".join(curvature)}')


def plot_seed(seed, truth, out_png):
    """One figure, both checkpoints overlaid per panel: rows = sources, columns = 4 parameters + location error."""
    fits, footers = {}, []
    for label, _, epoch, tag, fit_rel in CHECKPOINTS:
        tag_dir = os.path.join(PROJECT_ROOT, EXT_ROOT, tag.format(seed=seed))
        fits[label] = load_matched_fit(os.path.join(tag_dir, fit_rel), truth)
        print(f"  seed {seed} epoch {epoch}: params shape={fits[label][1].shape}, dtype={fits[label][1].dtype}")
        footers.append(footer_line(label, os.path.join(tag_dir, 'extension_scores.csv'), epoch, fits[label][1]))

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'legend.fontsize': 8, 'lines.linewidth': 1.7})
    fig, axes = plt.subplots(2, len(PARAMS) + 1, figsize=(5.0 * (len(PARAMS) + 1), 3.6 * 2), squeeze=False)
    for src, src_name in enumerate(SOURCE_NAMES):
        for col, (p_idx, title, ylabel, scale) in enumerate(PARAMS):
            axis = axes[src][col]
            dates = fits[CHECKPOINTS[0][0]][0]
            axis.plot(dates, np.full(len(dates), truth[src, p_idx] * scale), '-', color=COLOR_TRUTH, label='truth')
            for label, colour, *_ in CHECKPOINTS:
                run_dates, params = fits[label]
                axis.plot(run_dates, params[:, src, p_idx] * scale, '-', color=colour, label=label)
            axis.set_title(f'{src_name}: {title}')
            axis.set_ylabel(ylabel)
            axis.set_xlabel('Date')
            axis.tick_params(axis='x', rotation=30)
            if src == 0 and col == 0:
                axis.legend(loc='best', framealpha=0.9)
        # Last column: horizontal location error per epoch vs the (static) true centre
        axis = axes[src][-1]
        for label, colour, *_ in CHECKPOINTS:
            run_dates, params = fits[label]
            location_error_km = np.hypot(params[:, src, 0] - truth[src, 0], params[:, src, 1] - truth[src, 1])
            axis.plot(run_dates, location_error_km, '-', color=colour, label=label)
        axis.axhline(1.5, color='0.45', ls='-.', label='success threshold 1.5 km')
        axis.set_title(f'{src_name}: location error')
        axis.set_ylabel('Location error (km)')
        axis.set_xlabel('Date')
        axis.tick_params(axis='x', rotation=30)
        if src == 0:
            axis.legend(loc='best', framealpha=0.9)

    group = 'late rescue' if seed in LATE_SEEDS else 'early-separation control'
    footer = '\n'.join(footers)
    footer += ('\nSame run continued at constant lr 1.5e-5, penalties off, no hopping. Truth is static, so any '
               'epoch-to-epoch change is error. Scores are stack medians (extend_k2_polish.py score).')
    fig.suptitle(f'slot_resid seed {seed} ({group})   [K=2 Mogi, SNR 12, E1 album]\n'
                 f'parameter recovery at epoch 400 vs epoch 1400', fontsize=12)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.text(0.5, 0.012, footer, ha='center', va='bottom', fontsize=9, color='0.15')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='1,3,5,8,9,13,4,7,15')
    parser.add_argument('--fig-dir', default='figures/multisource_k2_polishext')
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(',')]

    print(f"[1/2] Loading truth {TRUTH}...")
    with open(os.path.join(PROJECT_ROOT, TRUTH)) as handle:
        truth = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km'], s['dV_m3']] for s in json.load(handle)['sources']])
    print(f"  Loaded: shape={truth.shape}, dtype={truth.dtype}")

    fig_dir = os.path.join(PROJECT_ROOT, args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    print(f"[2/2] Plotting {len(seeds)} seeds into {args.fig_dir}/...")
    for seed in seeds:
        group = 'late' if seed in LATE_SEEDS else 'control'
        out_png = os.path.join(fig_dir, f'E1_S3_K2_separated_n025_seed{seed}_{group}_param_recovery_ep400_vs_ep1400.png')
        if os.path.exists(out_png):
            print(f"  WARNING: overwriting {out_png}")
        plot_seed(seed, truth, out_png)
    print(f"Done. Output saved to {fig_dir}/")


if __name__ == '__main__':
    main()
