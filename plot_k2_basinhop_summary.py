#!/usr/bin/env python
# Usage:       python plot_k2_basinhop_summary.py [--fig-root figures/multisource_k2_basinhop] [--seed-trace 1] [--noise-tag n025|n050]
# Description: Figures for the basin-hopping test (generate_k2_basinhop_jobs.py), from compare_k2_basinhop.py's
#              basinhop_runs.csv and each run's basin_epochs.csv / basin_hops.csv:
#                1. k2_basinhop_seeds.png        worst-source location error per seed at the basin_best checkpoint, one
#                                                panel per arm; green = success; ringed = PILA's own pick (lowest
#                                                deterministic album misfit of 16). Truth used only for scoring.
#                2. k2_basinhop_trace_seed<S>.png one seed, residual-guided slot arm vs the warm-restart baseline:
#                                                album misfit and worst-source location error per epoch, hop windows
#                                                marked (green = window set a new best by > 0.5 %).
#              PNG only.
# Date:        2026-10-23

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_k2_keepbest_summary import worst_location_error_km

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = 'saved/multisource_k2_tune'
SNR_LABEL = {'n025': 'SNR 12', 'n050': 'SNR 6'}
TRUTH = 'synthetic/multisource/pila_stacks_v2/albums_snr/E1_S3_K2_separated_{noise}_truth.json'
ARM_PANELS = [  # (compare arm label, panel title, tag template)
    ('observe_std (warm restarts, rerun)', 'baseline: warm restarts (no hopping)', 'k2tune_snrE1{noise}bh_observe_seed{seed}'),
    ('bh_none', 'control: revert + LR re-warm 0.5x', 'k2tune_snrE1{noise}bh_none_seed{seed}'),
    ('bh_none_full', 'control: revert + LR re-warm 1.0x', 'k2tune_snrE1{noise}bh_none_full_seed{seed}'),
    ('bh_wnoise', 'weight noise on z_phy head', 'k2tune_snrE1{noise}bh_wnoise_seed{seed}'),
    ('bh_slot', 'move redundant slot: random spot', 'k2tune_snrE1{noise}bh_slot_seed{seed}'),
    ('bh_slot_resid', 'move redundant slot: residual-guided', 'k2tune_snrE1{noise}bh_slot_resid_seed{seed}'),
]


def plot_seeds(runs, out_path, noise):
    """2x3 panels, one per arm: worst-source location error per seed at basin_best, own-misfit pick ringed."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for axis, (arm, title, _template) in zip(axes.ravel(), ARM_PANELS):
        subset = runs[(runs['arm'] == arm) & (runs['checkpoint'] == 'basin_best')].set_index('seed').sort_index()
        if subset.empty:                                    # arm not run for this album
            axis.set_title(f'{title}\n(not run)', fontsize=10)
            axis.set_axis_off()
            continue
        seeds = subset.index.to_numpy()
        axis.scatter(seeds, subset['loc_max_km'], color=np.where(subset['success'], 'C2', 'C3'), zorder=3,
                     label='seed (green = success)')
        pick = subset['misfit'].idxmin()
        axis.scatter([pick], [subset.loc[pick, 'loc_max_km']], s=260, facecolors='none', edgecolors='k', linewidths=1.8,
                     zorder=4, label="PILA's own pick (lowest misfit of 16)")
        axis.axhline(1.5, color='k', ls=':', lw=1)
        axis.set_yscale('log')
        axis.set_title(f"{title}\nsuccess {int(subset['success'].sum())}/{len(subset)}, pick "
                       f"{'SUCCESS' if subset.loc[pick, 'success'] else 'fail'}", fontsize=10)
        axis.set_xticks(seeds)
        axis.set_xlabel('Seed')
    for axis in axes[:, 0]:
        axis.set_ylabel('Worst-source location error (km)')
    axes[0][0].legend(fontsize=8, loc='center left')
    fig.suptitle(f'{SNR_LABEL[noise]} two-source scene (E1_S3_K2_separated_{noise}), honest K=2, penalties off, 400 epochs — '
                 'dotted = 1.5 km limit (success also needs depth & dV < 50 %)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_dir(template, seed):
    models = glob.glob(os.path.join(PROJECT_ROOT, SAVE_ROOT, template.format(seed=seed), '*', 'K2', '*', '*', 'models'))
    if len(models) != 1:
        raise FileNotFoundError(f'{template.format(seed=seed)}: {len(models)} run dirs')
    return models[0]


def plot_trace(seed, out_path, noise):
    """Album misfit + worst-source location error per epoch: residual-guided slot arm vs the warm-restart baseline."""
    truth = [(s['xcen_km'], s['ycen_km']) for s in json.load(open(os.path.join(PROJECT_ROOT, TRUTH.format(noise=noise))))['sources']]
    baseline = pd.read_csv(os.path.join(_run_dir('k2tune_snrE1{noise}bh_observe_seed{seed}'.replace('{noise}', noise), seed), 'basin_epochs.csv'))
    hopping_dir = _run_dir('k2tune_snrE1{noise}bh_slot_resid_seed{seed}'.replace('{noise}', noise), seed)
    hopping = pd.read_csv(os.path.join(hopping_dir, 'basin_epochs.csv'))
    hops = pd.read_csv(os.path.join(hopping_dir, 'basin_hops.csv'))
    print(f"  Loaded: baseline {baseline.shape}, hopping {hopping.shape}, windows {hops.shape}")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for frame, color, label in ((baseline, '0.45', 'baseline: warm restarts (grey dashed = restart)'),
                                (hopping, 'C0', 'residual-guided slot hopping')):
        axes[0].plot(frame['epoch'], frame['misfit'], color=color, lw=1.2, label=label)
        axes[1].plot(frame['epoch'], worst_location_error_km(frame, truth), color=color, lw=1.2, label=label)
    axes[0].plot(hopping['epoch'], hopping['best_misfit'], color='C0', ls='--', lw=1, label='hopping: best so far (kept)')
    for _index, window in hops.iterrows():
        shade = 'C2' if bool(window['accepted']) else '0.85'
        for axis in axes:
            axis.axvspan(window['start_epoch'] - 0.5, window['end_epoch'] + 0.5, color=shade, alpha=0.12, lw=0)
            axis.axvline(window['start_epoch'], color='C0', lw=0.6, alpha=0.6)
        axes[0].text(window['start_epoch'] + 1, 1.0, f"{'polish' if window['kind'] == 'polish' else 'hop ' + str(window['hop'])}",
                     transform=axes[0].get_xaxis_transform(), va='top', fontsize=8, color='C0')
    for axis in axes:
        for restart in (101, 201, 301):
            axis.axvline(restart, color='0.45', ls='--', lw=0.8)
        axis.set_yscale('log')
    axes[0].set_ylabel('Album misfit (standardized, rec_loss units)')
    axes[1].axhline(1.5, color='k', ls=':', lw=1)
    axes[1].set_ylabel('Worst-source location error (km)')
    axes[1].set_xlabel('Epoch')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'{SNR_LABEL[noise]}, seed {seed}: green windows set a new best (> 0.5 %); each hop starts from the best state so far')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig-root', default=None,
                        help='default figures/multisource_k2_basinhop (n025) or figures/multisource_k2_basinhop_snr6 (n050)')
    parser.add_argument('--seed-trace', type=int, default=1)
    parser.add_argument('--noise-tag', default='n025', choices=list(SNR_LABEL))
    args = parser.parse_args()
    if args.fig_root is None:
        args.fig_root = {'n025': 'figures/multisource_k2_basinhop', 'n050': 'figures/multisource_k2_basinhop_snr6'}[args.noise_tag]
    fig_root = os.path.join(PROJECT_ROOT, args.fig_root)
    print("[1/2] Per-seed summary...")
    runs = pd.read_csv(os.path.join(fig_root, 'basinhop_runs.csv'))
    print(f"  Loaded: shape={runs.shape}")
    # The CSV must come from the same album (older SNR 12 CSVs predate the column and are SNR 12 by construction)
    csv_tags = set(runs['noise_tag']) if 'noise_tag' in runs else {'n025'}
    if csv_tags != {args.noise_tag}:
        raise ValueError(f"{args.fig_root}/basinhop_runs.csv is for {csv_tags}, not --noise-tag {args.noise_tag}")
    plot_seeds(runs, os.path.join(fig_root, 'k2_basinhop_seeds.png'), args.noise_tag)
    print(f"[2/2] Epoch trace, seed {args.seed_trace}...")
    plot_trace(args.seed_trace, os.path.join(fig_root, f'k2_basinhop_trace_seed{args.seed_trace}.png'), args.noise_tag)
    print(f"Done. Output saved to {args.fig_root}/")


if __name__ == '__main__':
    main()
