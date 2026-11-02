#!/usr/bin/env python
# Usage:       python plot_k2_keepbest_summary.py [--fig-root figures/multisource_k2_keepbest] [--seed-trace 12]
# Description: Figures for the noisy two-source test (generate_k2_keepbest_jobs.py), from compare_k2_keepbest.py's
#              keepbest_runs.csv and each run's per_epoch_sources.csv:
#                1. k2_keepbest_seeds.png   worst-source location error per seed, final vs lowest-loss epoch, for SNR 6/12 x
#                                           both ranges; the seed PILA would pick on its own loss (lowest val_rec_loss of
#                                           16) is ringed. Truth used only for scoring.
#                2. k2_keepbest_trace_seed<S>.png  one SNR-12 standard-range seed: train reconstruction loss and worst-source
#                                           location error per epoch, showing a separated low-loss state lost at a restart.
#              PNG only.
# Date:        2026-10-22

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = 'saved/multisource_k2_tune'
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'
NOISE_TAG = {6.0: 'n050', 11.9: 'n025'}
RANGE_LABEL = {'std': 'standard range ±2e8 m³', 'box60': 'tighter range (60 mm anchor)'}


def worst_location_error_km(frame, truth):
    """Per-epoch worst-source horizontal error (km) under the better of the two slot-to-source assignments."""
    slots = [frame[['s1_x_km', 's1_y_km']].to_numpy(), frame[['s2_x_km', 's2_y_km']].to_numpy()]
    dist = lambda slot, source: np.hypot(slot[:, 0] - source[0], slot[:, 1] - source[1])
    keep = np.maximum(dist(slots[0], truth[0]), dist(slots[1], truth[1]))
    swap = np.maximum(dist(slots[0], truth[1]), dist(slots[1], truth[0]))
    return np.minimum(keep, swap)


def plot_seeds(runs, out_path):
    """2x2 panels: SNR (rows) x range (columns), final vs lowest-loss location error per seed, own-loss pick ringed."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for row, snr in enumerate(sorted(NOISE_TAG, reverse=True)):
        for col, range_tag in enumerate(RANGE_LABEL):
            axis = axes[row][col]
            subset = runs[(runs['snr'] == snr) & (runs['range'] == range_tag)]
            final = subset[subset['epoch_choice'] == 'final'].set_index('seed')
            lowest = subset[subset['epoch_choice'] == 'lowest_loss'].set_index('seed')
            seeds = final.index.to_numpy()
            axis.scatter(seeds - 0.15, final['loc_max_km'], marker='x', color='0.45', label='final epoch')
            axis.scatter(seeds + 0.15, lowest['loc_max_km'], color=np.where(lowest['success'], 'C2', 'C3'),
                         label='lowest-loss epoch (green = success)')
            pick = lowest['lowest_val_rec_loss'].idxmin()
            axis.scatter([pick + 0.15], [lowest.loc[pick, 'loc_max_km']], s=260, facecolors='none', edgecolors='k',
                         linewidths=1.8, label="PILA's own pick (lowest loss of 16)")
            axis.axhline(1.5, color='k', ls=':', lw=1)
            axis.set_yscale('log')
            n_final, n_lowest = int(final['success'].sum()), int(lowest['success'].sum())
            axis.set_title(f'SNR {snr:g}, {RANGE_LABEL[range_tag]}\nsuccess: final {n_final}/16, lowest-loss {n_lowest}/16, '
                           f"pick {'SUCCESS' if lowest.loc[pick, 'success'] else 'fail'}", fontsize=10)
            axis.set_xticks(seeds)
            axis.set_xlabel('Seed')
            if col == 0:
                axis.set_ylabel('Worst-source location error (km)')
    axes[0][0].legend(fontsize=8, loc='center left')
    fig.suptitle('Noisy two-source scene (E1_S3_K2_separated), honest K=2, penalties off — dotted = 1.5 km success limit '
                 '(success also needs depth & dV < 50 %)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trace(seed, out_path):
    """Loss and worst-source location error per epoch for one SNR-12 standard-range seed."""
    truth_json = os.path.join(PROJECT_ROOT, SNR_DIR, 'E1_S3_K2_separated_n025_truth.json')
    truth = [(s['xcen_km'], s['ycen_km']) for s in json.load(open(truth_json))['sources']]
    models = glob.glob(os.path.join(PROJECT_ROOT, SAVE_ROOT, f'k2tune_snrE1n025kb_std_wr100_seed{seed}', '*', 'K2', '*', '*', 'models'))[0]
    frame = pd.read_csv(os.path.join(models, 'per_epoch_sources.csv'))
    lowest = json.load(open(os.path.join(models, 'lowest_loss', 'lowest_loss.json')))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(frame['epoch'], frame['rec_loss'], color='C0')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Train reconstruction loss')
    axes[1].plot(frame['epoch'], worst_location_error_km(frame, truth), color='C3')
    axes[1].axhline(1.5, color='k', ls=':', lw=1)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Worst-source location error (km)')
    axes[1].set_xlabel('Epoch')
    for axis in axes:
        for restart in (101, 201, 301):
            axis.axvline(restart, color='0.6', ls='--', lw=0.8)
        axis.axvline(lowest['epoch'], color='C2', lw=1.5)
    axes[0].text(lowest['epoch'], axes[0].get_ylim()[1], f" kept: epoch {lowest['epoch']} (lowest val loss)",
                 color='C2', va='top', fontsize=9)
    fig.suptitle(f'SNR 12, standard range, seed {seed}: grey dashed = learning-rate restarts')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig-root', default='figures/multisource_k2_keepbest')
    parser.add_argument('--seed-trace', type=int, default=12)
    args = parser.parse_args()
    fig_root = os.path.join(PROJECT_ROOT, args.fig_root)
    print("[1/2] Per-seed summary...")
    runs = pd.read_csv(os.path.join(fig_root, 'keepbest_runs.csv'))
    print(f"  Loaded: {runs.shape}")
    plot_seeds(runs, os.path.join(fig_root, 'k2_keepbest_seeds.png'))
    print(f"[2/2] Epoch trace, seed {args.seed_trace}...")
    plot_trace(args.seed_trace, os.path.join(fig_root, f'k2_keepbest_trace_seed{args.seed_trace}.png'))
    print(f"Done. Output saved to {args.fig_root}/")


if __name__ == '__main__':
    main()
