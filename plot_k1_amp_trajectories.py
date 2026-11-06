#!/usr/bin/env python
# Usage:       python plot_k1_amp_trajectories.py [--root saved/multisource_k1_amp] \
#                  [--out figures/multisource_k1_amp/k1_amp_trajectories.png]
# Description: Per-epoch source trajectories for the amplitude-vs-SNR K=1 test (generate_k1_amp_jobs.py), from each
#              run's per_epoch_sources.csv: (a) depth, (b) dV normalised by that album's TRUE dV, (c) location error,
#              (d) train reconstruction loss -- vs epoch, one colour per arm, one line per seed. Truth is used only
#              for the reference lines / normalisation (honest runs).
# Date:        2026-10-20

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'
DEPTH_BOUND_KM = 12.0
ARMS = {'E1_S2_K1_weak_n012_amp333': ('A: weak SNR 7.9 x3.33 (dV 8.4e6)', 'C2'),
        'E1_S2_K1_weak_n012': ('B: weak SNR 7.9 (dV 2.5e6)', 'C0'),
        'E1_S1_K1_strong_n050': ('C: strong SNR 6.6 (dV 8.4e6)', 'C1')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='saved/multisource_k1_amp')
    parser.add_argument('--out', default='figures/multisource_k1_amp/k1_amp_trajectories.png')
    args = parser.parse_args()

    print("[1/2] Loading per_epoch_sources.csv files...")
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    labelled = set()
    n_runs = 0
    for album, (label, colour) in ARMS.items():
        truth_path = os.path.join(PROJECT_ROOT, SNR_DIR, f'{album}_truth.json')
        if not os.path.exists(truth_path):
            raise FileNotFoundError(truth_path)
        src = json.load(open(truth_path))['sources'][0]
        pattern = os.path.join(PROJECT_ROOT, args.root, 'k1snr_seed*', album, 'K1', '*', '*', 'models',
                               'per_epoch_sources.csv')
        for csv_path in sorted(glob.glob(pattern)):
            rows = list(csv.DictReader(open(csv_path)))
            epoch = np.array([int(row['epoch']) for row in rows])
            depth_km = np.array([float(row['s1_depth_km']) for row in rows])
            dv_ratio = np.array([float(row['s1_dV_m3']) for row in rows]) / src['dV_m3']
            loc_err_km = np.hypot(np.array([float(row['s1_x_km']) for row in rows]) - src['xcen_km'],
                                  np.array([float(row['s1_y_km']) for row in rows]) - src['ycen_km'])
            rec_loss = np.array([float(row['rec_loss']) for row in rows])
            kwargs = dict(color=colour, lw=1.3, alpha=0.85, label=None if label in labelled else label)
            labelled.add(label)
            for axis, values in zip(axes, (depth_km, dv_ratio, loc_err_km, rec_loss)):
                axis.plot(epoch, values, **kwargs)
            n_runs += 1
    print(f"  Loaded: {n_runs} runs")

    print("[2/2] Plotting...")
    axes[0].axhline(5.0, color='k', ls='--', lw=1, label='truth 5 km')
    axes[0].axhline(DEPTH_BOUND_KM, color='grey', ls=':', lw=1, label='prior bound 12 km')
    axes[0].set_ylabel('Depth (km)')
    axes[1].axhline(1.0, color='k', ls='--', lw=1)
    axes[1].set_ylabel('dV / true dV')
    axes[2].axhline(1.5, color='grey', ls=':', lw=1)
    axes[2].set_ylabel('Location error (km)')
    axes[3].set_yscale('log')
    axes[3].set_ylabel('Train reconstruction loss')
    for axis in axes:
        axis.set_xlabel('Epoch')
    axes[0].legend(fontsize=8)
    fig.suptitle('K=1 standard PILA: same SNR, different absolute amplitude (lines = seeds 123, 1, 2)')
    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)             # PNG only (user preference)
    print(f"Done. Output saved to {args.out}")


if __name__ == '__main__':
    main()
