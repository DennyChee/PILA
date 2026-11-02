#!/usr/bin/env python
# Usage:       python plot_k1_dvbox_compare.py [--fig-root figures/multisource_k1_dvbox]
# Description: "Is the data-scaled dV box working?" figures for the WEAK single-source album (dV 2.5e6 m^3, peak 22 mm),
#              fixed box (+-2e8 m^3, K=1 SNR sweep) vs data-scaled box (+-2e8 * data std / 12 mm, dvbox rule arm):
#                1. k1_dvbox_vs_snr.png         location / depth / dV error vs SNR, both boxes, one marker per seed
#                                               (filled = success: loc < 1 km, depth < 30 %, dV < 50 %, not railed)
#                2. k1_dvbox_trajectories.png   per-epoch depth and dV/true dV (per_epoch_sources.csv): fixed box at
#                                               SNR 7.9 (amplitude-test arm B) vs data-scaled box at every SNR
#                3. k1_dvbox_maps_snr<..>.png   truth-vs-recon maps, fixed (top) vs scaled (bottom), seed 123
#              Truth is used only for scoring / reference lines (honest runs). PNG only.
# Date:        2026-10-22

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
NOISY_DIR = 'synthetic/multisource/pila_stacks_v2/albums'
FIXED_SUMMARY = 'figures/multisource_k1_snr/summary.csv'          # fixed box, standard recipe
FIXED_REPORTS = 'figures/multisource_k1_snr'                      # per-run reports, fixed box
FIXED_TRAJ_ROOT = 'saved/multisource_k1_amp'                      # arm B = weak n012 rerun with per-epoch logging
SCALED_ROOT = 'saved/multisource_k1_dvbox/rule'
DEPTH_BOUND_KM = 12.0
STRONG_ALBUMS = ['E1_S1_K1_strong', 'E1_S1_K1_strong_n050', 'E1_S1_K1_strong_n025', 'E1_S1_K1_strong_n012',
                 'E1_S1_K1_strong_n006']                            # SNR 3.3, 6.6, 13, 26, 53
STRONG_SCALED_SUMMARY = 'figures/dvbox_rerun/k1_strong/rule_strong_summary.csv'
WEAK_ALBUMS = ['E1_S2_K1_weak', 'E1_S2_K1_weak_n050', 'E1_S2_K1_weak_n025', 'E1_S2_K1_weak_n012',
               'E1_S2_K1_weak_n006']                                # SNR 1.0, 2.0, 4.0, 7.9, 15.9
FIXED_COLOUR, SCALED_COLOUR = '0.45', 'C3'


def read_rows(path, albums):
    """Rows of a summarize_k1_snr_runs.py CSV for the given albums, numeric fields parsed."""
    rows = []
    for row in csv.DictReader(open(os.path.join(PROJECT_ROOT, path))):
        if row['album'] in albums:
            for key in ('snr_peak_over_rms', 'loc_err_km', 'depth_err_pct', 'dV_err_pct'):
                row[key] = float(row[key])
            row['success'] = row['success'] == 'True'
            rows.append(row)
    return rows


def album_truth(album):
    """Truth JSON of an album (the n100 album lives in the noisy dir, the scaled-noise ones in albums_snr)."""
    folder = NOISY_DIR if album == 'E1_S2_K1_weak' else SNR_DIR
    return json.load(open(os.path.join(PROJECT_ROOT, folder, f'{album}_truth.json')))


def truth_source(album):
    """First (only) true source of an album, km / m^3."""
    return album_truth(album)['sources'][0]


def plot_vs_snr(fixed_rows, scaled_rows, out_path,
                source_label='Weak source (dV 2.5e6 m³, peak 22 mm)'):
    """Three panels of error vs SNR, both boxes; success fraction per SNR written above each panel's points."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = (('loc_err_km', 'Location error (km)', 1.0), ('depth_err_pct', '|Depth error| (%)', 30.0),
               ('dV_err_pct', '|dV error| (%)', 50.0))
    for rows, colour, label, offset in ((fixed_rows, FIXED_COLOUR, 'fixed box  ±2e8 m³', 0.96),
                                        (scaled_rows, SCALED_COLOUR, 'data-scaled box', 1.04)):
        for axis, (key, ylabel, threshold) in zip(axes, metrics):
            for success in (True, False):
                subset = [row for row in rows if row['success'] == success]
                axis.scatter([row['snr_peak_over_rms'] * offset for row in subset],
                             [max(abs(row[key]), 1e-2) for row in subset], s=55, color=colour,
                             facecolors=colour if success else 'none', linewidths=1.6,
                             label=f"{label} ({'success' if success else 'fail'})" if key == 'loc_err_km' else None)
            axis.axhline(threshold, color='k', ls=':', lw=1)
            axis.set_xscale('log')
            axis.set_yscale('log')
            axis.set_xlabel('SNR (peak / noise RMS)')
            axis.set_ylabel(ylabel)
    # success fractions per SNR, both boxes, on the first panel
    for snr in sorted({row['snr_peak_over_rms'] for row in fixed_rows}):
        n_fixed = sum(row['success'] for row in fixed_rows if row['snr_peak_over_rms'] == snr)
        n_scaled = sum(row['success'] for row in scaled_rows if row['snr_peak_over_rms'] == snr)
        axes[0].annotate(f'{n_fixed}/3 → {n_scaled}/3', (snr, 0.97), xycoords=('data', 'axes fraction'),
                         ha='center', va='top', fontsize=9)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 4)          # head-room so the success counts clear the markers
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'{source_label}, K=1 standard PILA, seeds 123/1/2 — '
                 'success counts: fixed → data-scaled (dotted = success threshold)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_trajectory(csv_path, source):
    """Epoch, depth (km), dV / true dV from one per_epoch_sources.csv."""
    rows = list(csv.DictReader(open(csv_path)))
    epoch = np.array([int(row['epoch']) for row in rows])
    depth_km = np.array([float(row['s1_depth_km']) for row in rows])
    dv_ratio = np.array([float(row['s1_dV_m3']) for row in rows]) / source['dV_m3']
    return epoch, depth_km, dv_ratio


def plot_trajectories(out_path):
    """Depth and dV/true vs epoch: fixed box at SNR 7.9 vs data-scaled box at every SNR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    arms = [(FIXED_TRAJ_ROOT, 'E1_S2_K1_weak_n012', 'fixed box, SNR 7.9', 'k', '--')]
    scaled_colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(WEAK_ALBUMS)))
    for album, colour in zip(WEAK_ALBUMS, scaled_colours):
        snr = album_truth(album).get('snr_peak_over_rms', 0.99)      # n100 truth predates the SNR field (0.99 per sweep CSV)
        arms.append((SCALED_ROOT, album, f'data-scaled, SNR {snr:.1f}', colour, '-'))
    n_runs = 0
    for root, album, label, colour, style in arms:
        source = truth_source(album)
        pattern = os.path.join(PROJECT_ROOT, root, 'k1snr_seed*', album, 'K1', '*', '*', 'models', 'per_epoch_sources.csv')
        for index, csv_path in enumerate(sorted(glob.glob(pattern))):
            epoch, depth_km, dv_ratio = load_trajectory(csv_path, source)
            kwargs = dict(color=colour, ls=style, lw=1.3, alpha=0.85, label=label if index == 0 else None)
            axes[0].plot(epoch, depth_km, **kwargs)
            axes[1].plot(epoch, dv_ratio, **kwargs)
            n_runs += 1
    print(f"  Trajectories: {n_runs} runs")
    axes[0].axhline(5.0, color='k', lw=1, label='true depth 5 km')
    axes[0].axhline(DEPTH_BOUND_KM, color='grey', ls=':', lw=1, label='prior bound 12 km')
    axes[0].set_ylabel('Depth (km)')
    axes[1].axhline(1.0, color='k', lw=1)
    axes[1].set_ylim(-12, 4)
    axes[1].set_ylabel('dV / true dV')
    for axis in axes:
        axis.set_xlabel('Epoch')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Weak source: per-epoch source depth and volume (lines = seeds 123, 1, 2)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_maps(album, scaled_report_dir, out_path):
    """Stack the fixed-box and data-scaled truth-vs-recon map PNGs (seed 123) for one album."""
    fixed_png = os.path.join(PROJECT_ROOT, FIXED_REPORTS, album, 'seed123', f'{album}_K1_maps_truth_vs_recon.png')
    scaled_png = os.path.join(scaled_report_dir, album, 'seed123', f'{album}_K1_maps_truth_vs_recon.png')
    for path in (fixed_png, scaled_png):
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}; skipping {album}")
            return
    images = [plt.imread(fixed_png), plt.imread(scaled_png)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 14 * sum(im.shape[0] / im.shape[1] for im in images) / 2 + 1))
    for axis, image, title in zip(axes, images, ('FIXED box ±2e8 m³', 'DATA-SCALED box')):
        axis.imshow(image)
        axis.set_title(title, fontsize=14, color=FIXED_COLOUR if 'FIXED' in title else SCALED_COLOUR)
        axis.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig-root', default='figures/multisource_k1_dvbox')
    args = parser.parse_args()
    fig_root = os.path.join(PROJECT_ROOT, args.fig_root)

    print("[1/3] Error vs SNR...")
    fixed_rows = read_rows(FIXED_SUMMARY, WEAK_ALBUMS)
    scaled_rows = read_rows(os.path.join(args.fig_root, 'rule_summary.csv'), WEAK_ALBUMS)
    print(f"  Loaded: fixed {len(fixed_rows)} runs, scaled {len(scaled_rows)} runs")
    plot_vs_snr(fixed_rows, scaled_rows, os.path.join(fig_root, 'k1_dvbox_vs_snr.png'))
    # Strong album (2026-10-22 rerun), if scored: shows the rule WIDENING the box at low SNR (noise-inflated std)
    if os.path.exists(os.path.join(PROJECT_ROOT, STRONG_SCALED_SUMMARY)):
        plot_vs_snr(read_rows(FIXED_SUMMARY, STRONG_ALBUMS), read_rows(STRONG_SCALED_SUMMARY, STRONG_ALBUMS),
                    os.path.join(fig_root, 'k1_dvbox_vs_snr_strong.png'),
                    source_label='Strong source (dV 8.4e6 m³, peak 72 mm)')

    print("[2/3] Per-epoch trajectories...")
    plot_trajectories(os.path.join(fig_root, 'k1_dvbox_trajectories.png'))

    print("[3/3] Fixed vs scaled maps (seed 123)...")
    for album, snr_tag in (('E1_S2_K1_weak_n012', 'snr7.9'), ('E1_S2_K1_weak_n006', 'snr15.9')):
        plot_maps(album, os.path.join(fig_root, 'rule'), os.path.join(fig_root, f'k1_dvbox_maps_{snr_tag}.png'))
    print(f"Done. Output saved to {args.fig_root}/")


if __name__ == '__main__':
    main()
