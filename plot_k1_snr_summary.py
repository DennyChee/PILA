#!/usr/bin/env python
# Usage:       python plot_k1_snr_summary.py [--k1-csv figures/multisource_k1_snr/summary.csv] \
#                  [--k2-dir figures/multisource_k1_snr/k2_reference] [--out figures/multisource_k1_snr/k1_vs_k2_snr_summary.png]
# Description: One-figure answer to "at what SNR does PILA recover the source(s) on TA69-ERA5 noise?". Panels vs
#              peak/RMS SNR (log x): (a) location error, (b) |depth error|, (c) |dV error|, (d) success fraction.
#              K=1 points = summarize_k1_snr_runs.py CSV (strong + weak albums, every seed). K=2 reference = the
#              penalties-ON `base` arm of the K=2 tuning study at SNR 3 / 6 / 12 (summarize_k2_seed_runs.py CSVs,
#              worst-matched slot per seed). Success in (d) uses the K=2 study's noisy criterion for BOTH
#              (loc < 1.5 km AND |depth| < 50 % AND |dV| < 50 %) so the two curves are directly comparable.
# Caveat:      the K=2 base arm ran 300 epochs (early_stop 300); the K=1 runs are the 150-epoch standard stage.
# Date:        2026-10-20

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# K=2 reference CSV -> peak/RMS SNR of its album (values from the albums' truth JSONs; n100 = half of n050)
K2_CSVS = {'k2_snr3_pen_base.csv': 2.98, 'k2_snr6_pen_base.csv': 5.97, 'k2_snr12_pen_base.csv': 11.93}


def success_noisy(loc_err_km, depth_err_pct, dv_err_pct):
    """K=2 study noisy criterion (summarize_k2_seed_runs.py --noisy), applied identically to K=1."""
    return loc_err_km < 1.5 and abs(depth_err_pct) < 50 and abs(dv_err_pct) < 50


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k1-csv', default='figures/multisource_k1_snr/summary.csv')
    parser.add_argument('--k2-dir', default='figures/multisource_k1_snr/k2_reference')
    parser.add_argument('--out', default='figures/multisource_k1_snr/k1_vs_k2_snr_summary.png')
    args = parser.parse_args()

    # --- [1/3] Load K=1 sweep results ---
    k1_path = os.path.join(PROJECT_ROOT, args.k1_csv)
    if not os.path.exists(k1_path):
        raise FileNotFoundError(f"{k1_path} (run summarize_k1_snr_runs.py first)")
    print(f"[1/3] Loading {args.k1_csv}...")
    k1 = {'strong': [], 'weak': []}
    for row in csv.DictReader(open(k1_path)):
        group = 'strong' if 'strong' in row['album'] else 'weak'
        k1[group].append(dict(snr=float(row['snr_peak_over_rms']), loc=float(row['loc_err_km']),
                              depth=abs(float(row['depth_err_pct'])), dv=abs(float(row['dV_err_pct'])),
                              railed=row['railed'] == 'True'))
    print(f"  Loaded: {len(k1['strong'])} strong + {len(k1['weak'])} weak runs")

    # --- [2/3] Load K=2 reference (worst slot per seed: the K=2 success needs BOTH sources) ---
    print(f"[2/3] Loading K=2 reference from {args.k2_dir}...")
    k2 = []
    for name, snr in K2_CSVS.items():
        path = os.path.join(PROJECT_ROOT, args.k2_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        for row in csv.DictReader(open(path)):
            k2.append(dict(snr=snr, loc=max(float(row['loc1_km']), float(row['loc2_km'])),
                           depth=float(row['depth_pct_max']), dv=float(row['dV_pct_max'])))
    print(f"  Loaded: {len(k2)} K=2 runs")

    # --- [3/3] Plot ---
    print("[3/3] Plotting...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    styles = {'strong': dict(color='C1', marker='o', label='K=1 strong (8.4e6 m³)'),
              'weak': dict(color='C0', marker='s', label='K=1 weak (2.5e6 m³)'),
              'k2': dict(color='k', marker='^', label='K=2 separated (worst slot)')}
    jitter_rng = np.random.default_rng(0)          # small x-jitter so seeds at one SNR do not overplot
    panels = [('loc', 'Location error (km)', 1.5), ('depth', '|Depth error| (%)', 50), ('dv', '|dV error| (%)', 50)]
    for axis, (key, ylabel, threshold) in zip(axes[:3], panels):
        for group, runs in list(k1.items()) + [('k2', k2)]:
            if not runs:
                continue
            snr = np.array([run['snr'] for run in runs]) * np.exp(jitter_rng.uniform(-0.04, 0.04, len(runs)))
            values = np.array([run[key] for run in runs])
            axis.scatter(snr, np.maximum(values, 1e-2), s=22, alpha=0.75, **styles[group])
        axis.axhline(threshold, color='grey', ls='--', lw=1)       # success threshold of the shared criterion
        axis.set_xscale('log'); axis.set_yscale('log')
        axis.set_xlabel('Peak / noise RMS (SNR)'); axis.set_ylabel(ylabel)
    axes[0].legend(fontsize=8, loc='lower left')

    # Success fraction per SNR level
    axis = axes[3]
    for group, runs in list(k1.items()) + [('k2', k2)]:
        levels = sorted({round(run['snr'], 2) for run in runs})
        fractions = [np.mean([success_noisy(run['loc'], run['depth'], run['dv'])
                              for run in runs if round(run['snr'], 2) == level]) for level in levels]
        counts = [sum(round(run['snr'], 2) == level for run in runs) for level in levels]
        axis.plot(levels, fractions, '-', **styles[group])
        for level, fraction, count in zip(levels, fractions, counts):
            axis.annotate(f'n={count}', (level, fraction), fontsize=7, xytext=(3, 3), textcoords='offset points')
    axis.set_xscale('log'); axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel('Peak / noise RMS (SNR)'); axis.set_ylabel('Success fraction')
    axis.set_title('loc<1.5 km, |depth|,|dV|<50 %', fontsize=9)
    fig.suptitle('Standard PILA on TA69 ERA5 + synthetic turbulence/white/ramp noise: single vs two sources')
    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)              # PNG only (user preference)
    print(f"Done. Output saved to {args.out}")


if __name__ == '__main__':
    main()
