#!/usr/bin/env python
# Usage:       python compare_k2_keepbest.py [--out-dir figures/multisource_k2_keepbest]
# Description: Scores the noisy two-source dV-range test (generate_k2_keepbest_jobs.py): E1_S3_K2_separated at SNR 6 / 12,
#              standard +-2e8 vs data-scaled (anchor 60 mm) dV range, 16 seeds, each scored at the FINAL epoch
#              (models/fit_K2.npz) and at the LOWEST val_rec_loss epoch (models/lowest_loss/fit_K2.npz).
#              Metrics and success rule are identical to summarize_k2_seed_runs.py --noisy: stack-median parameters,
#              Hungarian matching on horizontal position, success = both sources within 1.5 km AND worst depth / dV error
#              < 50 %. Also reported: location-only success (both < 1 km). Truth is used ONLY for scoring.
# Output:      keepbest_runs.csv (one row per run x epoch choice), keepbest_summary.csv (success counts), printed table
# Date:        2026-10-22

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = 'saved/multisource_k2_tune'
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'
ALBUMS = {'n050': 6.0, 'n025': 11.9}
RANGES = {'std': '±2e8 (standard)', 'box60': 'data-scaled, 60 mm anchor'}


def score_fit(fit_path, truth):
    """Matched errors of one fit_K2.npz vs truth [2 x (x km, y km, depth km, dV m3)]; same rules as summarize_k2_seed_runs."""
    with np.load(fit_path) as fit:
        params = np.median(fit['params_phys'], axis=0)                 # static scene: all epochs equivalent
        data_mm, physics_mm = fit['data_mm'], fit['xP_src_mm'].sum(axis=1)
    rows_idx, cols_idx = linear_sum_assignment(np.linalg.norm(truth[:, None, :2] - params[None, :, :2], axis=-1))
    loc_km = [float(np.hypot(*(params[b, :2] - truth[a, :2]))) for a, b in zip(rows_idx, cols_idx)]
    depth_pct = max(abs(params[b, 2] - truth[a, 2]) / truth[a, 2] * 100 for a, b in zip(rows_idx, cols_idx))
    dv_pct = max(abs(params[b, 3] - truth[a, 3]) / abs(truth[a, 3]) * 100 for a, b in zip(rows_idx, cols_idx))
    r2 = 1 - np.nansum((data_mm - physics_mm) ** 2) / np.nansum(data_mm ** 2)
    return {'loc_max_km': max(loc_km), 'depth_pct_max': float(depth_pct), 'dV_pct_max': float(dv_pct), 'physics_r2': float(r2),
            'success': bool(max(loc_km) < 1.5 and depth_pct < 50 and dv_pct < 50), 'loc_only_success': bool(max(loc_km) < 1.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='figures/multisource_k2_keepbest')
    args = parser.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[1/2] Scoring 2 SNR x 2 ranges x 16 seeds x {final, lowest-loss}...")
    rows = []
    for noise_tag, snr in ALBUMS.items():
        with open(os.path.join(PROJECT_ROOT, SNR_DIR, f'E1_S3_K2_separated_{noise_tag}_truth.json')) as handle:
            truth = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km'], s['dV_m3']] for s in json.load(handle)['sources']])
        for range_tag in RANGES:
            for seed in range(1, 17):
                models = glob.glob(os.path.join(PROJECT_ROOT, SAVE_ROOT, f'k2tune_snrE1{noise_tag}kb_{range_tag}_wr100_seed{seed}',
                                                '*', 'K2', '*', '*', 'models'))
                if len(models) != 1:
                    raise FileNotFoundError(f'{noise_tag} {range_tag} seed {seed}: {len(models)} run dirs')
                with open(os.path.join(models[0], 'lowest_loss', 'lowest_loss.json')) as handle:
                    lowest = json.load(handle)
                for choice, fit_path in (('final', os.path.join(models[0], 'fit_K2.npz')),
                                         ('lowest_loss', os.path.join(models[0], 'lowest_loss', 'fit_K2.npz'))):
                    rows.append({'snr': snr, 'range': range_tag, 'seed': seed, 'epoch_choice': choice,
                                 'epoch': 400 if choice == 'final' else lowest['epoch'],
                                 'lowest_val_rec_loss': lowest['val_rec_loss'], **score_fit(fit_path, truth)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, 'keepbest_runs.csv'), index=False, float_format='%.4g')

    print("[2/2] Success counts (of 16 seeds):")
    summary = (table.groupby(['snr', 'range', 'epoch_choice'])
               .agg(success=('success', 'sum'), loc_only=('loc_only_success', 'sum'),
                    median_loc_km=('loc_max_km', 'median'), median_lowest_epoch=('epoch', 'median')).reset_index())
    summary.to_csv(os.path.join(out_dir, 'keepbest_summary.csv'), index=False, float_format='%.4g')
    print(summary.to_string(index=False))
    winners = table[table['success']].sort_values(['snr', 'range', 'epoch_choice', 'seed'])
    print("\nSuccessful runs:")
    print(winners[['snr', 'range', 'epoch_choice', 'seed', 'epoch', 'loc_max_km', 'depth_pct_max', 'dV_pct_max', 'physics_r2']]
          .round(2).to_string(index=False))
    print(f"Done. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
