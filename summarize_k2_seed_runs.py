#!/usr/bin/env python
# Usage:       python summarize_k2_seed_runs.py --tag-template 'k2only_seed{seed}_cos300_nopen' --seeds 1-24 \
#                  --truth synthetic/multisource/clean_scenarios/K2_separated_static_truth.json
#              (noisy twin) --tag-template 'noisyE1_k2only_seed{seed}_cos300_nopen' --seeds 1-16 \
#                  --truth synthetic/multisource/pila_stacks_v2/albums/E1_S3_K2_separated_truth.json --noisy
# Description: Table for the honest K=2-only seed runs (run_sequential_album.py --from-scratch-k 2): per seed the final
#              reconstruction loss, physics R2 (vs the data), RMS(data - physics), the Hungarian-matched location /
#              depth / dV errors against the truth, and a success flag. The truth is used ONLY for scoring.
#              Noise-free success = final loss < 500 (seed 8 reached 188, failures sit above 1000). With --noisy the
#              loss floor is the noise level, so success = both sources within 1.5 km AND depth/dV within 50%.
# Date:        2026-10-08

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_seeds(text):
    """'1-8,123' -> [1..8, 123]."""
    seeds = []
    for part in text.split(','):
        if '-' in part:
            low, high = part.split('-')
            seeds += list(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='saved/multisource_k2_multiseed')
    parser.add_argument('--tag-template', required=True, help="e.g. 'k2only_seed{seed}_cos300_nopen'")
    parser.add_argument('--seeds', default='1-8,123')
    parser.add_argument('--truth', required=True)
    parser.add_argument('--noisy', action='store_true')
    parser.add_argument('--out-csv', default=None)
    args = parser.parse_args()
    truth_path = os.path.join(PROJECT_ROOT, args.truth) if not os.path.isabs(args.truth) else args.truth
    if not os.path.exists(truth_path):
        raise FileNotFoundError(truth_path)
    with open(truth_path) as handle:
        truth_sources = json.load(handle)['sources']
    truth = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km'], s['dV_m3']] for s in truth_sources])
    rows = []
    print(f"{'seed':>4} {'final loss':>11} {'R2':>6} {'RMS mm':>7} | {'loc1 km':>7} {'loc2 km':>7} {'dep% max':>8} {'dV% max':>8} | ok")
    for seed in parse_seeds(args.seeds):
        tag = args.tag_template.format(seed=seed)
        pattern = os.path.join(PROJECT_ROOT, args.root, tag, '*', 'K2', '*', '*', 'models', 'fit_K2.npz')
        fits = glob.glob(pattern)
        if not fits:
            print(f"{seed:4d} MISSING ({tag})")
            continue
        with np.load(fits[0]) as fit:
            params = np.median(fit['params_phys'], axis=0)      # static scene: epochs equivalent
            data_mm = fit['data_mm']; physics_mm = fit['xP_src_mm'].sum(axis=1)
        log_path = glob.glob(os.path.join(os.path.dirname(os.path.dirname(fits[0])), 'log', 'info.log'))[0]
        with open(log_path) as handle:
            losses = [float(m.group(1)) for line in handle
                      for m in [re.search(r'Epoch \d+ Summary - Loss: [\d.eE+-]+, Rec: ([\d.eE+-]+)', line)] if m]
        r2 = 1 - np.nansum((data_mm - physics_mm) ** 2) / np.nansum(data_mm ** 2)
        rms_mm = float(np.sqrt(np.nanmean((data_mm - physics_mm) ** 2)))
        rows_idx, cols_idx = linear_sum_assignment(np.linalg.norm(truth[:, None, :2] - params[None, :, :2], axis=-1))
        loc = [float(np.hypot(*(params[b, :2] - truth[a, :2]))) for a, b in zip(rows_idx, cols_idx)]
        depth_pct = max(abs(params[b, 2] - truth[a, 2]) / truth[a, 2] * 100 for a, b in zip(rows_idx, cols_idx))
        dV_pct = max(abs(params[b, 3] - truth[a, 3]) / abs(truth[a, 3]) * 100 for a, b in zip(rows_idx, cols_idx))
        ok = (max(loc) < 1.5 and depth_pct < 50 and dV_pct < 50) if args.noisy else losses[-1] < 500
        rows.append(dict(seed=seed, final_loss=losses[-1], r2=float(r2), rms_mm=rms_mm, loc1_km=loc[0], loc2_km=loc[1],
                         depth_pct_max=depth_pct, dV_pct_max=dV_pct, success=bool(ok), slots=params.round(3).tolist()))
        print(f"{seed:4d} {losses[-1]:11.0f} {r2:6.3f} {rms_mm:7.2f} | {loc[0]:7.2f} {loc[1]:7.2f} {depth_pct:8.0f} {dV_pct:8.0f} | {'*' if ok else ''}")
    n_ok = sum(r['success'] for r in rows)
    print(f"success: {n_ok}/{len(rows)} seeds ({'noisy criterion' if args.noisy else 'final loss < 500'})")
    if args.out_csv and rows:
        with open(args.out_csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(f"Done. Output saved to {args.out_csv}")


if __name__ == '__main__':
    main()
