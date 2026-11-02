#!/usr/bin/env python
# Usage:       python summarize_k1_snr_runs.py [--root saved/multisource_k1_snr] [--out-csv figures/multisource_k1_snr/summary.csv]
# Description: Scores the single-source SNR sweep (generate_k1_snr_jobs.py). For every run found under
#              <root>/k1snr_seed<S>/<album>/K1/*/*/models/fit_K1.npz: median over acquisitions of the recovered
#              x, y, depth, dV, against the truth JSON of that album (truth used ONLY for scoring); peak/RMS SNR;
#              railed flag (depth within 0.2 km of the 12 km prior bound); success flag.
#              Success = location error <= 1 km AND |depth error| <= 30 % AND |dV error| <= 50 % (static source). Thresholds are a pragmatic pass/fail marker, not a calibrated test.
# Assumptions: fit_K1.npz params_phys rows = [xcen_km, ycen_km, depth_km, dV_m3] (checked against the std stats.csv);
#              truth sources are static over the album, so the median over acquisitions is a fair point estimate.
# Date:        2026-10-20

import argparse
import csv
import glob
import json
import os
import re

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR = 'synthetic/multisource/pila_stacks_v2/albums'
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'
NONOISE_DIR = 'synthetic/multisource/pila_stacks_v2/albums_nonoise'
DEPTH_BOUND_KM = 12.0           # upper depth bound of the standard Mogi prior (railing check)
NOISE_RMS_FULL_MM = 21.82       # stack noise RMS of the E1 K1 albums at scale 1.0 (make_snr_scaled_albums.py output)


def load_truth(album):
    """Truth JSON for an album name (scaled twins live in albums_snr/). Returns (dict, peak/RMS SNR)."""
    path = os.path.join(PROJECT_ROOT, SNR_DIR if re.search(r'_n\d{3}', album) else NOISY_DIR, f'{album}_truth.json')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    truth = json.load(open(path))
    # Scaled twins record their own SNR (max |noise-free field| over all dates / stack noise RMS). The original album
    # (scale 1.0) gets exactly the same definition from its noise-free twin, so all SNRs are on one scale.
    snr = truth.get('snr_peak_over_rms')
    if len(truth['sources']) != 1:          # not scored by this study (main() skips it); no twin needed
        return truth, float('nan')
    if snr is None:
        nonoise_path = os.path.join(PROJECT_ROOT, NONOISE_DIR, f'{album}_nonoise.npz')
        if os.path.exists(nonoise_path):
            with np.load(nonoise_path) as twin:
                snr = float(np.abs(twin['los_asc_m']).max() * 1e3) / NOISE_RMS_FULL_MM
        else:
            # No twin (E2/E3 albums): truth's field_peak_mm is ~1.4 % below the twin max, so this SNR is approximate.
            print(f"  note: no noise-free twin for {album}; SNR from field_peak_mm (approximate)")
            snr = truth['field_peak_mm'] / NOISE_RMS_FULL_MM
    return truth, snr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='saved/multisource_k1_snr')
    parser.add_argument('--out-csv', default='figures/multisource_k1_snr/summary.csv')
    args = parser.parse_args()

    # --- [1/2] Find and score runs ---
    pattern = os.path.join(PROJECT_ROOT, args.root, 'k1snr_seed*', '*', 'K1', '*', '*', 'models', 'fit_K1.npz')
    # Keep only the newest timestamped run per (seed, album) so a rerun into the same root is not double-counted.
    newest_by_run = {}
    for fit_path in sorted(glob.glob(pattern)):           # sorted: later MMDD_HHMMSS overwrites earlier
        rel = os.path.relpath(fit_path, os.path.join(PROJECT_ROOT, args.root)).split(os.sep)
        newest_by_run[(rel[0], rel[1])] = fit_path
    fit_paths = sorted(newest_by_run.values())
    print(f"[1/2] Scoring {len(fit_paths)} runs under {args.root}...")
    rows = []
    for fit_path in fit_paths:
        rel = os.path.relpath(fit_path, os.path.join(PROJECT_ROOT, args.root)).split(os.sep)
        seed = int(rel[0].replace('k1snr_seed', ''))
        album = rel[1]
        truth, snr = load_truth(album)
        if len(truth['sources']) != 1:      # this study scores single-source albums only (skip null / K>=2 albums)
            continue
        src = truth['sources'][0]
        with np.load(fit_path) as fit:
            params = np.median(fit['params_phys'][:, 0, :], axis=0)       # [x_km, y_km, depth_km, dV_m3]
        loc_err_km = float(np.hypot(params[0] - src['xcen_km'], params[1] - src['ycen_km']))
        depth_err_pct = float(100 * (params[2] - src['depth_km']) / src['depth_km'])
        dv_err_pct = float(100 * (params[3] - src['dV_m3']) / abs(src['dV_m3']))
        success = loc_err_km <= 1.0 and abs(depth_err_pct) <= 30 and abs(dv_err_pct) <= 50
        rows.append(dict(album=album, seed=seed, snr_peak_over_rms=round(snr, 2),
                         x_km=round(params[0], 2), y_km=round(params[1], 2), depth_km=round(params[2], 2),
                         dV_m3=f'{params[3]:.3e}', loc_err_km=round(loc_err_km, 2), depth_err_pct=round(depth_err_pct, 1),
                         dV_err_pct=round(dv_err_pct, 1), railed=bool(params[2] > DEPTH_BOUND_KM - 0.2), success=success))
    if not rows:
        print("  No finished runs yet.")
        return

    # --- [2/2] Print table (sorted by album then SNR) and save CSV ---
    rows.sort(key=lambda row: (row['album'].split('_n')[0], row['snr_peak_over_rms'], row['seed']))
    print("[2/2] Results:")
    header = list(rows[0])
    # Album names are long (e.g. E1_S1_K1_strong_n050): give that column full width, truncate the others to 10.
    print('  ' + f"{'album':<22s}" + ' '.join(f'{key[:10]:>10s}' for key in header[1:]))
    for row in rows:
        print('  ' + f"{row['album']:<22s}" + ' '.join(f'{str(row[key])[:10]:>10s}' for key in header[1:]))
    out_csv = os.path.join(PROJECT_ROOT, args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done. Output saved to {args.out_csv}")


if __name__ == '__main__':
    main()
