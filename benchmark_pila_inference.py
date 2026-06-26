#!/usr/bin/env python
# Usage:       python benchmark_pila_inference.py [--saved saved] [--out comparison_out]
#                                                 [--n-timing 50] [--configs A B ...]
# Description: For each trained PILA InSAR config, time an isolated single-scene inference
#              (one amortized forward pass) and record BOTH fit metrics — physics-only
#              (the source, fair vs classical) and physics+residual (PILA's deployed
#              reconstruction). Closes publication blockers (b) physics+residual metric
#              logging and (c) the PILA side of the speed comparison vs classical SA/MCMC.
# Date:        2026-06-18
#
# Scientific notes:
#   * physics-only (x_P)   = pure physical source (Mogi/Okada/Sun69). This is the
#     apples-to-apples quantity for the classical SA/Bayesian comparison.
#   * physics+residual (x_PB) = x_P + low-rank z_aux augmentation = PILA's operational
#     output. Reported alongside, NOT as the classical comparison (classical has no
#     residual channel).
#   * Inference latency is the median of repeated forward passes after warm-up, on the
#     SAME machine the table reports. Classical per-scene optimisation time must be timed
#     separately (MATLAB SA + NBay MCMC) to complete the speed comparison.
#   * LOS sign: + = motion toward satellite. Metrics in mm; global affine z-score cancels.

import argparse
import csv
import glob
import os
import time

import numpy as np
import torch

# Reuse the EXACT inference path the verification figures use, so numbers match.
from plot_insar_results import build_model, hard_z_flags, _r2_rmse
from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from datasets.displacementGPS import time_feats
from utils import read_json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def latest_best_checkpoint(saved_root, config_name):
    """Most recent model_best.pth for a config, or None.

    Inputs : saved_root (str) PILA saved/ dir; config_name (str).
    Output : path to the newest run's model_best.pth (lexical timestamp sort), or None.
    """
    pattern = os.path.join(saved_root, '*', config_name, '*', 'models', 'model_best.pth')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def configs_from_comparison_table(out_dir):
    """Read the config names already present in comparison_out/comparison_table.csv.

    Keeps this benchmark aligned 1:1 with the PILA-vs-classical comparison set.
    Output : list[str] of unique config names (order preserved).
    """
    csv_path = os.path.join(out_dir, 'comparison_table.csv')
    if not os.path.exists(csv_path):
        return []
    seen = []
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            if row['config'] not in seen:
                seen.append(row['config'])
    return seen


def benchmark_one(config_name, ckpt_path, device, n_timing, epoch_index=-1):
    """Time isolated inference + compute both fit-metric pairs for one config.

    Inputs : config_name (str); ckpt_path (str) model_best.pth; device; n_timing (int)
             number of timed forward passes; epoch_index (int) time-series epoch (-1 = final).
    Output : dict with N points, inference latency (ms, median + p90), and
             physics-only / physics+residual R² and RMSE (mm).
    """
    cfg_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    config = read_json(cfg_path)
    insar_args = config['arch']['args']['insar']

    # --- Build model + restore weights/tau/r (same as plot_insar_results) ---
    model = build_model(config, ckpt_path, device)

    # --- Load the multilooked LOS field (the memoized points the model trained on) ---
    d = load_insar_mintpy(
        insar_args['timeseries'], insar_args['geometry'], insar_args.get('mask'),
        insar_args['lat0'], insar_args['lon0'],
        multilook=int(insar_args.get('multilook', 20)),
        coh_valid_frac=insar_args.get('coh_valid_frac', 0.5),
        bbox=insar_args.get('bbox'), verbose=False)

    date_str = d.dates[epoch_index]
    std_series = (d.los_points_mm - d.x_mean_global) / d.x_scale_global
    target_std = std_series[epoch_index].astype(np.float32)               # [N]

    tfeat_dim = config['arch']['args'].get('time_feat_dim', 4)
    tfeat = torch.tensor(time_feats(date_str, time_feat_dim=tfeat_dim),
                         dtype=torch.float32).unsqueeze(0).to(device)      # [1, T]
    x_in = torch.tensor(target_std, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N]
    hz_phy, hz_aux = hard_z_flags(config)

    def _one_forward():
        with torch.no_grad():
            return model(x_in, t=tfeat, inference=True,
                         hard_z_phy=hz_phy, hard_z_aux=hz_aux, const=None)

    # --- Warm-up (3 passes) so lazy init / caches don't pollute the timing ---
    for _ in range(3):
        _one_forward()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # --- Timed forward passes ---
    latencies_ms = np.empty(n_timing, dtype=np.float64)
    for i in range(n_timing):
        t_start = time.perf_counter()
        _latent_phy, _latent_aux, x_PB, x_P = _one_forward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies_ms[i] = (time.perf_counter() - t_start) * 1000.0

    # --- Fit metrics in mm (de-standardize with the global scaler) ---
    actual_mm = target_std * d.x_scale_global + d.x_mean_global
    phys_mm = x_P[0].cpu().numpy() * d.x_scale_global + d.x_mean_global    # physics-only
    full_mm = x_PB[0].cpu().numpy() * d.x_scale_global + d.x_mean_global   # physics+residual
    r2_phys, rmse_phys = _r2_rmse(phys_mm, actual_mm)
    r2_full, rmse_full = _r2_rmse(full_mm, actual_mm)

    return {
        'config': config_name,
        'epoch': date_str,
        'N': int(d.n_points),
        'infer_ms_median': round(float(np.median(latencies_ms)), 3),
        'infer_ms_p90': round(float(np.percentile(latencies_ms, 90)), 3),
        'r2_phys': round(float(r2_phys), 4),
        'rmse_phys_mm': round(float(rmse_phys), 3),
        'r2_full': round(float(r2_full), 4),
        'rmse_full_mm': round(float(rmse_full), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--saved', default=os.path.join(CURRENT_DIR, 'saved'))
    parser.add_argument('--out', default=os.path.join(CURRENT_DIR, 'comparison_out'))
    parser.add_argument('--n-timing', type=int, default=50,
                        help='number of timed forward passes per config (median reported)')
    parser.add_argument('--configs', nargs='*', default=None,
                        help='config names (default: those in comparison_table.csv)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[1/3] Device: {device}; timed passes per config: {args.n_timing}")

    config_names = args.configs or configs_from_comparison_table(args.out)
    if not config_names:
        raise SystemExit("No configs given and none found in comparison_table.csv "
                         "(pass --configs or run compare_pila_vs_classical.py first).")
    print(f"[2/3] Benchmarking {len(config_names)} configs ...")

    rows = []
    for idx, config_name in enumerate(config_names, 1):
        ckpt = latest_best_checkpoint(args.saved, config_name)
        if ckpt is None:
            print(f"  [{idx}/{len(config_names)}] {config_name}: NO checkpoint found, skipping")
            continue
        row = benchmark_one(config_name, ckpt, device, args.n_timing)
        rows.append(row)
        print(f"  [{idx}/{len(config_names)}] {config_name}: "
              f"N={row['N']:>5}  infer={row['infer_ms_median']:.2f} ms  "
              f"R2_phys={row['r2_phys']:+.3f}  R2_full={row['r2_full']:+.3f}")

    # --- Write CSV next to the comparison outputs ---
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, 'pila_inference_benchmark.csv')
    fieldnames = ['config', 'epoch', 'N', 'infer_ms_median', 'infer_ms_p90',
                  'r2_phys', 'rmse_phys_mm', 'r2_full', 'rmse_full_mm']
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Headline summary ---
    if rows:
        med_ms = np.median([r['infer_ms_median'] for r in rows])
        mean_dphi = np.mean([r['r2_full'] - r['r2_phys'] for r in rows])
        print(f"[3/3] Wrote {out_csv} ({len(rows)} configs)")
        print(f"  Median single-scene inference latency: {med_ms:.2f} ms ({device})")
        print(f"  Mean R² uplift from z_aux residual (full - phys): +{mean_dphi:.3f}")
    print("Done.")


if __name__ == '__main__':
    main()
