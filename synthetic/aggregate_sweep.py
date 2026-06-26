#!/usr/bin/env python
# Usage:       python -m synthetic.aggregate_sweep --out synthetic/snr_sweep_ml1 \
#                  --base-name marapi_mogi_buildup_combined --multilook 1
# Description: Aggregate all completed SNR-sweep points (per-point metrics written by
#              run_one_snr_point.py) into one CSV + a breakdown-vs-SNR curve. Reads each
#              eval/<tag>/<tag>_metrics.json and the matching cube truth json (for the
#              realized SNR), sorts by SNR, and plots location error / dV %err / LOS RMSE.
# Date:        2026-06-22

import argparse
import csv
import glob
import json
import os

import numpy as np

from synthetic import plots


def aggregate(out_dir, base_name, multilook):
    rows = []
    for mj in sorted(glob.glob(os.path.join(out_dir, 'eval', f'{base_name}_*', '*_metrics.json'))):
        m = json.load(open(mj))
        tag = m['name']
        truth_p = os.path.join('synthetic', 'cubes', tag, f'{tag}_truth.json')
        if not os.path.exists(truth_p):
            print(f"  skip {tag}: no truth json"); continue
        truth = json.load(open(truth_p))
        snr = truth['noise'].get('realized_snr_db', float('nan'))
        pp = m['per_param']
        row = {'tag': tag, 'realized_snr_db': snr, 'n_points': m.get('n_points'),
               'los_rmse_mm_peak': m['los_rmse_mm_peak'],
               'loc_err_km_peak': m['source_loc_err_km_peak'],
               'loc_err_km_strong': m['source_loc_err_km_strong_mean']}
        for p in pp:
            row[f'{p}_pcterr_peak'] = pp[p]['pct_err_peak']
        rows.append(row)

    if not rows:
        print("No completed points found."); return
    rows.sort(key=lambda r: r['realized_snr_db'])

    csv_path = os.path.join(out_dir, f'{base_name}_snr_sweep.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} points)")

    snr = [r['realized_snr_db'] for r in rows]
    loc_err_m = [r['loc_err_km_peak'] * 1000 for r in rows]
    amp_param = 'dV_pcterr_peak' if 'dV_pcterr_peak' in rows[0] else \
        ('opening_pcterr_peak' if 'opening_pcterr_peak' in rows[0] else None)
    secondary = {'LOS RMSE (mm)': [r['los_rmse_mm_peak'] for r in rows]}
    if amp_param:
        secondary[f'{amp_param.replace("_pcterr_peak","")} %err'] = [r[amp_param] for r in rows]
    cell_m = (40000.0 / 128) * (multilook or 1)
    plots.plot_breakdown_curve(out_dir, base_name, snr, loc_err_m,
                               secondary=secondary, breakdown_threshold_m=cell_m)
    print(f"Wrote {os.path.join(out_dir, base_name + '_breakdown_snr.png')}")
    # Print a compact table
    print(f"\n  {'SNR dB':>8}{'loc_err m':>12}{'dV %err':>10}{'LOS RMSE':>10}")
    for r in rows:
        dvp = r.get('dV_pcterr_peak', float('nan'))
        print(f"  {r['realized_snr_db']:>8.2f}{r['loc_err_km_peak']*1000:>12.0f}"
              f"{dvp:>10.1f}{r['los_rmse_mm_peak']:>10.2f}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate SNR-sweep points into CSV + breakdown curve.")
    ap.add_argument('--out', default=os.path.join('synthetic', 'snr_sweep_ml1'))
    ap.add_argument('--base-name', default='marapi_mogi_buildup_combined')
    ap.add_argument('--multilook', type=int, default=1)
    args = ap.parse_args()
    aggregate(args.out, args.base_name, args.multilook)


if __name__ == '__main__':
    main()
