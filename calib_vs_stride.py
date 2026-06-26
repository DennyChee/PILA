#!/usr/bin/env python
# Usage:       conda activate pila
#              python calib_vs_stride.py [--strides 10 20 30 40 50]
# Description: Map strided-UQ CALIBRATION vs member pixel-density on the synthetic okada-dike
#              cube (known truth). For each stride it reads the aggregated ensemble
#              (comparison_out/uq_marapi_okada_dike_mid_combined_uqstr<n>/uq_params.csv),
#              and reports per-stride: px/member, p5-p95 coverage of the 8 true params, the
#              fraction of members in the TRUE steep-dike basin vs the degenerate corner, and
#              the median dip/opening/width (basin diagnostics). Writes a summary CSV + plot.
# Date:        2026-06-24

import os, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
ENS_FMT = 'comparison_out/uq_marapi_okada_dike_mid_combined_uqstr{n}/uq_params.csv'
TRUTH = 'synthetic/cubes/marapi_okada_dike_combined/marapi_okada_dike_combined_truth.json'
PARAMS = ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening']
N_TOTAL_PX = 16384  # coherent px in the 128x128 synthetic cube


def load_truth():
    t = json.load(open(os.path.join(REPO, TRUTH)))
    return dict(zip(t['param_names'], t['params_per_epoch'][t['peak_epoch_index']]))


def load_members(csv_path):
    lines = [l for l in open(csv_path) if not l.startswith('#')]
    rows = [r for r in csv.DictReader(lines) if r['offset'].strip().lstrip('-').isdigit()]
    return rows


def analyze(stride, truth):
    csv_path = os.path.join(REPO, ENS_FMT.format(n=stride))
    if not os.path.exists(csv_path):
        return None
    rows = load_members(csv_path)
    m = len(rows)
    if m == 0:
        return None
    vals = {p: np.array([float(r[p]) for r in rows]) for p in PARAMS}
    # p5-p95 coverage of the 8 true params
    inside = sum(np.percentile(vals[p], 5) <= truth[p] <= np.percentile(vals[p], 95)
                 for p in PARAMS)
    # Basin classification per member: TRUE steep-dike (dip>50 AND opening<4 AND width>1500)
    # vs degenerate corner (the thin/shallow/high-opening trade-off).
    true_basin = np.sum((vals['dip'] > 50) & (vals['opening'] < 4) & (vals['width'] > 1500))
    return {
        'stride': stride, 'members': m, 'px_per_member': N_TOTAL_PX // stride,
        'coverage_p5_95': int(inside),
        'frac_true_basin': true_basin / m,
        'median_dip': float(np.median(vals['dip'])),
        'median_opening': float(np.median(vals['opening'])),
        'median_width': float(np.median(vals['width'])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strides', type=int, nargs='+', default=[10, 20, 30, 40, 50])
    ap.add_argument('--out', default='comparison_out/uq_calibration_vs_stride')
    args = ap.parse_args()
    out_dir = os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)
    truth = load_truth()
    print("truth(peak): " + ", ".join(f"{k}={truth[k]:.4g}" for k in PARAMS))
    print(f"\n{'stride':>6} {'mbrs':>5} {'px/mbr':>7} {'cover/8':>8} {'%true_basin':>12} "
          f"{'med_dip':>8} {'med_open':>9} {'med_wid':>8}")
    res = []
    for s in sorted(args.strides):
        r = analyze(s, truth)
        if r is None:
            print(f"{s:>6}  <no aggregated csv yet>"); continue
        res.append(r)
        print(f"{r['stride']:>6} {r['members']:>5} {r['px_per_member']:>7} "
              f"{r['coverage_p5_95']:>6}/8 {100*r['frac_true_basin']:>10.0f}% "
              f"{r['median_dip']:>8.1f} {r['median_opening']:>9.2f} {r['median_width']:>8.0f}")
    if not res:
        print("\nNo aggregated ensembles found yet."); return

    # Summary CSV
    csv_path = os.path.join(out_dir, 'calibration_vs_stride.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    print(f"\nWrote {csv_path}")

    # Plot: coverage + %true-basin vs px/member (truth: dip 85, opening 1.5, width 3000)
    px = [r['px_per_member'] for r in res]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(px, [r['coverage_p5_95'] for r in res], 'o-', label='p5-p95 coverage (/8)')
    ax[0].plot(px, [100*r['frac_true_basin'] for r in res], 's--', label='% members in true basin')
    ax[0].set_xlabel('pixels per ensemble member'); ax[0].set_ylabel('coverage / %')
    ax[0].set_title('UQ calibration vs member density'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(px, [r['median_dip'] for r in res], 'o-', label='median dip (truth 85)')
    ax[1].plot(px, [r['median_opening'] for r in res], 's-', label='median opening (truth 1.5)')
    ax[1].axhline(85, color='C0', ls=':', alpha=0.6); ax[1].axhline(1.5, color='C1', ls=':', alpha=0.6)
    ax[1].set_xlabel('pixels per ensemble member'); ax[1].set_ylabel('median value')
    ax[1].set_title('Ensemble-median basin (deg / m)'); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'calibration_vs_stride.{ext}'), dpi=150)
    print(f"Wrote {out_dir}/calibration_vs_stride.png/.pdf")


if __name__ == '__main__':
    main()
