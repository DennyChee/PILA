#!/usr/bin/env python
# Usage:       python summarize_sequential.py --stats <stats.csv> --thresholds <thresholds.json> \
#                  [--truth-dir synthetic/multisource/pila_stacks_v2/albums] [--out-dir figures/multisource_seq/step3]
# Description: Step 3 summary of the sequential multi-source PILA test:
#                * K_hat for every album (sequential stop with the 5 % bars) and a true-K vs K_hat table
#                  for the evaluation albums (scenarios S0..S5 x noise sets E1..E3);
#                * false-alarm rate on noise-only albums: held-out Marapi S0 albums AND leave-one-out over
#                  the 39 calibration albums (each album judged with bars computed WITHOUT it);
#                * dchi2 per step: calibration null distribution vs the bar vs evaluation albums;
#                * errors of accepted sources vs truth (matched by 3-D distance): horizontal location (km),
#                  depth (km), dV ratio, using the stack-median parameters of the K_hat fit;
#                * training / inference cost per K from compute_usage.json and infer_usage_K*.json.
# Units:       km, m^3, chi2 dimensionless, seconds.
# Date:        2026-10-06

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from sequential_select_k import select_k


def thresholds_from(rows, k_max, alpha, metric):
    return {str(k): float(np.percentile([float(r[f'{metric}_{k}']) for r in rows], 100 * (1 - alpha)))
            for k in range(1, k_max + 1)}


def accepted_source_errors(run_root, k_hat, truth):
    """Errors of the K_hat fit's sources vs truth (stack medians). Returns list of dicts."""
    if k_hat == 0 or not truth['sources']:
        return []
    seq = json.load(open(os.path.join(run_root, 'sequence.json')))
    with np.load(seq['runs'][str(k_hat)]['fit']) as f:
        med = np.median(f['params_phys'], axis=0)                      # [K, 4]
    t = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km']] for s in truth['sources']])
    cost = np.linalg.norm(t[:, None] - med[None, :, :3], axis=-1)
    ti, ri = linear_sum_assignment(cost)
    return [dict(loc_err_km=float(np.hypot(*(med[b, :2] - t[a, :2]))), depth_err_km=float(med[b, 2] - t[a, 2]),
                 dV_ratio=float(med[b, 3] / truth['sources'][a]['dV_m3'])) for a, b in zip(ti, ri)]


def timing_table(stat_rows):
    """Median training / inference seconds per K over all runs."""
    per_k = defaultdict(lambda: defaultdict(list))
    for r in stat_rows:
        seq = json.load(open(os.path.join(r['run'], 'sequence.json')))
        for k, run in seq['runs'].items():
            models = os.path.dirname(run['checkpoint'])
            usage_path = os.path.join(models, 'compute_usage.json')
            infer_path = os.path.join(models, f'infer_usage_K{k}.json')
            if os.path.exists(usage_path):
                u = json.load(open(usage_path))
                per_k[int(k)]['train_sec'].append(u['train_sec'])
                per_k[int(k)]['physics_ms_per_sample'].append(u['timing'].get('physics_decode_ms_per_sample', np.nan))
            if os.path.exists(infer_path):
                per_k[int(k)]['infer_ms_per_acq'].append(1e3 * json.load(open(infer_path))['sec_per_acquisition'])
    return {k: {m: float(np.nanmedian(v)) for m, v in d.items()} for k, d in sorted(per_k.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats', required=True)
    parser.add_argument('--thresholds', required=True)
    parser.add_argument('--truth-dir', default='synthetic/multisource/pila_stacks_v2/albums')
    parser.add_argument('--out-dir', default='figures/multisource_seq/step3')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cal = json.load(open(args.thresholds))
    alpha = cal['alpha']
    metric = cal.get('metric', 'dchi2')
    rows = list(csv.DictReader(open(args.stats)))
    cal_rows = [r for r in rows if r['album'].endswith('_cal')]
    eval_rows = [r for r in rows if not r['album'].endswith('_cal')]
    k_max = min(int(r['k_max']) for r in rows)
    print(f"[1/5] {len(cal_rows)} calibration + {len(eval_rows)} evaluation albums; bars {cal['thresholds']}")

    print("[2/5] False alarms on noise-only albums...")
    loo_k = []
    for i, r in enumerate(cal_rows):
        bars = thresholds_from(cal_rows[:i] + cal_rows[i + 1:], k_max, alpha, metric)
        loo_k.append(select_k(r, bars, metric))
    loo_fa = float(np.mean(np.array(loo_k) > 0))
    s0 = [r for r in eval_rows if '_S0_' in r['album']]
    s0_k = [select_k(r, cal['thresholds'], metric) for r in s0]
    print(f"  leave-one-out calibration: false-alarm rate {loo_fa:.1%} ({sum(k > 0 for k in loo_k)}/{len(loo_k)})")
    print(f"  held-out Marapi S0 albums: K_hat = {dict(zip([r['album'] for r in s0], s0_k))}")

    print("[3/5] Evaluation albums: true K vs K_hat, accepted-source errors...")
    table, errors, lines = defaultdict(int), defaultdict(list), []
    for r in sorted(eval_rows, key=lambda r: r['album']):
        truth = json.load(open(os.path.join(args.truth_dir, f"{r['album']}_truth.json")))
        k_hat = select_k(r, cal['thresholds'], metric)
        table[(truth['n_sources'], k_hat)] += 1
        errs = accepted_source_errors(r['run'], k_hat, truth)
        scenario = r['album'].split('_', 1)[1]
        errors[scenario].extend(errs)
        lines.append(f"  {r['album']:24s} true K={truth['n_sources']}  K_hat={k_hat}  " +
                     ' '.join(f"{metric}_{k}={float(r[f'{metric}_{k}']):8.2f}(bar {cal['thresholds'][str(k)]:.2f})"
                              for k in range(1, k_max + 1)) +
                     ('  railed@' + ','.join(str(k) for k in range(1, k_max + 1) if r.get(f'railed_{k}') == 'True')
                      if any(r.get(f'railed_{k}') == 'True' for k in range(1, k_max + 1)) else '') +
                     ('  | ' + '; '.join(f"loc {e['loc_err_km']:.2f} km, dz {e['depth_err_km']:+.2f} km, dV x{e['dV_ratio']:.2f}"
                                         for e in errs) if errs else ''))
    print('\n'.join(lines))
    true_ks = sorted({t for t, _ in table}); hat_ks = list(range(k_max + 1))
    print("  confusion (rows true K, cols K_hat): " + ' '.join(f"K_hat={k}" for k in hat_ks))
    for t in true_ks:
        print(f"    true K={t}: " + ' '.join(f"{table[(t, k)]:7d}" for k in hat_ks))

    print("[4/5] Timing per K...")
    timing = timing_table(rows)
    for k, d in timing.items():
        print(f"  K={k}: train {d.get('train_sec', np.nan):.1f} s, physics decode {d.get('physics_ms_per_sample', np.nan):.3f} ms/sample, "
              f"inference {d.get('infer_ms_per_acq', np.nan):.2f} ms/acquisition")

    print("[5/5] Figure + JSON...")
    fig, axes = plt.subplots(1, k_max, figsize=(6 * k_max, 4.8), squeeze=False)
    colors = plt.cm.tab10(np.arange(6))
    scen_names = sorted({r['album'].split('_', 1)[1] for r in eval_rows})
    for k in range(1, k_max + 1):
        ax = axes[0, k - 1]
        null = np.array([float(r[f'{metric}_{k}']) for r in cal_rows])
        lo = min(null.min(), min(float(r[f'{metric}_{k}']) for r in eval_rows))
        offset = 1 - min(lo, 0)                                          # symlog-free: shift to positive
        bins = np.logspace(np.log10(max(lo + offset, 1e-1)), np.log10(max(float(r[f'{metric}_{k}']) for r in rows) + offset), 30)
        ax.hist(null + offset, bins=bins, color='0.6', label='calibration (noise only)')
        ax.axvline(cal['thresholds'][str(k)] + offset, color='k', ls='--', label=f'{100 * (1 - alpha):.0f}th pct bar')
        for i, sc in enumerate(scen_names):
            vals = [float(r[f'{metric}_{k}']) + offset for r in eval_rows if r['album'].endswith(sc)]
            ax.plot(vals, [0.5 + 0.3 * i] * len(vals), 'o', color=colors[i % 10], label=sc)
        ax.set_xscale('log'); ax.set_xlabel(f'{metric}_{k} + {offset:.0f} (misfit improvement)')
        ax.set_ylabel('Albums'); ax.set_title(f'Step {k}: adding source {k}'); ax.legend(fontsize=7)
    plt.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'dchi2_vs_bar.png'), dpi=150); plt.close(fig)
    summary = dict(thresholds=cal['thresholds'], alpha=alpha, loo_false_alarm=loo_fa,
                   s0_k_hat=dict(zip([r['album'] for r in s0], s0_k)),
                   confusion={f'{t}->{k}': n for (t, k), n in table.items()}, errors=errors, timing=timing)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f"Done. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
