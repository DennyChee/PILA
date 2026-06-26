#!/usr/bin/env python
# Usage:       conda activate pila
#              python compare_subsample_schemes.py
# Description: Benchmark two PILA UQ subsampling schemes against the SAME synthetic truth on
#              the Marapi Okada-dike cube, at MATCHED member density (~819 px/member):
#                (A) strided disjoint decimation  (stride=20, 20 members)
#                (B) spatial block-bootstrap        (k=819, block=3px, 20 members)
#              For each scheme it reports, per source parameter: the true value, the ensemble
#              p5 / median / p95, whether truth is covered, and the p5-p95 interval WIDTH.
#              Independent-pixel/stride sampling treats spatially-correlated atmospheric noise
#              as independent and tends to give over-tight (under-covering) intervals; the
#              block bootstrap should widen them. Also reports overall coverage (/8) and the
#              fraction of members in the true steep-dike basin. Writes a table + bar plot.
# Date:        2026-06-24

import os
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
TRUTH = 'synthetic/cubes/marapi_okada_dike_combined/marapi_okada_dike_combined_truth.json'
PARAMS = ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening']
UNIT = {'xoff': 'm', 'yoff': 'm', 'depth': 'm', 'strike': 'deg', 'dip': 'deg',
        'length': 'm', 'width': 'm', 'opening': 'm'}

# (label, ensemble uq_params.csv) -- aggregate_uq output per scheme.
SCHEMES = [
    ('stride-20', 'comparison_out/uq_marapi_okada_dike_mid_combined_uqstr20/uq_params.csv'),
    ('bootstrap-k819-b3', 'comparison_out/uq_marapi_okada_dike_mid_combined_bootk819_b3/uq_params.csv'),
]


def load_truth():
    with open(os.path.join(REPO, TRUTH)) as fh:
        t = json.load(fh)
    return dict(zip(t['param_names'], t['params_per_epoch'][t['peak_epoch_index']]))


def load_members(csv_path):
    """Read the per-member rows from an aggregate_uq uq_params.csv (skip comment/summary)."""
    with open(csv_path) as fh:
        lines = [l for l in fh if not l.startswith('#')]
    rows = [r for r in csv.DictReader(lines) if r.get('offset', '').strip().lstrip('-').isdigit()]
    return rows


def analyze(csv_path, truth):
    rows = load_members(csv_path)
    vals = {p: np.array([float(r[p]) for r in rows]) for p in PARAMS}
    out = {'n_members': len(rows), 'param': {}}
    inside = 0
    for p in PARAMS:
        p5, p50, p95 = np.percentile(vals[p], [5, 50, 95])
        cov = bool(p5 <= truth[p] <= p95)
        inside += cov
        out['param'][p] = dict(p5=p5, p50=p50, p95=p95, width=p95 - p5, covered=cov)
    out['coverage'] = inside
    out['frac_true_basin'] = float(np.mean((vals['dip'] > 50) & (vals['opening'] < 4) & (vals['width'] > 1500)))
    out['vals'] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='comparison_out/subsample_scheme_comparison')
    args = ap.parse_args()
    out_dir = os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)

    truth = load_truth()
    results = {}
    for label, rel in SCHEMES:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            print(f"  MISSING: {label} -> {path} (aggregate it first); skipping")
            continue
        results[label] = analyze(path, truth)

    if len(results) < 2:
        raise SystemExit("Need both schemes aggregated to compare. Run aggregate_uq for each.")

    labels = list(results.keys())
    # ---------------- Console table ----------------
    print("\n" + "=" * 96)
    print(f"SUBSAMPLING SCHEME COMPARISON vs synthetic truth (matched density ~819 px/member)")
    print("=" * 96)
    for lab in labels:
        r = results[lab]
        print(f"  {lab:<20} members={r['n_members']:>3}  coverage={r['coverage']}/8  "
              f"true-basin={100*r['frac_true_basin']:.0f}%")
    print("-" * 96)
    hdr = f"{'param':<8}{'truth':>10}" + "".join(
        f"{lab+' p50':>16}{'p5-p95 width':>14}{'cov':>5}" for lab in labels)
    print(hdr)
    for p in PARAMS:
        line = f"{p:<8}{truth[p]:>10.4g}"
        for lab in labels:
            pr = results[lab]['param'][p]
            line += f"{pr['p50']:>16.4g}{pr['width']:>14.4g}{('Y' if pr['covered'] else '.'):>5}"
        print(line)
    print("-" * 96)

    # Mean interval-width ratio (bootstrap / stride): >1 means bootstrap intervals are wider.
    if 'stride-20' in results and any('bootstrap' in l for l in labels):
        boot = next(l for l in labels if 'bootstrap' in l)
        ratios = [results[boot]['param'][p]['width'] / results['stride-20']['param'][p]['width']
                  for p in PARAMS if results['stride-20']['param'][p]['width'] > 0]
        print(f"Median p5-p95 width ratio ({boot} / stride-20): {np.median(ratios):.2f}x "
              f"(>1 = bootstrap intervals wider / more conservative)")

    # ---------------- CSV ----------------
    csv_path = os.path.join(out_dir, 'subsample_scheme_comparison.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['param', 'unit', 'truth'] +
                   [f"{lab}_{s}" for lab in labels for s in ('p5', 'p50', 'p95', 'width', 'covered')])
        for p in PARAMS:
            row = [p, UNIT[p], f"{truth[p]:.6g}"]
            for lab in labels:
                pr = results[lab]['param'][p]
                row += [f"{pr['p5']:.6g}", f"{pr['p50']:.6g}", f"{pr['p95']:.6g}",
                        f"{pr['width']:.6g}", int(pr['covered'])]
            w.writerow(row)
    print(f"\nWrote {csv_path}")

    # ---------------- Plot: violins per param, both schemes side by side, truth line ----------------
    n_par = len(PARAMS)
    fig, axes = plt.subplots(1, n_par, figsize=(3.0 * n_par, 4.4), constrained_layout=True)
    colors = {'stride-20': 'C0'}
    for ci, lab in enumerate(labels):
        colors.setdefault(lab, f"C{ci}")
    for i, (p, ax) in enumerate(zip(PARAMS, axes)):
        for ci, lab in enumerate(labels):
            col = results[lab]['vals'][p]
            parts = ax.violinplot(col, positions=[ci], showmedians=True, widths=0.8)
            for body in parts['bodies']:
                body.set_facecolor(colors[lab]); body.set_alpha(0.5)
        ax.axhline(truth[p], color='red', lw=2.0, ls='--', zorder=4,
                   label='truth' if i == 0 else None)
        ax.set_title(p); ax.set_ylabel(f"{p} ({UNIT[p]})")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace('bootstrap-', 'boot\n') for l in labels], fontsize=7)
    axes[0].legend(loc='best', fontsize=8)
    fig.suptitle("PILA UQ subsampling: strided vs block-bootstrap (matched ~819 px/member)\n"
                 "red dashed = synthetic truth (peak epoch)", fontsize=12)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'subsample_scheme_comparison.{ext}'), dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir}/subsample_scheme_comparison.png/.pdf")


if __name__ == '__main__':
    main()
