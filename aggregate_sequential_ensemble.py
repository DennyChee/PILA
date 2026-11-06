#!/usr/bin/env python
# Usage:       python aggregate_sequential_ensemble.py --stats <ensemble_stats.csv> --thresholds <thresholds.json> \
#                  [--truth-dir synthetic/multisource/pila_stacks_v2/albums] [--cluster-km 3.0]
#                  [--out-dir figures/multisource_seq/step4]
# Description: Step 4: turns the ensemble members of each album into probabilities.
#                * each member selects K_hat with the calibrated bars (sequential stop);
#                * P(K = k) = fraction of members selecting k;
#                * the accepted sources of all members (stack-median x, y, depth, dV of the K_hat fit) are
#                  clustered across members (greedy, horizontal radius cluster_km); support of a cluster =
#                  number of distinct members with a source in it / M;
#                * each cluster is labelled real / spurious by matching to the truth (within cluster_km),
#                  giving a calibration table: support bin vs fraction of clusters that are real.
# Notes:       The ensemble spread measures how robust a source is to the seed and to which ~half of the
#              scene is used; it is not a calibrated Bayesian posterior until the calibration table shows it.
#              Units: km, m^3.
# Date:        2026-10-06

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sequential_select_k import select_k


def member_sources(run_root, k_hat):
    """Stack-median sources [K_hat, 4] of the member's K_hat fit."""
    if k_hat == 0:
        return np.zeros((0, 4))
    seq = json.load(open(os.path.join(run_root, 'sequence.json')))
    with np.load(seq['runs'][str(k_hat)]['fit']) as f:
        return np.median(f['params_phys'], axis=0)


def cluster(sources, radius_km):
    """Greedy clustering of (member, source) rows by horizontal distance to running cluster centroids."""
    clusters = []                                    # dicts: members set, rows list
    for member, src in sources:
        best, best_d = None, radius_km
        for c in clusters:
            d = np.hypot(*(np.mean([r[:2] for r in c['rows']], axis=0) - src[:2]))
            if d <= best_d:
                best, best_d = c, d
        if best is None:
            clusters.append({'members': {member}, 'rows': [src]})
        else:
            best['members'].add(member); best['rows'].append(src)
    return clusters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats', required=True)
    parser.add_argument('--thresholds', required=True)
    parser.add_argument('--truth-dir', default='synthetic/multisource/pila_stacks_v2/albums')
    parser.add_argument('--cluster-km', type=float, default=3.0)
    parser.add_argument('--out-dir', default='figures/multisource_seq/step4')
    args = parser.parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    cal = json.load(open(args.thresholds))
    bars, metric = cal['thresholds'], cal.get('metric', 'dchi2')
    rows = list(csv.DictReader(open(args.stats)))
    by_album = defaultdict(list)
    for r in rows:
        member = re.search(r'ens_m(\d+)', r['run'])
        if member:
            by_album[r['album']].append((int(member.group(1)), r))
    print(f"[1/3] {len(by_album)} albums, members per album: {sorted({len(v) for v in by_album.values()})}")

    print("[2/3] P(K) and source support per album...")
    results, calib = {}, []
    for album, members in sorted(by_album.items()):
        truth = json.load(open(os.path.join(args.truth_dir, f'{album}_truth.json')))
        n_members = len(members)
        k_hats, pooled = [], []
        for member, r in members:
            k_hat = select_k(r, bars, metric)
            k_hats.append(k_hat)
            pooled += [(member, s) for s in member_sources(r['run'], k_hat)]
        p_k = {k: float(np.mean(np.array(k_hats) == k)) for k in range(0, max(k_hats + [0]) + 1)}
        clusters = cluster(pooled, args.cluster_km)
        out = []
        for c in clusters:
            centre = np.median(np.array(c['rows']), axis=0)
            support = len(c['members']) / n_members
            is_real = any(np.hypot(centre[0] - s['xcen_km'], centre[1] - s['ycen_km']) <= args.cluster_km
                          for s in truth['sources'])
            calib.append((support, is_real))
            out.append(dict(support=support, x_km=float(centre[0]), y_km=float(centre[1]),
                            depth_km=float(centre[2]), dV_m3=float(centre[3]), real=bool(is_real)))
        out.sort(key=lambda c: -c['support'])
        results[album] = dict(true_k=truth['n_sources'], members=n_members, p_k=p_k, clusters=out)
        print(f"  {album:24s} true K={truth['n_sources']}  P(K)=" + ', '.join(f"{k}:{p:.1f}" for k, p in p_k.items()))
        for c in out:
            print(f"      support {c['support']:.1f}  ({c['x_km']:6.2f}, {c['y_km']:6.2f}) km, depth {c['depth_km']:.2f} km, "
                  f"dV {c['dV_m3']:.2e} m^3  {'REAL' if c['real'] else 'spurious'}")

    print("[3/3] Support calibration + figure...")
    edges = [0, 0.3, 0.6, 0.9, 1.01]
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [real for s, real in calib if lo <= s < hi]
        table.append(dict(support_lo=lo, support_hi=min(hi, 1.0), n=len(sel), frac_real=float(np.mean(sel)) if sel else None))
        print(f"  support [{lo:.1f}, {min(hi, 1.0):.1f}]: {len(sel)} clusters, fraction real "
              f"{np.mean(sel) if sel else float('nan'):.2f}")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [0.5 * (t['support_lo'] + t['support_hi']) for t in table if t['n']]
    ax.plot([0, 1], [0, 1], 'k--', label='perfectly calibrated')
    ax.plot(xs, [t['frac_real'] for t in table if t['n']], 'o-', label='ensemble support')
    for t, x in zip([t for t in table if t['n']], xs):
        ax.text(x, t['frac_real'] + 0.04, f"n={t['n']}", ha='center', fontsize=8)
    ax.set_xlabel('Support (fraction of members keeping the source)'); ax.set_ylabel('Fraction that are real sources')
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.1); ax.legend(); ax.set_title('Is support a probability?')
    plt.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'support_calibration.png'), dpi=150); plt.close(fig)
    with open(os.path.join(args.out_dir, 'ensemble_summary.json'), 'w') as f:
        json.dump(dict(albums=results, support_calibration=table, cluster_km=args.cluster_km), f, indent=1)
    print(f"Done in {time.time() - t0:.1f}s. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
