#!/usr/bin/env python
# Usage:       python plot_multisource_training.py --runs saved/multisource/clean_detr/<ts> \
#                  saved/multisource/clean_detr_depletion/<ts> saved/multisource/clean_fixed_baseline/<ts> \
#                  --out figures/multisource/training_curves_clean.png
# Description: Overlay the validation curves (log.csv) of several Stage-1 multi-source runs:
#              count accuracy, recall, duplicate rate, location / depth / log10 dV errors,
#              existence and parameter training losses. Works on runs still in progress.
# Date:        2026-10-05

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = [('count_acc', 'Count accuracy (fraction)', False),
          ('recall', 'Recall (fraction of true sources found)', False),
          ('dup_rate', 'Duplicate rate (fraction of K≥1 scenes)', False),
          ('err_xy_km', 'Median location error (km)', True),
          ('err_depth_km', 'Median depth error (km)', True),
          ('err_log10dV', 'Median |log10 dV error|', True),
          ('exist', 'Train existence BCE (unitless)', True),
          ('param', 'Train parameter L1 (normalised)', True),
          ('sign_acc', 'Sign accuracy (fraction)', False)]


def read_log(run_dir):
    """Read log.csv of a run into a dict of float arrays."""
    path = os.path.join(run_dir, 'log.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) if r[key] not in ('', 'nan') else np.nan for r in rows])
            for key in rows[0]} if rows else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    print(f"[1/2] Reading {len(args.runs)} run logs...")
    logs = {}
    for run in args.runs:
        name = os.path.basename(os.path.dirname(os.path.normpath(run)))
        logs[name] = read_log(run)
        n = len(logs[name].get('step', []))
        print(f"  {name}: {n} evals, last step {logs[name]['step'][-1] if n else '-'}")

    print("[2/2] Plotting...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, (key, label, log_y) in zip(axes.ravel(), PANELS):
        for name, log in logs.items():
            if log and key in log:
                ax.plot(log['step'], log[key], 'o-', ms=3, label=name)
        ax.set_xlabel('Training step'); ax.set_ylabel(label)
        if log_y:
            ax.set_yscale('log')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Stage-1 multi-source training: validation metrics (noise-free, peak/σ 2–10)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Done. Output saved to {args.out}")


if __name__ == '__main__':
    main()
