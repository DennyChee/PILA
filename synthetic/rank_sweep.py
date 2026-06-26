#!/usr/bin/env python
# Usage:       python -m synthetic.rank_sweep \
#                  synthetic/eval/<name>/<name>_alpha_sweep_depth_amp_summary.csv \
#                  [--score worst] [--objectives depth amp] [--top 10] [--no-heatmap]
# Description: Rank the combos of a sweep_alpha summary CSV by a JOINT score across the
#              swept objectives (find the "best of both"). Each objective's blended MAE is
#              normalised by its RAW MAE -> fraction of raw error remaining (lower=better,
#              1.0=no improvement); the per-objective normalised errors are then combined
#              (worst / sum / mean). Different parameters have different units (depth in m,
#              dV in m^3), so normalising first is what makes them comparable. Also writes a
#              combined-score heatmap over the two swept alpha axes (faceted by a 3rd if any).
# Date:        2026-06-26

import argparse
import os
import sys

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic.sweep_alpha import faceted_heatmap


def main():
    ap = argparse.ArgumentParser(
        description="Rank a sweep_alpha summary CSV by a joint normalised score across the "
                    "swept objectives, and write a combined-score heatmap.")
    ap.add_argument('csv', help='Path to a *_alpha_sweep_*_summary.csv from sweep_alpha.')
    ap.add_argument('--objectives', nargs='+', default=None,
                    help='Objective tokens to combine (default: the swept axes that have '
                         '{tok}_mae_raw/blend columns, e.g. depth amp).')
    ap.add_argument('--score', choices=['worst', 'sum', 'mean'], default='worst',
                    help="Combine rule for the normalised errors. 'worst' = minimise the "
                         "worse-off objective (the natural best-of-both); 'sum'/'mean' = "
                         "total/average reduction. Default: worst.")
    ap.add_argument('--top', type=int, default=10, help='How many ranked rows to print.')
    ap.add_argument('--no-heatmap', action='store_true', help='Skip the combined-score heatmap.')
    ap.add_argument('--out', default=None, help='Output dir for the heatmap (default: CSV dir).')
    args = ap.parse_args()

    # --- [1/4] Load ---
    print(f"[1/4] Loading {args.csv} ...")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)
    df = pd.read_csv(args.csv)
    print(f"  shape={df.shape}")

    # --- [2/4] Identify swept axes and objectives ---
    alpha_cols = [c for c in df.columns if c.startswith('alpha_') and c != 'alpha_tag']
    swept = [c[len('alpha_'):] for c in alpha_cols if df[c].nunique() > 1]
    fixed = {c[len('alpha_'):]: df[c].iloc[0] for c in alpha_cols if df[c].nunique() == 1}
    if not swept:
        raise ValueError("No varying alpha_* column found — is this a sweep summary CSV?")

    if args.objectives:
        objectives = args.objectives
    else:
        objectives = [t for t in swept
                      if f'{t}_mae_blend' in df.columns and f'{t}_mae_raw' in df.columns]
    if not objectives:
        raise ValueError(f"No usable objectives; columns: {list(df.columns)}")
    for t in objectives:
        for suf in ('_mae_raw', '_mae_blend'):
            if f'{t}{suf}' not in df.columns:
                raise ValueError(f"objective '{t}' missing column {t}{suf}")
    print(f"[2/4] swept axes={swept}  fixed={fixed}  objectives={objectives}")

    # --- [3/4] Normalise each objective by its raw error, then combine ---
    norm_cols = []
    for t in objectives:
        raw = df[f'{t}_mae_raw'].clip(lower=1e-30)        # avoid /0 for a trivial param
        df[f'{t}_norm'] = df[f'{t}_mae_blend'] / raw
        norm_cols.append(f'{t}_norm')
    if args.score == 'worst':
        df['score'] = df[norm_cols].max(axis=1)
    elif args.score == 'sum':
        df['score'] = df[norm_cols].sum(axis=1)
    else:
        df['score'] = df[norm_cols].mean(axis=1)

    show = [f'alpha_{t}' for t in swept] + norm_cols + ['score']
    ranked = df.sort_values('score').reset_index(drop=True)
    print(f"[3/4] Top {args.top} by '{args.score}' (normalised error = fraction of raw, lower better):")
    print(ranked[show].head(args.top).to_string(index=False))

    best = ranked.iloc[0]
    coords = '  '.join(f'{t}={best[f"alpha_{t}"]:g}' for t in swept)
    print(f"\nBEST ({args.score}): {coords}")
    for t in objectives:
        print(f"  {t:<8}: blended {best[f'{t}_norm']:.0%} of raw  "
              f"(MAE {best[f'{t}_mae_blend']:.4g} vs raw {best[f'{t}_mae_raw']:.4g})")

    # --- [4/4] Combined-score heatmap over the two swept alpha axes ---
    if args.no_heatmap:
        print("[4/4] Heatmap skipped (--no-heatmap).")
        return
    if len(swept) < 2:
        print("[4/4] Only one swept axis — no 2-D heatmap.")
        return
    if len(swept) > 3:
        print(f"[4/4] {len(swept)} swept axes — too many to plot; use the CSV.")
        return

    print("[4/4] Writing combined-score heatmap...")
    a1, a2 = swept[0], swept[1]
    facet = swept[2] if len(swept) == 3 else None
    row_vals = sorted(df[f'alpha_{a1}'].unique())
    col_vals = sorted(df[f'alpha_{a2}'].unique())
    facet_vals = sorted(df[f'alpha_{facet}'].unique()) if facet else [None]
    panel_grids = {fv: {} for fv in facet_vals}
    for _, r in df.iterrows():
        fv = r[f'alpha_{facet}'] if facet else None
        # keep the best (min) score if duplicates ever map to a cell
        key = (r[f'alpha_{a1}'], r[f'alpha_{a2}'])
        prev = panel_grids[fv].get(key, np.inf)
        panel_grids[fv][key] = min(prev, r['score'])

    out_dir = args.out or os.path.dirname(os.path.abspath(args.csv))
    name = os.path.basename(args.csv).split('_alpha_sweep_')[0]
    fac_note = f', faceted by {facet}' if facet else ''
    stem = os.path.join(out_dir, f'{name}_sweep_{"_".join(swept)}_combined_{args.score}')
    faceted_heatmap(stem, f'{name}: combined score ({args.score} of {"+".join(objectives)} '
                          f'norm err){fac_note}',
                    row_vals, col_vals, facet_vals, facet, panel_grids,
                    cbar_label=f'{args.score} normalised error (lower=better)',
                    row_label=f'alpha[{a1}]', col_label=f'alpha[{a2}]')
    print(f"  -> {stem}.png / .pdf")
    print("Done.")


if __name__ == '__main__':
    main()
