#!/usr/bin/env python
# Usage:       python -m synthetic.sweep_alpha \
#                  --truth synthetic/cubes/<name>/<name>_truth.json \
#                  --ckpt  saved/synth/<name>/.../models/model_best.pth \
#                  --locs 0.0 0.1 0.3 0.5 --amps 0.0 0.1 0.3 0.5
#              # --ckpt may also be a DIRECTORY -> newest model_best.pth under it is used.
# Description: 2-D inference-time temporal-blending sweep for ONE synthetic cube. Runs
#              synthetic.evaluate for every (alpha_loc, alpha_amp) combination, each into
#              its own alpha-tagged subfolder (a{alpha}_loc{l}_amp{m}/), then collects the
#              recovery metrics into a summary CSV and two heatmaps over the (loc, amp)
#              grid: source-location error and amplitude-parameter error.
# Scientific notes:
#   * alpha_loc smooths the source LOCATION/GEOMETRY across epochs, alpha_amp smooths the
#     source AMPLITUDE (dV for Mogi/Sun69, opening for Okada). They mainly affect DIFFERENT
#     metrics, so the two heatmaps should be read independently: pick the alpha_loc that
#     minimises location error, and the alpha_amp that minimises amplitude error.
#   * For a MOVING source a large alpha_loc lags the true migration -> location error can
#     get WORSE with smoothing. The (loc=0, amp=0) corner is the raw, no-blending baseline.
#   * --alpha is only the enabler/fallback; for Mogi & Okada every parameter is in the loc
#     or amp group, so its value is unused. Left at 0 by default.
# Date:        2026-06-26

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Headless backend BEFORE importing evaluate (which pulls in matplotlib via plots) so this
# runs on an HPC compute node with no display.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic.evaluate import (evaluate, alpha_tag, LOC_GROUPS, PHYSICS_ATTRS,
                                ROOT as EVAL_ROOT)


def resolve_ckpt(path):
    """Return an exact checkpoint path. If `path` is a directory, pick the most recently
    modified model_best.pth beneath it (so the caller can pass the experiment dir)."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, '**', 'model_best.pth'), recursive=True)
        if not cands:
            raise FileNotFoundError(f"No model_best.pth found under directory: {path}")
        newest = max(cands, key=os.path.getmtime)
        print(f"  --ckpt is a directory; using newest checkpoint: {newest}")
        return newest
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def heatmap(out_stem, title, locs, amps, grid, cbar_label, cmap='viridis'):
    """Save a (loc rows x amp cols) heatmap (png @150dpi + pdf), annotate each cell, and
    outline the minimum-error cell in red. grid is a dict keyed by (loc, amp) -> value."""
    matrix = np.array([[grid.get((l, a), np.nan) for a in amps] for l in locs], dtype=float)
    fig, ax = plt.subplots(figsize=(1.8 + 1.15 * len(amps), 1.8 + 1.0 * len(locs)))
    im = ax.imshow(matrix, cmap=cmap, origin='upper', aspect='auto')
    ax.set_xticks(range(len(amps)))
    ax.set_xticklabels([f'{a:g}' for a in amps])
    ax.set_yticks(range(len(locs)))
    ax.set_yticklabels([f'{l:g}' for l in locs])
    ax.set_xlabel('alpha_amp  (amplitude smoothing)')
    ax.set_ylabel('alpha_loc  (location smoothing)')
    # Annotate each cell; pick black/white text for contrast against the colormap.
    for i in range(len(locs)):
        for j in range(len(amps)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            txt_color = 'white' if im.norm(val) < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=txt_color, fontsize=8)
    # Outline the best (minimum) cell.
    if np.isfinite(matrix).any():
        bi, bj = np.unravel_index(np.nanargmin(matrix), matrix.shape)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                   edgecolor='red', lw=2.0))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_stem + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_stem + '.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="2-D (alpha_loc x alpha_amp) temporal-blending sweep for one synthetic "
                    "cube: runs synthetic.evaluate per combo, writes a summary CSV + heatmaps.")
    ap.add_argument('--truth', required=True, help='Ground-truth sidecar JSON.')
    ap.add_argument('--ckpt', required=True,
                    help='model_best.pth, OR a directory (newest model_best.pth is used).')
    ap.add_argument('--locs', type=float, nargs='+', default=[0.0, 0.1, 0.3, 0.5],
                    help='alpha_loc grid (space-separated). Default: 0.0 0.1 0.3 0.5')
    ap.add_argument('--amps', type=float, nargs='+', default=[0.0, 0.1, 0.3, 0.5],
                    help='alpha_amp grid (space-separated). Default: 0.0 0.1 0.3 0.5')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='Enabler/fallback weight (unused for Mogi/Okada). Default: 0.0')
    ap.add_argument('--out', default=None,
                    help='Base output dir (default synthetic/eval/<name>). Combos go into '
                         '<out>/<alpha_tag>/; the summary CSV + heatmaps go into <out>/.')
    args = ap.parse_args()

    t_start = time.time()

    # --- [1/4] Resolve inputs ---
    print("[1/4] Resolving inputs...")
    if not os.path.exists(args.truth):
        raise FileNotFoundError(f"Truth JSON not found: {args.truth}")
    ckpt = resolve_ckpt(args.ckpt)
    with open(args.truth, 'r') as f:
        truth = json.load(f)
    name = truth['name']
    cfg_path = os.path.join(os.path.dirname(ckpt), 'config.json')
    with open(cfg_path, 'r') as f:
        physics = json.load(f)['arch']['args']['physics']
    amp_params = LOC_GROUPS[physics]['amp']     # params whose error tracks alpha_amp
    base_dir = args.out or os.path.join(EVAL_ROOT, 'synthetic', 'eval', name)
    os.makedirs(base_dir, exist_ok=True)
    print(f"  cube={name}  physics={physics}  amp_params={amp_params}")
    print(f"  locs={args.locs}  amps={args.amps}  -> {len(args.locs) * len(args.amps)} combos")
    print(f"  base output dir: {base_dir}")

    # --- [2/4] Run every (loc, amp) combo through synthetic.evaluate ---
    print("[2/4] Running sweep...")
    rows = []
    loc_err_grid = {}       # (loc, amp) -> blended source-location error (km)
    amp_err_grid = {}       # (loc, amp) -> blended amplitude-param MAE (mean over amp_params)
    n_combo = len(args.locs) * len(args.amps)
    k = 0
    for loc in args.locs:
        for amp in args.amps:
            k += 1
            tag = alpha_tag(args.alpha, loc, amp)
            print(f"  ({k}/{n_combo}) {tag} ...", flush=True)
            try:
                m = evaluate(args.truth, ckpt, out_dir=args.out,
                             alpha=args.alpha, alpha_loc=loc, alpha_amp=amp)
            except Exception as exc:        # keep the sweep going if one combo fails
                print(f"    !! FAILED ({type(exc).__name__}: {exc}) — skipping")
                continue

            raw, blend = m['raw'], m['temporal']
            # Mean strong-epoch MAE over the amplitude parameter(s).
            amp_mae_raw = float(np.mean([raw['per_param'][p]['mae_strong_epochs']
                                         for p in amp_params]))
            amp_mae_blend = float(np.mean([blend['per_param'][p]['mae_strong_epochs']
                                           for p in amp_params]))
            loc_err_grid[(loc, amp)] = blend['source_loc_err_km_strong_mean']
            amp_err_grid[(loc, amp)] = amp_mae_blend
            rows.append({
                'alpha_loc': loc, 'alpha_amp': amp, 'alpha_tag': tag,
                'loc_err_km_raw': raw['source_loc_err_km_strong_mean'],
                'loc_err_km_blend': blend['source_loc_err_km_strong_mean'],
                'loc_err_km_peak_blend': blend['source_loc_err_km_peak'],
                'amp_mae_raw': amp_mae_raw,
                'amp_mae_blend': amp_mae_blend,
                'los_rmse_mm_blend': blend['los_rmse_mm_all'],
            })

    if not rows:
        raise RuntimeError("All combos failed — nothing to summarise.")

    # --- [3/4] Write summary CSV ---
    print("[3/4] Writing summary CSV...")
    df = pd.DataFrame(rows).sort_values(['alpha_loc', 'alpha_amp']).reset_index(drop=True)
    csv_path = os.path.join(base_dir, f'{name}_alpha_sweep_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  shape={df.shape}  ->  {csv_path}")

    # --- [4/4] Heatmaps over the (loc, amp) plane ---
    print("[4/4] Writing heatmaps...")
    amp_label = '+'.join(amp_params)
    heatmap(os.path.join(base_dir, f'{name}_sweep_loc_err'),
            f'{name}: source-location error (km), blended',
            args.locs, args.amps, loc_err_grid,
            cbar_label='source loc err (km), strong-mean')
    heatmap(os.path.join(base_dir, f'{name}_sweep_amp_err'),
            f'{name}: {amp_label} error, blended',
            args.locs, args.amps, amp_err_grid,
            cbar_label=f'{amp_label} MAE (strong epochs)')

    # --- Report best combos ---
    best_loc = df.loc[df['loc_err_km_blend'].idxmin()]
    best_amp = df.loc[df['amp_mae_blend'].idxmin()]
    raw_corner = df[(df['alpha_loc'] == 0.0) & (df['alpha_amp'] == 0.0)]
    print("\n" + "=" * 70)
    print(f"ALPHA SWEEP SUMMARY: {name} ({physics})")
    print("=" * 70)
    if not raw_corner.empty:
        rc = raw_corner.iloc[0]
        print(f"baseline (loc=0, amp=0): loc_err={rc['loc_err_km_blend']:.4f} km   "
              f"{amp_label}_MAE={rc['amp_mae_blend']:.4g}")
    print(f"best location : alpha_loc={best_loc['alpha_loc']:g} alpha_amp={best_loc['alpha_amp']:g}"
          f"  -> loc_err={best_loc['loc_err_km_blend']:.4f} km")
    print(f"best amplitude: alpha_loc={best_amp['alpha_loc']:g} alpha_amp={best_amp['alpha_amp']:g}"
          f"  -> {amp_label}_MAE={best_amp['amp_mae_blend']:.4g}")
    print(f"\nDone in {time.time() - t_start:.1f}s. Outputs in {base_dir}")


if __name__ == '__main__':
    main()
