#!/usr/bin/env python
# Usage:       python -m synthetic.sweep_alpha \
#                  --truth synthetic/cubes/<name>/<name>_truth.json \
#                  --ckpt  saved/synth/<name>/.../models/model_best.pth \
#                  --locs 0 0.1 0.2 0.3 0.4 0.5 0.7 0.9 \
#                  --amps 0 0.1 0.2 0.3 0.4 0.5 0.7 0.9 \
#                  [--per-combo-plots]
#              # --ckpt may also be a DIRECTORY -> newest model_best.pth under it is used.
# Description: 2-D inference-time temporal-blending sweep for ONE synthetic cube. The
#              encoder output is INDEPENDENT of the blend weights, so the model is loaded
#              and every epoch encoded ONCE (encode_pass); each (alpha_loc, alpha_amp)
#              combo then only re-blends + re-decodes (blend_decode) -- cheap, so a dense
#              grid is practical. Collects recovery metrics into a summary CSV and two
#              heatmaps over the (loc, amp) grid: source-location error and amplitude error.
# Scientific notes:
#   * alpha_loc smooths source LOCATION/GEOMETRY across epochs; alpha_amp smooths source
#     AMPLITUDE (dV for Mogi/Sun69, opening for Okada). They mainly affect DIFFERENT
#     metrics -> read the two heatmaps independently.
#   * For a MOVING source a large alpha_loc lags the true migration, so location error can
#     get WORSE with smoothing. The (loc=0, amp=0) corner is the raw, no-blending baseline.
#   * --alpha is only the enabler/fallback; for Mogi & Okada every parameter is in the loc
#     or amp group, so its value is unused (left at 0).
# Date:        2026-06-26

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Headless backend BEFORE importing evaluate/plots so this runs on an HPC node (no display).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic import plots
from synthetic.evaluate import (encode_pass, blend_decode, build_alpha_vec, alpha_tag,
                                _recovery_metrics, LOC_GROUPS, LOC_KEYS, PHYSICS_ATTRS,
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
    fig, ax = plt.subplots(figsize=(1.8 + 1.05 * len(amps), 1.8 + 0.95 * len(locs)))
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
                    color=txt_color, fontsize=7)
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
                    "cube. Encodes once, re-blends per combo; writes a summary CSV + heatmaps.")
    ap.add_argument('--truth', required=True, help='Ground-truth sidecar JSON.')
    ap.add_argument('--ckpt', required=True,
                    help='model_best.pth, OR a directory (newest model_best.pth is used).')
    ap.add_argument('--locs', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                    help='alpha_loc grid (space-separated).')
    ap.add_argument('--amps', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                    help='alpha_amp grid (space-separated).')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='Enabler/fallback weight (unused for Mogi/Okada). Default: 0.0')
    ap.add_argument('--out', default=None,
                    help='Base output dir (default synthetic/eval/<name>). Per-combo '
                         'metrics/plots go into <out>/<alpha_tag>/; CSV + heatmaps into <out>/.')
    ap.add_argument('--per-combo-plots', action='store_true',
                    help='Also write per-combo trajectory + param-recovery comparison plots '
                         '(off by default to keep a dense sweep fast/uncluttered).')
    args = ap.parse_args()

    t_start = time.time()

    # --- [1/5] Resolve inputs + truth ---
    print("[1/5] Resolving inputs...")
    if not os.path.exists(args.truth):
        raise FileNotFoundError(f"Truth JSON not found: {args.truth}")
    ckpt = resolve_ckpt(args.ckpt)
    with open(args.truth, 'r') as f:
        truth = json.load(f)
    cfg_path = os.path.join(os.path.dirname(ckpt), 'config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    name = truth['name']
    physics = cfg['arch']['args']['physics']
    attrs = PHYSICS_ATTRS[physics]
    amp_params = LOC_GROUPS[physics]['amp']     # params whose error tracks alpha_amp
    truth_names = truth['param_names']
    truth_params = np.asarray(truth['params_per_epoch'])
    signal_rms = np.asarray(truth['signal_rms_mm_per_epoch'])
    n_epoch = truth_params.shape[0]
    col = [attrs.index(p) for p in truth_names]
    kx, ky = LOC_KEYS[physics]
    ix, iy = truth_names.index(kx), truth_names.index(ky)
    base_dir = args.out or os.path.join(EVAL_ROOT, 'synthetic', 'eval', name)
    os.makedirs(base_dir, exist_ok=True)
    print(f"  cube={name}  physics={physics}  amp_params={amp_params}")
    print(f"  locs={args.locs}  amps={args.amps}  -> {len(args.locs) * len(args.amps)} combos")

    # --- [2/5] Encode ONCE (alpha-independent) ---
    print("[2/5] Encoding (once)...")
    ctx = encode_pass(cfg, ckpt)
    n = min(ctx['params_raw'].shape[0], n_epoch)
    if ctx['params_raw'].shape[0] != n_epoch:
        print(f"  WARNING: {ctx['params_raw'].shape[0]} inferred epochs vs {n_epoch} truth epochs")
    tru = truth_params[:n]; rms = signal_rms[:n]
    peak = int(truth.get('peak_epoch_index', int(np.argmax(rms))))
    strong = rms >= 0.5 * rms.max()
    x_scale = ctx['x_scale']

    # RAW recovery is alpha-independent -> compute once.
    inferred_raw = ctx['params_raw'][:n, col]
    m_raw = _recovery_metrics(inferred_raw, ctx['pred_raw'], ctx['targ'], x_scale,
                              tru, rms, peak, strong, truth_names, physics)
    amp_mae_raw = float(np.mean([m_raw['per_param'][p]['mae_strong_epochs'] for p in amp_params]))
    if not strong.any():
        print("  WARNING: no strong-signal epochs (rms >= 0.5*max) — strong-mean metrics NaN.")

    # --- [3/5] Sweep: blend + decode per combo (cheap) ---
    print("[3/5] Sweeping combos...")
    rows = []
    loc_err_grid, amp_err_grid = {}, {}
    n_combo = len(args.locs) * len(args.amps)
    k = 0
    for loc in args.locs:
        for amp in args.amps:
            k += 1
            tag = alpha_tag(args.alpha, loc, amp)
            print(f"  ({k}/{n_combo}) {tag} ...", flush=True)
            alpha_vec = build_alpha_vec(attrs, physics, args.alpha, loc, amp)
            params_blend, pred_blend = blend_decode(ctx, alpha_vec)
            inferred_blend = params_blend[:n, col]
            m_blend = _recovery_metrics(inferred_blend, pred_blend, ctx['targ'], x_scale,
                                        tru, rms, peak, strong, truth_names, physics)
            amp_mae_blend = float(np.mean([m_blend['per_param'][p]['mae_strong_epochs']
                                           for p in amp_params]))

            loc_err_grid[(loc, amp)] = m_blend['source_loc_err_km_strong_mean']
            amp_err_grid[(loc, amp)] = amp_mae_blend
            rows.append({
                'alpha_loc': loc, 'alpha_amp': amp, 'alpha_tag': tag,
                'loc_err_km_raw': m_raw['source_loc_err_km_strong_mean'],
                'loc_err_km_blend': m_blend['source_loc_err_km_strong_mean'],
                'loc_err_km_peak_blend': m_blend['source_loc_err_km_peak'],
                'amp_mae_raw': amp_mae_raw,
                'amp_mae_blend': amp_mae_blend,
                'los_rmse_mm_blend': m_blend['los_rmse_mm_all'],
            })

            # Per-combo drill-down (metrics json always; plots only if requested).
            combo_dir = os.path.join(base_dir, tag)
            os.makedirs(combo_dir, exist_ok=True)
            with open(os.path.join(combo_dir, f'{name}_metrics_temporal.json'), 'w') as f:
                json.dump({'name': name, 'physics': physics,
                           'alpha_default': args.alpha, 'alpha_loc': loc, 'alpha_amp': amp,
                           'alpha_per_param': {attrs[i]: float(alpha_vec[i])
                                               for i in range(len(attrs))},
                           'raw': m_raw, 'temporal': m_blend}, f, indent=2)
            if args.per_combo_plots:
                lbl = f"alpha_loc={loc:g}, alpha_amp={amp:g}"
                truth_xy = np.column_stack([tru[:, ix], tru[:, iy]]) / 1000.0
                raw_xy = np.column_stack([inferred_raw[:, ix], inferred_raw[:, iy]]) / 1000.0
                blend_xy = np.column_stack([inferred_blend[:, ix], inferred_blend[:, iy]]) / 1000.0
                plots.plot_trajectory_compare(combo_dir, name, truth_xy, raw_xy, blend_xy,
                                              rms, alpha_label=lbl)
                plots.plot_param_recovery_compare(combo_dir, name, truth_names,
                                                  truth['param_units'], tru, inferred_raw,
                                                  inferred_blend, rms, alpha_label=lbl)

    # --- [4/5] Summary CSV ---
    print("[4/5] Writing summary CSV...")
    df = pd.DataFrame(rows).sort_values(['alpha_loc', 'alpha_amp']).reset_index(drop=True)
    csv_path = os.path.join(base_dir, f'{name}_alpha_sweep_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  shape={df.shape}  ->  {csv_path}")

    # --- [5/5] Heatmaps over the (loc, amp) plane ---
    print("[5/5] Writing heatmaps...")
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
    print("\n" + "=" * 72)
    print(f"ALPHA SWEEP SUMMARY: {name} ({physics})")
    print("=" * 72)
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
