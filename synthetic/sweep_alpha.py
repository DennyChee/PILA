#!/usr/bin/env python
# Usage:       python -m synthetic.sweep_alpha \
#                  --truth synthetic/cubes/<name>/<name>_truth.json \
#                  --ckpt  saved/synth/<name>/.../models/model_best.pth \
#                  --depths 0 0.1 0.2 0.3 0.4 0.5 0.7 0.9 \
#                  --amps   0 0.1 0.2 0.3 0.4 0.5 0.7 0.9 \
#                  [--locs 0] [--per-combo-plots]
#              # --ckpt may also be a DIRECTORY -> newest model_best.pth under it is used.
# Description: temporal-blending alpha sweep for ONE synthetic cube over the
#              (alpha_loc x alpha_depth x alpha_amp) grid. The encoder output is INDEPENDENT
#              of the blend weights, so the model is loaded and every epoch encoded ONCE
#              (encode_pass); each combo then only re-blends + re-decodes (blend_decode), so
#              a dense grid is cheap. Collects recovery metrics into a summary CSV and, for
#              each alpha_loc, two heatmaps over the (alpha_depth x alpha_amp) plane: depth
#              error and amplitude error.
# Scientific notes:
#   * alpha_depth smooths the SOURCE DEPTH, alpha_amp the SOURCE AMPLITUDE (dV / opening).
#     Depth and amplitude are the two halves of the depth-amplitude trade-off; pooling depth
#     across epochs (higher alpha_depth) can break that ambiguity, while amplitude must stay
#     free (low alpha_amp) to track real episodic change. alpha_loc handles horizontal
#     position/geometry and is usually held LOW (default grid [0]) for a moving source.
#   * The (depth=0, amp=0) corner of each heatmap is the raw, no-blending baseline.
#   * --alpha is only the enabler/fallback; for Mogi & Okada every parameter is in the loc,
#     depth, or amp group, so its value is unused (left at 0).
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


def heatmap(out_stem, title, row_vals, col_vals, grid, cbar_label,
            row_label, col_label, cmap='viridis'):
    """Save a (row x col) heatmap (png @150dpi + pdf), annotate each cell, outline the
    minimum-error cell in red. grid is a dict keyed by (row_val, col_val) -> value."""
    matrix = np.array([[grid.get((r, c), np.nan) for c in col_vals] for r in row_vals],
                      dtype=float)
    fig, ax = plt.subplots(figsize=(1.8 + 1.05 * len(col_vals), 1.8 + 0.95 * len(row_vals)))
    im = ax.imshow(matrix, cmap=cmap, origin='upper', aspect='auto')
    ax.set_xticks(range(len(col_vals)))
    ax.set_xticklabels([f'{c:g}' for c in col_vals])
    ax.set_yticks(range(len(row_vals)))
    ax.set_yticklabels([f'{r:g}' for r in row_vals])
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    for i in range(len(row_vals)):
        for j in range(len(col_vals)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            txt_color = 'white' if im.norm(val) < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=txt_color, fontsize=7)
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
        description="Temporal-blending alpha sweep (loc x depth x amp) for one synthetic "
                    "cube. Encodes once, re-blends per combo; writes a summary CSV + "
                    "depth/amp heatmaps per alpha_loc.")
    ap.add_argument('--truth', required=True, help='Ground-truth sidecar JSON.')
    ap.add_argument('--ckpt', required=True,
                    help='model_best.pth, OR a directory (newest model_best.pth is used).')
    ap.add_argument('--locs', type=float, nargs='+', default=[0.0],
                    help='alpha_loc grid (position/geometry). Default: [0.0] (held fixed).')
    ap.add_argument('--depths', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                    help='alpha_depth grid (source depth). Heatmap rows.')
    ap.add_argument('--amps', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                    help='alpha_amp grid (source amplitude). Heatmap cols.')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='Enabler/fallback weight (unused for Mogi/Okada). Default: 0.0')
    ap.add_argument('--out', default=None,
                    help='Base output dir (default synthetic/eval/<name>). Per-combo '
                         'metrics/plots go into <out>/<alpha_tag>/; CSV + heatmaps into <out>/.')
    ap.add_argument('--per-combo-plots', action='store_true',
                    help='Also write per-combo truth/raw/blended trajectory + param-recovery '
                         'plots (off by default to keep a dense sweep fast/uncluttered).')
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
    depth_params = LOC_GROUPS[physics]['depth']     # error tracks alpha_depth
    amp_params = LOC_GROUPS[physics]['amp']          # error tracks alpha_amp
    truth_names = truth['param_names']
    truth_params = np.asarray(truth['params_per_epoch'])
    signal_rms = np.asarray(truth['signal_rms_mm_per_epoch'])
    n_epoch = truth_params.shape[0]
    col = [attrs.index(p) for p in truth_names]
    kx, ky = LOC_KEYS[physics]
    ix, iy = truth_names.index(kx), truth_names.index(ky)
    base_dir = args.out or os.path.join(EVAL_ROOT, 'synthetic', 'eval', name)
    os.makedirs(base_dir, exist_ok=True)
    n_combo = len(args.locs) * len(args.depths) * len(args.amps)
    print(f"  cube={name}  physics={physics}  depth_params={depth_params}  amp_params={amp_params}")
    print(f"  locs={args.locs}  depths={args.depths}  amps={args.amps}  -> {n_combo} combos")

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
    if not strong.any():
        print("  WARNING: no strong-signal epochs (rms >= 0.5*max) — strong-mean metrics NaN.")

    # RAW recovery is alpha-independent -> compute once.
    inferred_raw = ctx['params_raw'][:n, col]
    m_raw = _recovery_metrics(inferred_raw, ctx['pred_raw'], ctx['targ'], x_scale,
                              tru, rms, peak, strong, truth_names, physics)

    def _group_mae(metric, params):
        """Mean strong-epoch MAE over a group of parameters (e.g. depth or amplitude)."""
        return float(np.mean([metric['per_param'][p]['mae_strong_epochs'] for p in params]))

    depth_mae_raw = _group_mae(m_raw, depth_params)
    amp_mae_raw = _group_mae(m_raw, amp_params)

    # --- [3/5] Sweep: blend + decode per (loc, depth, amp) combo (cheap) ---
    print("[3/5] Sweeping combos...")
    rows = []
    depth_err = {}      # (loc, depth_a, amp_a) -> blended depth-param MAE
    amp_err = {}        # (loc, depth_a, amp_a) -> blended amplitude-param MAE
    k = 0
    for loc in args.locs:
        for depth_a in args.depths:
            for amp_a in args.amps:
                k += 1
                tag = alpha_tag(args.alpha, loc, depth_a, amp_a)
                print(f"  ({k}/{n_combo}) {tag} ...", flush=True)
                alpha_vec = build_alpha_vec(attrs, physics, args.alpha, loc, depth_a, amp_a)
                params_blend, pred_blend = blend_decode(ctx, alpha_vec)
                inferred_blend = params_blend[:n, col]
                m_blend = _recovery_metrics(inferred_blend, pred_blend, ctx['targ'], x_scale,
                                            tru, rms, peak, strong, truth_names, physics)

                depth_err[(loc, depth_a, amp_a)] = _group_mae(m_blend, depth_params)
                amp_err[(loc, depth_a, amp_a)] = _group_mae(m_blend, amp_params)
                rows.append({
                    'alpha_loc': loc, 'alpha_depth': depth_a, 'alpha_amp': amp_a,
                    'alpha_tag': tag,
                    'loc_err_km_raw': m_raw['source_loc_err_km_strong_mean'],
                    'loc_err_km_blend': m_blend['source_loc_err_km_strong_mean'],
                    'depth_mae_raw': depth_mae_raw,
                    'depth_mae_blend': depth_err[(loc, depth_a, amp_a)],
                    'amp_mae_raw': amp_mae_raw,
                    'amp_mae_blend': amp_err[(loc, depth_a, amp_a)],
                    'los_rmse_mm_blend': m_blend['los_rmse_mm_all'],
                })

                # Per-combo drill-down (metrics json always; plots only if requested).
                combo_dir = os.path.join(base_dir, tag)
                os.makedirs(combo_dir, exist_ok=True)
                with open(os.path.join(combo_dir, f'{name}_metrics_temporal.json'), 'w') as f:
                    json.dump({'name': name, 'physics': physics, 'alpha_default': args.alpha,
                               'alpha_loc': loc, 'alpha_depth': depth_a, 'alpha_amp': amp_a,
                               'alpha_per_param': {attrs[i]: float(alpha_vec[i])
                                                   for i in range(len(attrs))},
                               'raw': m_raw, 'temporal': m_blend}, f, indent=2)
                if args.per_combo_plots:
                    lbl = f"alpha_loc={loc:g}, alpha_depth={depth_a:g}, alpha_amp={amp_a:g}"
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
    df = pd.DataFrame(rows).sort_values(['alpha_loc', 'alpha_depth', 'alpha_amp'])
    df = df.reset_index(drop=True)
    csv_path = os.path.join(base_dir, f'{name}_alpha_sweep_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  shape={df.shape}  ->  {csv_path}")

    # --- [5/5] depth & amplitude heatmaps over (alpha_depth x alpha_amp), one per alpha_loc ---
    print("[5/5] Writing heatmaps...")
    depth_label = '+'.join(depth_params)
    amp_label = '+'.join(amp_params)
    for loc in args.locs:
        suffix = '' if len(args.locs) == 1 else f'_loc{loc:g}'
        loc_note = '' if len(args.locs) == 1 else f' (alpha_loc={loc:g})'
        sub_depth = {(d, a): depth_err[(loc, d, a)] for d in args.depths for a in args.amps}
        sub_amp = {(d, a): amp_err[(loc, d, a)] for d in args.depths for a in args.amps}
        heatmap(os.path.join(base_dir, f'{name}_sweep_depth_err{suffix}'),
                f'{name}: {depth_label} error, blended{loc_note}',
                args.depths, args.amps, sub_depth,
                cbar_label=f'{depth_label} MAE (strong epochs)',
                row_label='alpha_depth  (depth smoothing)',
                col_label='alpha_amp  (amplitude smoothing)')
        heatmap(os.path.join(base_dir, f'{name}_sweep_amp_err{suffix}'),
                f'{name}: {amp_label} error, blended{loc_note}',
                args.depths, args.amps, sub_amp,
                cbar_label=f'{amp_label} MAE (strong epochs)',
                row_label='alpha_depth  (depth smoothing)',
                col_label='alpha_amp  (amplitude smoothing)')

    # --- Report best combos ---
    best_depth = df.loc[df['depth_mae_blend'].idxmin()]
    best_amp = df.loc[df['amp_mae_blend'].idxmin()]
    base = df[(df['alpha_loc'] == 0.0) & (df['alpha_depth'] == 0.0) & (df['alpha_amp'] == 0.0)]
    print("\n" + "=" * 74)
    print(f"ALPHA SWEEP SUMMARY: {name} ({physics})")
    print("=" * 74)
    if not base.empty:
        b = base.iloc[0]
        print(f"baseline (all 0): {depth_label}_MAE={b['depth_mae_blend']:.4g}   "
              f"{amp_label}_MAE={b['amp_mae_blend']:.4g}   loc_err={b['loc_err_km_blend']:.4f} km")
    print(f"best {depth_label:<8}: loc={best_depth['alpha_loc']:g} depth={best_depth['alpha_depth']:g} "
          f"amp={best_depth['alpha_amp']:g}  -> MAE={best_depth['depth_mae_blend']:.4g}")
    print(f"best {amp_label:<8}: loc={best_amp['alpha_loc']:g} depth={best_amp['alpha_depth']:g} "
          f"amp={best_amp['alpha_amp']:g}  -> MAE={best_amp['amp_mae_blend']:.4g}")
    print(f"\nDone in {time.time() - t_start:.1f}s. Outputs in {base_dir}")


if __name__ == '__main__':
    main()
