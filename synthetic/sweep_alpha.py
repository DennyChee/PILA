#!/usr/bin/env python
# Usage:       # Okada: sweep depth x opening, fix the other geometry knobs
#              python -m synthetic.sweep_alpha \
#                  --truth synthetic/cubes/<name>/<name>_truth.json \
#                  --ckpt  saved/synth/<name>/.../models/model_best.pth \
#                  --sweep depth opening --vals1 0 0.1 0.3 0.5 0.7 \
#                  --fix width=0.3 dip=0.3 [--per-combo-plots]
#              # Mogi: sweep the depth param 'd' x amplitude 'dV'
#              python -m synthetic.sweep_alpha --truth ... --ckpt ... --sweep d dV
#              # --ckpt may also be a DIRECTORY -> newest model_best.pth under it is used.
# Description: 2-D per-parameter temporal-blending sweep for ONE synthetic cube. You pick
#              TWO parameters to vary (the heatmap axes) and optionally --fix others; every
#              remaining parameter gets the --alpha fallback (default 0 = raw). The encoder
#              output is INDEPENDENT of the blend weights, so the model is loaded + every
#              epoch encoded ONCE (encode_pass); each combo only re-blends + re-decodes
#              (blend_decode), so a dense grid is cheap. Writes a summary CSV and a recovery
#              heatmap for EACH swept parameter over the (alpha[P1] x alpha[P2]) plane.
# Scientific notes:
#   * Use this to map the depth-amplitude trade-off per parameter: e.g. pooling a geometry
#     param (higher alpha) can break the trade-off and lower its error, while amplitude
#     (opening / dV) usually wants a LOW alpha to track real episodic change. The
#     (0, 0) corner is the raw, no-blending baseline.
#   * Parameter names must be exact (Mogi depth is 'd'); unknown names error out.
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
                                parse_param_alphas, _recovery_metrics, LOC_KEYS,
                                PHYSICS_ATTRS, ROOT as EVAL_ROOT)


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
        description="2-D per-parameter temporal-blending sweep for one synthetic cube: pick "
                    "two params as heatmap axes, fix others, encode once and re-blend per "
                    "combo; writes a summary CSV + a recovery heatmap per swept parameter.")
    ap.add_argument('--truth', required=True, help='Ground-truth sidecar JSON.')
    ap.add_argument('--ckpt', required=True,
                    help='model_best.pth, OR a directory (newest model_best.pth is used).')
    ap.add_argument('--sweep', nargs=2, required=True, metavar=('P1', 'P2'),
                    help='Two parameter names to vary (heatmap row, col). Exact names; '
                         'Mogi depth is "d".')
    ap.add_argument('--vals1', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                    help='alpha grid for P1 (heatmap rows).')
    ap.add_argument('--vals2', type=float, nargs='+', default=None,
                    help='alpha grid for P2 (heatmap cols). Default: same as --vals1.')
    ap.add_argument('--fix', nargs='+', default=None, metavar='NAME=VALUE',
                    help='Fix OTHER parameters at given alphas, e.g. --fix width=0.3 dip=0.3. '
                         'Anything not swept or fixed uses --alpha.')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='Fallback weight for parameters that are neither swept nor fixed. '
                         'Default: 0.0 (raw per-epoch).')
    ap.add_argument('--out', default=None,
                    help='Base output dir (default synthetic/eval/<name>).')
    ap.add_argument('--per-combo-plots', action='store_true',
                    help='Also write per-combo truth/raw/blended trajectory + param-recovery '
                         'plots (off by default to keep a dense sweep fast/uncluttered).')
    args = ap.parse_args()

    t_start = time.time()

    # --- [1/5] Resolve inputs + truth + validate parameter names ---
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
    p1, p2 = args.sweep
    vals1 = args.vals1
    vals2 = args.vals2 if args.vals2 is not None else args.vals1
    fixed = parse_param_alphas(args.fix) or {}
    for nm in [p1, p2, *fixed]:
        if nm not in attrs:
            raise ValueError(f"'{nm}' is not a {physics} parameter; valid names: {attrs}")
    if p1 == p2:
        raise ValueError("--sweep needs two DIFFERENT parameter names.")
    controlled = sorted(set([p1, p2, *fixed]))   # params we report MAE for

    truth_names = truth['param_names']
    truth_params = np.asarray(truth['params_per_epoch'])
    signal_rms = np.asarray(truth['signal_rms_mm_per_epoch'])
    n_epoch = truth_params.shape[0]
    col = [attrs.index(p) for p in truth_names]
    kx, ky = LOC_KEYS[physics]
    ix, iy = truth_names.index(kx), truth_names.index(ky)
    base_dir = args.out or os.path.join(EVAL_ROOT, 'synthetic', 'eval', name)
    os.makedirs(base_dir, exist_ok=True)
    n_combo = len(vals1) * len(vals2)
    print(f"  cube={name}  physics={physics}")
    print(f"  sweep: {p1} (rows) x {p2} (cols)   fixed: {fixed or '{}'}   fallback alpha={args.alpha}")
    print(f"  vals1={vals1}  vals2={vals2}  -> {n_combo} combos")

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

    inferred_raw = ctx['params_raw'][:n, col]
    m_raw = _recovery_metrics(inferred_raw, ctx['pred_raw'], ctx['targ'], x_scale,
                              tru, rms, peak, strong, truth_names, physics)
    mae_raw = {p: m_raw['per_param'][p]['mae_strong_epochs'] for p in controlled}

    # --- [3/5] Sweep: blend + decode per (v1, v2) combo (cheap) ---
    print("[3/5] Sweeping combos...")
    rows = []
    err_grid = {p1: {}, p2: {}}     # param -> {(v1, v2): blended MAE}
    k = 0
    for v1 in vals1:
        for v2 in vals2:
            k += 1
            param_alphas = {**fixed, p1: v1, p2: v2}
            tag = alpha_tag(args.alpha, param_alphas=param_alphas)
            print(f"  ({k}/{n_combo}) {tag} ...", flush=True)
            alpha_vec = build_alpha_vec(attrs, physics, args.alpha, param_alphas=param_alphas)
            params_blend, pred_blend = blend_decode(ctx, alpha_vec)
            inferred_blend = params_blend[:n, col]
            m_blend = _recovery_metrics(inferred_blend, pred_blend, ctx['targ'], x_scale,
                                        tru, rms, peak, strong, truth_names, physics)
            mae_blend = {p: m_blend['per_param'][p]['mae_strong_epochs'] for p in controlled}
            err_grid[p1][(v1, v2)] = mae_blend[p1]
            err_grid[p2][(v1, v2)] = mae_blend[p2]

            row = {f'alpha_{p1}': v1, f'alpha_{p2}': v2, 'alpha_tag': tag,
                   'loc_err_km_blend': m_blend['source_loc_err_km_strong_mean'],
                   'los_rmse_mm_blend': m_blend['los_rmse_mm_all']}
            for p in controlled:
                row[f'{p}_mae_raw'] = mae_raw[p]
                row[f'{p}_mae_blend'] = mae_blend[p]
            rows.append(row)

            combo_dir = os.path.join(base_dir, tag)
            os.makedirs(combo_dir, exist_ok=True)
            with open(os.path.join(combo_dir, f'{name}_metrics_temporal.json'), 'w') as f:
                json.dump({'name': name, 'physics': physics, 'alpha_default': args.alpha,
                           'alpha_params': param_alphas,
                           'alpha_per_param': {attrs[i]: float(alpha_vec[i])
                                               for i in range(len(attrs))},
                           'raw': m_raw, 'temporal': m_blend}, f, indent=2)
            if args.per_combo_plots:
                lbl = f"{p1}={v1:g}, {p2}={v2:g}" + (f"  fixed {fixed}" if fixed else '')
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
    df = pd.DataFrame(rows).sort_values([f'alpha_{p1}', f'alpha_{p2}']).reset_index(drop=True)
    csv_path = os.path.join(base_dir, f'{name}_alpha_sweep_{p1}_{p2}_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  shape={df.shape}  ->  {csv_path}")

    # --- [5/5] One recovery heatmap per swept parameter, over (alpha[P1] x alpha[P2]) ---
    print("[5/5] Writing heatmaps...")
    for p in (p1, p2):
        heatmap(os.path.join(base_dir, f'{name}_sweep_{p1}_{p2}_{p}_err'),
                f'{name}: {p} error, blended',
                vals1, vals2, err_grid[p],
                cbar_label=f'{p} MAE (strong epochs)',
                row_label=f'alpha[{p1}]', col_label=f'alpha[{p2}]')

    # --- Report best combos ---
    print("\n" + "=" * 74)
    print(f"ALPHA SWEEP SUMMARY: {name} ({physics})   sweep {p1} x {p2}   fixed {fixed or '{}'}")
    print("=" * 74)
    base = df[(df[f'alpha_{p1}'] == 0.0) & (df[f'alpha_{p2}'] == 0.0)]
    if not base.empty:
        b = base.iloc[0]
        print(f"baseline ({p1}=0,{p2}=0): "
              + "  ".join(f"{p}_MAE={b[f'{p}_mae_blend']:.4g}" for p in controlled))
    for p in (p1, p2):
        best = df.loc[df[f'{p}_mae_blend'].idxmin()]
        print(f"best {p:<8}: {p1}={best[f'alpha_{p1}']:g} {p2}={best[f'alpha_{p2}']:g}"
              f"  -> {p}_MAE={best[f'{p}_mae_blend']:.4g}")
    print(f"\nDone in {time.time() - t_start:.1f}s. Outputs in {base_dir}")


if __name__ == '__main__':
    main()
