#!/usr/bin/env python
# Usage:       # Mogi: 3-way view -> facet by loc, axes depth x amp (group aliases ok)
#              python -m synthetic.sweep_alpha \
#                  --truth synthetic/cubes/<name>/<name>_truth.json \
#                  --ckpt  saved/synth/<name>/.../models/model_best.pth \
#                  --sweep depth amp --facet loc \
#                  --vals1 0 0.3 0.5 0.7 --vals2 0 0.1 0.3 --facet-vals 0 0.3 0.6
#              # Okada: per-parameter, sweep depth x opening, fix width & dip
#              python -m synthetic.sweep_alpha --truth ... --ckpt ... \
#                  --sweep depth opening --fix width=0.3 dip=0.3
#              # --ckpt may also be a DIRECTORY -> newest model_best.pth under it is used.
# Description: per-parameter / per-group temporal-blending sweep for ONE synthetic cube.
#              Pick TWO axes to vary (--sweep, heatmap row & col) and OPTIONALLY a third
#              (--facet) rendered as side-by-side heatmap panels; --fix pins others, the
#              rest use --alpha (default 0 = raw). Each axis token is a GROUP alias
#              (loc/depth/amp) or an exact parameter name (Mogi depth is 'd'). The encoder
#              output is alpha-independent, so the model is loaded + every epoch encoded
#              ONCE; each combo only re-blends + re-decodes -> a dense grid is cheap.
#              Writes a summary CSV and one faceted recovery heatmap per swept axis.
# Scientific notes:
#   * Maps the depth-amplitude trade-off: pooling geometry (higher alpha) can lower its
#     error, while amplitude (dV / opening) usually wants LOW alpha to track real change.
#     The all-zero corner is the raw, no-blending baseline.
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
from synthetic.evaluate import (encode_pass, blend_decode, build_alpha_vec,
                                parse_param_alphas, _recovery_metrics, LOC_GROUPS,
                                LOC_KEYS, PHYSICS_ATTRS, ROOT as EVAL_ROOT)


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


def resolve_token(token, physics, attrs):
    """A sweep/fix/facet token is either a GROUP alias (loc/depth/amp) or an exact parameter
    name. Return the list of parameter names it controls (e.g. Mogi 'loc' -> [xcen, ycen],
    'depth' -> [d])."""
    groups = LOC_GROUPS.get(physics, {})
    if token in groups:
        return list(groups[token])
    if token in attrs:
        return [token]
    raise ValueError(f"'{token}' is neither a group (loc/depth/amp) nor a {physics} "
                     f"parameter {attrs}")


def faceted_heatmap(out_stem, suptitle, row_vals, col_vals, facet_vals, facet_token,
                    panel_grids, cbar_label, row_label, col_label, cmap='viridis'):
    """Row of heatmaps sharing one colour scale: one panel per facet value (panel_grids
    keyed facet_val -> {(row_val, col_val): metric}). facet_vals=[None] -> single panel.
    Annotates every cell, outlines each panel's minimum-error cell in red. png+pdf."""
    finite = [v for g in panel_grids.values() for v in g.values() if np.isfinite(v)]
    vmin, vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)
    nfac = len(facet_vals)
    fig, axes = plt.subplots(1, nfac, squeeze=False,
                             figsize=(1.6 + 1.0 * len(col_vals) * nfac,
                                      1.9 + 0.9 * len(row_vals)))
    im = None
    for k, fv in enumerate(facet_vals):
        ax = axes[0][k]
        M = np.array([[panel_grids[fv].get((r, c), np.nan) for c in col_vals]
                      for r in row_vals], dtype=float)
        im = ax.imshow(M, cmap=cmap, origin='upper', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(col_vals)))
        ax.set_xticklabels([f'{c:g}' for c in col_vals])
        ax.set_xlabel(col_label)
        if k == 0:
            ax.set_yticks(range(len(row_vals)))
            ax.set_yticklabels([f'{r:g}' for r in row_vals])
            ax.set_ylabel(row_label)
        else:
            ax.set_yticks([])
        if fv is not None:
            ax.set_title(f'{facet_token}={fv:g}')
        for i in range(len(row_vals)):
            for j in range(len(col_vals)):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        color='white' if im.norm(v) < 0.5 else 'black', fontsize=6)
        if np.isfinite(M).any():
            bi, bj = np.unravel_index(np.nanargmin(M), M.shape)
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                       edgecolor='red', lw=1.5))
    fig.colorbar(im, ax=axes.ravel().tolist(), label=cbar_label, shrink=0.85)
    fig.suptitle(suptitle)
    fig.savefig(out_stem + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_stem + '.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Per-parameter/per-group temporal-blending sweep for one synthetic cube. "
                    "Two --sweep axes (+ optional --facet) over group aliases (loc/depth/amp) "
                    "or exact param names; encodes once, re-blends per combo; writes a CSV + "
                    "faceted recovery heatmaps.")
    ap.add_argument('--truth', required=True, help='Ground-truth sidecar JSON.')
    ap.add_argument('--ckpt', required=True,
                    help='model_best.pth, OR a directory (newest model_best.pth is used).')
    ap.add_argument('--sweep', nargs=2, required=True, metavar=('P1', 'P2'),
                    help='Two axes to vary (heatmap row, col): group alias or exact name.')
    ap.add_argument('--facet', default=None, metavar='P3',
                    help='Optional third axis rendered as side-by-side heatmap panels.')
    ap.add_argument('--vals1', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                    help='alpha grid for P1 (rows).')
    ap.add_argument('--vals2', type=float, nargs='+', default=None,
                    help='alpha grid for P2 (cols). Default: same as --vals1.')
    ap.add_argument('--facet-vals', type=float, nargs='+', default=None,
                    help='alpha grid for the facet axis. Default: [0] if --facet given.')
    ap.add_argument('--fix', nargs='+', default=None, metavar='NAME=VALUE',
                    help='Fix OTHER axes/params at given alphas, e.g. --fix width=0.3 dip=0.3.')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='Fallback weight for anything not swept/faceted/fixed. Default 0.0.')
    ap.add_argument('--refit-amp', action='store_true',
                    help='Coupling (option A): re-solve the amplitude (dV/opening) per epoch '
                         'from the data with the blended geometry held fixed.')
    ap.add_argument('--out', default=None, help='Base output dir (default synthetic/eval/<name>).')
    ap.add_argument('--per-combo-plots', action='store_true',
                    help='Also write per-combo truth/raw/blended trajectory + recovery plots.')
    args = ap.parse_args()

    t_start = time.time()

    # --- [1/5] Resolve inputs + truth + axis tokens ---
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
    facet = args.facet
    vals1 = args.vals1
    vals2 = args.vals2 if args.vals2 is not None else args.vals1
    facet_vals = (args.facet_vals if args.facet_vals is not None else [0.0]) if facet else [None]
    fixed = parse_param_alphas(args.fix) or {}

    # token -> the parameter names it controls (validates names early)
    axis_tokens = [p1, p2] + ([facet] if facet else [])
    if len(set(axis_tokens)) != len(axis_tokens):
        raise ValueError(f"--sweep / --facet axes must be distinct, got {axis_tokens}")
    tok_params = {tok: resolve_token(tok, physics, attrs)
                  for tok in axis_tokens + list(fixed)}
    report_tokens = axis_tokens + [t for t in fixed if t not in axis_tokens]

    truth_names = truth['param_names']
    truth_params = np.asarray(truth['params_per_epoch'])
    signal_rms = np.asarray(truth['signal_rms_mm_per_epoch'])
    n_epoch = truth_params.shape[0]
    # Guard: checkpoint physics must match the truth cube (catches a stale --ckpt).
    if sorted(truth_names) != sorted(attrs):
        raise ValueError(
            f"checkpoint physics is {physics} (params {attrs}), but truth cube '{name}' "
            f"has params {truth_names} -- --ckpt and --truth do not match.")
    col = [attrs.index(p) for p in truth_names]
    kx, ky = LOC_KEYS[physics]
    ix, iy = truth_names.index(kx), truth_names.index(ky)
    base_dir = args.out or os.path.join(EVAL_ROOT, 'synthetic', 'eval', name)
    os.makedirs(base_dir, exist_ok=True)
    n_combo = len(vals1) * len(vals2) * len(facet_vals)
    print(f"  cube={name}  physics={physics}")
    print(f"  sweep: {p1} (rows) x {p2} (cols)" + (f"  facet: {facet}" if facet else '')
          + f"   fixed: {fixed or '{}'}   fallback alpha={args.alpha}")
    print(f"  vals1={vals1}  vals2={vals2}" + (f"  facet_vals={facet_vals}" if facet else '')
          + f"  -> {n_combo} combos")

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

    def token_mae(metric, tok):
        """Mean strong-epoch MAE over the parameters a token controls."""
        return float(np.mean([metric['per_param'][p]['mae_strong_epochs']
                              for p in tok_params[tok]]))

    mae_raw = {tok: token_mae(m_raw, tok) for tok in report_tokens}

    # --- [3/5] Sweep over (P1 x P2 x facet) ---
    print("[3/5] Sweeping combos...")
    rows = []
    # err_grids[axis_token][facet_val][(v1, v2)] = blended MAE of that axis token
    err_grids = {p1: {fv: {} for fv in facet_vals}, p2: {fv: {} for fv in facet_vals}}
    k = 0
    for fv in facet_vals:
        for v1 in vals1:
            for v2 in vals2:
                k += 1
                # token -> value for this combo
                tok_val = {**fixed, p1: v1, p2: v2}
                if facet:
                    tok_val[facet] = fv
                # expand tokens to per-parameter alphas
                param_alphas = {}
                for tok, val in tok_val.items():
                    for nm in tok_params[tok]:
                        param_alphas[nm] = val
                tag = "a{:g}".format(args.alpha) + ''.join(
                    f"_{tok}{tok_val[tok]:g}" for tok in sorted(tok_val))
                if args.refit_amp:
                    tag += '_refitamp'
                print(f"  ({k}/{n_combo}) {tag} ...", flush=True)

                alpha_vec = build_alpha_vec(attrs, physics, args.alpha, param_alphas=param_alphas)
                params_blend, pred_blend = blend_decode(ctx, alpha_vec, refit_amp=args.refit_amp)
                inferred_blend = params_blend[:n, col]
                m_blend = _recovery_metrics(inferred_blend, pred_blend, ctx['targ'], x_scale,
                                            tru, rms, peak, strong, truth_names, physics)
                mae_blend = {tok: token_mae(m_blend, tok) for tok in report_tokens}
                err_grids[p1][fv][(v1, v2)] = mae_blend[p1]
                err_grids[p2][fv][(v1, v2)] = mae_blend[p2]

                row = {f'alpha_{tok}': tok_val[tok] for tok in report_tokens}
                row['alpha_tag'] = tag
                row['loc_err_km_blend'] = m_blend['source_loc_err_km_strong_mean']
                row['los_rmse_mm_blend'] = m_blend['los_rmse_mm_all']
                for tok in report_tokens:
                    row[f'{tok}_mae_raw'] = mae_raw[tok]
                    row[f'{tok}_mae_blend'] = mae_blend[tok]
                rows.append(row)

                combo_dir = os.path.join(base_dir, tag)
                os.makedirs(combo_dir, exist_ok=True)
                with open(os.path.join(combo_dir, f'{name}_metrics_temporal.json'), 'w') as f:
                    json.dump({'name': name, 'physics': physics, 'alpha_default': args.alpha,
                               'alpha_tokens': tok_val,
                               'alpha_per_param': {attrs[i]: float(alpha_vec[i])
                                                   for i in range(len(attrs))},
                               'raw': m_raw, 'temporal': m_blend}, f, indent=2)
                if args.per_combo_plots:
                    lbl = ', '.join(f'{tok}={tok_val[tok]:g}' for tok in sorted(tok_val))
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
    sort_cols = [f'alpha_{t}' for t in axis_tokens]
    df = pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)
    tagname = '_'.join(axis_tokens)
    csv_path = os.path.join(base_dir, f'{name}_alpha_sweep_{tagname}_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  shape={df.shape}  ->  {csv_path}")

    # --- [5/5] One faceted recovery heatmap per swept axis (rows=P1, cols=P2, panels=facet) ---
    print("[5/5] Writing heatmaps...")
    fac_note = f', faceted by {facet}' if facet else ''
    for tok in (p1, p2):
        faceted_heatmap(
            os.path.join(base_dir, f'{name}_sweep_{tagname}_{tok}_err'),
            f'{name}: {tok} error (blended){fac_note}',
            vals1, vals2, facet_vals, facet, err_grids[tok],
            cbar_label=f'{tok} MAE (strong epochs)',
            row_label=f'alpha[{p1}]', col_label=f'alpha[{p2}]')

    # --- Report ---
    print("\n" + "=" * 74)
    print(f"ALPHA SWEEP SUMMARY: {name} ({physics})   sweep {p1} x {p2}"
          + (f" facet {facet}" if facet else '') + f"   fixed {fixed or '{}'}")
    print("=" * 74)
    for tok in (p1, p2):
        best = df.loc[df[f'{tok}_mae_blend'].idxmin()]
        coords = '  '.join(f'{t}={best[f"alpha_{t}"]:g}' for t in axis_tokens)
        print(f"best {tok:<8}: {coords}  -> {tok}_MAE={best[f'{tok}_mae_blend']:.4g} "
              f"(raw {mae_raw[tok]:.4g})")
    print(f"\nDone in {time.time() - t_start:.1f}s. Outputs in {base_dir}")


if __name__ == '__main__':
    main()
