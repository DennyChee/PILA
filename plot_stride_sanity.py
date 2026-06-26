#!/usr/bin/env python
# Usage:       conda activate pila
#              python plot_stride_sanity.py [--strides 10 20 30 40 50] [--member-offset 0]
# Description: Visual sanity check of the strided-UQ PILA inversions on the synthetic
#              Marapi Okada-dike cube. For one representative ensemble member per stride it
#              renders the PEAK-epoch LOS field as observed | PILA prediction (x_PB) |
#              residual (obs - pred), drawn as a scatter on the coarse grid (single decimated
#              pixels are invisible in an imshow montage, so we use sized markers). Annotates
#              each row with the member's pixel count, fit R^2 and residual RMS. One figure,
#              one row per stride, so the strides can be compared at a glance.
# Date:        2026-06-24
#
# Scientific notes (flagged):
#   * "Peak epoch" is taken from the truth JSON's peak_epoch_index (the epoch of largest
#     true deformation), so every stride is shown at the SAME date for a fair comparison.
#   * Observed and prediction share one diverging colour scale (RdBu_r, + toward satellite);
#     the residual uses its own, tighter scale so the misfit structure is visible.
#   * R^2 / RMS are computed on the member's OWN decimated pixels at the peak epoch only --
#     a single-epoch fit diagnostic, not a time-series-wide score.

import os
import sys
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless HPC node
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Reuse the existing, validated full-inference routine (handles stride/offset masking).
from synthetic.validate_inference import run_inference_full

TRUTH = 'synthetic/cubes/marapi_okada_dike_combined/marapi_okada_dike_combined_truth.json'
RUNS_FMT = 'saved/stride_uq/marapi_okada_dike_mid_combined_uqstr{n}/*off{off}/*/models/model_best.pth'


def fit_stats(obs_mm, pred_mm):
    """Peak-epoch fit diagnostics on one member's pixels. Returns (R^2, RMS_mm)."""
    ss_res = np.nansum((obs_mm - pred_mm) ** 2)
    ss_tot = np.nansum((obs_mm - np.nanmean(obs_mm)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    rms = float(np.sqrt(np.nanmean((obs_mm - pred_mm) ** 2)))
    return r2, rms


def load_member(stride, offset):
    """Run peak-epoch inference for one ensemble member. Returns dict or None if missing."""
    pattern = os.path.join(CURRENT_DIR, RUNS_FMT.format(n=stride, off=offset))
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    ckpt = hits[0]
    with open(os.path.join(os.path.dirname(ckpt), 'config.json')) as fh:
        config = json.load(fh)
    obs, pb, _pp, data, dates = run_inference_full(config, ckpt)
    return {'stride': stride, 'obs': obs, 'pred': pb, 'mask_d': data.mask_d,
            'dates': dates, 'n_points': obs.shape[1]}


def main():
    ap = argparse.ArgumentParser(description="Per-stride peak-epoch fit sanity figure")
    ap.add_argument('--strides', type=int, nargs='+', default=[10, 20, 30, 40, 50])
    ap.add_argument('--member-offset', type=int, default=0,
                    help="which ensemble member (offset) to show per stride (default 0)")
    ap.add_argument('--truth', default=TRUTH)
    ap.add_argument('--out', default='comparison_out/uq_stride_sanity')
    args = ap.parse_args()

    out_dir = os.path.join(CURRENT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # --- Peak epoch from truth (same date for every stride) ---
    truth_path = args.truth if os.path.isabs(args.truth) else os.path.join(CURRENT_DIR, args.truth)
    with open(truth_path) as fh:
        truth = json.load(fh)
    peak_idx = int(truth['peak_epoch_index'])
    peak_date = truth['epochs'][peak_idx] if peak_idx < len(truth.get('epochs', [])) else f"idx{peak_idx}"
    print(f"[1/3] Peak epoch index {peak_idx} ({peak_date})")

    # --- Run inference per stride ---
    print(f"[2/3] Running inference for member off{args.member_offset} per stride ...")
    members = []
    for stride in sorted(args.strides):
        m = load_member(stride, args.member_offset)
        if m is None:
            print(f"  stride {stride}: NO checkpoint for offset {args.member_offset}; skipping")
            continue
        m['obs_peak'] = m['obs'][peak_idx]
        m['pred_peak'] = m['pred'][peak_idx]
        m['res_peak'] = m['obs_peak'] - m['pred_peak']
        m['r2'], m['rms'] = fit_stats(m['obs_peak'], m['pred_peak'])
        rows_d, cols_d = np.where(m['mask_d'])  # grid coords of this member's pixels
        m['rows'], m['cols'] = rows_d, cols_d
        members.append(m)
        print(f"  stride {stride}: N={m['n_points']} px  R2={m['r2']:.3f}  RMS={m['rms']:.1f} mm")
    if not members:
        raise SystemExit("No members found for any requested stride.")

    # --- Shared colour scales ---
    # obs/pred: robust 99th-pct of |signal| so the dike core is visible without a single hot
    # pixel saturating the map; residual: its own tighter scale.
    sig = np.concatenate([np.concatenate([m['obs_peak'], m['pred_peak']]) for m in members])
    vmax = float(np.nanpercentile(np.abs(sig), 99)) + 1e-9
    rmax = float(np.nanpercentile(np.abs(np.concatenate([m['res_peak'] for m in members])), 99)) + 1e-9
    Hd, Wd = members[0]['mask_d'].shape

    # --- Figure: one row per stride, columns = observed | prediction | residual ---
    print("[3/3] Rendering sanity figure ...")
    n_row = len(members)
    fig, axes = plt.subplots(n_row, 3, figsize=(11, 3.4 * n_row), squeeze=False)
    col_titles = ['observed (LOS)', 'PILA prediction (x_PB)', 'residual (obs - pred)']
    sc_sig = sc_res = None
    for ri, m in enumerate(members):
        panels = [(m['obs_peak'], vmax, 'RdBu_r'),
                  (m['pred_peak'], vmax, 'RdBu_r'),
                  (m['res_peak'], rmax, 'RdBu_r')]
        # Marker size grows with stride so sparser members stay visible.
        msize = 4 + m['stride'] * 0.6
        for ci, (vals, vm, cmap) in enumerate(panels):
            ax = axes[ri][ci]
            sc = ax.scatter(m['cols'], m['rows'], c=vals, cmap=cmap, vmin=-vm, vmax=vm,
                            s=msize, marker='s', edgecolors='none')
            ax.set_xlim(0, Wd); ax.set_ylim(Hd, 0)  # origin='upper'
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            if ci < 2:
                sc_sig = sc
            else:
                sc_res = sc
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=11)
        axes[ri][0].set_ylabel(f"stride {m['stride']}\nN={m['n_points']} px\n"
                               f"R$^2$={m['r2']:.3f}\nRMS={m['rms']:.1f} mm",
                               fontsize=9, rotation=0, ha='right', va='center', labelpad=30)

    fig.suptitle(f"PILA strided-UQ sanity: peak-epoch fit ({peak_date}), member off{args.member_offset}\n"
                 f"Marapi Okada-dike synthetic  (+ toward satellite, mm)", fontsize=12)
    fig.subplots_adjust(left=0.19, right=0.9, top=0.93, bottom=0.04, hspace=0.08, wspace=0.05)
    # Two shared colourbars: signal (obs/pred) and residual.
    cax1 = fig.add_axes([0.92, 0.55, 0.012, 0.33]); fig.colorbar(sc_sig, cax=cax1, label='LOS (mm)')
    cax2 = fig.add_axes([0.92, 0.12, 0.012, 0.33]); fig.colorbar(sc_res, cax=cax2, label='residual (mm)')

    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'stride_sanity_peak.{ext}')
        fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Done. Wrote {out_dir}/stride_sanity_peak.png/.pdf")


if __name__ == '__main__':
    main()
