#!/usr/bin/env python
# Usage:       python plot_sequential_album.py --album-root saved/multisource_seq/<tag>/<album> \
#                  [--truth synthetic/multisource/pila_stacks_v2/albums/<album>_truth.json] [--out <png>]
# Description: Diagnostics for one album of the sequential multi-source PILA runs (all K found under
#              album-root): per K, the stack-median recovered sources vs the truth (location / depth /
#              dV errors after optimal matching), the stack-mean data, physics fit and residual maps, the
#              per-acquisition dV of every source, and the per-phase training time per K.
# Notes:       Location error = horizontal distance between matched true and recovered sources (km).
#              Matching = assignment minimising total 3-D distance (scipy linear_sum_assignment); extra
#              recovered sources (K > true K) are reported as "unmatched" with their |dV|.
#              Units: km, m^3, LOS mm (+ toward satellite).
# Date:        2026-10-06

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

PHASES = ['data_fetch', 'encode', 'physics_decode', 'residual_decode', 'loss', 'backward',
          'optimizer_step', 'bookkeeping', 'validation', 'checkpoint_save']


def load_fits(album_root):
    """{K: (fit npz dict, compute_usage dict)} for every K run under album_root."""
    fits = {}
    for path in sorted(glob.glob(os.path.join(album_root, 'K*', '*', '*', 'models', 'fit_K*.npz'))):
        k = int(os.path.basename(path)[5:-4])
        with np.load(path) as f:
            fit = {key: f[key] for key in f.files}
        usage_path = os.path.join(os.path.dirname(path), 'compute_usage.json')
        usage = json.load(open(usage_path)) if os.path.exists(usage_path) else {}
        fits[k] = (fit, usage)
    if not fits:
        raise FileNotFoundError(f"no fit_K*.npz under {album_root}")
    return fits


def remove_ramp(maps, fit):
    """
    Remove each map's least-squares offset + planar ramp (the nuisance model) over the COHERENT
    pixels only (same points as PILA's loss); off-mask pixels stay NaN. maps [T, H, W] -> same shape.
    """
    mask = fit['mask'].astype(bool)
    G = np.stack([np.ones(mask.sum()), fit['xE_km'][mask], fit['yN_km'][mask]], axis=1)
    vals = maps[:, mask]                                             # [T, N_coherent]
    out = np.full(maps.shape, np.nan)
    out[:, mask] = vals - (vals @ np.linalg.pinv(G).T) @ G.T
    return out


def match_sources(true_sources, rec_median):
    """Optimal matching; returns rows of (true idx or None, rec idx, errors dict)."""
    rows = []
    if true_sources:
        t = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km']] for s in true_sources])
        cost = np.linalg.norm(t[:, None, :] - rec_median[None, :, :3], axis=-1)
        ti, ri = linear_sum_assignment(cost)
        for a, b in zip(ti, ri):
            src = true_sources[a]
            rows.append((a, b, dict(
                loc_err_km=float(np.hypot(rec_median[b, 0] - src['xcen_km'], rec_median[b, 1] - src['ycen_km'])),
                depth_err_km=float(rec_median[b, 2] - src['depth_km']),
                dV_ratio=float(rec_median[b, 3] / src['dV_m3']))))
        matched = set(ri)
    else:
        matched = set()
    for b in range(rec_median.shape[0]):
        if b not in matched:
            rows.append((None, b, dict(unmatched_abs_dV_m3=float(abs(rec_median[b, 3])))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album-root', required=True)
    parser.add_argument('--truth', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    album = os.path.basename(os.path.normpath(args.album_root))
    truth_path = args.truth or os.path.join('synthetic/multisource/pila_stacks_v2/albums', f'{album}_truth.json')
    if not os.path.exists(truth_path):
        raise FileNotFoundError(truth_path)
    truth = json.load(open(truth_path))
    fits = load_fits(args.album_root)
    ks = sorted(fits)
    print(f"[1/3] {album}: true K = {truth['n_sources']}, fitted K = {ks}")

    print("[2/3] Recovered sources (stack median over acquisitions) vs truth:")
    summary = {}
    for k in ks:
        fit, usage = fits[k]
        med = np.median(fit['params_phys'], axis=0)                    # [K, 4]
        spread = np.percentile(fit['params_phys'], [5, 95], axis=0)    # [2, K, 4]
        resid = remove_ramp(fit['data_mm'] - fit['xP_src_mm'].sum(axis=1), fit)
        rms = float(np.sqrt(np.nanmean(resid ** 2)))
        rows = match_sources(truth['sources'], med)
        summary[k] = dict(median=med.tolist(), residual_rms_mm=rms, matches=rows,
                          train_sec=usage.get('train_sec'), timing=usage.get('timing', {}))
        print(f"  K={k}: physics-only residual RMS (offset+ramp removed) {rms:.2f} mm; train {usage.get('train_sec', float('nan')):.0f}s")
        for b in range(k):
            p5, p95 = spread[0, b], spread[1, b]
            print(f"    source {b + 1}: x {med[b, 0]:6.2f} [{p5[0]:.2f},{p95[0]:.2f}] km, y {med[b, 1]:6.2f} km, "
                  f"depth {med[b, 2]:5.2f} [{p5[2]:.2f},{p95[2]:.2f}] km, dV {med[b, 3]:.3e} m^3")
        for t_idx, r_idx, err in rows:
            label = f"true {t_idx + 1}" if t_idx is not None else 'unmatched'
            print(f"      {label} <- recovered {r_idx + 1}: " + ', '.join(f"{a}={v:.3g}" for a, v in err.items()))

    print("[3/3] Figure...")
    n_rows = len(ks) + 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.3 * n_rows))
    fit0 = fits[ks[0]][0]
    extent = [fit0['xE_km'][0, 0], fit0['xE_km'][0, -1], fit0['yN_km'][-1, 0], fit0['yN_km'][0, 0]]
    data_mean = np.nanmean(remove_ramp(fit0['data_mm'], fit0), axis=0)        # offset + ramp removed
    vmax = np.nanpercentile(np.abs(data_mean), 99.5)

    def mark(ax, rec=None):
        for s in truth['sources']:
            ax.plot(s['xcen_km'], s['ycen_km'], '^', mfc='none', mec='k', ms=11, mew=1.8)
        if rec is not None:
            for b in range(rec.shape[0]):
                ax.plot(rec[b, 0], rec[b, 1], 'o', mfc='none', mec='lime', ms=12, mew=2)
                ax.text(rec[b, 0] + 1, rec[b, 1] + 1, str(b + 1), color='lime', fontsize=10, weight='bold')

    ax = axes[0, 0]
    im = ax.imshow(data_mean, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent)
    mark(ax); ax.set_title(f'{album}\nData minus per-acquisition offset+ramp, mean of {fit0["data_mm"].shape[0]} (▲ truth)', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label('LOS (mm)')
    # timing bars per K
    ax = axes[0, 1]
    bottoms = np.zeros(len(ks))
    for phase in PHASES:
        vals = np.array([fits[k][1].get('timing', {}).get(f'{phase}_sec', 0.0) for k in ks])
        ax.bar([f'K={k}' for k in ks], vals, bottom=bottoms, label=phase)
        bottoms += vals
    ax.set_ylabel('Training wall-clock (s)'); ax.set_title('Training time per phase', fontsize=9)
    ax.legend(fontsize=6)
    ax = axes[0, 2]
    ax.plot(ks, [summary[k]['residual_rms_mm'] for k in ks], 'o-')
    ax.set_xlabel('Number of sources K'); ax.set_ylabel('Physics-only residual RMS (mm)')
    ax.set_xticks(ks); ax.set_title('Misfit vs K', fontsize=9)
    axes[0, 3].axis('off')
    axes[0, 3].text(0, 0.95, '\n'.join([f"truth: {len(truth['sources'])} source(s)"] +
                                       [f"  ({s['xcen_km']:.1f}, {s['ycen_km']:.1f}) km, {s['depth_km']:.1f} km, "
                                        f"dV {s['dV_m3']:.2e} m³, peak/σ {s['peak_ratio']}" for s in truth['sources']]),
                    va='top', fontsize=9, family='monospace')

    for row, k in enumerate(ks, start=1):
        fit = fits[k][0]
        med = np.array(summary[k]['median'])
        xP_mean = np.nanmean(fit['xP_src_mm'].sum(axis=1), axis=0)
        res_mean = np.nanmean(remove_ramp(fit['data_mm'] - fit['xP_src_mm'].sum(axis=1), fit), axis=0)
        for col, (img, title) in enumerate([(xP_mean, f'K={k}: physics fit (stack mean), ○ recovered'),
                                            (res_mean, f'K={k}: residual (offset+ramp removed, stack mean)')]):
            ax = axes[row, col]
            im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent)
            mark(ax, med); ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046).set_label('LOS (mm)')
        ax = axes[row, 2]
        for b in range(k):
            ax.plot(fit['params_phys'][:, b, 3], 'o-', ms=3, label=f'source {b + 1}')
        for s in truth['sources']:
            ax.axhline(s['dV_m3'], color='k', ls='--', lw=0.8)
        ax.set_xlabel('Acquisition index'); ax.set_ylabel('dV (m³)'); ax.legend(fontsize=7)
        ax.set_title(f'K={k}: dV per acquisition (dashed = truth)', fontsize=9)
        ax = axes[row, 3]
        for b in range(k):
            ax.plot(fit['params_phys'][:, b, 2], 'o-', ms=3, label=f'source {b + 1}')
        for s in truth['sources']:
            ax.axhline(s['depth_km'], color='k', ls='--', lw=0.8)
        ax.set_xlabel('Acquisition index'); ax.set_ylabel('Depth (km)'); ax.legend(fontsize=7)
        ax.set_title(f'K={k}: depth per acquisition (dashed = truth)', fontsize=9)
        for a in axes[row, :2]:
            a.set_xlabel('East (km)'); a.set_ylabel('North (km)')
    axes[0, 0].set_xlabel('East (km)'); axes[0, 0].set_ylabel('North (km)')
    plt.tight_layout()
    out = args.out or os.path.join(args.album_root, f'{album}_sequential.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    with open(os.path.join(args.album_root, 'summary.json'), 'w') as f:
        json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != 'matches'} | {
            'matches': [[t, r, e] for t, r, e in v['matches']]} for k, v in summary.items()}, f, indent=1, default=str)
    print(f"Done. Output saved to {out}")


if __name__ == '__main__':
    main()
