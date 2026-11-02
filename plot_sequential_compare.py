#!/usr/bin/env python
# Usage:       python plot_sequential_compare.py --album-root saved/multisource_seq_seedtest/A_d3/E1_S3_K2_separated \
#                  [--truth <album>_truth.json] [--out <png>]
# Description: Truth-vs-reconstruction figure for one sequential multi-source album. For every fitted K:
#              row 1 = combined deformation (noise-free truth | noisy data stack mean | PILA reconstruction
#              sum of sources | data minus reconstruction); then one row per pair of truth/recovered
#              INDIVIDUAL sources (truth source i | its matched recovered source), plus unmatched recovered
#              sources when K > true K.
# Notes:       Truth fields are re-rendered noise-free from the truth JSON (Mogi, same grid/LOS as the
#              album builder, synthetic/multisource/make_pila_stack.py). Every map has its own least-squares
#              offset + planar ramp removed over coherent pixels (the nuisance PILA fits per acquisition,
#              and the album's reference-pixel offset), so panels are directly comparable.
#              Matching = optimal 3-D assignment (match_sources in plot_sequential_album.py).
#              Units: km, mm LOS (+ toward satellite), RdBu_r.
# Date:        2026-10-07

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic', 'multisource'))
from plot_sequential_album import load_fits, remove_ramp, match_sources   # noqa: E402
from make_pila_stack import load_ta69_geometry                              # noqa: E402
from synthetic.multisource.torch_generator import mogi_los_mm_torch          # noqa: E402  (same as builder)
import torch                                                                # noqa: E402


def truth_source_maps_mm(truth, fit):
    """Noise-free LOS map [n_true, H, W] (mm) for each truth source, on the album's TA69 grid."""
    geom = load_ta69_geometry()
    grid = geom['grid']
    xE = torch.tensor(grid['xE_m']); yN = torch.tensor(grid['yN_m'])
    los = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]))
    maps = []
    for src in truth['sources']:
        unit_mm_per_m3 = mogi_los_mm_torch(
            xE, yN, los, torch.tensor([[src['xcen_km'] * 1e3]], dtype=torch.float64),
            torch.tensor([[src['ycen_km'] * 1e3]], dtype=torch.float64),
            torch.tensor([[src['depth_km'] * 1e3]], dtype=torch.float64),
            torch.tensor([[1.0]], dtype=torch.float64))[0, 0].numpy()
        maps.append((unit_mm_per_m3 * src['dV_m3']).reshape(grid['shape']))
    maps = np.stack(maps)
    if maps.shape[1:] != fit['mask'].shape:
        raise ValueError(f"truth grid {maps.shape[1:]} != fit grid {fit['mask'].shape}")
    return maps


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
    fit0 = fits[ks[0]][0]
    print(f"[1/3] {album}: true K={truth['n_sources']}, fitted K={ks}")

    # --- Truth maps (ramp+offset removed, same nuisance as data) ---
    truth_maps = remove_ramp(truth_source_maps_mm(truth, fit0), fit0)        # [n_true, H, W]
    truth_total = np.nansum(truth_maps, axis=0)
    truth_total[~fit0['mask'].astype(bool)] = np.nan
    data_mean = np.nanmean(remove_ramp(fit0['data_mm'], fit0), axis=0)
    vmax = np.nanpercentile(np.abs(data_mean), 99.5)
    vmax = max(vmax, np.nanmax(np.abs(truth_total)))
    extent = [fit0['xE_km'][0, 0], fit0['xE_km'][0, -1], fit0['yN_km'][-1, 0], fit0['yN_km'][0, 0]]
    print(f"  truth peak |LOS| = {np.nanmax(np.abs(truth_total)):.1f} mm, colour limit +-{vmax:.1f} mm")

    # --- Layout: per K, 1 combined row + ceil(n_pairs/2) rows of (truth_i, recovered_i) pairs ---
    n_true = len(truth['sources'])
    block_rows = {k: 1 + int(np.ceil((n_true + max(0, k - n_true)) / 2)) for k in ks}
    total_rows = sum(block_rows.values())
    print("[2/3] Plotting...")
    fig, axes = plt.subplots(total_rows, 4, figsize=(20, 4.2 * total_rows), squeeze=False)
    row0 = 0

    def show(ax, img, title, rec_xy=None, true_xy=None):
        im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
        for x_km, y_km in (true_xy or []):
            ax.plot(x_km, y_km, '^', mfc='none', mec='k', ms=10, mew=1.6)
        for x_km, y_km, label in (rec_xy or []):
            ax.plot(x_km, y_km, 'o', mfc='none', mec='lime', ms=11, mew=1.8)
            ax.text(x_km + 1, y_km + 1, label, color='lime', fontsize=9, weight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046).set_label('LOS (mm)')

    all_true_xy = [(s['xcen_km'], s['ycen_km']) for s in truth['sources']]
    for k in ks:
        fit = fits[k][0]
        med = np.median(fit['params_phys'], axis=0)                          # [K, 4]
        rec_src = remove_ramp(np.nanmean(fit['xP_src_mm'], axis=0), fit)     # [K, H, W] stack-mean per source
        rec_total = np.nansum(rec_src, axis=0)
        rec_total[~fit['mask'].astype(bool)] = np.nan
        resid = data_mean - rec_total
        rms_resid = float(np.sqrt(np.nanmean(resid ** 2)))
        rms_vs_truth = float(np.sqrt(np.nanmean((rec_total - truth_total) ** 2)))
        print(f"  K={k}: residual RMS {rms_resid:.2f} mm; recon-vs-truth(noise-free) RMS {rms_vs_truth:.2f} mm")
        all_rec_xy = [(med[b, 0], med[b, 1], str(b + 1)) for b in range(k)]
        # --- Row 1: combined deformation ---
        r = row0
        show(axes[r, 0], truth_total, f'K={k}: TRUTH combined (noise-free)', true_xy=all_true_xy)
        show(axes[r, 1], data_mean, f'K={k}: DATA stack mean (noisy)', true_xy=all_true_xy)
        show(axes[r, 2], rec_total, f'K={k}: RECON combined (sum of {k} sources)', rec_xy=all_rec_xy, true_xy=all_true_xy)
        show(axes[r, 3], resid, f'K={k}: data - recon (RMS {rms_resid:.1f} mm)')
        # --- Individual sources: pairs (truth_i, matched recovered) ---
        pairs = match_sources(truth['sources'], med)
        pairs = sorted(pairs, key=lambda p: (p[0] is None, p[0] if p[0] is not None else p[1]))
        panels = []
        for t_idx, r_idx, err in pairs:
            if t_idx is not None:
                s = truth['sources'][t_idx]
                panels.append((truth_maps[t_idx], f"TRUTH src {t_idx + 1}: ({s['xcen_km']:.0f},{s['ycen_km']:.0f}) km, "
                                                  f"z {s['depth_km']:.1f} km, dV {s['dV_m3']:.1e}", [(s['xcen_km'], s['ycen_km'])], None))
            else:
                panels.append((np.full(truth_total.shape, np.nan), 'TRUTH: none (extra source)', None, None))
            panels.append((rec_src[r_idx], f"RECON src {r_idx + 1}: ({med[r_idx, 0]:.1f},{med[r_idx, 1]:.1f}) km, "
                                           f"z {med[r_idx, 2]:.1f} km, dV {med[r_idx, 3]:.1e}"
                                           + (f"\nloc err {err['loc_err_km']:.1f} km" if 'loc_err_km' in err else '\nUNMATCHED'),
                           None, [(med[r_idx, 0], med[r_idx, 1], str(r_idx + 1))]))
        for idx, (img, title, true_xy, rec_xy) in enumerate(panels):
            ax = axes[row0 + 1 + idx // 4, idx % 4]
            show(ax, img, title, rec_xy=rec_xy, true_xy=true_xy)
        for idx in range(len(panels), 4 * (block_rows[k] - 1)):
            axes[row0 + 1 + idx // 4, idx % 4].axis('off')
        row0 += block_rows[k]
    fig.suptitle(f'{album}  ({os.path.basename(os.path.dirname(args.album_root.rstrip("/")))})  '
                 f'offset+ramp removed from every map; ▲ truth, ○ recovered', fontsize=12, y=1.0)
    plt.tight_layout()
    out = args.out or os.path.join(args.album_root, f'{album}_compare.png')
    fig.savefig(out, dpi=100, bbox_inches='tight'); plt.close(fig)
    print(f"[3/3] Done. Output saved to {out}")


if __name__ == '__main__':
    main()
