#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.plot_multisource_preview \
#                  --data synthetic/multisource/preview
# Description: Diagnostic figures for the Stage-1 multi-source synthetic data written by
#              generate_multisource.py: (1) per-K example scenes showing each source's clean
#              LOS field, their sum, the noise and the observed map; (2) distributions of the
#              sampled truth (K, peak ratio, depth, dV, location, pair separation in units of
#              the shallower depth) and of the noise RMS for TRAIN vs held-out TEST pools.
# Date:        2026-10-05

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_examples(data, meta, out_dir):
    """One figure per K: rows = example scenes, cols = sources 1..K_max, clean sum, noise, observed."""
    k_max = meta['prior']['k_max']
    extent = [data['xE_km'].min(), data['xE_km'].max(), data['yN_km'].min(), data['yN_km'].max()]
    for k in range(k_max + 1):
        idx_list = np.where(data['k'] == k)[0]
        n_rows = len(idx_list)
        col_titles = [f'Source {s + 1}' for s in range(k_max)] + ['Clean sum', 'Noise', 'Observed']
        fig, axes = plt.subplots(n_rows, len(col_titles), figsize=(3.0 * len(col_titles), 2.8 * n_rows),
                                 squeeze=False)
        for row, scene_idx in enumerate(idx_list):
            panels = list(data['fields_mm'][scene_idx]) + [data['clean_mm'][scene_idx],
                                                           data['noise_mm'][scene_idx],
                                                           data['obs_mm'][scene_idx]]
            # One symmetric colour scale per row, set by the observed map, so signal and
            # noise amplitudes are directly comparable along the row.
            vmax_mm = np.percentile(np.abs(data['obs_mm'][scene_idx]), 99.5)
            params = data['params'][scene_idx]
            extra = data['extra'][scene_idx]
            for col, (ax, panel) in enumerate(zip(axes[row], panels)):
                im = ax.imshow(panel, cmap='RdBu_r', vmin=-vmax_mm, vmax=vmax_mm,
                               extent=extent, origin='upper')
                if col < k_max and not data['exist'][scene_idx, col]:
                    ax.text(0.5, 0.5, 'no source', transform=ax.transAxes,
                            ha='center', va='center', color='0.4')
                elif col < k_max:
                    x_km, y_km, d_km, dV_m3 = params[col]
                    ax.set_title(f'S{col + 1}: d={d_km:.1f} km, dV={dV_m3:.1e} m³\n'
                                 f'peak/σ={extra[col, 0]:.2f} (RMS-SNR {extra[col, 1]:.1f} dB)',
                                 fontsize=8)
                # Mark true source centres on the sum / observed maps.
                if col >= k_max and col != k_max + 1:
                    for s in range(k):
                        marker = '^' if params[s, 3] > 0 else 'v'   # up = inflation
                        ax.plot(params[s, 0], params[s, 1], marker, mfc='none', mec='k', ms=8)
                        ax.text(params[s, 0] + 1, params[s, 1] + 1, str(s + 1), fontsize=8)
                if col >= k_max:
                    ax.set_title(col_titles[col], fontsize=10)
                elif not data['exist'][scene_idx, col]:
                    ax.set_title(col_titles[col], fontsize=8)
                if col == len(panels) - 1:
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cb.set_label('LOS (mm)')
                ax.set_xlabel('East (km)', fontsize=8)
                ax.set_ylabel('North (km)', fontsize=8)
                ax.tick_params(labelsize=7)
            sep = data['min_sep_dshallow'][scene_idx]
            sep_txt = f'min sep {sep:.2f}·d_shallow' if np.isfinite(sep) else ''
            axes[row, -1].text(1.45, 0.5, f'scene {scene_idx}\n{sep_txt}', transform=axes[row, -1].transAxes,
                               fontsize=8, va='center')
        noise_tag = 'NOISE-FREE, ' if meta.get('noise') == 'none' else ''
        fig.suptitle(f'{noise_tag}K = {k} source(s): clean per-source LOS fields, sum, noise, observed '
                     f'(LOS + toward satellite; ▲ inflation ▼ deflation)', y=1.0)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'fig1_examples_K{k}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_distributions(stats, meta, out_dir):
    """Truth-parameter and noise distributions from the statistics draws."""
    prior = meta['prior']
    train = stats['split'] == 'train'
    exist = ~np.isnan(stats['params'][:, :, 0])
    params, extra = stats['params'], stats['extra']
    p_tr, e_tr, x_tr = params[train][exist[train]], extra[train][exist[train]], exist[train]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    ax = axes[0, 0]
    ax.bar(range(prior['k_max'] + 1), np.bincount(stats['k'][train], minlength=prior['k_max'] + 1))
    ax.set_xlabel('Number of sources K'); ax.set_ylabel('Scenes'); ax.set_title('K distribution (train)')

    ax = axes[0, 1]
    bins = np.geomspace(*prior['peak_ratio'], 40)
    for slot in range(prior['k_max']):
        ax.hist(extra[train][:, slot, 0][x_tr[:, slot]], bins=bins, histtype='step', label=f'slot {slot + 1}')
    ax.set_xscale('log')
    ax.set_xlabel('Peak ratio max|LOS_k| / σ_ref'); ax.set_ylabel('Sources')
    ax.set_title('Peak ratio by slot (slots ordered by peak ratio)'); ax.legend()

    ax = axes[0, 2]
    ax.hist(p_tr[:, 2], bins=40)
    ax.set_xlabel('Depth (km)'); ax.set_ylabel('Sources'); ax.set_title('Depth')

    ax = axes[1, 0]
    dV = p_tr[:, 3]
    log_abs_dV = np.log10(np.abs(dV))
    bins = np.linspace(np.floor(log_abs_dV.min()), np.ceil(log_abs_dV.max()), 50)   # data-driven
    ax.hist(np.log10(dV[dV > 0]), bins=bins, histtype='step', label=f'inflation ({(dV > 0).mean():.0%})')
    ax.hist(np.log10(-dV[dV < 0]), bins=bins, histtype='step', label='deflation')
    ax.set_xlabel('log10 |dV| (m³)'); ax.set_ylabel('Sources'); ax.set_title('Volume change'); ax.legend()

    ax = axes[1, 1]
    sc = ax.scatter(p_tr[:, 2], np.log10(np.abs(dV)), c=np.log10(e_tr[:, 0]), s=3, cmap='viridis')
    ax.set_xlabel('Depth (km)'); ax.set_ylabel('log10 |dV| (m³)')
    ax.set_title('dV vs depth (colour = log10 peak ratio)')
    cb = fig.colorbar(sc, ax=ax); cb.set_label('log10 peak ratio')

    ax = axes[1, 2]
    ax.scatter(p_tr[:, 0], p_tr[:, 1], s=2, alpha=0.3)
    ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)'); ax.set_aspect('equal')
    ax.set_xlim(-20, 20); ax.set_ylim(-20, 20); ax.set_title('Source locations (grid = ±20 km)')

    ax = axes[2, 0]
    sep = stats['min_sep_dshallow'][train & (stats['k'] >= 2)]
    ax.hist(np.clip(sep, 0, 20), bins=60)
    ax.axvline(prior['sep_reject_dshallow'], color='r', ls='--', label='reject < 0.8 d (4a)')
    ax.axvline(prior['sep_flag_dshallow'], color='orange', ls='--', label='flag < 1.6 d (8a)')
    frac_flag = np.mean(sep < prior['sep_flag_dshallow'])
    ax.set_xlabel('Min pair 3-D separation / shallower depth (clipped at 20)')
    ax.set_ylabel('Scenes (K≥2)')
    ax.set_title(f'Pascal separation: {frac_flag:.1%} of K≥2 scenes flagged'); ax.legend()

    ax = axes[2, 1]
    sc = ax.scatter(e_tr[:, 0], e_tr[:, 1], c=p_tr[:, 2], s=2, cmap='viridis')
    ax.set_xscale('log')
    ax.set_xlabel('Peak ratio max|LOS_k| / σ_ref'); ax.set_ylabel('Scene-RMS SNR (dB)')
    ax.set_title('Peak ratio vs old scene-RMS SNR (colour = depth)')
    cb = fig.colorbar(sc, ax=ax); cb.set_label('Depth (km)')

    ax = axes[2, 2]
    if meta.get('noise') == 'none':
        # No noise to show: plot per-source peak |LOS| in mm instead (peak ratio x sigma_ref).
        peak_mm = e_tr[:, 0] * meta['sigma_ref_mm']
        ax.hist(peak_mm, bins=np.geomspace(peak_mm.min(), peak_mm.max(), 40))
        ax.set_xscale('log')
        ax.set_xlabel('Per-source peak |LOS| (mm)'); ax.set_ylabel('Sources')
        ax.set_title(f'Source amplitude (noise-free; min {peak_mm.min():.1f} mm)')
    else:
        _plot_noise_rms(ax, stats, meta)

    noise_tag = 'NOISE-FREE ' if meta.get('noise') == 'none' else ''
    fig.suptitle(f'{noise_tag}Stage-1 multi-source synthetic: truth and noise distributions '
                 f'({train.sum()} train / {(~train).sum()} test scenes)')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_noise_rms(ax, stats, meta):
    """Histogram of per-scene noise RMS (mm) for TRAIN vs held-out TEST noise pools."""
    bins = np.linspace(0, np.percentile(stats['noise_rms_mm'], 99.5), 50)
    for split in ['train', 'test']:
        sel = stats['split'] == split
        ax.hist(stats['noise_rms_mm'][sel], bins=bins, histtype='step', density=True,
                label=f"{split} (median {np.median(stats['noise_rms_mm'][sel]):.1f} mm)")
    ax.axvline(meta['sigma_ref_mm'], color='k', ls=':', label=f"σ_ref = {meta['sigma_ref_mm']:.1f} mm")
    ax.set_xlabel('Noise RMS per scene (mm)'); ax.set_ylabel('Density')
    ax.set_title('Noise RMS: train vs held-out test epochs'); ax.legend()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='synthetic/multisource/preview')
    args = parser.parse_args()

    for name in ['preview_scenes.npz', 'stats_draws.npz', 'meta.json']:
        if not os.path.exists(os.path.join(args.data, name)):
            raise FileNotFoundError(os.path.join(args.data, name))

    print(f"[1/3] Loading preview data from {args.data}...")
    with np.load(os.path.join(args.data, 'preview_scenes.npz')) as npz:
        data = dict(npz)
    with np.load(os.path.join(args.data, 'stats_draws.npz')) as npz:
        stats = dict(npz)
    with open(os.path.join(args.data, 'meta.json')) as f:
        meta = json.load(f)
    print(f"  Loaded: obs shape={data['obs_mm'].shape}, dtype={data['obs_mm'].dtype}; "
          f"{len(stats['k'])} stats draws")

    print("[2/3] Plotting per-K example scenes...")
    plot_examples(data, meta, args.data)
    print("[3/3] Plotting truth / noise distributions...")
    plot_distributions(stats, meta, args.data)
    print(f"Done. Output saved to {args.data}/")


if __name__ == '__main__':
    main()
