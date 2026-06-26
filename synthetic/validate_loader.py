#!/usr/bin/env python
# Usage:       python -m synthetic.validate_loader \
#                  --spec synthetic/specs/marapi_mogi_buildup_combined.json --multilook 4
# Description: Validate the MULTILOOK / LOADER stage — what PILA actually ingests after
#              downsampling. Runs the production loader (load_insar_mintpy) on the saved
#              synthetic cube and ALSO multilooks each reconstructed component (deformation,
#              trop, turbulent, orbit, white, combined) with the same mask-aware block
#              average, to quantify and visualise:
#                * full-resolution field  ->  multilooked coherent-cell field
#                * how multilook suppresses each noise component (white >> correlated APS)
#                * SNR at full resolution vs post-multilook (the value PILA sees)
#                * the standardized point set PILA trains on (N cells, global z-score)
# Scientific notes:
#   * Mask-aware block average over `multilook` x `multilook` px; white noise variance
#     drops ~1/M^2, correlated APS far less -> SNR generally IMPROVES after multilook.
#   * SNR convention matches synthetic/noise.py: 20*log10(signal_RMS/noise_RMS).
# Date:        2026-06-22

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.preprocessing.insar_mintpy import load_insar_mintpy, mask_aware_multilook
from synthetic.validate_generation import reconstruct, COMPONENT_LABELS
from synthetic import noise as noisemod


def _rms(field, mask):
    v = field[mask] if mask is not None else field.ravel()
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0


def multilook_layers(layers, fine_mask, factor, coh_valid_frac=0.5):
    """Mask-aware block-average every reconstructed layer -> coarse cubes + coarse mask."""
    if fine_mask is None:
        fine_mask = np.ones(next(iter(layers.values())).shape[1:], dtype=bool)
    coarse = {}
    cmask = None
    for key, cube in layers.items():
        cc, cm = mask_aware_multilook(cube.astype(np.float64), fine_mask, factor, coh_valid_frac)
        coarse[key] = cc
        cmask = cm
    return coarse, cmask


def snr_table(layers_fine, fine_mask, layers_coarse, coarse_mask, peak):
    """Per-component RMS (mm) full-res vs multilooked, + signal/noise SNR both ways."""
    rows = []
    comp_keys = ['deformation'] + [k for k, _ in COMPONENT_LABELS] + ['combined']
    for key in comp_keys:
        # noise components: RMS averaged over active (non-zero-signal) epochs; signal: peak
        if key == 'deformation':
            f = _rms(layers_fine[key][peak] * 1000, fine_mask)
            c = _rms(layers_coarse[key][peak] * 1000, coarse_mask)
        else:
            n = layers_fine[key].shape[0]
            f = np.mean([_rms(layers_fine[key][i] * 1000, fine_mask) for i in range(n)])
            c = np.mean([_rms(layers_coarse[key][i] * 1000, coarse_mask) for i in range(n)])
        rows.append((key, f, c, (c / f if f > 0 else float('nan'))))

    sig_f = _rms(layers_fine['deformation'][peak] * 1000, fine_mask)
    sig_c = _rms(layers_coarse['deformation'][peak] * 1000, coarse_mask)
    n_ep = layers_fine['combined'].shape[0]
    noise_f = np.mean([_rms(layers_fine['combined'][i] * 1000, fine_mask) for i in range(n_ep)])
    noise_c = np.mean([_rms(layers_coarse['combined'][i] * 1000, coarse_mask) for i in range(n_ep)])
    snr_f = 20 * np.log10(sig_f / noise_f) if noise_f > 0 else float('nan')
    snr_c = 20 * np.log10(sig_c / noise_c) if noise_c > 0 else float('nan')
    return rows, (snr_f, snr_c, sig_f, sig_c, noise_f, noise_c)


def plot_fullres_vs_multilook(out_dir, name, total_fine, total_coarse, fine_mask,
                              coarse_mask, peak, date):
    """Peak-epoch total field: full resolution vs multilooked coherent cells (mm)."""
    f = np.where(fine_mask, total_fine[peak] * 1000, np.nan) if fine_mask is not None else total_fine[peak] * 1000
    c = np.where(coarse_mask, total_coarse[peak] * 1000, np.nan)
    vmax = np.nanmax(np.abs(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, d, t in [(axes[0], f, f'Full resolution {f.shape[0]}x{f.shape[1]}'),
                     (axes[1], c, f'Multilooked {c.shape[0]}x{c.shape[1]} (PILA input grid)')]:
        im = ax.imshow(d, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label('LOS (mm), + toward sat')
    fig.suptitle(f'{name}: total field, epoch {peak} ({date}) — multilook downsampling', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_fullres_vs_multilook.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{name}_fullres_vs_multilook.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_suppression(out_dir, name, rows):
    """Bar chart: per-component RMS full-res vs multilooked, with suppression ratio."""
    keys = [r[0] for r in rows]; fine = [r[1] for r in rows]; coarse = [r[2] for r in rows]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, fine, w, label='full resolution', color='C0')
    ax.bar(x + w / 2, coarse, w, label='multilooked', color='C1')
    for i, r in enumerate(rows):
        if r[1] > 0:
            ax.text(i, max(r[1], r[2]) * 1.02, f'×{r[3]:.2f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20)
    ax.set_ylabel('RMS (mm)'); ax.set_title(f'{name}: multilook RMS suppression by component '
                                            f'(×ratio = multilook/full-res)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_multilook_suppression.png'), dpi=150)
    fig.savefig(os.path.join(out_dir, f'{name}_multilook_suppression.pdf'))
    plt.close(fig)


def plot_pila_input(out_dir, name, data, peak):
    """Scatter of the standardized coherent-cell LOS PILA actually trains on (peak epoch)."""
    los_std = (data.los_points_mm[peak] - data.x_mean_global) / data.x_scale_global
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(data.xE_pts, data.yN_pts, c=data.los_points_mm[peak],
                          cmap='RdBu_r', s=14)
    axes[0].set_title(f'Multilooked LOS at {data.n_points} coherent cells (mm)')
    cb0 = fig.colorbar(sc0, ax=axes[0]); cb0.set_label('LOS (mm)')
    sc1 = axes[1].scatter(data.xE_pts, data.yN_pts, c=los_std, cmap='RdBu_r', s=14)
    axes[1].set_title('Standardized (global z-score) — actual PILA input')
    cb1 = fig.colorbar(sc1, ax=axes[1]); cb1.set_label('z-score')
    for ax in axes:
        ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)'); ax.set_aspect('equal')
    fig.suptitle(f'{name}: what PILA ingests at epoch {peak} '
                 f'(global mean {data.x_mean_global:.2f} mm, scale {data.x_scale_global:.2f} mm)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_pila_input_points.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{name}_pila_input_points.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Validate the multilook/loader stage of the synthetic pipeline.")
    ap.add_argument('--spec', required=True)
    ap.add_argument('--multilook', type=int, default=4, help='Block factor (Marapi grid uses 4).')
    ap.add_argument('--coh-valid-frac', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    name = spec['name']
    base = spec['base_scene']
    out_dir = args.out or os.path.join('synthetic', 'validation', name, 'loader')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/5] Reconstructing full-resolution layers")
    layers, dates, fine_mask, pnames, params, sumcheck, n_epoch = reconstruct(spec)
    peak = int(np.argmax([_rms(layers['deformation'][i], fine_mask) for i in range(n_epoch)]))

    print(f"[2/5] Multilooking each layer (factor={args.multilook})")
    coarse, coarse_mask = multilook_layers(layers, fine_mask, args.multilook, args.coh_valid_frac)
    print(f"  full-res {layers['total'].shape[1:]}  ->  coarse {coarse['total'].shape[1:]}; "
          f"coherent cells = {int(coarse_mask.sum())}")

    print(f"[3/5] SNR / RMS table (full-res vs multilook)")
    rows, (snr_f, snr_c, sig_f, sig_c, noise_f, noise_c) = snr_table(
        layers, fine_mask, coarse, coarse_mask, peak)
    print(f"  {'component':<12}{'fullres RMS':>14}{'multilook RMS':>16}{'ratio':>9}")
    for k, fr, cr, ra in rows:
        print(f"  {k:<12}{fr:>13.2f} {cr:>15.2f} {ra:>8.3f}")
    print(f"\n  SIGNAL (peak) RMS  full {sig_f:.1f} mm -> multilook {sig_c:.1f} mm")
    print(f"  NOISE  (mean) RMS  full {noise_f:.1f} mm -> multilook {noise_c:.1f} mm")
    print(f"  SNR  full-res = {snr_f:.2f} dB   post-multilook = {snr_c:.2f} dB   "
          f"(gain {snr_c - snr_f:+.2f} dB)")

    print(f"[4/5] Loading the actual PILA point set (load_insar_mintpy)")
    data = load_insar_mintpy(spec_h5(spec), base['geometry'], base.get('mask'),
                             base['lat0'], base['lon0'], multilook=args.multilook,
                             coh_valid_frac=args.coh_valid_frac, verbose=False)
    print(f"  N coherent cells = {data.n_points}; global mean {data.x_mean_global:.3f} mm, "
          f"scale {data.x_scale_global:.3f} mm")

    print(f"[5/5] Plots")
    plot_fullres_vs_multilook(out_dir, name, layers['total'], coarse['total'],
                              fine_mask, coarse_mask, peak, dates[peak])
    plot_suppression(out_dir, name, rows)
    plot_pila_input(out_dir, name, data, peak)

    metrics = {
        'name': name, 'multilook': args.multilook, 'n_coherent_cells': int(coarse_mask.sum()),
        'peak_epoch': peak, 'snr_db_fullres': snr_f, 'snr_db_multilook': snr_c,
        'snr_gain_db': snr_c - snr_f,
        'signal_rms_mm_fullres': sig_f, 'signal_rms_mm_multilook': sig_c,
        'noise_rms_mm_fullres': noise_f, 'noise_rms_mm_multilook': noise_c,
        'component_rms_mm': {k: {'fullres': fr, 'multilook': cr, 'ratio': ra}
                             for k, fr, cr, ra in rows},
        'x_mean_global_mm': float(data.x_mean_global), 'x_scale_global_mm': float(data.x_scale_global),
    }
    with open(os.path.join(out_dir, f'{name}_loader_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Done. Loader validation in {out_dir}")


def spec_h5(spec):
    """Path to the saved synthetic cube for a spec."""
    name = spec['name']
    return os.path.join('synthetic', 'cubes', name, f'{name}.h5')


if __name__ == '__main__':
    main()
