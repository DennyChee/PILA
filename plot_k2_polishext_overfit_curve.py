#!/usr/bin/env python
# Usage:       python plot_k2_polishext_overfit_curve.py [--fig-dir figures/multisource_k2_polishext]
# Description: Late-rescue test, overfitting check. For every dumped checkpoint of the polish extension (epoch 400, every
#              20 to 600 [k2ext], every 50 to 1400 [k2ext2]) computes the RMS of the physics prediction (sum of the two
#              Mogi sources) against (a) the noisy album the run trained on and (b) the noise-free twin
#              albums_nonoise/E1_S3_K2_separated_nonoise.npz (truth, used only for this diagnostic). Falling (a) with rising
#              (b) = the run is fitting noise; the minimum of (b) is where it starts. Also plots the depth / dV errors from
#              extension_scores.csv. Writes polishext_signal_vs_noise_fit.csv + .png. PNG only.
# Date:        2026-10-26

import argparse
import glob
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT = 'saved/multisource_k2_polishext'
NOISY_ALBUM = 'synthetic/multisource/pila_stacks_v2/albums_snr/E1_S3_K2_separated_n025.npz'
CLEAN_ALBUM = 'synthetic/multisource/pila_stacks_v2/albums_nonoise/E1_S3_K2_separated_nonoise.npz'
LATE_SEEDS = [1, 3, 5, 8, 9, 13]          # separated at epoch >= 266 in the original run
CONTROL_SEEDS = [4, 7, 15]                # separated at epoch 131-207


def checkpoint_fits(seed):
    """All dumped fits of one seed's extension, as {epoch: path} (epoch 400 init, k2ext 420-600, k2ext2 650-1400)."""
    fits = {400: os.path.join(PROJECT_ROOT, EXT_ROOT, f'k2ext_slot_resid_seed{seed}', 'init', 'fit_K2_epoch400.npz')}
    for tag in (f'k2ext_slot_resid_seed{seed}', f'k2ext2_slot_resid_seed{seed}'):
        for path in glob.glob(os.path.join(PROJECT_ROOT, EXT_ROOT, tag, 'train', '*', '*', 'models', 'fit_K2_epoch*.npz')):
            fits.setdefault(int(path.split('epoch')[-1].split('.')[0]), path)   # epoch 600 is in both; keep k2ext's
    for epoch, path in fits.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    return dict(sorted(fits.items()))


def load_scores(seed):
    """Depth / dV errors per epoch from both extension_scores.csv files (epoch 600 duplicate dropped)."""
    frames = [pd.read_csv(os.path.join(PROJECT_ROOT, EXT_ROOT, tag.format(seed=seed), 'extension_scores.csv'))
              for tag in ('k2ext_slot_resid_seed{seed}', 'k2ext2_slot_resid_seed{seed}')]
    return pd.concat(frames).drop_duplicates('epoch').set_index('epoch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig-dir', default='figures/multisource_k2_polishext')
    args = parser.parse_args()
    fig_dir = os.path.join(PROJECT_ROOT, args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    out_csv = os.path.join(fig_dir, 'polishext_signal_vs_noise_fit.csv')
    out_png = os.path.join(fig_dir, 'polishext_signal_vs_noise_fit.png')
    for path in (out_csv, out_png):
        if os.path.exists(path):
            print(f"  WARNING: overwriting {path}")

    # --- Load the noisy album and its noise-free twin (same grid, same dates, static signal) ---
    print("[1/3] Loading noisy album and noise-free twin...")
    for path in (NOISY_ALBUM, CLEAN_ALBUM):
        if not os.path.exists(os.path.join(PROJECT_ROOT, path)):
            raise FileNotFoundError(path)
    with np.load(os.path.join(PROJECT_ROOT, NOISY_ALBUM)) as album:
        noisy_mm = album['los_asc_m'].astype(np.float64) * 1e3
    with np.load(os.path.join(PROJECT_ROOT, CLEAN_ALBUM)) as album:
        clean_mm = album['los_asc_m'].astype(np.float64) * 1e3
    print(f"  Loaded: noisy shape={noisy_mm.shape}, clean shape={clean_mm.shape}, dtype={noisy_mm.dtype}")
    if noisy_mm.shape != clean_mm.shape:
        raise ValueError('noisy and noise-free albums differ in shape')

    # --- Score every checkpoint ---
    print("[2/3] Scoring every checkpoint against noisy data and noise-free signal...")
    t0 = time.time()
    rows = []
    for seed in LATE_SEEDS + CONTROL_SEEDS:
        scores = load_scores(seed)
        for epoch, path in checkpoint_fits(seed).items():
            with np.load(path) as fit:
                mask = fit['mask']
                data_mm = fit['data_mm'][:, mask].astype(np.float64)
                physics_mm = fit['xP_src_mm'].sum(axis=1)[:, mask].astype(np.float64)
            # The fit's data may be re-referenced; the per-pixel album-vs-fit offset carries the noise-free signal into
            # the same frame (noisy - data is a per-epoch constant, so clean - offset = clean in the fit's reference).
            offset_mm = noisy_mm[:, mask] - data_mm
            if np.ptp(offset_mm, axis=1).max() > 1e-3:
                raise ValueError(f'{path}: fit data is not the album up to a per-epoch constant')
            signal_mm = clean_mm[:, mask] - offset_mm
            rows.append(dict(seed=seed, group='late' if seed in LATE_SEEDS else 'control', epoch=epoch,
                             rms_physics_vs_data_mm=np.sqrt(np.mean((physics_mm - data_mm) ** 2)),
                             rms_physics_vs_signal_mm=np.sqrt(np.mean((physics_mm - signal_mm) ** 2)),
                             noise_rms_mm=np.sqrt(np.mean((data_mm - signal_mm) ** 2)),
                             depth_pct_max=scores.loc[epoch, 'depth_pct_max'],
                             dV_pct_max=scores.loc[epoch, 'dV_pct_max'], misfit=scores.loc[epoch, 'misfit']))
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False, float_format='%.5g')
    print(f"  {len(table)} checkpoints scored in {time.time() - t0:.1f}s")

    # --- Plot: misfit to data, error vs signal (minimum marked), depth / dV error ---
    print("[3/3] Plotting...")
    panels = [('rms_physics_vs_data_mm', 'Physics vs noisy data (what training sees)', 'RMS (mm)'),
              ('rms_physics_vs_signal_mm', 'Physics vs noise-free signal (truth)', 'RMS (mm)'),
              ('depth_pct_max', 'Depth error, worst source', 'Depth error (%)'),
              ('dV_pct_max', 'dV error, worst source', r'$\Delta V$ error (%)')]
    colours = plt.cm.tab10(np.arange(len(LATE_SEEDS + CONTROL_SEEDS)))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    for colour, seed in zip(colours, LATE_SEEDS + CONTROL_SEEDS):
        seed_rows = table[table.seed == seed]
        style = '-' if seed in LATE_SEEDS else '--'
        label = f'seed {seed} ({"late" if seed in LATE_SEEDS else "control"})'
        for axis, (column, _, _) in zip(axes, panels):
            axis.plot(seed_rows.epoch, seed_rows[column], style, color=colour, label=label, lw=1.5)
        # Where this run stops learning signal: lowest error vs the noise-free signal
        best = seed_rows.loc[seed_rows.rms_physics_vs_signal_mm.idxmin()]
        axes[1].plot(best.epoch, best.rms_physics_vs_signal_mm, 'o', color=colour, ms=6)
    for axis, (_, title, ylabel) in zip(axes, panels):
        axis.set_title(title)
        axis.set_xlabel('Training epoch')
        axis.set_ylabel(ylabel)
        axis.axvline(600, color='0.6', lw=0.8, ls=':')
    axes[0].legend(fontsize=7, ncol=2)
    noise_rms_mm = table.noise_rms_mm.iloc[0]
    fig.suptitle('K=2 Mogi, SNR 12, E1 album: polish extension 400 -> 1400 epochs (constant lr 1.5e-5, penalties off)\n'
                 f'dots = minimum error vs noise-free signal; noise RMS {noise_rms_mm:.2f} mm; '
                 'dotted line = k2ext / k2ext2 boundary', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # --- Console summary: where each seed's signal error bottoms out ---
    for seed in LATE_SEEDS + CONTROL_SEEDS:
        seed_rows = table[table.seed == seed].set_index('epoch')
        best_epoch = seed_rows.rms_physics_vs_signal_mm.idxmin()
        print(f"  seed {seed:2d}: signal error min {seed_rows.loc[best_epoch, 'rms_physics_vs_signal_mm']:.2f} mm "
              f"at epoch {best_epoch}; ep400 {seed_rows.loc[400, 'rms_physics_vs_signal_mm']:.2f}, "
              f"ep1400 {seed_rows.loc[1400, 'rms_physics_vs_signal_mm']:.2f}")
    print(f"Done. Output saved to {fig_dir}/")


if __name__ == '__main__':
    main()
