#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_stage1_noisy_data \
#                  --out synthetic/multisource/stage1_ta69_noisy_pk0.5-10 \
#                  --geometry synthetic/marapi_ta69_grid/geo_geometryRadar.h5 \
#                  --peak-ratio-min 0.5 --peak-ratio-max 10 --ramp-sigma 0.3
# Description: Build a NOISY Stage-1 data directory on a given idealized grid (default: the
#              Marapi TA69-matched ascending grid) using the batched torch generator:
#              (1) sigma_ref = mean RMS of referenced noise-only maps (APS + ramp, TRAIN pool),
#              (2) scale-only input scaler from TRAIN scenes, (3) fixed val (TRAIN noise pool)
#              and test (HELD-OUT noise pool) .npz, (4) meta.json, (5) figures of what the
#              network sees and of the noise statistics (train vs held-out).
# Scientific notes:
#   * Marapi = W. Sumatra (-0.38, 100.47); the grid geometry is taken from real S1 TA69.
#   * Referencing: whole observed map minus its value at a random reference pixel; random
#     planar ramp (ramp_sigma mm/km per axis) added before referencing. Labels unchanged.
#   * Peak ratio = max|LOS_k| of the source's own clean field / sigma_ref (log-uniform).
# Date:        2026-10-05

import argparse
import copy
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from synthetic.multisource import generate_multisource as gen
from synthetic.multisource.dataset import FixedSceneDataset, load_stage1_meta
from synthetic.multisource.make_stage1_data import plot_network_input
from synthetic.multisource.torch_generator import TorchSceneGenerator

CHUNK = 250


def draw_noise_only(tgen, n_scenes, split, seed):
    """Referenced APS + ramp maps with no sources, [n, H, W] mm."""
    out = []
    for start in range(0, n_scenes, CHUNK):
        g = torch.Generator().manual_seed(seed + start)
        n = min(CHUNK, n_scenes - start)
        out.append(tgen.sample_ramp_and_reference(tgen.sample_noise(n, g, split), g))
    return torch.cat(out).numpy()


def draw_set(tgen, n_scenes, split, seed):
    """Fixed scene set in FixedSceneDataset format (params NaN in empty slots)."""
    parts = {key: [] for key in ['obs_mm', 'params', 'extra', 'exist', 'k']}
    for start in range(0, n_scenes, CHUNK):
        batch = tgen.sample(min(CHUNK, n_scenes - start), seed=seed + start, split=split)
        params = batch['params'].clone()
        params[~batch['exist']] = float('nan')
        for key, val in [('obs_mm', batch['obs_mm']), ('params', params), ('extra', batch['extra']),
                         ('exist', batch['exist']), ('k', batch['k'])]:
            parts[key].append(val.numpy())
    return {key: np.concatenate(val) for key, val in parts.items()}


def plot_noise_stats(noise_train, noise_test, meta, out_dir):
    """Noise-only example maps and per-scene RMS distributions (train vs held-out test)."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    extent = [-24.3, 23.9, -20.6, 21.0]
    for col in range(4):
        for row, (arr, name) in enumerate([(noise_train, 'train pool'), (noise_test, 'held-out test pool')]):
            ax = axes[row, col]
            if col < 3:
                vmax = np.percentile(np.abs(arr[col]), 99)
                im = ax.imshow(arr[col], cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent, origin='upper')
                cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label('LOS noise (mm)')
                ax.set_title(f'{name}: example {col + 1} (RMS {np.sqrt(np.mean(arr[col] ** 2)):.1f} mm)')
                ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
            elif row == 0:
                rms_tr = np.sqrt((noise_train ** 2).mean(axis=(1, 2)))
                rms_te = np.sqrt((noise_test ** 2).mean(axis=(1, 2)))
                bins = np.linspace(0, np.percentile(np.r_[rms_tr, rms_te], 99.5), 50)
                ax.hist(rms_tr, bins, histtype='step', density=True, label=f'train (median {np.median(rms_tr):.1f} mm)')
                ax.hist(rms_te, bins, histtype='step', density=True, label=f'test (median {np.median(rms_te):.1f} mm)')
                ax.axvline(meta['sigma_ref_mm'], color='k', ls=':', label=f"σ_ref {meta['sigma_ref_mm']:.1f} mm")
                ax.set_xlabel('Per-scene noise RMS (mm)'); ax.set_ylabel('Density'); ax.legend(fontsize=8)
                ax.set_title('Noise RMS (APS + ramp, referenced)')
            else:
                peak = np.abs(noise_train).reshape(len(noise_train), -1).max(1) / meta['sigma_ref_mm']
                ax.hist(peak, bins=50)
                ax.set_xlabel('Noise-only peak |LOS| / σ_ref'); ax.set_ylabel('Scenes')
                ax.set_title(f'How "tall" pure noise gets (median {np.median(peak):.1f})')
    fig.suptitle(f'Noise-only maps on the TA69-matched grid: APS epoch-pair differences + ramp '
                 f'(σ {meta["noise_cfg"]["ramp_sigma_mm_per_km"]} mm/km) − reference pixel', y=1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig5_noise_only.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--geometry', default='synthetic/marapi_ta69_grid/geo_geometryRadar.h5')
    parser.add_argument('--peak-ratio-min', type=float, default=0.5)
    parser.add_argument('--peak-ratio-max', type=float, default=10.0)
    parser.add_argument('--ramp-sigma', type=float, default=0.3, help='ramp gradient sigma, mm/km')
    parser.add_argument('--n-val', type=int, default=2000)
    parser.add_argument('--n-test', type=int, default=2000)
    parser.add_argument('--n-scaler', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    for fname in ['val.npz', 'test.npz', 'meta.json']:
        if os.path.exists(os.path.join(args.out, fname)):
            raise FileExistsError(f"{os.path.join(args.out, fname)} exists; choose a new --out")
    os.makedirs(args.out, exist_ok=True)

    print("[1/6] Building meta skeleton and noise pools...")
    prior = copy.deepcopy(gen.PRIOR)
    prior['peak_ratio'] = (args.peak_ratio_min, args.peak_ratio_max)
    pool = gen.NoisePool(prior['n_holdout_epochs'])
    meta = {'prior': prior, 'noise': 'marapi',
            'noise_cfg': {'components': list(gen.NOISE_COMPONENTS), 'reference': 'ref_pixel',
                          'ramp_sigma_mm_per_km': args.ramp_sigma},
            'grid': {'geometry_h5': args.geometry, 'lat0_deg': gen.LAT0_DEG, 'lon0_deg': gen.LON0_DEG},
            'noise_epochs': {s: e.tolist() for s, e in pool.epochs.items()},
            'sigma_ref_mm': 1.0, 'input_scaler': {'mean_mm': 0.0, 'std_mm': 1.0}, 'seed': args.seed,
            'params_columns': ['xcen_km', 'ycen_km', 'depth_km', 'dV_m3'],
            'extra_columns': ['peak_ratio', 'unused'],
            'slot_order': 'descending per-source peak ratio',
            'reference_frame': 'obs = clean + APS + ramp, minus value at a random reference pixel'}
    del pool

    print("[2/6] sigma_ref from 2000 referenced noise-only TRAIN maps...")
    t0 = time.time()
    tgen = TorchSceneGenerator(meta)
    noise_train = draw_noise_only(tgen, 2000, 'train', seed=args.seed * 1000 + 1)
    noise_test = draw_noise_only(tgen, 500, 'test', seed=args.seed * 1000 + 2)
    meta['sigma_ref_mm'] = float(np.mean(np.sqrt((noise_train ** 2).mean(axis=(1, 2)))))
    print(f"  sigma_ref = {meta['sigma_ref_mm']:.2f} mm ({time.time() - t0:.1f}s)")

    print(f"[3/6] Scale-only input scaler from {args.n_scaler} TRAIN scenes...")
    tgen = TorchSceneGenerator(meta)                          # rebuild with real sigma_ref
    sq_sum, n_px = 0.0, 0
    for start in range(0, args.n_scaler, CHUNK):
        obs = tgen.sample(min(CHUNK, args.n_scaler - start), seed=10**8 + start)['obs_mm']
        sq_sum += float((obs.double() ** 2).sum()); n_px += obs.numel()
    meta['input_scaler'] = {'mean_mm': 0.0, 'std_mm': float(np.sqrt(sq_sum / n_px))}
    print(f"  RMS scale = {meta['input_scaler']['std_mm']:.2f} mm")

    print(f"[4/6] Fixed val ({args.n_val}, train pool) and test ({args.n_test}, held-out pool)...")
    tgen = TorchSceneGenerator(meta)                          # final scaler
    for name, n, split, seed in [('val', args.n_val, 'train', 2 * 10**8), ('test', args.n_test, 'test', 3 * 10**8)]:
        arrays = draw_set(tgen, n, split, seed)
        np.savez_compressed(os.path.join(args.out, f'{name}.npz'), **arrays)
        print(f"  Saved {name}.npz: obs shape={arrays['obs_mm'].shape}, dtype={arrays['obs_mm'].dtype}")

    print("[5/6] Writing meta.json...")
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    meta = load_stage1_meta(args.out)                         # exercise the JSON round-trip

    print("[6/6] Figures...")
    tgen = TorchSceneGenerator(meta)
    batch = tgen.sample(256, seed=4 * 10**8)
    train_items = [{key: batch[key][i] for key in ['x', 'params', 'exist', 'k']} for i in range(256)]
    val_ds = FixedSceneDataset(os.path.join(args.out, 'val.npz'), meta)
    test_ds = FixedSceneDataset(os.path.join(args.out, 'test.npz'), meta)
    extent = (float(tgen.xE_m.min()) / 1e3, float(tgen.xE_m.max()) / 1e3,
              float(tgen.yN_m.min()) / 1e3, float(tgen.yN_m.max()) / 1e3)
    plot_network_input(train_items, val_ds, test_ds, meta, args.out, extent=extent)
    plot_noise_stats(noise_train, noise_test, meta, args.out)
    print(f"Done. Output saved to {args.out}/")


if __name__ == '__main__':
    main()
