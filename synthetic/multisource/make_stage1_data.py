#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_stage1_data \
#                  --out synthetic/multisource/stage1_clean_pk2-10 --noise none \
#                  --peak-ratio-min 2 --peak-ratio-max 10 --n-val 2000 --n-test 2000
# Description: Build a Stage-1 data directory for supervised multi-source PILA: (1) the
#              global SCALE-ONLY input scaler (RMS of observed LOS, mm; mean fixed 0) from TRAIN
#              scenes, (2) FIXED validation (train noise pool) and test (held-out noise
#              pool) scenes as .npz, (3) meta.json consumed by dataset.py, and (4) figures
#              showing exactly what the network receives (standardized input maps, value
#              histograms, target tables) plus a DataLoader speed / shape check.
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
from torch.utils.data import DataLoader

from synthetic.multisource import generate_multisource as gen
from synthetic.multisource.dataset import OnTheFlySceneDataset, FixedSceneDataset, load_stage1_meta


def draw_fixed_set(rng, grid, noise_pool, sigma_ref_mm, prior, noise_mode, split, n_scenes):
    """Draw n_scenes with K ~ U{0..K_max}; keep only what is saved (no per-source fields)."""
    keep = ['obs_mm', 'params', 'extra', 'exist', 'k', 'min_sep_dshallow', 'sep_flagged']
    rows = {key: [] for key in keep}
    for scene_idx in range(n_scenes):
        k = int(rng.integers(0, prior['k_max'] + 1))
        scene = gen.make_scene(rng, grid, noise_pool, sigma_ref_mm, k, split, noise_mode, prior)
        for key in keep:                       # drop fields/clean/noise maps immediately
            rows[key].append(scene[key])
        if (scene_idx + 1) % 500 == 0:
            print(f"    {split}: {scene_idx + 1}/{n_scenes}")
    out = {key: np.stack(rows[key]) if key in ('obs_mm', 'params', 'extra', 'exist')
           else np.array(rows[key]) for key in keep}
    out['obs_mm'] = out['obs_mm'].astype(np.float32)
    return out


def plot_network_input(train_items, val_ds, test_ds, meta, out_dir, extent=(-20, 20, -20, 20)):
    """Figure 3: standardized input maps per K, as the network sees them, with target text."""
    k_max = meta['prior']['k_max']
    n_cols = 4
    fig, axes = plt.subplots(k_max + 1, n_cols, figsize=(4.0 * n_cols, 3.6 * (k_max + 1)), squeeze=False)
    extent = list(extent)                    # map extent in km (E_min, E_max, N_min, N_max)
    for k in range(k_max + 1):
        items_k = [it for it in train_items if int(it['k']) == k][:n_cols]
        for col, item in enumerate(items_k):
            ax = axes[k, col]
            x_std = item['x'][0].numpy()
            vmax = max(np.abs(x_std).max(), 1e-6)
            im = ax.imshow(x_std, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent, origin='upper')
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label('Standardized LOS (unitless)')
            lines = [f'K={k}  x: shape {tuple(item["x"].shape)}']
            for slot in range(k_max):
                if item['exist'][slot]:
                    x_km, y_km, d_km, dV = item['params'][slot].tolist()
                    lines.append(f'S{slot + 1}: ({x_km:+.1f},{y_km:+.1f}) km, d={d_km:.1f} km, '
                                 f'dV={dV:.1e} m³')
                    ax.plot(x_km, y_km, '^' if dV > 0 else 'v', mfc='none', mec='k', ms=8)
                else:
                    lines.append(f'S{slot + 1}: empty (exist=False)')
            ax.set_title('\n'.join(lines), fontsize=7, loc='left')
            ax.set_xlabel('East (km)', fontsize=8); ax.set_ylabel('North (km)', fontsize=8)
            ax.tick_params(labelsize=7)
    scaler = meta['input_scaler']
    fig.suptitle(f'What the network receives: x = (obs_mm − {scaler["mean_mm"]:.2f}) / {scaler["std_mm"]:.2f} '
                 f'(noise={meta["noise"]}, peak/σ {meta["prior"]["peak_ratio"][0]}–{meta["prior"]["peak_ratio"][1]}); '
                 'colour scale per panel; ▲ inflation ▼ deflation', y=1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_network_input.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure 4: standardized value distributions and split balance.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    train_vals = np.concatenate([it['x'].numpy().ravel() for it in train_items])
    val_vals = np.concatenate([val_ds[i]['x'].numpy().ravel() for i in range(min(300, len(val_ds)))])
    test_vals = np.concatenate([test_ds[i]['x'].numpy().ravel() for i in range(min(300, len(test_ds)))])
    lim = np.percentile(np.abs(train_vals), 99.9)
    bins = np.linspace(-lim, lim, 120)
    for vals, name in [(train_vals, 'train (on-the-fly)'), (val_vals, 'val (fixed)'), (test_vals, 'test (fixed)')]:
        axes[0].hist(vals, bins=bins, histtype='step', density=True,
                     label=f'{name}: mean {vals.mean():+.3f}, std {vals.std():.3f}')
    axes[0].set_yscale('log'); axes[0].set_xlabel('Standardized pixel value (unitless)')
    axes[0].set_ylabel('Density (log)'); axes[0].set_title('Input pixel distribution'); axes[0].legend(fontsize=8)

    for offset, (ds_arr, name) in zip([-0.2, 0.2], [(val_ds.arrays, 'val'), (test_ds.arrays, 'test')]):
        axes[1].bar(np.arange(k_max + 1) + offset, np.bincount(ds_arr['k'], minlength=k_max + 1),
                    width=0.4, label=name)
    axes[1].set_xlabel('Number of sources K'); axes[1].set_ylabel('Scenes'); axes[1].set_title('K balance')
    axes[1].legend()

    for ds_arr, name in [(val_ds.arrays, 'val'), (test_ds.arrays, 'test')]:
        ratios = ds_arr['extra'][:, :, 0][ds_arr['exist']]
        axes[2].hist(ratios, bins=np.geomspace(*meta['prior']['peak_ratio'], 30), histtype='step', label=name)
    axes[2].set_xscale('log'); axes[2].set_xlabel('Peak ratio max|LOS_k| / σ_ref')
    axes[2].set_ylabel('Sources'); axes[2].set_title('Source amplitude balance'); axes[2].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_input_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--noise', choices=['marapi', 'none'], default='none')
    parser.add_argument('--peak-ratio-min', type=float, default=2.0)
    parser.add_argument('--peak-ratio-max', type=float, default=10.0)
    parser.add_argument('--n-val', type=int, default=2000)
    parser.add_argument('--n-test', type=int, default=2000)
    parser.add_argument('--n-scaler', type=int, default=2000, help='train scenes for the input scaler')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    for fname in ['val.npz', 'test.npz', 'meta.json']:
        if os.path.exists(os.path.join(args.out, fname)):
            raise FileExistsError(f"{os.path.join(args.out, fname)} exists; choose a new --out")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prior = copy.deepcopy(gen.PRIOR)
    prior['peak_ratio'] = (args.peak_ratio_min, args.peak_ratio_max)

    print("[1/6] Loading grid, noise pools and sigma_ref...")
    grid = gen.load_grid()
    noise_pool = gen.NoisePool(prior['n_holdout_epochs'])
    sigma_ref_mm = gen.estimate_sigma_ref(noise_pool, rng)
    print(f"  sigma_ref = {sigma_ref_mm:.2f} mm (sets source amplitudes even when noise='none')")

    print(f"[2/6] Global input scaler from {args.n_scaler} TRAIN scenes...")
    t0 = time.time()
    scaler_set = draw_fixed_set(rng, grid, noise_pool, sigma_ref_mm, prior, args.noise, 'train', args.n_scaler)
    # Scale-only (mean fixed at 0): zero deformation must map to zero input, otherwise empty
    # scenes carry an arbitrary offset set by the inflation/deflation mix. Scale = pixel RMS.
    input_scaler = {'mean_mm': 0.0, 'std_mm': float(np.sqrt(np.mean(scaler_set['obs_mm'] ** 2)))}
    del scaler_set                                            # free ~130 MB before val/test
    print(f"  scale-only: mean=0 (fixed), RMS scale={input_scaler['std_mm']:.3f} mm "
          f"({time.time() - t0:.1f}s)")

    print(f"[3/6] Fixed validation set ({args.n_val}, train noise pool)...")
    val = draw_fixed_set(rng, grid, noise_pool, sigma_ref_mm, prior, args.noise, 'train', args.n_val)
    np.savez_compressed(os.path.join(args.out, 'val.npz'), **val)
    print(f"  Saved val.npz: obs shape={val['obs_mm'].shape}, dtype={val['obs_mm'].dtype}")

    print(f"[4/6] Fixed test set ({args.n_test}, HELD-OUT noise pool)...")
    test = draw_fixed_set(rng, grid, noise_pool, sigma_ref_mm, prior, args.noise, 'test', args.n_test)
    np.savez_compressed(os.path.join(args.out, 'test.npz'), **test)
    print(f"  Saved test.npz: obs shape={test['obs_mm'].shape}, dtype={test['obs_mm'].dtype}")

    meta = {'prior': prior, 'noise': args.noise, 'sigma_ref_mm': sigma_ref_mm,
            'input_scaler': input_scaler, 'seed': args.seed,
            'noise_epochs': {key: val_list.tolist() for key, val_list in noise_pool.epochs.items()},
            'params_columns': ['xcen_km', 'ycen_km', 'depth_km', 'dV_m3'],
            'extra_columns': ['peak_ratio', 'snr_db_rms'],
            'slot_order': 'descending per-source peak ratio',
            'reference_frame': ('obs = sum of source fields (no spatial re-reference)'
                                if args.noise == 'none' else 'noise median removed; obs not re-referenced')}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=float)

    print("[5/6] DataLoader check (on-the-fly train set, meta re-read from JSON)...")
    meta = load_stage1_meta(args.out)                         # exercise the JSON round-trip
    train_ds = OnTheFlySceneDataset(meta, scenes_per_epoch=256)
    loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    t0 = time.time()
    train_items = []
    for batch in loader:
        for i in range(batch['x'].shape[0]):
            train_items.append({key: batch[key][i] for key in batch})
    elapsed_s = time.time() - t0
    print(f"  batch x: {tuple(batch['x'].shape)} {batch['x'].dtype}; params {tuple(batch['params'].shape)}; "
          f"exist {tuple(batch['exist'].shape)}; k {tuple(batch['k'].shape)}")
    print(f"  {len(train_items)} scenes in {elapsed_s:.1f}s -> {len(train_items) / elapsed_s:.0f} scenes/s "
          "(single process)")

    print("[6/6] Plotting network-input figures...")
    val_ds = FixedSceneDataset(os.path.join(args.out, 'val.npz'), meta)
    test_ds = FixedSceneDataset(os.path.join(args.out, 'test.npz'), meta)
    plot_network_input(train_items, val_ds, test_ds, meta, args.out)
    print(f"Done. Output saved to {args.out}/")


if __name__ == '__main__':
    main()
