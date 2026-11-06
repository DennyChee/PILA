#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.build_ta69_quiet_windows \
#                  --noisy-data synthetic/multisource/stage1_ta69_noisy_pk0.5-10 \
#                  --out synthetic/multisource/real_ta69_quiet_windows
# Description: Test-3 data: REAL noise windows from the Marapi (W. Sumatra) Sentinel-1
#              ascending track 69 MintPy stack, cut far from known volcanoes where no
#              volcanic source is expected (target K = 0). Each geocoded epoch is
#              mask-aware multilooked x4 (matching the TA69 training grid), 128x128 windows
#              are cut, and each window-epoch is re-referenced exactly like the training
#              scenes (minus the value at one random valid reference pixel). Masked pixels
#              are set to 0 AFTER referencing (training scenes have no masked pixels).
#              Figures: frame velocity map with volcanoes and chosen windows; example window
#              epochs; real vs synthetic noise RMS and peak distributions.
# Scientific notes / assumptions (flagged):
#   * Volcano coordinates below are approximate (Global Volcanism Program, rounded).
#   * "Quiet" = every window pixel >= min-dist-km (default 15 km; a 5-km-deep Mogi field
#     decays to ~3% of its peak there) from every listed volcano, >= 80% coherent pixels,
#     and lowest de-meaned LOS velocity RMS; non-volcanic deformation cannot be excluded.
#   * Data = geo_timeseries_SET_ERA5_ramp_demErr.h5 (ERA5, ramp and DEM-error corrected),
#     i.e. residual noise is expected to be SMALLER than the uncorrected synthetic APS.
#   * Each epoch is cumulative since 2022-01-01 (MintPy REF_DATE) = APS(t) - APS(t0) + signal,
#     analogous to the synthetic epoch-pair differences.
# Date:        2026-10-05

import argparse
import json
import os
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.preprocessing.insar_mintpy import mask_aware_multilook
from synthetic.multisource.dataset import load_stage1_meta

TA69_GEO_DIR = '/eos-rs/INSAR_processing/denny/Marapi/S1_TA69/mintpy/geo'
TS_FILE = 'geo_timeseries_SET_ERA5_ramp_demErr.h5'
MULTILOOK = 4
WIN_PX = 128
# Approximate volcano positions (lat, lon) in/near the TA69 frame -- W. Sumatra.
VOLCANOES = {'Marapi': (-0.381, 100.473), 'Singgalang': (-0.380, 100.330),
             'Tandikat': (-0.433, 100.317), 'Talang': (-0.978, 100.679),
             'Maninjau (caldera)': (-0.350, 100.200)}


def geo_axes(attrs, n_rows, n_cols, factor):
    """Pixel-centre lat/lon vectors of the multilooked grid."""
    lon = float(attrs['X_FIRST']) + float(attrs['X_STEP']) * factor * (np.arange(n_cols) + 0.5)
    lat = float(attrs['Y_FIRST']) + float(attrs['Y_STEP']) * factor * (np.arange(n_rows) + 0.5)
    return lat, lon


def km_between(lat1, lon1, lat2, lon2):
    """Small-distance approximation, km."""
    return np.hypot((lat1 - lat2) * 110.57, (lon1 - lon2) * 111.32 * np.cos(np.radians(lat1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noisy-data', required=True, help='Stage-1 noisy data dir (for sigma_ref, scaler)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--min-dist-km', type=float, default=15.0,
                        help='min distance from ANY window pixel to a listed volcano (km)')
    parser.add_argument('--min-valid-frac', type=float, default=0.8)
    parser.add_argument('--n-windows', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ts_path = os.path.join(TA69_GEO_DIR, TS_FILE)
    for path in [ts_path, os.path.join(TA69_GEO_DIR, 'geo_maskTempCoh.h5'),
                 os.path.join(TA69_GEO_DIR, 'geo_velocity.h5')]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if os.path.exists(os.path.join(args.out, 'windows.npz')):
        raise FileExistsError(f"{args.out}/windows.npz exists; choose a new --out")
    os.makedirs(args.out, exist_ok=True)
    meta = load_stage1_meta(args.noisy_data)
    rng = np.random.default_rng(args.seed)

    print("[1/5] Multilooking mask and velocity x4...")
    with h5py.File(os.path.join(TA69_GEO_DIR, 'geo_maskTempCoh.h5'), 'r') as f:
        mask_full = f['mask'][:].astype(bool)
    with h5py.File(os.path.join(TA69_GEO_DIR, 'geo_velocity.h5'), 'r') as f:
        vel_full_m_yr = f['velocity'][:]
    vel_ml, mask_ml = mask_aware_multilook(vel_full_m_yr[None] * 1e3, mask_full, MULTILOOK)   # mm/yr
    vel_ml, mask_ml = vel_ml[0], mask_ml & np.isfinite(vel_ml[0])
    with h5py.File(ts_path, 'r') as f:
        attrs = dict(f.attrs)
        dates = [d.decode() for d in f['date'][:]]
        n_epoch = len(dates)
    lat, lon = geo_axes(attrs, *mask_ml.shape, MULTILOOK)
    print(f"  multilooked grid {mask_ml.shape}, valid {mask_ml.mean():.1%}, {n_epoch} epochs {dates[0]}..{dates[-1]}")

    print(f"[2/5] Choosing {args.n_windows} quiet windows (>= {args.min_dist_km} km from volcanoes)...")
    candidates = []
    for r0 in range(0, mask_ml.shape[0] - WIN_PX + 1, 16):
        for c0 in range(0, mask_ml.shape[1] - WIN_PX + 1, 16):
            win_mask = mask_ml[r0:r0 + WIN_PX, c0:c0 + WIN_PX]
            if win_mask.mean() < args.min_valid_frac:
                continue
            # Distance from the CLOSEST window pixel to the nearest volcano (km).
            lat_win, lon_win = np.meshgrid(lat[r0:r0 + WIN_PX], lon[c0:c0 + WIN_PX], indexing='ij')
            d_min = min(float(km_between(lat_win, lon_win, v_lat, v_lon).min())
                        for v_lat, v_lon in VOLCANOES.values())
            if d_min < args.min_dist_km:
                continue
            # Velocity RMS after removing the window mean (offset is irrelevant after referencing).
            vel_win = vel_ml[r0:r0 + WIN_PX, c0:c0 + WIN_PX][win_mask]
            vel_rms = float(np.sqrt(np.nanmean((vel_win - np.nanmean(vel_win)) ** 2)))
            candidates.append((vel_rms, r0, c0, win_mask.mean(), d_min))
    candidates.sort()
    chosen = []
    for cand in candidates:                                   # lowest velocity RMS, no overlap
        if all(abs(cand[1] - ch[1]) >= WIN_PX or abs(cand[2] - ch[2]) >= WIN_PX for ch in chosen):
            chosen.append(cand)
        if len(chosen) == args.n_windows:
            break
    if not chosen:
        raise RuntimeError("No window satisfies the distance/validity constraints")
    for vel_rms, r0, c0, frac, dist in chosen:
        print(f"  window rows {r0}-{r0 + WIN_PX}, cols {c0}-{c0 + WIN_PX}: centre ({lat[r0 + 64]:.3f}, "
              f"{lon[c0 + 64]:.3f}), valid {frac:.0%}, de-meaned velocity RMS {vel_rms:.1f} mm/yr, "
              f"nearest volcano {dist:.0f} km")

    print(f"[3/5] Extracting {len(chosen)} windows x {n_epoch} epochs (epoch by epoch)...")
    t0 = time.time()
    obs_list, valid_list, info = [], [], []
    with h5py.File(ts_path, 'r') as f:
        ts = f['timeseries']
        for e_idx in range(n_epoch):
            epoch_ml, _ = mask_aware_multilook(ts[e_idx][None].astype(np.float32) * 1e3, mask_full, MULTILOOK)
            for w_idx, (_, r0, c0, _, _) in enumerate(chosen):
                win = epoch_ml[0, r0:r0 + WIN_PX, c0:c0 + WIN_PX].astype(np.float64)
                valid = np.isfinite(win) & mask_ml[r0:r0 + WIN_PX, c0:c0 + WIN_PX]
                ref_flat = rng.choice(np.flatnonzero(valid))   # random valid reference pixel
                win = win - win.ravel()[ref_flat]
                win[~valid] = 0.0                             # masked -> 0 after referencing
                obs_list.append(win.astype(np.float32)); valid_list.append(valid)
                info.append((w_idx, e_idx))
    obs_mm = np.stack(obs_list)
    print(f"  Loaded: shape={obs_mm.shape}, dtype={obs_mm.dtype} ({time.time() - t0:.1f}s)")

    print("[4/5] Saving windows.npz (FixedSceneDataset-compatible, K=0 everywhere)...")
    n = len(obs_mm)
    np.savez_compressed(os.path.join(args.out, 'windows.npz'), obs_mm=obs_mm,
                        params=np.full((n, 3, 4), np.nan, np.float32),
                        extra=np.full((n, 3, 2), np.nan, np.float32),
                        exist=np.zeros((n, 3), bool), k=np.zeros(n, int),
                        valid=np.stack(valid_list), window_idx=np.array([i[0] for i in info]),
                        epoch_idx=np.array([i[1] for i in info]), dates=np.array(dates))
    with open(os.path.join(args.out, 'windows.json'), 'w') as f:
        json.dump({'source': ts_path, 'multilook': MULTILOOK, 'volcanoes_lat_lon': VOLCANOES,
                   'min_dist_km': args.min_dist_km,
                   'windows': [{'rows': [r0, r0 + WIN_PX], 'cols': [c0, c0 + WIN_PX],
                                'centre_lat_lon': [float(lat[r0 + 64]), float(lon[c0 + 64])],
                                'valid_frac': float(fr), 'velocity_rms_mm_yr': float(v)}
                               for v, r0, c0, fr, _ in chosen]}, f, indent=2)

    print("[5/5] Figures...")
    fig, ax = plt.subplots(figsize=(7, 9))
    vmax = np.nanpercentile(np.abs(vel_ml[mask_ml]), 98)
    im = ax.imshow(np.where(mask_ml, vel_ml, np.nan), cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   extent=[lon[0], lon[-1], lat[-1], lat[0]], origin='upper')
    cb = fig.colorbar(im, ax=ax); cb.set_label('LOS velocity (mm/yr), + toward satellite')
    for name, (v_lat, v_lon) in VOLCANOES.items():
        ax.plot(v_lon, v_lat, 'k^', ms=8); ax.text(v_lon + 0.01, v_lat, name, fontsize=8)
    for w_idx, (_, r0, c0, _, _) in enumerate(chosen):
        ax.add_patch(plt.Rectangle((lon[c0], lat[r0 + WIN_PX - 1]), lon[c0 + WIN_PX - 1] - lon[c0],
                                   lat[r0] - lat[r0 + WIN_PX - 1], fill=False, ec='lime', lw=2))
        ax.text(lon[c0] + 0.01, lat[r0] - 0.03, f'W{w_idx}', color='lime', fontsize=11, weight='bold')
    ax.set_xlabel('Longitude (deg)'); ax.set_ylabel('Latitude (deg)')
    ax.set_title('Marapi S1 TA69 (ascending) velocity, x4 multilooked\nquiet windows (green), volcanoes (▲)')
    plt.tight_layout(); fig.savefig(os.path.join(args.out, 'fig_windows_map.png'), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(len(chosen), 4, figsize=(18, 4.2 * len(chosen)), squeeze=False)
    for w_idx in range(len(chosen)):
        sel = np.where(np.array([i[0] for i in info]) == w_idx)[0]
        for col, e_pick in enumerate(np.linspace(1, len(sel) - 1, 4).astype(int)):
            arr = obs_mm[sel[e_pick]]
            vm = max(np.percentile(np.abs(arr), 99), 1e-3)
            im = axes[w_idx, col].imshow(np.where(valid_list[sel[e_pick]], arr, np.nan), cmap='RdBu_r',
                                         vmin=-vm, vmax=vm, origin='upper')
            cb = fig.colorbar(im, ax=axes[w_idx, col], fraction=0.046); cb.set_label('LOS (mm)')
            axes[w_idx, col].set_title(f'W{w_idx} {dates[info[sel[e_pick]][1]]} (RMS {np.sqrt(np.mean(arr ** 2)):.1f} mm)')
            axes[w_idx, col].set_xlabel('Column (ml px)'); axes[w_idx, col].set_ylabel('Row (ml px)')
    fig.suptitle('Real TA69 quiet windows (re-referenced, masked = grey/NaN)', y=1.0)
    plt.tight_layout(); fig.savefig(os.path.join(args.out, 'fig_window_examples.png'), dpi=120,
                                    bbox_inches='tight'); plt.close(fig)

    # Real vs synthetic noise statistics.
    with np.load(os.path.join(args.noisy_data, 'test.npz')) as z:
        syn_obs, syn_k = z['obs_mm'], z['k']
    syn_noise = syn_obs[syn_k == 0]
    rms_real = np.sqrt((obs_mm ** 2).mean(axis=(1, 2)))
    rms_syn = np.sqrt((syn_noise ** 2).mean(axis=(1, 2)))
    peak_real = np.abs(obs_mm).reshape(n, -1).max(1)
    peak_syn = np.abs(syn_noise).reshape(len(syn_noise), -1).max(1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (a, b, lab) in zip(axes, [(rms_real, rms_syn, 'Per-map RMS (mm)'),
                                      (peak_real, peak_syn, 'Per-map peak |LOS| (mm)')]):
        bins = np.linspace(0, np.percentile(np.r_[a, b], 99.5), 50)
        ax.hist(a, bins, histtype='step', density=True, label=f'REAL TA69 quiet windows (median {np.median(a):.1f})')
        ax.hist(b, bins, histtype='step', density=True, label=f'synthetic K=0 test (median {np.median(b):.1f})')
        ax.set_xlabel(lab); ax.set_ylabel('Density'); ax.legend(fontsize=8)
    axes[0].axvline(meta['sigma_ref_mm'], color='k', ls=':')
    fig.suptitle('Is the synthetic training noise realistic? Real vs synthetic noise-only maps')
    plt.tight_layout(); fig.savefig(os.path.join(args.out, 'fig_real_vs_synthetic_noise.png'), dpi=150); plt.close(fig)
    print(f"  real RMS median {np.median(rms_real):.1f} mm vs synthetic {np.median(rms_syn):.1f} mm")
    print(f"Done. Output saved to {args.out}/")


if __name__ == '__main__':
    main()
