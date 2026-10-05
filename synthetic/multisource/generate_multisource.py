#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.generate_multisource \
#                  --out synthetic/multisource/preview --n-per-k 4 --n-stats 4000 --seed 0
# Description: Stage-1 (supervised) training data for multi-source PILA. Each scene is a
#              single 128x128 Marapi LOS map containing K ~ U{0..K_max} Mogi sources plus
#              realistic Marapi APS noise, with full ground truth (per-source parameters,
#              per-source clean LOS fields, noise field, existence mask). This step writes a
#              small PREVIEW set plus parameter/noise statistics and diagnostic figures so
#              the data can be checked by eye before the full dataset is generated.
# Scientific conventions / assumptions (all flagged in the figures):
#   * Grid: idealized Marapi grid (synthetic/marapi_grid), 128x128, ~312 m, all-valid mask,
#     single LOS geometry (inc 33.1 deg, az -102 deg). LOS positive = toward satellite.
#   * Forward model: synthetic.forward_numpy.mogi (independent of PILA's torch decoder).
#   * Noise: each of the four Marapi APS components (trop, turbulent, orbit, white) is an
#     interferogram-like difference of two random epochs N_c[i] - N_c[j], drawn from a
#     TRAIN or TEST epoch pool (disjoint) so noise maps cannot be memorised. The spatial
#     median of the summed noise is removed (unobservable constant offset). Clean source
#     fields are NOT re-referenced, so source labels stay exact.
#   * Per-source detectability = peak ratio = max|LOS_k| / sigma_ref, where sigma_ref is the
#     mean per-scene noise RMS of the TRAIN pool (fixed scalar). The ratio is drawn
#     LOG-uniform in [0.5, 10] (equal mass per octave, so marginal sources are well
#     represented) and dV is solved exactly (LOS is linear in dV). The scene-RMS SNR of
#     the existing framework, 20 log10(RMS_k / sigma_ref), is recorded for reference only.
#   * Pascal et al. (2014) separation rule, geometry form (no assumed overpressure): Mogi is
#     valid for chamber radius a <= d/5, so with a_max = d_shallow/5 a pair with 3-D centre
#     distance < 4 a_max = 0.8 d_shallow is REJECTED (summation invalid) and < 8 a_max =
#     1.6 d_shallow is FLAGGED (parameters biased by interaction).
# Date:        2026-10-05

import argparse
import json
import os
import time

import numpy as np

from synthetic.forward_numpy import mogi
from synthetic.generate_timeseries import load_full_res_geometry
from synthetic.noise import load_marapi_noise

# --- Fixed scene / prior settings -------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRID_DIR = os.path.join(PROJECT_ROOT, 'synthetic', 'marapi_grid')   # CWD-independent
GEOMETRY_H5 = os.path.join(GRID_DIR, 'geo_geometryRadar.h5')
LAT0_DEG, LON0_DEG = -0.38, 100.47          # grid centre (same as existing Marapi specs)
NOISE_COMPONENTS = ('trop', 'turbulent', 'orbit', 'white')

PRIOR = {
    'k_max': 3,                              # K ~ U{0..k_max}
    'xcen_km': (-15.0, 15.0),                # source East offset, km (grid is +-20 km)
    'ycen_km': (-15.0, 15.0),                # source North offset, km
    'depth_km': (1.0, 10.0),                 # >= 1 km: 0.5 km sources are sub-pixel-sharp
    'peak_ratio': (0.5, 10.0),               # max|LOS_k| / sigma_ref, drawn log-uniform
    'p_inflation': 0.7,                      # P(dV > 0)
    'sep_reject_dshallow': 0.8,              # Pascal 2014 4 radii with a = d/5
    'sep_flag_dshallow': 1.6,                # Pascal 2014 8 radii with a = d/5
    'n_holdout_epochs': 15,                  # noise epochs reserved for TEST scenes
    'max_placement_tries': 500,
}


def load_grid():
    """
    Load the Marapi idealized grid as flattened ENU coordinates and LOS unit vectors.

    Returns
    -------
    grid : dict  xE_m, yN_m, losE, losN, losU (all [n_px] float64), shape (L, W).
    """
    if not os.path.exists(GEOMETRY_H5):
        raise FileNotFoundError(GEOMETRY_H5)
    xE_m, yN_m, losE, losN, losU, shape = load_full_res_geometry(GEOMETRY_H5, LAT0_DEG, LON0_DEG)
    # Plots use imshow(origin='upper') with marker coords in km: requires row 0 = north.
    if not (yN_m[0, 0] > yN_m[-1, 0]):
        raise ValueError("Grid is not north-up (row 0 must be the northernmost row)")
    return {'xE_m': xE_m.ravel(), 'yN_m': yN_m.ravel(), 'losE': losE.ravel(),
            'losN': losN.ravel(), 'losU': losU.ravel(), 'shape': shape}


def mogi_los_mm(grid, xcen_km, ycen_km, depth_km, dV_m3):
    """LOS displacement (mm, + toward satellite) of one Mogi source, as an [L, W] map."""
    uE, uN, uU = mogi(grid['xE_m'], grid['yN_m'], xcen_km * 1e3, ycen_km * 1e3,
                      depth_km * 1e3, dV_m3)
    los_mm = grid['losE'] * uE + grid['losN'] * uN + grid['losU'] * uU
    return los_mm.reshape(grid['shape'])


def separation_dshallow(src_a, src_b):
    """3-D centre distance between two sources divided by the shallower depth."""
    dist_km = np.sqrt((src_a['xcen_km'] - src_b['xcen_km']) ** 2 +
                      (src_a['ycen_km'] - src_b['ycen_km']) ** 2 +
                      (src_a['depth_km'] - src_b['depth_km']) ** 2)
    return dist_km / min(src_a['depth_km'], src_b['depth_km'])


# --- Noise pools -------------------------------------------------------------
class NoisePool:
    """
    Marapi APS component cubes split into disjoint TRAIN / TEST epoch pools.

    A noise realisation = sum over components of (N_c[i] - N_c[j]) with i != j drawn
    independently per component from the chosen pool, minus its spatial median.
    """

    def __init__(self, n_holdout, seed=12345):
        print("  Loading Marapi noise components...")
        # float32 halves memory; output scenes are float32 anyway.
        self.cubes_mm = {c: (load_marapi_noise((c,)) * 1e3).astype(np.float32)
                         for c in NOISE_COMPONENTS}
        n_epoch = next(iter(self.cubes_mm.values())).shape[0]
        if n_epoch < n_holdout + 3:
            raise ValueError(f"Need > n_holdout + 2 noise epochs; have {n_epoch}")
        for c, cube in self.cubes_mm.items():
            print(f"    {c:10s} shape={cube.shape}, dtype={cube.dtype}")
        # Fixed-seed split so TRAIN/TEST epochs are identical across runs.
        perm = np.random.default_rng(seed).permutation(n_epoch)
        self.epochs = {'test': np.sort(perm[:n_holdout]), 'train': np.sort(perm[n_holdout:])}
        print(f"    epoch pools: train={len(self.epochs['train'])}, test={len(self.epochs['test'])}")

    def sample(self, rng, split):
        """Return (noise_mm [L, W], epoch_pairs dict) for one scene."""
        pool = self.epochs[split]
        total_mm = 0.0
        pairs = {}
        for c in NOISE_COMPONENTS:
            i, j = rng.choice(pool, size=2, replace=False)   # ordered pair -> random sign
            total_mm = total_mm + (self.cubes_mm[c][i] - self.cubes_mm[c][j])
            pairs[c] = (int(i), int(j))
        total_mm = total_mm - np.median(total_mm)            # remove unobservable offset
        return total_mm.astype(np.float32), pairs


def estimate_sigma_ref(noise_pool, rng, n_draws=2000):
    """Reference noise RMS (mm): mean per-realisation RMS over TRAIN-pool draws."""
    rms = [np.sqrt(np.mean(noise_pool.sample(rng, 'train')[0] ** 2)) for _ in range(n_draws)]
    return float(np.mean(rms))


# --- Source sampling ----------------------------------------------------------
def sample_sources(rng, grid, k, sigma_ref_mm, prior=PRIOR):
    """
    Draw K Mogi sources satisfying the Pascal (2014) reject rule.

    Returns
    -------
    sources : list of dict (sorted by descending peak ratio) with xcen_km, ycen_km,
              depth_km, dV_m3, peak_ratio, snr_db_rms, field_mm [L, W].
    n_rejected : int  number of placements rejected for separation < reject limit.
    """
    sources, n_rejected = [], 0
    while len(sources) < k:
        for _ in range(prior['max_placement_tries']):
            xcen_km = rng.uniform(*prior['xcen_km'])
            ycen_km = rng.uniform(*prior['ycen_km'])
            depth_km = rng.uniform(*prior['depth_km'])
            candidate = {'xcen_km': xcen_km, 'ycen_km': ycen_km, 'depth_km': depth_km}
            # Pascal 2014 (geometry form): reject if closer than 0.8 * shallower depth.
            if any(separation_dshallow(candidate, other) < prior['sep_reject_dshallow']
                   for other in sources):
                n_rejected += 1
                continue
            # Log-uniform peak ratio: equal sampling effort per octave of detectability.
            peak_ratio = np.exp(rng.uniform(*np.log(prior['peak_ratio'])))
            sign = 1.0 if rng.random() < prior['p_inflation'] else -1.0
            # LOS is linear in dV: solve dV exactly for the target peak |LOS|.
            unit_field_mm = mogi_los_mm(grid, xcen_km, ycen_km, depth_km, 1.0)
            dV_m3 = sign * peak_ratio * sigma_ref_mm / np.abs(unit_field_mm).max()
            field_mm = unit_field_mm * dV_m3
            snr_db_rms = 20 * np.log10(np.sqrt(np.mean(field_mm ** 2)) / sigma_ref_mm)
            candidate.update({'dV_m3': dV_m3, 'peak_ratio': peak_ratio, 'snr_db_rms': snr_db_rms,
                              'field_mm': field_mm.astype(np.float32)})
            sources.append(candidate)
            break
        else:
            raise RuntimeError(f"Could not place source {len(sources) + 1}/{k} "
                               f"after {prior['max_placement_tries']} tries")
    sources.sort(key=lambda s: -s['peak_ratio'])             # canonical slot order
    return sources, n_rejected


def min_pair_separation(sources):
    """Min over pairs of 3-D centre distance / shallower depth; inf if K < 2."""
    return min((separation_dshallow(sources[i], sources[j])
                for i in range(len(sources)) for j in range(i + 1, len(sources))),
               default=np.inf)


def make_scene(rng, grid, noise_pool, sigma_ref_mm, k, split, noise_mode='marapi', prior=PRIOR):
    """One scene: K sources + noise. Returns dict of arrays and truth."""
    sources, n_rejected = sample_sources(rng, grid, k, sigma_ref_mm, prior)
    if noise_mode == 'none':
        # Noise-free debug mode: identical source distribution, zero noise.
        noise_mm, pairs = np.zeros(grid['shape'], np.float32), {}
    else:
        noise_mm, pairs = noise_pool.sample(rng, split)
    L, W = grid['shape']
    k_max = prior['k_max']
    fields_mm = np.zeros((k_max, L, W), np.float32)
    params = np.full((k_max, 4), np.nan, np.float32)        # xcen_km, ycen_km, depth_km, dV_m3
    extra = np.full((k_max, 2), np.nan, np.float32)         # peak_ratio, snr_db_rms
    for s_idx, src in enumerate(sources):
        fields_mm[s_idx] = src['field_mm']
        params[s_idx] = [src['xcen_km'], src['ycen_km'], src['depth_km'], src['dV_m3']]
        extra[s_idx] = [src['peak_ratio'], src['snr_db_rms']]
    clean_mm = fields_mm.sum(axis=0)
    return {'obs_mm': clean_mm + noise_mm, 'clean_mm': clean_mm, 'noise_mm': noise_mm,
            'fields_mm': fields_mm, 'params': params, 'extra': extra, 'k': k,
            'exist': np.arange(k_max) < k, 'min_sep_dshallow': min_pair_separation(sources),
            # Pascal 2014 flag: parameters of this pair likely biased by source interaction.
            'sep_flagged': bool(min_pair_separation(sources) < prior['sep_flag_dshallow']),
            'n_rejected': n_rejected, 'noise_pairs': pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='synthetic/multisource/preview')
    parser.add_argument('--n-per-k', type=int, default=4, help='preview scenes per K value')
    parser.add_argument('--n-stats', type=int, default=4000,
                        help='scenes drawn for parameter/noise statistics (fields not saved)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', choices=['marapi', 'none'], default='marapi',
                        help="'none' = noise-free scenes with the SAME source distribution "
                             "(dV still scaled by the Marapi sigma_ref)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    k_max = PRIOR['k_max']

    print("[1/5] Loading Marapi grid and LOS geometry...")
    grid = load_grid()
    print(f"  Loaded: shape={grid['shape']}, x range {grid['xE_m'].min()/1e3:.1f}.."
          f"{grid['xE_m'].max()/1e3:.1f} km, LOS unit (E,N,U)=({grid['losE'][0]:.3f},"
          f"{grid['losN'][0]:.3f},{grid['losU'][0]:.3f})")

    print("[2/5] Building TRAIN/TEST noise pools and reference noise RMS...")
    t0 = time.time()
    noise_pool = NoisePool(PRIOR['n_holdout_epochs'])
    sigma_ref_mm = estimate_sigma_ref(noise_pool, rng)
    print(f"  sigma_ref = {sigma_ref_mm:.2f} mm  ({time.time() - t0:.1f}s)")

    print(f"[3/5] Generating preview: {args.n_per_k} scenes per K in 0..{k_max} (TRAIN pool)...")
    preview = [make_scene(rng, grid, noise_pool, sigma_ref_mm, k, 'train', args.noise)
               for k in range(k_max + 1) for _ in range(args.n_per_k)]
    np.savez_compressed(
        os.path.join(args.out, 'preview_scenes.npz'),
        obs_mm=np.stack([s['obs_mm'] for s in preview]),
        clean_mm=np.stack([s['clean_mm'] for s in preview]),
        noise_mm=np.stack([s['noise_mm'] for s in preview]),
        fields_mm=np.stack([s['fields_mm'] for s in preview]),
        params=np.stack([s['params'] for s in preview]),
        extra=np.stack([s['extra'] for s in preview]),
        k=np.array([s['k'] for s in preview]),
        exist=np.stack([s['exist'] for s in preview]),
        min_sep_dshallow=np.array([s['min_sep_dshallow'] for s in preview]),
        sep_flagged=np.array([s['sep_flagged'] for s in preview]),
        xE_km=grid['xE_m'].reshape(grid['shape']) / 1e3,
        yN_km=grid['yN_m'].reshape(grid['shape']) / 1e3)
    print(f"  Saved {len(preview)} preview scenes")

    print(f"[4/5] Drawing {args.n_stats} scenes for statistics (train) + {args.n_stats // 4} (test)...")
    t0 = time.time()
    rows = []
    for split, n in [('train', args.n_stats), ('test', args.n_stats // 4)]:
        for _ in range(n):
            k = int(rng.integers(0, k_max + 1))
            s = make_scene(rng, grid, noise_pool, sigma_ref_mm, k, split, args.noise)
            rows.append({'split': split, 'k': k, 'params': s['params'], 'extra': s['extra'],
                         'min_sep_dshallow': s['min_sep_dshallow'], 'n_rejected': s['n_rejected'],
                         'sep_flagged': s['sep_flagged'],
                         'noise_rms_mm': float(np.sqrt(np.mean(s['noise_mm'] ** 2))),
                         'clean_peak_mm': float(np.abs(s['clean_mm']).max())})
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez_compressed(
        os.path.join(args.out, 'stats_draws.npz'),
        split=np.array([r['split'] for r in rows]), k=np.array([r['k'] for r in rows]),
        params=np.stack([r['params'] for r in rows]), extra=np.stack([r['extra'] for r in rows]),
        min_sep_dshallow=np.array([r['min_sep_dshallow'] for r in rows]),
        sep_flagged=np.array([r['sep_flagged'] for r in rows]),
        n_rejected=np.array([r['n_rejected'] for r in rows]),
        noise_rms_mm=np.array([r['noise_rms_mm'] for r in rows]),
        clean_peak_mm=np.array([r['clean_peak_mm'] for r in rows]))

    print("[5/5] Writing metadata...")
    meta = {'prior': PRIOR, 'sigma_ref_mm': sigma_ref_mm, 'seed': args.seed, 'noise': args.noise,
            'noise_epochs': {k: v.tolist() for k, v in noise_pool.epochs.items()},
            'params_columns': ['xcen_km', 'ycen_km', 'depth_km', 'dV_m3'],
            'extra_columns': ['peak_ratio', 'snr_db_rms'],
            'slot_order': 'descending per-source peak ratio',
            'notes': ['empty slots: params/extra NaN, fields_mm 0 -> mask losses with exist',
                      'obs_mm is NOT spatially re-referenced; only the noise median is removed',
                      'dV is derived from depth and peak ratio, so the dV prior is depth-dependent',
                      'peak ratio is the SAMPLED-pixel peak (under-samples d~1 km by up to ~7%)']}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"Done. Output saved to {args.out}/")


if __name__ == '__main__':
    main()
