#!/usr/bin/env python
# Usage:       python classical_k2_floor.py [--albums E1n025,E2n035,E1nonoise] [--starts 40] [--epoch-starts 6]
#                  [--out-dir figures/classical_k2_floor]
# Description: Data-floor benchmark for the K=2 SNR-12 PILA results. Classical unweighted least-squares fit of two static
#              Mogi sources to the SAME albums PILA saw, with the SAME forward function and grid that generated them
#              (synthetic.multisource.torch_generator.mogi_los_mm_torch on the separable TA69 grid). Source locations and
#              depths (6 numbers) are searched non-linearly (scipy least_squares, multi-start from random points in a wide
#              box, lowest misfit kept); dV of both sources + one constant offset are solved linearly at every step
#              (LOS is linear in dV; the free offset makes the reference pixel irrelevant). Two estimators per album:
#              "stack" = one fit to the time-mean stack; "per-epoch" = one fit per date (started from the stack solution
#              + --epoch-starts random starts), then the median of the parameters (same scoring rule as PILA).
#              Truth is used ONLY for scoring. The noise-free album must return the truth (sanity check).
# Date:        2026-10-26

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares, linear_sum_assignment

from synthetic.multisource.make_pila_stack import load_ta69_geometry
from synthetic.multisource.torch_generator import mogi_los_mm_torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STACKS = 'synthetic/multisource/pila_stacks_v2'
ALBUMS = {  # key -> (album npz, truth json)
    'E1n025': (f'{STACKS}/albums_snr/E1_S3_K2_separated_n025.npz', f'{STACKS}/albums_snr/E1_S3_K2_separated_n025_truth.json'),
    'E1n050': (f'{STACKS}/albums_snr/E1_S3_K2_separated_n050.npz', f'{STACKS}/albums_snr/E1_S3_K2_separated_n050_truth.json'),
    'E2n035': (f'{STACKS}/albums_snr/E2_S3_K2_separated_n035.npz', f'{STACKS}/albums_snr/E2_S3_K2_separated_n035_truth.json'),
    'E1nonoise': (f'{STACKS}/albums_nonoise/E1_S3_K2_separated_nonoise.npz',
                  f'{STACKS}/albums_nonoise/E1_S3_K2_separated_nonoise_truth.json'),
}
# Random-start box (km): whole scene horizontally, depth 1-12 km; no truth information
X_BOX_KM, Y_BOX_KM, DEPTH_BOX_KM = (-12.0, 12.0), (-12.0, 12.0), (1.0, 12.0)


class MogiK2:
    """Two-Mogi LOS forward model on the album grid with dV + constant offset solved linearly."""

    def __init__(self, geom):
        grid = geom['grid']
        self.xE_m = torch.tensor(grid['xE_m'], dtype=torch.float64)
        self.yN_m = torch.tensor(grid['yN_m'], dtype=torch.float64)
        self.los = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]), dtype=torch.float64)

    def unit_fields(self, geometry_km):
        """LOS fields (mm per m^3) of the two sources [2, N]; geometry_km = (x1, y1, d1, x2, y2, d2) km."""
        g = torch.tensor(np.asarray(geometry_km, dtype=np.float64).reshape(1, 2, 3) * 1e3)
        return mogi_los_mm_torch(self.xE_m, self.yN_m, self.los, g[..., 0], g[..., 1], g[..., 2],
                                 torch.ones(1, 2, dtype=torch.float64))[0].numpy()

    def solve_linear(self, geometry_km, data_mm):
        """Best dV (m^3) of both sources + offset (mm) for fixed geometry; returns (coefficients, residual mm)."""
        design = np.column_stack([self.unit_fields(geometry_km).T, np.ones(data_mm.size)])
        # Scale the dV columns to O(1) so lstsq is well conditioned (unit fields are ~1e-5 mm per m^3)
        column_scale = np.abs(design).max(axis=0)
        coeffs, *_ = np.linalg.lstsq(design / column_scale, data_mm, rcond=None)
        coeffs = coeffs / column_scale
        return coeffs, design @ coeffs - data_mm

    def fit(self, data_mm, starts_km, depth_bounds_km=(0.3, 20.0)):
        """Multi-start non-linear fit; returns (params [2 x (x km, y km, depth km, dV m3)], offset mm, RMS mm).
        depth_bounds_km: solver depth bounds (default unchanged; classical_k2_perdate.py passes PILA's 0.5-12 km)."""
        lower = [X_BOX_KM[0], Y_BOX_KM[0], depth_bounds_km[0]] * 2
        upper = [X_BOX_KM[1], Y_BOX_KM[1], depth_bounds_km[1]] * 2
        best = None
        for start in starts_km:
            result = least_squares(lambda g: self.solve_linear(g, data_mm)[1], start, bounds=(lower, upper),
                                   x_scale=1.0, diff_step=1e-4)
            if best is None or result.cost < best.cost:
                best = result
        coeffs, residual = self.solve_linear(best.x, data_mm)
        params = np.column_stack([best.x.reshape(2, 3), coeffs[:2]])
        return params, coeffs[2], float(np.sqrt(np.mean(residual ** 2)))


def random_starts(rng, n_starts):
    """n_starts random 6-vectors (x, y, depth) x 2 inside the start box; no truth used."""
    starts = []
    for _ in range(n_starts):
        starts.append(np.concatenate([[rng.uniform(*X_BOX_KM), rng.uniform(*Y_BOX_KM), rng.uniform(*DEPTH_BOX_KM)]
                                      for _ in range(2)]))
    return starts


def score(params, truth):
    """Match fitted sources to truth (Hungarian on location) and return per-source signed errors + success flag."""
    truth_idx, fit_idx = linear_sum_assignment(np.linalg.norm(truth[:, None, :2] - params[None, :, :2], axis=-1))
    matched = np.empty_like(params)
    matched[truth_idx] = params[fit_idx]
    out = {}
    for src, name in enumerate(['west', 'east']):
        out[f'{name}_loc_km'] = float(np.hypot(*(matched[src, :2] - truth[src, :2])))
        out[f'{name}_depth_pct'] = float(100 * (matched[src, 2] / truth[src, 2] - 1))
        out[f'{name}_dV_pct'] = float(100 * (matched[src, 3] / truth[src, 3] - 1))
    out['success'] = bool(max(out['west_loc_km'], out['east_loc_km']) <= 1.5 and
                          max(abs(out['west_depth_pct']), abs(out['east_depth_pct'])) <= 50 and
                          max(abs(out['west_dV_pct']), abs(out['east_dV_pct'])) <= 50)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--albums', default='E1nonoise,E1n025,E2n035')
    parser.add_argument('--starts', type=int, default=40, help='random starts for the stack fit')
    parser.add_argument('--epoch-starts', type=int, default=6, help='extra random starts per date (+ stack solution)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', default='figures/classical_k2_floor')
    args = parser.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'classical_k2_floor.csv')
    if os.path.exists(out_csv):
        print(f"  WARNING: overwriting {out_csv}")

    print("[1/3] Loading TA69 geometry (same separable grid the albums were rendered on)...")
    model = MogiK2(load_ta69_geometry())
    rng = np.random.default_rng(args.seed)

    rows = []
    album_keys = args.albums.split(',')
    for step, key in enumerate(album_keys, start=1):
        album_path, truth_path = (os.path.join(PROJECT_ROOT, path) for path in ALBUMS[key])
        for path in (album_path, truth_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        with np.load(album_path) as album:
            data_mm = album['los_asc_m'].astype(np.float64).reshape(album['los_asc_m'].shape[0], -1) * 1e3
        with open(truth_path) as handle:
            truth = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km'], s['dV_m3']]
                              for s in json.load(handle)['sources']])
        print(f"[2/3] ({step}/{len(album_keys)}) {key}: data shape={data_mm.shape}, dtype={data_mm.dtype}")

        # --- Estimator 1: static fit to the time-mean stack ---
        t0 = time.time()
        stack_params, _, stack_rms = model.fit(data_mm.mean(axis=0), random_starts(rng, args.starts))
        rows.append(dict(album=key, estimator='stack', rms_mm=stack_rms, **score(stack_params, truth)))
        print(f"  stack fit: RMS {stack_rms:.3f} mm, {rows[-1]} ({time.time() - t0:.1f}s)")

        # --- Estimator 2: one fit per date, then median of parameters (slots matched to the stack fit first) ---
        t0 = time.time()
        start_from_stack = np.concatenate([stack_params[0, :3], stack_params[1, :3]])
        epoch_params, epoch_rms = [], []
        for epoch_index in range(data_mm.shape[0]):
            params, _, rms = model.fit(data_mm[epoch_index], [start_from_stack] + random_starts(rng, args.epoch_starts))
            # Keep slot order consistent across dates: match each date's sources to the stack solution by location
            _, order = linear_sum_assignment(np.linalg.norm(stack_params[:, None, :2] - params[None, :, :2], axis=-1))
            epoch_params.append(params[order])
            epoch_rms.append(rms)
        median_params = np.median(np.array(epoch_params), axis=0)
        rows.append(dict(album=key, estimator='per-epoch median', rms_mm=float(np.median(epoch_rms)),
                         **score(median_params, truth)))
        print(f"  per-epoch median: {rows[-1]} ({time.time() - t0:.1f}s)")
        np.save(os.path.join(out_dir, f'classical_k2_per_epoch_params_{key}.npy'), np.array(epoch_params))

        if key == 'E1nonoise':   # sanity check: noise-free album must give back the truth
            errors = [abs(rows[-2][f'{name}_{q}']) for name in ('west', 'east') for q in ('depth_pct', 'dV_pct')]
            if max(errors) > 0.5 or max(rows[-2]['west_loc_km'], rows[-2]['east_loc_km']) > 0.01:
                raise RuntimeError(f'noise-free stack fit does not recover truth: {rows[-2]}')
            print("  SANITY PASS: noise-free album recovers truth")

    print("[3/3] Writing results...")
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format='%.4g')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"Done. Output saved to {out_dir}/")


if __name__ == '__main__':
    main()
