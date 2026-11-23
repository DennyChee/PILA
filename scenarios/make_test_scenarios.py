#!/usr/bin/env python
# Usage:       python -m scenarios.make_test_scenarios --out runs/_test_scenarios [--grid-from <scenario.npz>]
# Description: PIPELINE TEST scenarios only (plumbing coverage, not the Phase 4 science catalogue). Builds a small
#              set of synthetic scenarios that exercise every branch of the comparison pipeline, in the common
#              scenario format, plus one REAL MintPy row, and registers them in <out>/manifest.csv:
#                t1_mogi_k1_static_clean     Mogi K=1, fixed source, growing dV, noise-free (every method should win)
#                t2_mogi_k1_moving_noisy     Mogi K=1, source moving 4 km, correlated noise
#                t3_sun69_k1_noisy           Sun69 K=1 (second physics model through every adapter)
#                t4_okada_k1_noisy           Okada K=1 dike, opening grows (8 parameters, centroid depth)
#                t5_mogi_k3_static_noisy     Mogi K=3 (more than two sources)
#                t6_mogi_k1_masked_noisy     Mogi K=1 with ~25 % of pixels masked (masking path; vbICA must refuse)
#                real_sierranegra_td128      real MintPy track (Sierra Negra S1 TD128, multilook 20; no truth)
#              Grid and LOS geometry are copied from an existing scenario (default: a TA69 ascending album).
#              Forward fields: independent NumPy models (scenarios/forward_numpy.py); every field is referenced to
#              one corner pixel (like the albums); noise = Gaussian white noise smoothed over ~4 px, rescaled to the
#              requested RMS, also referenced to the same pixel. Seeds fixed.
# Date:        2026-11-23
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scenarios.manifest import add_scenario  # noqa: E402
from scenarios.schema import make_source, read_scenario, source_los_m, write_scenario, write_truth  # noqa: E402

N_DATES = 12
REF_PIXEL = (5, 5)                      # corner pixel, far from every source below
NOISE_SMOOTH_PX = 4.0


def dates_list(n_dates=N_DATES, start='2023-01-06', step_days=12):
    first = np.datetime64(start)
    return [str(first + np.timedelta64(step_days * k, 'D')).replace('-', '') for k in range(n_dates)]


def ramp(n_dates, start, end):
    return np.linspace(start, end, n_dates)


CASES = {
    # name: (list of (model, per-date param function), noise rms mm, masked)
    't1_mogi_k1_static_clean': ([('mogi', lambda n: np.stack([np.full(n, 2000.0), np.full(n, -1500.0),
                                                              np.full(n, 4000.0), ramp(n, 2e6, 2e7)], 1))], 0.0, False),
    't2_mogi_k1_moving_noisy': ([('mogi', lambda n: np.stack([ramp(n, -3000.0, 1000.0), np.full(n, 1000.0),
                                                              np.full(n, 5000.0), np.full(n, 1.5e7)], 1))], 3.0, False),
    't3_sun69_k1_noisy': ([('sun69', lambda n: np.stack([np.full(n, -1000.0), np.full(n, 2000.0), np.full(n, 4000.0),
                                                         np.full(n, 1500.0), ramp(n, 3e6, 2.5e7)], 1))], 3.0, False),
    't4_okada_k1_noisy': ([('okada', lambda n: np.stack([np.full(n, 1500.0), np.full(n, -2000.0), np.full(n, 3000.0),
                                                         np.full(n, 30.0), np.full(n, 75.0), np.full(n, 6000.0),
                                                         np.full(n, 3000.0), ramp(n, 0.2, 2.0)], 1))], 3.0, False),
    't5_mogi_k3_static_noisy': ([('mogi', lambda n: np.stack([np.full(n, -7000.0), np.full(n, -4000.0),
                                                              np.full(n, 4000.0), ramp(n, 5e6, 2e7)], 1)),
                                 ('mogi', lambda n: np.stack([np.full(n, 6000.0), np.full(n, -3000.0),
                                                              np.full(n, 5000.0), ramp(n, 1e7, 2.5e7)], 1)),
                                 ('mogi', lambda n: np.stack([np.full(n, 0.0), np.full(n, 7000.0),
                                                              np.full(n, 3500.0), ramp(n, -2e6, -1.2e7)], 1))],
                                3.0, False),
    't6_mogi_k1_masked_noisy': ([('mogi', lambda n: np.stack([np.full(n, 1000.0), np.full(n, 0.0),
                                                              np.full(n, 4500.0), ramp(n, 5e6, 2e7)], 1))], 3.0, True),
}
REAL_ROW = dict(scenario_id='real_sierranegra_td128', kind='real', family='real_sierranegra', models='mogi',
                n_sources=1, tracks='desc', split='test',
                timeseries='${MINTPY_ROOT}/SierraNegra/S1_TD128/mintpy/geo/geo_timeseries_SET_ERA5_demErr.h5',
                geometry='${MINTPY_ROOT}/SierraNegra/S1_TD128/mintpy/geo/geo_geometryRadar.h5',
                mask='${MINTPY_ROOT}/SierraNegra/S1_TD128/mintpy/geo/geo_maskTempCoh.h5',
                lat0=-0.83, lon0=-91.17, multilook=20, coh_valid_frac=0.5,
                notes='pipeline test: real MintPy branch (no truth); prep settings from configs/phys_smpl/SierraNegra_Mogi_InSAR_A.json')


def make_mask(n_rows, n_cols, rng):
    """~25 % of pixels masked in smooth blobs (like decorrelated areas); the reference pixel stays valid."""
    field = gaussian_filter(rng.standard_normal((n_rows, n_cols)), 8.0)
    mask = field > np.quantile(field, 0.25)
    mask[REF_PIXEL] = True
    return mask


def build(name, grid, out_dir, manifest_path, seed):
    """Generate one test scenario + truth and register it."""
    specs, noise_rms_mm, masked = CASES[name]
    rng = np.random.default_rng(seed)
    dates = dates_list()
    n_rows, n_cols = grid['y_m'].size, grid['x_m'].size
    track = grid['tracks'][0]
    clean_m = np.zeros((N_DATES, n_rows, n_cols))
    sources = []
    for model, param_fn in specs:
        params = param_fn(N_DATES)
        peaks = []
        for t in range(N_DATES):
            field = source_los_m(model, params[t], grid['x_m'], grid['y_m'], grid['incidence_deg'],
                                 grid[f'heading_{track}_deg'])
            peaks.append(float(np.abs(field).max()) * 1e3)                # own peak, unreferenced (truth convention)
            clean_m[t] += field
        sources.append(make_source(model, params, peaks))
    clean_m -= clean_m[:, REF_PIXEL[0]:REF_PIXEL[0] + 1, REF_PIXEL[1]:REF_PIXEL[1] + 1]
    noise_m = np.zeros_like(clean_m)
    if noise_rms_mm > 0:
        for t in range(N_DATES):
            smooth = gaussian_filter(rng.standard_normal((n_rows, n_cols)), NOISE_SMOOTH_PX)
            noise_m[t] = smooth / smooth.std() * noise_rms_mm / 1e3
        noise_m -= noise_m[:, REF_PIXEL[0]:REF_PIXEL[0] + 1, REF_PIXEL[1]:REF_PIXEL[1] + 1]
    data_m = clean_m + noise_m
    mask = make_mask(n_rows, n_cols, rng) if masked else None
    if mask is not None:
        data_m[:, ~mask] = np.nan                                         # masked pixels carry no data
    scenario_path = os.path.join(out_dir, f'{name}.npz')
    truth_path = os.path.join(out_dir, f'{name}_truth.json')
    write_scenario(scenario_path, grid['x_m'], grid['y_m'], {track: data_m}, grid['incidence_deg'],
                   {track: grid[f'heading_{track}_deg']}, dates, mask=mask, clean_los_m_by_track={track: clean_m},
                   ref_date='20230101', ref_pixel=REF_PIXEL)
    realised_rms_mm = float(np.sqrt(np.nanmean(noise_m[:, mask] ** 2 if mask is not None else noise_m ** 2)) * 1e3)
    write_truth(truth_path, name, [track], dates, sources, ref_date='20230101',
                noise=dict(rms_mm=realised_rms_mm, type='smoothed_gaussian', smooth_px=NOISE_SMOOTH_PX, seed=seed),
                notes='pipeline test scenario (scenarios/make_test_scenarios.py), not part of the science catalogue')
    add_scenario(dict(scenario_id=name, kind='synthetic', family=f"test_{name.split('_', 1)[1]}",
                      models=';'.join(model for model, _ in specs), n_sources=len(specs), tracks=track, split='test',
                      scenario_path=os.path.relpath(scenario_path, REPO_ROOT), truth_path=os.path.relpath(truth_path, REPO_ROOT),
                      geometry=track, notes='pipeline test'), manifest_path)
    peak_mm = max(max(s['peak_los_mm']) for s in sources)
    print(f"  {name}: K={len(specs)} peak {peak_mm:.1f} mm, noise {realised_rms_mm:.2f} mm, "
          f"mask {'%.0f %% valid' % (100 * mask.mean()) if mask is not None else 'none'}")


def main():
    parser = argparse.ArgumentParser(description='Generate pipeline test scenarios + manifest')
    parser.add_argument('--out', required=True)
    parser.add_argument('--grid-from', default=os.path.join(REPO_ROOT, 'runs', '_pilot_scenarios', 'G1_N4_moving_n0000.npz'))
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out)
    manifest_path = os.path.join(out_dir, 'manifest.csv')
    if os.path.exists(manifest_path):
        raise FileExistsError(f"{manifest_path} exists; use a new --out")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[1/3] Grid + geometry from {args.grid_from} ...")
    grid = read_scenario(args.grid_from)
    print(f"  grid {grid['y_m'].size} x {grid['x_m'].size}, track {grid['tracks'][0]}")
    print(f"[2/3] Building {len(CASES)} synthetic test scenarios ({N_DATES} dates each) ...")
    for index, name in enumerate(CASES):
        build(name, grid, out_dir, manifest_path, seed=100 + index)
    print("[3/3] Registering the real MintPy row ...")
    add_scenario(REAL_ROW, manifest_path)
    print(f"Done. Output saved to {out_dir}/ (manifest.csv)")


if __name__ == '__main__':
    main()
