#!/usr/bin/env python
# Usage:       python classical_k2_perdate.py --album G1_N1_staggered_n0250 [--starts 12] [--seed 0]
#                  [--albums-dir synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2]
#                  [--out-root saved/multisource_k2_nonstatic_classical_v2]
# Description: Classical per-date benchmark for the non-static K=2 albums. Each date is fitted ON ITS OWN with two Mogi
#              sources by unweighted least squares: locations and depths (6 numbers) searched non-linearly from
#              --starts COLD random starts (no stack solution, no information from other dates, never the truth),
#              dV of both sources + one constant offset solved linearly (so the reference pixel does not matter).
#              Same forward function and grid that made the albums (classical_k2_floor.MogiK2 on the separable TA69
#              grid). K = 2 is fixed, as in PILA. Scored with synthetic/multisource/nonstatic_scoring.score_run, the
#              rule PILA is scored with.
# Output:      <out-root>/<album>/per_date_params.npy [T, 2, 4] (x km, y km, depth km, dV m^3), per_date_rms.npy [T],
#              score_summary.json, score_source_dates.csv, timing.json
# Date:        2026-10-26

import argparse
import json
import os
import time

import numpy as np

import classical_k2_floor as floor
from synthetic.multisource.make_pila_stack import load_ta69_geometry
from synthetic.multisource.nonstatic_scoring import score_run

# Same box as PILA's prior (configs/mogi_paras_symdV.json): x, y +-15 km, depth 0.5-12 km. MogiK2.fit and random_starts
# read the x/y module constants at call time (solver bounds + starts); depth bounds are passed to fit explicitly.
# Remaining asymmetry (by design, stated in the write-up): classical dV is unbounded (PILA +-2e8 m^3) and classical
# solves a free constant offset per date (PILA has its low-rank residual instead).
floor.X_BOX_KM, floor.Y_BOX_KM, floor.DEPTH_BOX_KM = (-15.0, 15.0), (-15.0, 15.0), (1.0, 12.0)   # starts: depth 1-12
SOLVER_DEPTH_BOUNDS_KM = (0.5, 12.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', required=True)
    parser.add_argument('--albums-dir', default='synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2')
    parser.add_argument('--out-root', default='saved/multisource_k2_nonstatic_classical_v2')   # v2: PILA depth box
    parser.add_argument('--starts', type=int, default=12, help='cold random starts per date')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    album_path = os.path.join(args.albums_dir, f'{args.album}.npz')
    truth_path = os.path.join(args.albums_dir, f'{args.album}_truth.json')
    for path in (album_path, truth_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    out_dir = os.path.join(args.out_root, args.album)
    if os.path.exists(os.path.join(out_dir, 'score_summary.json')):
        raise FileExistsError(f'{out_dir} already scored; not overwriting')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/3] Loading {album_path} and the TA69 geometry...")
    with np.load(album_path) as album:
        data_mm = album['los_asc_m'].astype(np.float64).reshape(album['los_asc_m'].shape[0], -1) * 1e3
    print(f"  Loaded: shape={data_mm.shape}, dtype={data_mm.dtype}")
    model = floor.MogiK2(load_ta69_geometry())
    rng = np.random.default_rng(args.seed)

    print(f"[2/3] Per-date fits ({args.starts} cold starts each)...")
    t0 = time.time()
    params, rms = [], []
    for date_index in range(data_mm.shape[0]):
        fit_params, _, fit_rms = model.fit(data_mm[date_index], floor.random_starts(rng, args.starts),
                                            depth_bounds_km=SOLVER_DEPTH_BOUNDS_KM)
        params.append(fit_params)
        rms.append(fit_rms)
        print(f"  date {date_index + 1}/{data_mm.shape[0]}: RMS {fit_rms:.2f} mm ({time.time() - t0:.0f}s)")
    params, rms = np.array(params), np.array(rms)
    fit_seconds = time.time() - t0
    np.save(os.path.join(out_dir, 'per_date_params.npy'), params)
    np.save(os.path.join(out_dir, 'per_date_rms.npy'), rms)

    print("[3/3] Scoring with the shared per-date rule...")
    truth = json.load(open(truth_path))
    summary, table = score_run(params, truth)
    table.to_csv(os.path.join(out_dir, 'score_source_dates.csv'), index=False, float_format='%.6g')
    with open(os.path.join(out_dir, 'score_summary.json'), 'w') as handle:
        json.dump(dict(summary, method='classical_per_date', starts=args.starts, seed=args.seed), handle, indent=1)
    with open(os.path.join(out_dir, 'timing.json'), 'w') as handle:
        json.dump(dict(fit_seconds=fit_seconds, seconds_per_date=fit_seconds / data_mm.shape[0]), handle, indent=1)
    print(f"  success {summary['n_success']}/{summary['n_active']} active source-dates "
          f"({summary['success_frac']:.2f}), false detections {summary['n_false_detection']}/{summary['n_inactive']}")
    print(f"Done. Output saved to {out_dir}/ ({fit_seconds:.0f}s)")


if __name__ == '__main__':
    main()
