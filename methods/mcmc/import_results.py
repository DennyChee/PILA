#!/usr/bin/env python
# Usage:       python -m methods.mcmc.import_results --run runs/mcmc/<id>/<run_id> [--manifest scenarios/manifest.csv]
# Description: Per-epoch MCMC adapter, step 3 of 3 (export.py -> run_iclassic_perdate.m -> THIS). Reads
#              results/<scenario>_<date>.mat (kept start per date) and writes the common method output
#              (methods/contract.py) into the same run directory:
#                params      = m_best (lowest-misfit retained MCMC sample; iclassic's best fit)
#                params_lo / params_med / params_hi = 2.5 / 50 / 97.5 % posterior (band_pct [2.5, 97.5])
#                offset_m    = fitted constant LOS offset per date (NaN if FitOffset was off)
#                recon       = model LOS at the fitted points (sources + offset) on the loader grid
#                phys recon  = sources only (recon - offset)
#              Dates that were not run (date subset) are NaN everywhere.
# Conventions: iclassic parameter order = scenarios.schema.PARAM_NAMES (xoff/xcen, yoff/ycen, depth, ...);
#              SI units; okadaMod depth = centroid (checked by gate_iclassic_multi.m + forward cross-check).
# Date:        2026-11-23
import argparse
import json
import os
import sys

import numpy as np
from scipy.io import loadmat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.contract import grid_axes_m, pixel_fingerprint, write_method_output  # noqa: E402
from methods.mcmc.export import load_points  # noqa: E402
from scenarios.manifest import DEFAULT_MANIFEST, load_manifest  # noqa: E402
from scenarios.schema import PARAM_NAMES  # noqa: E402


def _struct_field(record, name):
    """Field of a MATLAB struct loaded with squeeze_me (handles 0-d object arrays)."""
    value = record[name]
    return value.item() if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1 else value


def import_run(run_dir, row):
    """Convert results/*.mat to the contract outputs in run_dir; return run_dir."""
    with open(os.path.join(run_dir, 'export.json')) as handle:
        info = json.load(handle)
    model, n_sources, fit_offset = info['model'], info['n_sources'], info['fit_offset']
    n_one = len(PARAM_NAMES[model])
    dates = info['dates']
    n_dates = len(dates)
    print(f"[1/3] Reading {n_dates} per-date results for {info['scenario_id']} (model={model}, K={n_sources}) ...")
    params = np.full((n_dates, n_sources, n_one), np.nan)
    bands = {label: np.full((n_dates, n_sources, n_one), np.nan) for label in ('post_lo', 'post_med', 'post_hi')}
    offset_m = np.full(n_dates, np.nan)
    stack = load_points(row)                                                       # same points as the export
    n_points = stack.los_points_mm.shape[1]
    recon_pts_m = np.full((n_dates, n_points), np.nan)
    phys_pts_m = np.full((n_dates, n_points), np.nan)
    n_found = 0
    date_seconds = []
    for t, input_file in enumerate(info['files']):
        result_path = os.path.join(run_dir, 'results', os.path.basename(input_file))
        if not os.path.exists(result_path):
            continue
        loaded = loadmat(result_path, squeeze_me=True, struct_as_record=False)
        record = loaded['R']
        date_seconds.append(float(loaded['date_sec']) if 'date_sec' in loaded else float('nan'))
        n_found += 1
        m_best = np.atleast_1d(np.asarray(record.m_best, dtype=np.float64))
        params[t] = m_best[:n_sources * n_one].reshape(n_sources, n_one)
        for label in bands:
            values = np.atleast_1d(np.asarray(getattr(record, label), dtype=np.float64))
            bands[label][t] = values[:n_sources * n_one].reshape(n_sources, n_one)
        los_mod = np.asarray(record.los_mod_m, dtype=np.float64).ravel()
        if los_mod.size != n_points:
            raise ValueError(f"{result_path}: {los_mod.size} model points != {n_points} exported points")
        recon_pts_m[t] = los_mod
        if fit_offset:
            offset_m[t] = m_best[-1]
            phys_pts_m[t] = los_mod - offset_m[t]
        else:
            phys_pts_m[t] = los_mod
    if n_found == 0:
        raise FileNotFoundError(f"no results in {run_dir}/results")
    print(f"  {n_found}/{n_dates} dates have results")

    print("[2/3] Scattering point fields to the loader grid ...")
    rows, cols = np.nonzero(stack.mask_d)

    def to_grid(points_m):
        grid = np.full((n_dates,) + stack.mask_d.shape, np.nan, dtype=np.float32)
        grid[:, rows, cols] = points_m
        return grid

    track = info.get('track') or (row['tracks'].split(';')[0] if row.get('tracks') else 'asc')
    per_date = np.asarray(date_seconds, dtype=np.float64)            # stored in every result .mat (resume-safe)
    timing = dict(train_s=0.0, infer_s=float(np.nansum(per_date)), total_s=float(np.nansum(per_date)),
                  per_date_s=float(np.nanmean(per_date)) if per_date.size else float('nan'),
                  hardware='cpu (MATLAB)', n_dates_run=int(n_found))
    meta = dict(method='mcmc', scenario_id=info['scenario_id'], run_id=os.path.basename(os.path.normpath(run_dir)),
                model=model, n_sources=n_sources, fit_offset=fit_offset, n_starts=info['n_starts'], seeds=info['seeds'],
                n_bay=info['n_bay'], start_mode=info['start_mode'], noise_scale=info['noise_scale'],
                prior_file=info['prior_file'], best_estimate='m_best (lowest-misfit retained MCMC sample)',
                code='iclassic_multi.m (insar-classical-inversion)', n_points=int(n_points),
                pixel_mask_sha1=pixel_fingerprint(stack.mask_d))
    print("[3/3] Writing contract outputs ...")
    write_method_output(run_dir, dates, [model] * n_sources, params, {track: to_grid(recon_pts_m)},
                        *grid_axes_m(stack.xE_grid, stack.yN_grid, stack.mask_d), timing, meta,
                        params_lo=bands['post_lo'], params_med=bands['post_med'], params_hi=bands['post_hi'],
                        band_pct=[2.5, 97.5], offset_m=offset_m, phys_by_track={track: to_grid(phys_pts_m)})
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Import per-epoch MCMC results into the common output format')
    parser.add_argument('--run', required=True)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    with open(os.path.join(args.run, 'export.json')) as handle:
        scenario_id = json.load(handle)['scenario_id']
    rows = {row['scenario_id']: row for row in load_manifest(args.manifest)}
    import_run(args.run, rows[scenario_id])
    print(f"Done. Output saved to {args.run}")


if __name__ == '__main__':
    main()
