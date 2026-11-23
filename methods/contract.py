#!/usr/bin/env python
# Usage:       from methods.contract import write_method_output, read_method_output, validate_run_dir
#              python -m methods.contract --validate runs/<method>/<scenario>/<run_id>
# Description: The ONE output format every method writes (PILA, per-epoch MCMC, vbICA, later supervised), so a
#              single scorer can judge them all by the same rule. One run directory per (method, scenario, run):
#
#                runs/<method>/<scenario_id>/<run_id>/
#                  params.npz   dates [T] 'YYYYMMDD'
#                               models [K] ('mogi'|'sun69'|'okada'), param_names [K, P_max] ('' padded)
#                               params [T, K, P_max] float64  best estimate, SI units (m, m^3, deg), NaN-padded
#                               params_lo / params_med / params_hi [T, K, P_max]  (optional: posterior / spread band)
#                               band_pct [2]   percentiles of params_lo / params_hi (e.g. [2.5, 97.5])
#                               offset_m [T]   fitted constant LOS offset per date, metres (NaN if not fitted)
#                  recon.npz    los_<track>_m [T, H, W] float32   the method's reconstructed LOS (metres, NaN off-mask)
#                               phys_los_<track>_m [T, H, W]      (optional) physics-only part (no residual/offset)
#                               x_m [W], y_m [H]                  grid of the reconstruction
#                  timing.json  {"train_s", "infer_s", "total_s", "n_dates", "per_date_s", "hardware", ...}
#                  meta.json    {"method", "scenario_id", "run_id", "seed", "git_commit", "config", "created", ...}
#
# Conventions: parameter names/order per model = scenarios.schema.PARAM_NAMES; Okada depth = CENTROID depth;
#              K is the number of fitted source slots (constant over dates; an "off" slot has amplitude ~0).
# Date:        2026-11-23
import argparse
import datetime
import json
import os
import subprocess

import numpy as np

from scenarios.schema import PARAM_NAMES

REQUIRED_FILES = ('params.npz', 'recon.npz', 'timing.json', 'meta.json')


def pixel_fingerprint(mask):
    """Short SHA-1 of a boolean pixel mask [H, W] (+ its shape): recorded by every method so compare.py can prove
    all methods fitted the same pixels."""
    import hashlib
    mask = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    return hashlib.sha1(mask.tobytes() + str(mask.shape).encode()).hexdigest()[:16]


def grid_axes_m(east_km_grid, north_km_grid, mask):
    """
    1-D map axes (metres) of a regular loader grid, fitted from the VALID cells only.

    The PILA loader zeroes the coordinates of masked cells, so a masked first row / column must not be read as an
    axis. Inputs: east_km_grid, north_km_grid [H, W] (km, from the loader), mask [H, W] bool (True = valid).
    Output: (x_m [W], y_m [H]) from least-squares lines x = a + b*col, y = c + d*row over the valid cells.
    """
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size < 2:
        raise ValueError("need at least two valid cells to recover the grid axes")
    x_fit = np.polyfit(cols, np.asarray(east_km_grid)[rows, cols] * 1e3, 1)
    y_fit = np.polyfit(rows, np.asarray(north_km_grid)[rows, cols] * 1e3, 1)
    n_rows, n_cols = np.asarray(mask).shape
    return np.polyval(x_fit, np.arange(n_cols)), np.polyval(y_fit, np.arange(n_rows))


def _json_safe(obj):
    """Replace NaN / inf floats by None (strict JSON) recursively."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def git_commit(repo_dir=None):
    """Current git commit hash of the PILA repo (or 'unknown'), recorded in meta.json for reproducibility."""
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_dir, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def write_method_output(run_dir, dates, models, params, recon_by_track, x_m, y_m, timing, meta,
                        params_lo=None, params_med=None, params_hi=None, band_pct=None, offset_m=None,
                        phys_by_track=None, overwrite=False):
    """
    Write params.npz / recon.npz / timing.json / meta.json for one method run.

    Inputs:  run_dir; dates [T]; models [K]; params [T, K, P_max] SI; recon_by_track {track: [T, H, W] m};
             x_m [W], y_m [H]; timing dict (seconds); meta dict (method, scenario_id, ...);
             optional posterior band params_lo / params_med / params_hi [T, K, P_max] with band_pct [lo, hi]
             percentiles, offset_m [T] (m), phys_by_track {track: [T, H, W] m}.
    Output:  run_dir. Refuses to overwrite existing outputs unless overwrite=True.
    """
    if not overwrite and any(os.path.exists(os.path.join(run_dir, name)) for name in REQUIRED_FILES):
        raise FileExistsError(f"{run_dir} already holds method outputs; not overwriting")
    os.makedirs(run_dir, exist_ok=True)
    params = np.asarray(params, dtype=np.float64)
    n_dates, n_sources, p_max = params.shape
    names = np.full((n_sources, p_max), '', dtype='<U16')
    for k, model in enumerate(models):
        names[k, :len(PARAM_NAMES[model])] = PARAM_NAMES[model]
    arrays = dict(dates=np.asarray([str(d).replace('-', '') for d in dates], dtype='<U8'),
                  models=np.asarray(models, dtype='<U8'), param_names=names, params=params,
                  offset_m=np.full(n_dates, np.nan) if offset_m is None else np.asarray(offset_m, np.float64))
    for label, band in (('params_lo', params_lo), ('params_med', params_med), ('params_hi', params_hi)):
        if band is not None:
            arrays[label] = np.asarray(band, dtype=np.float64)
    if params_lo is not None or params_hi is not None:
        if band_pct is None:
            raise ValueError("band_pct (percentiles of params_lo / params_hi) is required with a band")
        arrays['band_pct'] = np.asarray(band_pct, dtype=np.float64)
    np.savez_compressed(os.path.join(run_dir, 'params.npz'), **arrays)
    recon = dict(x_m=np.asarray(x_m, np.float64), y_m=np.asarray(y_m, np.float64))
    for track, cube in recon_by_track.items():
        recon[f'los_{track}_m'] = np.asarray(cube, np.float32)
    for track, cube in (phys_by_track or {}).items():
        recon[f'phys_los_{track}_m'] = np.asarray(cube, np.float32)
    np.savez_compressed(os.path.join(run_dir, 'recon.npz'), **recon)
    timing = dict(timing, n_dates=int(n_dates))
    with open(os.path.join(run_dir, 'timing.json'), 'w') as handle:
        json.dump(_json_safe(timing), handle, indent=2)
    meta = dict(meta)
    meta.setdefault('git_commit', git_commit())
    meta.setdefault('created', datetime.datetime.now().isoformat(timespec='seconds'))
    with open(os.path.join(run_dir, 'meta.json'), 'w') as handle:
        json.dump(_json_safe(meta), handle, indent=2, default=str)
    problems = validate_run_dir(run_dir)
    if problems:
        raise ValueError(f"{run_dir} written but invalid: {problems}")
    return run_dir


def read_method_output(run_dir):
    """Load a run directory: dict with params (npz dict), recon (npz dict), timing, meta."""
    with np.load(os.path.join(run_dir, 'params.npz'), allow_pickle=False) as handle:
        params = {key: handle[key] for key in handle.files}
    with np.load(os.path.join(run_dir, 'recon.npz'), allow_pickle=False) as handle:
        recon = {key: handle[key] for key in handle.files}
    with open(os.path.join(run_dir, 'timing.json')) as handle:
        timing = json.load(handle)
    with open(os.path.join(run_dir, 'meta.json')) as handle:
        meta = json.load(handle)
    params['models'] = [str(m) for m in params['models']]
    params['dates'] = [str(d) for d in params['dates']]
    return dict(params=params, recon=recon, timing=timing, meta=meta)


def validate_run_dir(run_dir):
    """Return a list of contract violations for one run directory (empty = valid)."""
    missing = [name for name in REQUIRED_FILES if not os.path.exists(os.path.join(run_dir, name))]
    if missing:
        return [f"missing {missing}"]
    out = read_method_output(run_dir)
    problems = []
    params = out['params']
    values = params['params']
    if values.ndim != 3:
        return [f"params must be [T, K, P_max], got {values.shape}"]
    n_dates, n_sources, _ = values.shape
    if len(params['dates']) != n_dates:
        problems.append(f"{len(params['dates'])} dates vs params T={n_dates}")
    if len(params['models']) != n_sources:
        problems.append(f"{len(params['models'])} models vs params K={n_sources}")
    for k, model in enumerate(params['models']):
        if model not in PARAM_NAMES:
            problems.append(f"slot {k}: unknown model {model!r}")
            continue
        n_par = len(PARAM_NAMES[model])
        if list(params['param_names'][k, :n_par]) != PARAM_NAMES[model]:
            problems.append(f"slot {k}: param_names {list(params['param_names'][k, :n_par])} != {PARAM_NAMES[model]}")
    for label in ('params_lo', 'params_med', 'params_hi'):
        if label in params and params[label].shape != values.shape:
            problems.append(f"{label} shape {params[label].shape} != {values.shape}")
    if params['offset_m'].shape != (n_dates,):
        problems.append(f"offset_m shape {params['offset_m'].shape} != ({n_dates},)")
    tracks = [key[4:-2] for key in out['recon'] if key.startswith('los_') and key.endswith('_m')]
    if not tracks:
        problems.append("recon.npz has no los_<track>_m")
    for track in tracks:
        cube = out['recon'][f'los_{track}_m']
        expected = (n_dates, out['recon']['y_m'].size, out['recon']['x_m'].size)
        if cube.shape != expected:
            problems.append(f"recon los_{track}_m shape {cube.shape} != {expected}")
    for key in ('method', 'scenario_id'):
        if key not in out['meta']:
            problems.append(f"meta.json missing {key}")
    if 'total_s' not in out['timing']:
        problems.append("timing.json missing total_s")
    return problems


def main():
    parser = argparse.ArgumentParser(description='Validate a method run directory against the output contract')
    parser.add_argument('--validate', required=True)
    args = parser.parse_args()
    problems = validate_run_dir(args.validate)
    print("VALID" if not problems else "INVALID:\n  " + "\n  ".join(problems))
    raise SystemExit(0 if not problems else 1)


if __name__ == '__main__':
    main()
