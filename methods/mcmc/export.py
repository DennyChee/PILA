#!/usr/bin/env python
# Usage:       python -m methods.mcmc.export --scenario-id <id> --out runs/mcmc/<id>/<run_id> \
#                  [--manifest scenarios/manifest.csv] [--model mogi] [--n-sources K] \
#                  [--prior-file configs/mogi_paras_symdV.json] --offset fit|none [--n-starts 1] [--n-bay 10000]
# Description: Per-epoch MCMC adapter, step 1 of 3 (export -> MATLAB run_iclassic_perdate.m -> import).
#              Writes ONE exported .mat per date in the layout iclassic_insar.m / iclassic_multi.m read, using the
#              SAME pixels PILA fits: every scenario (synthetic npz or real MintPy) is read through PILA's own
#              loader with the manifest's preparation settings, so per-point coordinates, LOS unit vectors and
#              values are identical to PILA's. Also writes export.json (dates, K, model, bounds, options) for the
#              MATLAB driver and the importer.
# Conventions: x/y in metres (local ENU), LOS in METRES (+ toward satellite), bounds SI from the shared prior box
#              (methods/priors.py). Truth is never read here.
# Date:        2026-11-23
import argparse
import json
import os
import sys

import numpy as np
from scipy.io import savemat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.pila.make_config import MODELS, insar_block  # noqa: E402
from methods.priors import prior_bounds_si  # noqa: E402
from pila.datasets.preprocessing.insar_mintpy import load_insar_mintpy  # noqa: E402
from scenarios.manifest import DEFAULT_MANIFEST, load_manifest, resolve_row  # noqa: E402
from scenarios.schema import PARAM_NAMES  # noqa: E402


def load_points(row):
    """Read a manifest scenario through PILA's loader; return the loader result (points PILA fits)."""
    block = insar_block(resolve_row(row))
    return load_insar_mintpy(block['timeseries'], block['geometry'], block.get('mask'), block['lat0'], block['lon0'],
                             multilook=block.get('multilook', 1), coh_valid_frac=block.get('coh_valid_frac', 0.5),
                             bbox=block.get('bbox'), verbose=False, ref_date=block.get('ref_date'))


def write_perdate_mats(inputs_dir, tag, dates, x_m, y_m, los_e, los_n, los_u, los_m, model, n_sources, bounds_one):
    """
    Write one iclassic_multi input .mat per date; return the list of paths.

    Inputs:  inputs_dir; tag (config_name in the mat); dates [T] ('YYYY-MM-DD' or 'YYYYMMDD'); x_m, y_m [N] point
             coordinates (m, local ENU); los_e/n/u [N] LOS unit vector; los_m [T, N] LOS values (m, + toward sat);
             model; n_sources; bounds_one [2, P] SI bounds of ONE source (shared prior box).
    """
    os.makedirs(inputs_dir, exist_ok=True)
    paths = []
    for t, date in enumerate(dates):
        date_tag = str(date).replace('-', '')
        path = os.path.join(inputs_dir, f"{tag}_{date_tag}.mat")
        savemat(path, {
            'x_east_m': np.asarray(x_m, float).reshape(-1, 1), 'y_north_m': np.asarray(y_m, float).reshape(-1, 1),
            'lon': np.zeros((np.size(x_m), 1)), 'lat': np.zeros((np.size(x_m), 1)),   # plotting-only fields
            'los_obs_m': np.asarray(los_m[t], float).reshape(-1, 1),
            'losE': np.asarray(los_e, float).reshape(-1, 1), 'losN': np.asarray(los_n, float).reshape(-1, 1),
            'losU': np.asarray(los_u, float).reshape(-1, 1),
            'lat0': 0.0, 'lon0': 0.0, 'model': model, 'epoch_yyyymmdd': date_tag,
            'epoch_iso': f"{date_tag[:4]}-{date_tag[4:6]}-{date_tag[6:]}",
            'config_name': tag, 'n_points': int(np.size(x_m)), 'n_sources': int(n_sources),
            'bounds': bounds_one, 'param_names': np.array(PARAM_NAMES[model], dtype=object).reshape(1, -1)})
        paths.append(path)
    return paths


def export(row, out_dir, model, n_sources, prior_file, fit_offset, n_starts, n_bay, seed0=1):
    """Write inputs/<scenario>_<date>.mat for every date + export.json; return the export dict."""
    inputs_dir = os.path.join(out_dir, 'inputs')
    if os.path.exists(os.path.join(out_dir, 'export.json')):
        raise FileExistsError(f"{out_dir} already holds an export; not overwriting")
    os.makedirs(inputs_dir, exist_ok=True)
    print(f"[1/3] Loading {row['scenario_id']} through PILA's loader ...")
    stack = load_points(row)
    los_m = stack.los_points_mm.astype(np.float64) / 1000.0                     # [T, N] mm -> m
    x_m, y_m = stack.xE_pts * 1e3, stack.yN_pts * 1e3                            # km -> m
    print(f"  Loaded: shape={los_m.shape}, dtype={los_m.dtype}; {len(stack.dates)} dates")
    bounds_one = prior_bounds_si(model, prior_file)                              # [2, P] SI
    print(f"[2/3] Writing {len(stack.dates)} per-date inputs (model={model}, K={n_sources}, offset={fit_offset}) ...")
    files = write_perdate_mats(inputs_dir, row['scenario_id'], stack.dates, x_m, y_m, stack.losE_pts, stack.losN_pts,
                               stack.losU_pts, los_m, model, n_sources, bounds_one)
    files = [os.path.relpath(path, out_dir) for path in files]
    info = dict(scenario_id=row['scenario_id'], model=model, n_sources=int(n_sources), fit_offset=bool(fit_offset),
                n_starts=int(n_starts), seeds=[seed0 + s for s in range(n_starts)], n_bay=int(n_bay),
                start_mode='random', noise_scale='data_std', prior_file=prior_file,
                bounds_one_source=bounds_one.tolist(), dates=[d.replace('-', '') for d in stack.dates], files=files,
                n_points=int(x_m.size), grid_shape=list(stack.mask_d.shape), track=insar_block(resolve_row(row))['geometry'])
    with open(os.path.join(out_dir, 'export.json'), 'w') as handle:
        json.dump(info, handle, indent=2)
    print(f"[3/3] export.json written ({len(files)} dates, {x_m.size} points per date)")
    return info


def main():
    parser = argparse.ArgumentParser(description='Export a scenario for per-epoch MCMC (iclassic_multi.m)')
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--model', default=None, help='default: first model of the manifest row')
    parser.add_argument('--n-sources', type=int, default=None, help='default: manifest n_sources (oracle K)')
    parser.add_argument('--prior-file', default=None, help='default: the same prior file PILA uses for the model')
    parser.add_argument('--offset', required=True, choices=('fit', 'none'),
                        help="'fit' = free per-date constant offset (the agreed comparison rule); explicit on purpose")
    parser.add_argument('--n-starts', type=int, default=1, help='cold random starts per date (best kept)')
    parser.add_argument('--n-bay', type=int, default=10000)
    args = parser.parse_args()
    rows = {row['scenario_id']: row for row in load_manifest(args.manifest)}
    row = rows[args.scenario_id]
    model = args.model or row['models'].split(';')[0]
    n_sources = args.n_sources if args.n_sources is not None else int(row['n_sources'])
    prior_file = args.prior_file or MODELS[model][3]
    if n_sources < 1:
        raise ValueError("K = 0 (noise-only) scenarios are not inverted with oracle K; see the K-unknown track")
    export(row, args.out, model, n_sources, prior_file, args.offset == 'fit', args.n_starts, args.n_bay)
    print(f"Done. Output saved to {args.out}")


if __name__ == '__main__':
    main()
