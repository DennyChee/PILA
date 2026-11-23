#!/usr/bin/env python
# Usage:       python -m methods.vbica.perepoch export --run <vbica run dir> --n-comp 5 --rule overlap_cluster \
#                  [--n-groups K] [--groups "0,2;1"] --offset fit|none [--n-starts 1] [--n-bay 10000] [--manifest ...]
#              matlab: run_iclassic_perdate('<run>/group_<g>', '<MCMC_REPO>')   for every group g
#              python -m methods.vbica.perepoch import --run <vbica run dir> [--manifest ...]
# Description: vbICA adapter, step 4 (decided 2026-11-23): group the ICs of one source, REBUILD that source's
#              signal per date (sum of its IC fields), and invert EVERY DATE of it with the same per-epoch
#              SA + MCMC as the MCMC method (iclassic_multi.m, K = 1 per group, free per-date offset, shared prior
#              box). Import writes the common method output (methods/contract.py) with one slot per group.
# Notes:       * IC fields lack vbICA's per-epoch spatial mean (centering); the free per-date offset of each group
#                inversion absorbs constants, and the reconstruction adds the removed mean back.
#              * recon  = sum_g (fitted source LOS + offset_g) + mu_epoch; phys = sum_g fitted source LOS.
#              * Pixels are matched to the scenario grid by their own coordinates (ICA.llh = scenario x_m / y_m),
#                LOS unit vectors come from the scenario geometry; no truth is read anywhere here.
# Date:        2026-11-23
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.io import loadmat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import re  # noqa: E402

from methods.contract import pixel_fingerprint, write_method_output  # noqa: E402
from methods.mcmc.export import write_perdate_mats  # noqa: E402
from methods.pila.make_config import MODELS  # noqa: E402
from methods.priors import prior_bounds_si  # noqa: E402
from methods.vbica.grouping import group_ics, load_ic_fields  # noqa: E402
from scenarios.manifest import DEFAULT_MANIFEST, load_manifest, resolve_row  # noqa: E402
from scenarios.schema import PARAM_NAMES, los_unit_vectors, read_scenario  # noqa: E402


def pixel_index(llh, scenario):
    """Scenario (row, col) of every vbICA pixel from its own coordinates; error if any is off-grid."""
    x_m, y_m = np.asarray(scenario['x_m'], float), np.asarray(scenario['y_m'], float)
    cols = np.argmin(np.abs(llh[:, 0][:, None] - x_m[None, :]), axis=1)
    rows = np.argmin(np.abs(llh[:, 1][:, None] - y_m[None, :]), axis=1)
    if np.max(np.abs(x_m[cols] - llh[:, 0])) > 1e-6 or np.max(np.abs(y_m[rows] - llh[:, 1])) > 1e-6:
        raise ValueError("vbICA pixel coordinates do not coincide with the scenario grid")
    return rows, cols


def export_groups(run_dir, row, n_comp, rule, n_groups=None, groups=None, fit_offset=True, n_starts=1, n_bay=10000,
                  model=None):
    """Group ICs, rebuild per-group fields, write per-date MCMC inputs per group; return the decision record."""
    with open(os.path.join(run_dir, 'export.json')) as handle:
        info = json.load(handle)
    ic = load_ic_fields(os.path.join(run_dir, f'ic_fields_N{n_comp:02d}.mat'))
    print(f"[1/3] {info['scenario_id']}: N={n_comp}, converged={ic['converged']} (1 = yes, 0 = hit max_steps, nan = unknown)")
    groups_found, record = group_ics(ic['ic_fields'], rule, n_groups=n_groups, groups=groups)
    record.update(n_comp=n_comp, converged=ic['converged'], groups=groups_found)
    print(f"[2/3] rule={rule}: groups {groups_found}")
    scenario = read_scenario(resolve_row(row)['scenario_path'])
    track = info['track']
    rows, cols = pixel_index(ic['llh'], scenario)
    n_grid = scenario['y_m'].size * scenario['x_m'].size
    if rows.size != n_grid:
        raise ValueError(f"vbICA used {rows.size} of {n_grid} pixels: the pixel set differs from the other methods")
    los_e, los_n, los_u = (np.broadcast_to(c, scenario['x_m'].shape[:0] + (scenario['y_m'].size, scenario['x_m'].size))
                           for c in los_unit_vectors(scenario['incidence_deg'], scenario[f'heading_{track}_deg']))
    x_pts, y_pts = scenario['x_m'][cols], scenario['y_m'][rows]
    if n_groups is not None and n_groups < 1:
        raise ValueError("K = 0 (noise-only) scenarios are not inverted with oracle K; see the K-unknown track")
    model = model or row['models'].split(';')[0]
    bounds_one = prior_bounds_si(model, MODELS[model][3])
    for g, members in enumerate(groups_found):
        group_dir = os.path.join(run_dir, f'group_{g}')
        if os.path.exists(os.path.join(group_dir, 'export.json')):
            raise FileExistsError(f"{group_dir} already exported")
        field_m = ic['ic_fields'][members].sum(axis=0)                           # [T, M] exact group sum
        files = write_perdate_mats(os.path.join(group_dir, 'inputs'), f"{info['scenario_id']}_g{g}", info['dates'],
                                   x_pts, y_pts, los_e[rows, cols], los_n[rows, cols], los_u[rows, cols], field_m,
                                   model, 1, bounds_one)
        with open(os.path.join(group_dir, 'export.json'), 'w') as handle:
            json.dump(dict(scenario_id=info['scenario_id'], model=model, n_sources=1, fit_offset=bool(fit_offset),
                           n_starts=int(n_starts), seeds=list(range(1, n_starts + 1)), n_bay=int(n_bay),
                           start_mode='random', noise_scale='data_std', prior_file=MODELS[model][3],
                           ic_members=members, dates=info['dates'],
                           files=[os.path.relpath(f, group_dir) for f in files]), handle, indent=1)
    with open(os.path.join(run_dir, f'grouping_N{n_comp:02d}.json'), 'w') as handle:
        json.dump(record, handle, indent=1, default=float)
    print(f"[3/3] wrote {len(groups_found)} group exports; run run_iclassic_perdate on each group_<g>/")
    return record


def import_groups(run_dir, row):
    """Combine the per-group MCMC results into the common method output (one slot per group)."""
    with open(os.path.join(run_dir, 'export.json')) as handle:
        info = json.load(handle)
    group_dirs = sorted(glob.glob(os.path.join(run_dir, 'group_*')), key=lambda p: int(p.rsplit('_', 1)[1]))
    grouping_files = sorted(glob.glob(os.path.join(run_dir, 'grouping_N*.json')))
    if len(grouping_files) != 1:
        raise ValueError(f"{run_dir}: expected exactly one grouping_N*.json, found {grouping_files} (one N per run directory)")
    grouping = grouping_files[0]
    n_comp = int(os.path.basename(grouping)[len('grouping_N'):-5])
    ic = load_ic_fields(os.path.join(run_dir, f'ic_fields_N{n_comp:02d}.mat'))
    scenario = read_scenario(resolve_row(row)['scenario_path'])
    rows, cols = pixel_index(ic['llh'], scenario)
    dates = info['dates']
    n_dates, n_groups = len(dates), len(group_dirs)
    group_info = []
    for group_dir in group_dirs:
        with open(os.path.join(group_dir, 'export.json')) as handle:
            group_info.append(json.load(handle))
    model = group_info[0]['model']
    n_one = len(PARAM_NAMES[model])
    params = np.full((n_dates, n_groups, n_one), np.nan)
    bands = {key: np.full((n_dates, n_groups, n_one), np.nan) for key in ('post_lo', 'post_med', 'post_hi')}
    offsets = np.zeros((n_dates, n_groups))
    phys_pts = np.zeros((n_dates, rows.size))
    seconds = 0.0
    print(f"[1/2] Importing {n_groups} group inversions ...")
    for g, (group_dir, ginfo) in enumerate(zip(group_dirs, group_info)):
        for t, input_file in enumerate(ginfo['files']):
            result = os.path.join(group_dir, 'results', os.path.basename(input_file))
            if not os.path.exists(result):
                params[t, g] = np.nan
                phys_pts[t] = np.nan
                continue
            loaded = loadmat(result, squeeze_me=True, struct_as_record=False)
            record = loaded['R']
            seconds += float(loaded['date_sec']) if 'date_sec' in loaded else 0.0
            m_best = np.atleast_1d(np.asarray(record.m_best, float))
            params[t, g] = m_best[:n_one]
            for key in bands:
                bands[key][t, g] = np.atleast_1d(np.asarray(getattr(record, key), float))[:n_one]
            offsets[t, g] = m_best[-1] if ginfo['fit_offset'] else 0.0
            phys_pts[t] += np.asarray(record.los_mod_m, float).ravel() - offsets[t, g]
    recon_pts = phys_pts + offsets.sum(axis=1)[:, None] + ic['mu_epoch'][:, None]
    shape = (n_dates,) + scenario[f"los_{info['track']}_m"].shape[1:]
    recon, phys = np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)
    recon[:, rows, cols], phys[:, rows, cols] = recon_pts, phys_pts
    with open(grouping) as handle:
        record = json.load(handle)
    vbica_log = os.path.join(run_dir, f'vbica_N{n_comp:02d}.log')
    decomposition_s = float('nan')
    if os.path.exists(vbica_log):
        with open(vbica_log) as handle:
            match = re.search(r'driver total[^:]*:\s*([0-9.]+)\s*s', handle.read())
        decomposition_s = float(match.group(1)) if match else float('nan')
    timing = dict(train_s=decomposition_s, infer_s=seconds,
                  total_s=(decomposition_s if np.isfinite(decomposition_s) else 0.0) + seconds,
                  per_date_s=seconds / max(n_dates, 1), hardware='cpu (MATLAB)',
                  note='train_s = vbICA decomposition (driver total, excl. plotting); infer_s = per-epoch inversions')
    meta = dict(method='vbica', scenario_id=info['scenario_id'], run_id=os.path.basename(os.path.normpath(run_dir)),
                n_comp=n_comp, converged=ic['converged'], grouping=record, model=model, vbica_log=vbica_log,
                fit_offset=bool(group_info[0]['fit_offset']), n_points=int(rows.size),
                pixel_mask_sha1=pixel_fingerprint(scenario['mask'] if 'mask' in scenario else
                                                  np.ones((scenario['y_m'].size, scenario['x_m'].size), bool)),
                best_estimate='m_best per group (lowest-misfit MCMC sample)')
    print("[2/2] Writing contract outputs ...")
    write_method_output(run_dir, dates, [model] * n_groups, params, {info['track']: recon}, scenario['x_m'],
                        scenario['y_m'], timing, meta, params_lo=bands['post_lo'], params_med=bands['post_med'],
                        params_hi=bands['post_hi'], band_pct=[2.5, 97.5], offset_m=offsets.sum(axis=1),
                        phys_by_track={info['track']: phys})
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='vbICA IC grouping + per-epoch inversion adapter')
    parser.add_argument('step', choices=('export', 'import'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--n-comp', type=int)
    parser.add_argument('--rule', default='overlap_cluster')
    parser.add_argument('--n-groups', type=int, default=None, help='default: manifest n_sources (oracle K)')
    parser.add_argument('--groups', default=None, help='explicit groups, e.g. "0,2;1" (0-based IC indices)')
    parser.add_argument('--offset', choices=('fit', 'none'), help="required for export: 'fit' = agreed comparison rule")
    parser.add_argument('--n-starts', type=int, default=1)
    parser.add_argument('--n-bay', type=int, default=10000)
    args = parser.parse_args()
    with open(os.path.join(args.run, 'export.json')) as handle:
        scenario_id = json.load(handle)['scenario_id']
    row = {r['scenario_id']: r for r in load_manifest(args.manifest)}[scenario_id]
    if args.step == 'export':
        groups = [[int(k) for k in part.split(',')] for part in args.groups.split(';')] if args.groups else None
        n_groups = args.n_groups if args.n_groups is not None else int(row['n_sources'])
        if args.offset is None:
            parser.error('--offset fit|none is required for export (explicit on purpose)')
        export_groups(args.run, row, args.n_comp, args.rule, n_groups, groups, args.offset == 'fit', args.n_starts, args.n_bay)
    else:
        import_groups(args.run, row)
    print(f"Done. Output saved to {args.run}")


if __name__ == '__main__':
    main()
