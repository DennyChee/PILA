#!/usr/bin/env python
# Usage:       python -m methods.vbica.export --scenario-id <id> --out runs/vbica/<id>/<run_id> \
#                  [--manifest scenarios/manifest.csv] [--max-steps 6000]
# Description: vbICA adapter, step 1: build a self-contained vbICA WORKING DIRECTORY inside the run directory, so
#              vbICA runs on a new scenario WITHOUT writing anything into the vbICA repo (dirs.dir_main = pwd in
#              vbICA_driver_testing.m; every scenario / data / output path is relative to the working directory):
#                <run>/workdir/Code -> ${VBICA_REPO}/Code                      (symlink: vbICA's own code)
#                <run>/workdir/Data/PILA/<case>/Sentinel_no_los_msk.mat        los_sub [H x W x T] m, x, y, date_list
#                <run>/workdir/Data/PILA/<case>/Sentinel_los_msk.mat           los_msk (all True for dense scenes)
#                <run>/workdir/Scenarios/PILA/<case>/{dataset,parameter_files} cloned from the vbICA synthetic
#                                                                              template, with the SAME edits as the
#                                                                              legacy bench (make_pila_scenarios.py)
#              Synthetic scenarios only (real data: vbICA reads MintPy through its own scenario setup).
# Conventions (READ -- silent-failure risks, from the vbICA loader load_InSARLOS_from_matfile_saveRAM.m):
#   * los_sub rows NORTH-FIRST, x ascending, y stored ASCENDING (the loader pairs meshgrid(x', fliplr(y)) with
#     los_sub(:)); asserted below by a placement round trip.
#   * Units METRES, + toward satellite (the loader never converts). x / y are the local grid in METRES
#     (vbICA uses coordinates for plots / ICA.llh only).
#   * date_list in decimal years.
# Date:        2026-11-23
import argparse
import datetime
import json
import os
import re
import shutil
import sys

import numpy as np
from scipy.io import savemat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pila.paths import resolve_tokens  # noqa: E402
from scenarios.manifest import DEFAULT_MANIFEST, load_manifest, resolve_row  # noqa: E402
from scenarios.schema import read_scenario  # noqa: E402

TEMPLATE_REL = os.path.join('Scenarios', 'Synthetic', 'mogi_overlap_ramp_noise')   # inside ${VBICA_REPO}
LEGACY_TEMPLATE_DATA = 'Data/Synthetic/mogi_overlap_ramp_noise/'
LEGACY_TEMPLATE_SCEN = 'Scenarios/Synthetic/mogi_overlap_ramp_noise/'


def decimal_year(date_yyyymmdd):
    """'YYYYMMDD' -> decimal year (vbICA date_list convention)."""
    day = datetime.datetime.strptime(date_yyyymmdd, '%Y%m%d')
    start = datetime.datetime(day.year, 1, 1)
    length = (datetime.datetime(day.year + 1, 1, 1) - start).total_seconds()
    return day.year + (day - start).total_seconds() / length


def _patch(path, substitutions):
    """Apply regex substitutions (multiline) to a text file; each pattern must match at least once."""
    with open(path) as handle:
        text = handle.read()
    for pattern, replacement in substitutions:
        text, count = re.subn(pattern, replacement, text, flags=re.M)
        if count == 0:
            raise ValueError(f"{path}: pattern {pattern!r} not found (template changed?)")
    with open(path, 'w') as handle:
        handle.write(text)


def vbica_grid(scenario, track):
    """
    Scenario cube -> vbICA layout. Returns (los_sub [H, W, T] m north-first, x ascending, y ascending,
    row_flip, col_flip) where the flips map vbICA rows/cols back to the scenario's.
    """
    cube = np.asarray(scenario[f'los_{track}_m'], dtype=np.float64)              # [T, H, W]
    x_m, y_m = np.asarray(scenario['x_m'], float), np.asarray(scenario['y_m'], float)
    row_flip = bool(y_m[0] < y_m[-1])          # scenario rows south-first -> flip to north-first
    col_flip = bool(x_m[0] > x_m[-1])          # x must be ascending
    if row_flip:
        cube, y_m = cube[:, ::-1, :], y_m[::-1]
    if col_flip:
        cube, x_m = cube[:, :, ::-1], x_m[::-1]
    los_sub = np.transpose(cube, (1, 2, 0))                                     # [H, W, T], rows north-first
    y_ascending = y_m[::-1].copy()                                              # stored ascending (loader convention)
    return np.ascontiguousarray(los_sub), x_m.copy(), y_ascending, row_flip, col_flip


def assert_orientation(los_sub, x_asc, y_asc, cube_scenario_order, x_scn, y_scn):
    """
    Replay the vbICA loader indexing (meshgrid(x', fliplr(y)) with column-major (:)) and check that every pixel
    lands at its scenario coordinate with its scenario value (a transposed / flipped square grid fails here).
    """
    n_rows, n_cols, _ = los_sub.shape
    lon_mat = np.tile(x_asc[None, :], (n_rows, 1))
    lat_mat = np.tile(y_asc[::-1][:, None], (1, n_cols))
    lon_flat, lat_flat = lon_mat.ravel(order='F'), lat_mat.ravel(order='F')
    value_flat = los_sub[:, :, -1].ravel(order='F')
    rows = np.searchsorted(-y_scn, -lat_flat) if y_scn[0] > y_scn[-1] else np.searchsorted(y_scn, lat_flat)
    cols = np.searchsorted(x_scn, lon_flat) if x_scn[0] < x_scn[-1] else np.searchsorted(-x_scn, -lon_flat)
    expected = cube_scenario_order[-1, rows, cols]
    if not np.array_equal(value_flat, expected):
        raise AssertionError("vbICA orientation round trip FAILED: pixels do not land at their scenario coordinates")


def export(row, run_dir, vbica_repo, max_steps=6000):
    """Build <run_dir>/workdir for one synthetic scenario; return the export info dict."""
    if row['kind'] != 'synthetic':
        raise ValueError("vbICA export supports synthetic scenarios; real data use vbICA's own MintPy scenario setup")
    workdir = os.path.join(run_dir, 'workdir')
    if os.path.exists(os.path.join(run_dir, 'export.json')):
        raise FileExistsError(f"{run_dir} already holds an export; not overwriting")
    template = os.path.join(vbica_repo, TEMPLATE_REL)
    for path in (os.path.join(vbica_repo, 'Code'), template, os.path.join(vbica_repo, 'vbICA_driver_testing.m')):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    resolved = resolve_row(row)
    case = f"pila_{row['scenario_id']}"
    data_rel, scen_rel = f'Data/PILA/{case}/', f'Scenarios/PILA/{case}/'
    print(f"[1/4] Reading scenario {resolved['scenario_path']} ...")
    scenario = read_scenario(resolved['scenario_path'])
    track = scenario['tracks'][0]
    print(f"  Loaded: los_{track}_m shape={scenario[f'los_{track}_m'].shape}")
    if 'mask' in scenario and not scenario['mask'].all():
        raise NotImplementedError("masked synthetic scenes: vbICA needs a dense grid here (decide NaN handling first)")

    print("[2/4] Writing vbICA input mats (north-first rows, ascending axes, metres) ...")
    los_sub, x_asc, y_asc, row_flip, col_flip = vbica_grid(scenario, track)
    assert_orientation(los_sub, x_asc, y_asc, np.asarray(scenario[f'los_{track}_m'], np.float64),
                       np.asarray(scenario['x_m'], float), np.asarray(scenario['y_m'], float))
    date_list = np.array([decimal_year(d) for d in scenario['dates']])[:, None]
    os.makedirs(os.path.join(workdir, data_rel), exist_ok=True)
    savemat(os.path.join(workdir, data_rel, 'Sentinel_no_los_msk.mat'),
            {'los_sub': los_sub, 'x': x_asc[None, :], 'y': y_asc[None, :], 'date_list': date_list}, do_compression=True)
    savemat(os.path.join(workdir, data_rel, 'Sentinel_los_msk.mat'),
            {'los_msk': np.ones_like(los_sub, dtype=bool), 'x': x_asc[None, :], 'y': y_asc[None, :],
             'date_list': date_list}, do_compression=True)

    print("[3/4] Cloning the vbICA synthetic template (read-only) into the working directory ...")
    scen_dir = os.path.join(workdir, scen_rel)
    shutil.copytree(os.path.join(template, 'parameter_files'), os.path.join(scen_dir, 'parameter_files'))
    shutil.copytree(os.path.join(template, 'dataset'), os.path.join(scen_dir, 'dataset'))
    os.makedirs(os.path.join(scen_dir, 'figures'), exist_ok=True)
    _patch(os.path.join(scen_dir, 'dataset', 'data_input_file.txt'), [(re.escape(LEGACY_TEMPLATE_SCEN), scen_rel)])
    _patch(os.path.join(scen_dir, 'dataset', 'InSAR_files', 'Sentinel.txt'), [(re.escape(LEGACY_TEMPLATE_DATA), data_rel)])
    _patch(os.path.join(scen_dir, 'parameter_files', 'scen_parameters.m'),
           [(r'^\s*options\.scen\.downsample_perc_lon\s*=.*$', '        options.scen.downsample_perc_lon  = 100;'),
            (r'^\s*options\.scen\.downsample_perc_lat\s*=.*$', '        options.scen.downsample_perc_lat  = 100;')])
    _patch(os.path.join(scen_dir, 'parameter_files', 'decompositionICA_parameters.m'),
           [(r'^init_parameters\.max_steps\s*=.*$', f'init_parameters.max_steps = {int(max_steps)};')])
    code_link = os.path.join(workdir, 'Code')
    if not os.path.exists(code_link):
        os.symlink(os.path.join(vbica_repo, 'Code'), code_link)

    info = dict(scenario_id=row['scenario_id'], case=case, scen_rel='Scenarios/PILA/', track=track,
                vbica_repo=vbica_repo, max_steps=int(max_steps), dates=scenario['dates'],
                grid_shape=list(los_sub.shape[:2]), row_flip=row_flip, col_flip=col_flip,
                x_m_vbica=x_asc.tolist(), y_m_vbica_ascending=y_asc.tolist())
    with open(os.path.join(run_dir, 'export.json'), 'w') as handle:
        json.dump(info, handle, indent=1)
    print(f"[4/4] export.json written; workdir {workdir}")
    return info


def main():
    parser = argparse.ArgumentParser(description='Build a self-contained vbICA working directory for a scenario')
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--vbica-repo', default='${VBICA_REPO}')
    parser.add_argument('--max-steps', type=int, default=6000)
    args = parser.parse_args()
    rows = {row['scenario_id']: row for row in load_manifest(args.manifest)}
    export(rows[args.scenario_id], os.path.abspath(args.out), resolve_tokens(args.vbica_repo), args.max_steps)
    print(f"Done. Output saved to {args.out}")


if __name__ == '__main__':
    main()
