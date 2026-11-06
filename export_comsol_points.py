#!/usr/bin/env python
# Usage:       conda activate pila
#              python export_comsol_points.py [--configs configs/comsol/per_chamber]
#                  [--out exported_comsol] [--only cid1000_flat] [--no-joint] [--overwrite]
# Description: Export the EXACT LOS points (and per-point LOS unit vectors) each per-chamber
#              PILA run of the COMSOL wk6 catalogue fit, as .mat files the classical MATLAB
#              inversion (iclassic_insar.m, 'exported' mode) reads. Writes one single-track
#              .mat per PILA config and, per (chamber, surface), one JOINT asc+desc .mat
#              (points stacked, each keeping its own LOS vector) for the single-track
#              LOS-tradeoff test.
# Date:        2026-10-02
#
# Scientific conventions (READ):
#   * Points come from PILA's own loader (load_insar_mintpy, native .npz branch) with the
#     SAME insar args as the config: 201x201 @100 m, multilook 1, no mask -> N = 40401.
#   * Coordinates: local frame, metres, origin = chamber axis (npz x_m / y_m). lat0/lon0 are
#     meaningless for these scenes; lon/lat are filled with the local km coordinates (they
#     are plotting aids only in iclassic_insar and never enter the fit).
#   * LOS: metres, + = motion TOWARD the satellite. COMSOL maps are noise-free.
#   * Bounds: built from the config's paras JSON (configs/comsol_mogi_paras.json) with the
#     same helper as the field benchmark, so the classical box equals PILA's decoder box.

import argparse
import glob
import json
import os
import re
import time

import numpy as np
from scipy.io import savemat

# PILA's own loader -> byte-identical points to what PILA trained on.
from datasets.preprocessing.insar_mintpy import load_insar_mintpy
# Shared bound builder -> identical search box for classical and PILA.
from export_pila_points import PHYSICS_TO_MODEL, PARAS_KEY, build_classical_bounds

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME_PATTERN = re.compile(r'comsol_wk6_cid(\d+)_(flat|cone)_(asc|desc)_mogi')


def load_config_points(config_path):
    """
    Load one per-chamber PILA config and the LOS points it was trained on.

    Input : config_path  str  path to a configs/comsol/per_chamber/*.json
    Output: dict with config_name, cid, surface, track, model, bounds [2 x n_par] (SI),
            param_names, and the point arrays x_east_m / y_north_m (m), los_obs_m (m),
            losE / losN / losU (unit vector components), all float64 [N].
    """
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    match = CONFIG_NAME_PATTERN.fullmatch(config_name)
    if match is None:
        raise ValueError(f"unexpected config name {config_name!r}")
    with open(config_path) as fh:
        config = json.load(fh)
    arch_args = config['arch']['args']
    model = PHYSICS_TO_MODEL[arch_args['physics']]
    if model != 'mogi':
        raise ValueError(f"{config_name}: only Mogi COMSOL runs exist, got {model}")

    # --- Search box from the SAME paras JSON PILA's decoder used ---
    paras_path = arch_args[PARAS_KEY[model]]
    if not os.path.isabs(paras_path):
        paras_path = os.path.join(CURRENT_DIR, paras_path)
    with open(paras_path) as fh:
        paras = json.load(fh)
    bounds, param_names = build_classical_bounds(model, paras)

    # --- Points: same loader call as model/model_phys_smpl.py (npz branch) ---
    ins = arch_args['insar']
    # Only these keys are forwarded below; any other loader option (stride, bbox, ref_date,
    # far_field, bootstrap...) would make PILA's points differ from the export -> refuse.
    unsupported = set(ins) - {'timeseries', 'geometry', 'lat0', 'lon0', 'multilook',
                              'coh_valid_frac', 'verbose'}
    if unsupported:
        raise ValueError(f"{config_name}: insar keys {sorted(unsupported)} are not forwarded "
                         f"by this exporter; extend it before exporting")
    if not os.path.exists(ins['timeseries']):
        raise FileNotFoundError(f"{config_name}: scene not found: {ins['timeseries']}")
    scene = load_insar_mintpy(
        ins['timeseries'], ins['geometry'], ins.get('mask'),
        ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 1),
        coh_valid_frac=ins.get('coh_valid_frac', 0.5), bbox=ins.get('bbox'), verbose=False)
    if len(scene.dates) != 1:
        # Per-chamber npz scenes carry exactly one map; anything else means a wrong input.
        raise ValueError(f"{config_name}: expected 1 sample, got {len(scene.dates)}")

    los_mm = scene.los_points_mm[0].astype(np.float64)               # [N] mm
    finite = np.isfinite(los_mm)
    if not finite.all():
        raise ValueError(f"{config_name}: {int((~finite).sum())} non-finite LOS points "
                         f"(COMSOL maps should be complete)")
    return dict(
        config_name=config_name, cid=int(match[1]), surface=match[2], track=match[3],
        model=model, bounds=bounds, param_names=param_names,
        x_east_m=scene.xE_pts.astype(np.float64) * 1000.0,           # km -> m
        y_north_m=scene.yN_pts.astype(np.float64) * 1000.0,
        los_obs_m=los_mm / 1000.0,                                   # mm -> m
        losE=scene.losE_pts.astype(np.float64),
        losN=scene.losN_pts.astype(np.float64),
        losU=scene.losU_pts.astype(np.float64))


def write_export(out_path, points, config_name, track):
    """Write the .mat iclassic_insar 'exported' mode reads (fields as export_pila_points)."""
    column = lambda values: np.asarray(values, dtype=np.float64).reshape(-1, 1)
    savemat(out_path, {
        'x_east_m': column(points['x_east_m']),
        'y_north_m': column(points['y_north_m']),
        # Local KM (not degrees) in place of lon/lat: saved-only in iclassic, never used in
        # the fit. Note the result's source_lonlat is degrees about (0,0) -- don't mix them.
        'lon': column(points['x_east_m'] / 1000.0),
        'lat': column(points['y_north_m'] / 1000.0),
        'los_obs_m': column(points['los_obs_m']),
        'losE': column(points['losE']),
        'losN': column(points['losN']),
        'losU': column(points['losU']),
        'lat0': 0.0, 'lon0': 0.0, 'multilook': 1,
        'epoch_yyyymmdd': '20000101', 'epoch_iso': '2000-01-01',     # npz placeholder date
        'model': points['model'], 'volcano': 'comsol_wk6', 'track': track,
        'config_name': config_name, 'cid': int(points['cid']), 'surface': points['surface'],
        'n_points': int(points['x_east_m'].size),
        'bounds': points['bounds'],                                  # [2 x n_par] SI
        'param_names': np.array(points['param_names'], dtype=object).reshape(1, -1),
    }, do_compression=True)


def main():
    parser = argparse.ArgumentParser(description='Export COMSOL wk6 points for iclassic_insar')
    parser.add_argument('--configs', default='configs/comsol/per_chamber')
    parser.add_argument('--out', default='exported_comsol')
    parser.add_argument('--only', default=None,
                        help='substring filter on config names, e.g. cid1000_flat')
    parser.add_argument('--no-joint', action='store_true', help='skip the joint asc+desc exports')
    parser.add_argument('--overwrite', action='store_true', help='replace existing .mat files')
    args = parser.parse_args()
    t0 = time.time()

    # --- [1/3] Find configs ---
    if not os.path.isdir(args.configs):
        raise FileNotFoundError(args.configs)
    config_paths = sorted(glob.glob(os.path.join(args.configs, 'comsol_wk6_cid*_mogi.json')))
    if args.only:
        config_paths = [path for path in config_paths if args.only in os.path.basename(path)]
    if not config_paths:
        raise RuntimeError(f"no configs matched in {args.configs} (only={args.only})")
    print(f"[1/3] {len(config_paths)} configs in {args.configs}")
    single_dir = os.path.join(args.out, 'single')
    joint_dir = os.path.join(args.out, 'joint')
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(joint_dir, exist_ok=True)

    # --- [2/3] Single-track exports (keep points in memory for the joint stack) ---
    print(f"[2/3] Writing single-track exports to {single_dir} ...")
    by_chamber = {}                                   # (cid, surface) -> {track: points}
    n_written, n_skipped = 0, 0
    for index, config_path in enumerate(config_paths, start=1):
        points = load_config_points(config_path)
        by_chamber.setdefault((points['cid'], points['surface']), {})[points['track']] = points
        out_path = os.path.join(single_dir, f"{points['config_name']}.mat")
        if os.path.exists(out_path) and not args.overwrite:
            n_skipped += 1
            continue
        write_export(out_path, points, points['config_name'], points['track'])
        n_written += 1
        print(f"  [{index}/{len(config_paths)}] {points['config_name']}: "
              f"N={points['x_east_m'].size}, peak |LOS|={1000 * np.abs(points['los_obs_m']).max():.1f} mm")
    print(f"  wrote {n_written}, skipped {n_skipped} existing (use --overwrite to replace)")

    # --- [3/3] Joint asc+desc exports ---
    if args.no_joint:
        print("[3/3] Skipping joint exports (--no-joint)")
    else:
        print(f"[3/3] Writing joint asc+desc exports to {joint_dir} ...")
        n_joint = 0
        for (cid, surface), tracks in sorted(by_chamber.items()):
            if set(tracks) != {'asc', 'desc'}:
                print(f"  WARNING: cid{cid} {surface} has tracks {sorted(tracks)}; no joint export")
                continue
            asc, desc = tracks['asc'], tracks['desc']
            # Both tracks must share the grid and the search box, or the stack is meaningless.
            if not (np.array_equal(asc['x_east_m'], desc['x_east_m'])
                    and np.array_equal(asc['y_north_m'], desc['y_north_m'])
                    and np.array_equal(asc['bounds'], desc['bounds'])):
                raise ValueError(f"cid{cid} {surface}: asc/desc grids or bounds differ")
            joint_name = f"comsol_wk6_cid{cid}_{surface}_joint_mogi"
            out_path = os.path.join(joint_dir, f"{joint_name}.mat")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            # Stack points; each keeps its own track's LOS unit vector. The single
            # likelihood sigma = std(all LOS) treats both tracks as equally weighted, so
            # joint MCMC interval widths are NOT comparable to single-track ones.
            joint = dict(asc)
            for key in ('x_east_m', 'y_north_m', 'los_obs_m', 'losE', 'losN', 'losU'):
                joint[key] = np.concatenate([asc[key], desc[key]])
            write_export(out_path, joint, joint_name, 'joint')
            n_joint += 1
        print(f"  wrote {n_joint} joint exports")

    print(f"Done in {time.time() - t0:.1f}s. Output saved to {args.out}/")


if __name__ == '__main__':
    main()
