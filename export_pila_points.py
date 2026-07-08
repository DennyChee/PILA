#!/usr/bin/env python
# Usage:       conda activate pila
#              python export_pila_points.py [--configs configs/phys_smpl] \
#                  [--out exported] [--config-glob "*_A.json"]
# Description: Export the EXACT multilooked MintPy LOS points (and per-point LOS unit
#              vectors) that each PILA InSAR run fit, for the same target epoch PILA
#              inverted, so the classical MATLAB inversion (iclassic_insar.m) fits an
#              identical dataset. One <ConfigName>.mat is written per PILA config.
# Date:        2026-06-18
#
# Scientific conventions (READ — these affect interpretation):
#   * LOS sign: positive = motion TOWARD the satellite (range decrease), MintPy
#     convention. Per-point LOS unit vectors come from PILA's enu2los helper.
#   * Coordinates are local ENU (azimuthal-equidistant about the config lat0/lon0),
#     exported in METRES so the MATLAB forward models (which work in metres) need no
#     rescaling. LOS is exported in METRES (mm/1000) for the same reason.
#   * The target epoch is read from the PILA run's figures/*_params.txt (# epoch line),
#     so the exported field is the very same epoch PILA inverted.

import argparse
import glob
import json
import os
import time

import numpy as np
import pyproj
from scipy.io import savemat

# Reuse PILA's own loader so the exported points are byte-for-byte the points PILA trained on.
from datasets.preprocessing.insar_mintpy import load_insar_mintpy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map the config 'physics' field to a short classical-model tag.
PHYSICS_TO_MODEL = {
    'Mogi_LOS': 'mogi',
    'Okada_LOS': 'okada',
    'Sun69_LOS': 'sun69',
}

# Classical-inversion parameter order (matches the MATLAB forward models in
# iclassic_insar.m) and the PILA paras-JSON key each maps from. The third entry
# is the unit transform from the PILA paras value to the physical SI value the
# MATLAB forward models expect (metres / degrees / m / m^3).
#   'km'  : km -> m         (x/y/depth/length/width/radius)
#   'deg' : unchanged       (strike/dip)
#   'm'   : unchanged       (opening)
#   'dV'  : val*1e5 - 1e7   (PILA's dV magnitude transform -> m^3; see Physics_Mogi.rescale)
CLASSICAL_PARAMS = {
    'mogi':  [('xoff_m', 'xcen', 'km'), ('yoff_m', 'ycen', 'km'),
              ('depth_m', 'd', 'km'), ('volume_m3', 'dV', 'dV')],
    'sun69': [('xoff_m', 'xcen', 'km'), ('yoff_m', 'ycen', 'km'),
              ('depth_m', 'depth', 'km'), ('radius_m', 'radius', 'km'),
              ('volume_m3', 'dV', 'dV')],
    'okada': [('xoff_m', 'xoff', 'km'), ('yoff_m', 'yoff', 'km'),
              ('depth_m', 'depth', 'km'), ('strike_deg', 'strike', 'deg'),
              ('dip_deg', 'dip', 'deg'), ('length_m', 'length', 'km'),
              ('width_m', 'width', 'km'), ('opening_m', 'opening', 'm')],
}

# config['arch']['args'] key holding the paras-JSON path, per model.
PARAS_KEY = {'mogi': 'mogi_paras', 'okada': 'okada_paras', 'sun69': 'sun69_paras'}


def _transform_bound(value, kind, rng=None):
    """Map a PILA paras-JSON bound to the physical SI unit the MATLAB model uses.

    For dV the affine is physical = value*scale + shift; scale/shift come from the
    paras-JSON dV entry (`rng`) and default to the legacy map (1e5, -1e7). This keeps
    the classical search box identical to PILA's decoder for any dV parameterization
    (e.g. the symmetric-about-0 range in configs/mogi_paras_symdV.json).
    """
    if kind == 'km':
        return value * 1000.0          # km -> m
    if kind == 'dV':
        dv_scale = float((rng or {}).get('scale', 1e5))
        dv_shift = float((rng or {}).get('shift', -1e7))
        return value * dv_scale + dv_shift   # PILA dV magnitude transform -> m^3
    return value                       # 'deg' / 'm' : unchanged


def build_classical_bounds(model, paras):
    """
    Build the classical-inversion bounds [2 x n_par] (row 0 = lower, row 1 = upper)
    and the parameter-name list, in the MATLAB forward-model's physical units, from
    the PILA paras-JSON dict. Guarantees both methods search the IDENTICAL space.
    """
    names, lo, hi = [], [], []
    for mat_name, json_key, kind in CLASSICAL_PARAMS[model]:
        rng = paras[json_key]
        names.append(mat_name)
        lo.append(_transform_bound(rng['min'], kind, rng))
        hi.append(_transform_bound(rng['max'], kind, rng))
    return np.array([lo, hi], dtype=np.float64), names


def find_pila_epoch(config_name, saved_root):
    """
    Read the target epoch ('YYYY-MM-DD') from the most recent PILA run for this config.

    Looks for saved/**/<config_name>/<timestamp>/figures/*_params.txt and parses the
    '# epoch:' header line. Returns None if no PILA run/figure is found.
    """
    pattern = os.path.join(saved_root, '*', config_name, '*', 'figures', '*_params.txt')
    matches = sorted(glob.glob(pattern))  # timestamp dirs sort chronologically
    if not matches:
        return None, None
    params_txt = matches[-1]  # most recent run
    epoch_iso = None
    with open(params_txt, 'r') as fh:
        for line in fh:
            if line.lower().startswith('# epoch:'):
                epoch_iso = line.split(':', 1)[1].strip()
                break
    return epoch_iso, params_txt


def parse_volcano_track(config_name):
    """Split e.g. 'LaPalma_Mogi_TA60_A' -> (volcano, track). model handled separately."""
    parts = config_name.split('_')
    volcano = parts[0]
    track = parts[2] if len(parts) >= 3 else ''   # e.g. 'LaPalma_Mogi_TA60_A' -> 'TA60'
    return volcano, track


def main():
    ap = argparse.ArgumentParser(description='Export PILA multilooked LOS points for classical inversion.')
    ap.add_argument('--configs', default=os.path.join(CURRENT_DIR, 'configs', 'phys_smpl'),
                    help='Directory of PILA phys_smpl config JSONs.')
    ap.add_argument('--config-glob', default='*_A.json',
                    help='Glob within --configs to select configs to export.')
    ap.add_argument('--saved-root', default=os.path.join(CURRENT_DIR, 'saved'),
                    help='Root of PILA saved/ runs (used to read the target epoch).')
    ap.add_argument('--out', default=os.path.join(CURRENT_DIR, 'exported'),
                    help='Output directory for the per-config .mat files.')
    ap.add_argument('--only', default=None,
                    help='Optional substring filter on config name (e.g. LaPalma).')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg_paths = sorted(glob.glob(os.path.join(args.configs, args.config_glob)))
    if args.only:
        cfg_paths = [p for p in cfg_paths if args.only in os.path.basename(p)]
    if not cfg_paths:
        raise FileNotFoundError(f"No configs matched {args.config_glob} in {args.configs}")

    print(f"[0/{len(cfg_paths)}] Exporting {len(cfg_paths)} config(s) to {args.out}/")
    n_ok, n_skip = 0, 0

    for i, cfg_path in enumerate(cfg_paths, 1):
        config_name = os.path.splitext(os.path.basename(cfg_path))[0]
        print(f"\n[{i}/{len(cfg_paths)}] {config_name}")
        t0 = time.time()

        with open(cfg_path, 'r') as fh:
            config = json.load(fh)

        # --- Resolve model + InSAR inputs from the config ---
        physics = config['arch']['args'].get('physics', '')
        model = PHYSICS_TO_MODEL.get(physics)
        if model is None:
            print(f"  SKIP: physics '{physics}' is not a LOS InSAR source model.")
            n_skip += 1
            continue
        ins = config['arch']['args'].get('insar')
        if ins is None:
            print("  SKIP: config has no arch.args.insar block.")
            n_skip += 1
            continue

        volcano, track = parse_volcano_track(config_name)

        # --- Classical-inversion bounds from the SAME PILA paras JSON (fair search space) ---
        paras_rel = config['arch']['args'].get(PARAS_KEY[model])
        paras_path = paras_rel if os.path.isabs(paras_rel) else os.path.join(CURRENT_DIR, paras_rel)
        with open(paras_path, 'r') as fh:
            paras = json.load(fh)
        bounds, param_names = build_classical_bounds(model, paras)

        # --- Target epoch: the one PILA inverted (from its params.txt) ---
        epoch_iso, params_txt = find_pila_epoch(config_name, args.saved_root)
        if epoch_iso is None:
            print(f"  WARNING: no PILA run/params.txt found for {config_name}; "
                  f"defaulting to the LAST epoch of the time series.")

        # --- Load the multilooked LOS field exactly as PILA does ---
        for key in ('timeseries', 'geometry'):
            if not os.path.exists(ins[key]):
                raise FileNotFoundError(f"{config_name}: {key} not found: {ins[key]}")
        d = load_insar_mintpy(
            ins['timeseries'], ins['geometry'], ins.get('mask'),
            ins['lat0'], ins['lon0'],
            multilook=ins.get('multilook', 20),
            coh_valid_frac=ins.get('coh_valid_frac', 0.5),
            bbox=ins.get('bbox'), verbose=False)
        print(f"  Loaded: N points={d.n_points}, n_epoch={len(d.dates)}, "
              f"dates[{d.dates[0]} .. {d.dates[-1]}]")

        # --- Select the epoch index ---
        if epoch_iso is not None and epoch_iso in d.dates:
            epoch_idx = d.dates.index(epoch_iso)
        elif epoch_iso is not None:
            # Nearest date (defensive; epoch should normally match exactly).
            tgt = np.datetime64(epoch_iso)
            all_dt = np.array([np.datetime64(s) for s in d.dates])
            epoch_idx = int(np.argmin(np.abs(all_dt - tgt)))
            print(f"  WARNING: epoch {epoch_iso} not exact; using nearest {d.dates[epoch_idx]}")
        else:
            epoch_idx = len(d.dates) - 1
        epoch_used = d.dates[epoch_idx]
        epoch_yyyymmdd = epoch_used.replace('-', '')
        print(f"  Epoch: PILA={epoch_iso}  used={epoch_used} (index {epoch_idx})")

        # --- Assemble per-point arrays (drop any non-finite LOS at this epoch) ---
        los_mm = d.los_points_mm[epoch_idx]                  # [N], mm
        x_east_m = d.xE_pts.astype(np.float64) * 1000.0      # km -> m
        y_north_m = d.yN_pts.astype(np.float64) * 1000.0
        finite = np.isfinite(los_mm)
        if not finite.all():
            print(f"  Dropping {int((~finite).sum())} non-finite LOS points "
                  f"({int(finite.sum())} kept).")
        los_obs_m = (los_mm[finite] / 1000.0).astype(np.float64)   # mm -> m
        x_east_m = x_east_m[finite]
        y_north_m = y_north_m[finite]
        losE = d.losE_pts.astype(np.float64)[finite]
        losN = d.losN_pts.astype(np.float64)[finite]
        losU = d.losU_pts.astype(np.float64)[finite]

        # --- Recover lon/lat per point (inverse of the local-ENU aeqd projection) ---
        aeqd = f"+proj=aeqd +lat_0={ins['lat0']} +lon_0={ins['lon0']} +datum=WGS84 +units=m +no_defs"
        to_lonlat = pyproj.Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
        lon, lat = to_lonlat.transform(x_east_m, y_north_m)

        # --- Write the .mat the MATLAB driver reads ---
        out_mat = os.path.join(args.out, f"{config_name}.mat")
        savemat(out_mat, {
            'x_east_m': x_east_m.reshape(-1, 1),
            'y_north_m': y_north_m.reshape(-1, 1),
            'lon': np.asarray(lon).reshape(-1, 1),
            'lat': np.asarray(lat).reshape(-1, 1),
            'los_obs_m': los_obs_m.reshape(-1, 1),
            'losE': losE.reshape(-1, 1),
            'losN': losN.reshape(-1, 1),
            'losU': losU.reshape(-1, 1),
            'lat0': float(ins['lat0']),
            'lon0': float(ins['lon0']),
            'multilook': int(ins.get('multilook', 20)),
            'epoch_yyyymmdd': epoch_yyyymmdd,
            'epoch_iso': epoch_used,
            'model': model,
            'volcano': volcano,
            'track': track,
            'config_name': config_name,
            'n_points': int(x_east_m.size),
            'bounds': bounds,                          # [2 x n_par] lower; upper (SI units)
            'param_names': np.array(param_names, dtype=object).reshape(1, -1),
        }, do_compression=True)
        print(f"  Saved: {out_mat}  (N={x_east_m.size}, model={model})  "
              f"[{time.time() - t0:.1f}s]")
        n_ok += 1

    print(f"\nDone. Exported {n_ok} config(s), skipped {n_skip}. Output in {args.out}/")


if __name__ == '__main__':
    main()
