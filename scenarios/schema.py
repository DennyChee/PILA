#!/usr/bin/env python
# Usage:       from scenarios.schema import write_scenario, read_scenario, write_truth, read_truth, validate_scenario
#              python -m scenarios.schema --validate <scenario.npz> [--truth <truth.json>]
# Description: ONE scenario format and ONE truth format for every method in the comparison (PILA, per-epoch
#              MCMC, vbICA, later supervised).
#
#   SYNTHETIC scenario = <id>.npz in the PILA-native npz scene schema (read directly by
#   pila.datasets.preprocessing.insar_mintpy._read_npz_scene, so PILA needs no conversion):
#     x_m [W], y_m [H]                 map axes, metres, local ENU frame (source params live in the same frame)
#     los_<track>_m [T, H, W] float32  LOS displacement, METRES, + = motion TOWARD the satellite
#     incidence_deg                    scalar or [H, W], from vertical
#     heading_<track>_deg              scalar or [H, W], flight heading, deg clockwise from North
#     dates [T]                        'YYYYMMDD' acquisition dates (displacement relative to ref_date)
#     mask [H, W] bool (optional)      True = valid pixel
#     height_m [H, W] (optional)       surface elevation, metres (topography correction)
#     clean_los_<track>_m [T, H, W]    (optional, synthetic only) the noise-free signal, for scoring only
#     ref_date, ref_pixel (optional)   reference date 'YYYYMMDD'; reference pixel (row, col) or (-1, -1)
#   <track> is 'asc' or 'desc'; a scenario may carry both (joint asc+desc).
#
#   REAL scenario = NO npz. Each method reads the MintPy stack itself (user decision 2026-11-23); the manifest
#   row records the files and the preparation settings (multilook, ref_date, mask, bbox, lat0/lon0) that every
#   method must apply identically.
#
#   TRUTH (synthetic only) = <id>_truth.json, schema 'pila-truth/1':
#     {"schema": "pila-truth/1", "scenario_id": str, "tracks": [...], "dates": [T], "ref_date": str,
#      "frame": "local ENU metres; origin = scene origin of x_m/y_m",
#      "sources": [ {"model": "mogi"|"sun69"|"okada",
#                    "param_names": [...],          # FORWARD_MODELS order, see PARAM_NAMES
#                    "param_units": [...],
#                    "params": [[...P] * T],        # per-date parameters, SI units
#                    "peak_los_mm": [T]},           # this source's own peak |LOS| per date (visibility)
#                   ... K entries ],
#      "noise": {...free-form, at least "rms_mm"...}, "notes": str}
#
# Conventions (READ -- these affect interpretation):
#   * Units SI everywhere in files: metres, m^3, degrees; LOS in metres in the npz, mm only in derived numbers.
#   * Okada depth = fault CENTROID depth (as pila/physics/okada and scenarios/forward_numpy). iclassic's okadaMod
#     also uses the centroid since its 2026-09-23 restore (d = depth + W/2 sin(dip)); it was TOP-EDGE only in the
#     2026-09-21..23 copy. The MCMC gate test checks okadaMod against forward_numpy numerically.
#   * A source that is OFF on a date keeps its position with amplitude 0 (dV / opening = 0), so K is constant.
#   * LOS unit vector from incidence + heading exactly as the PILA npz reader: azimuth = 90 - heading, then
#     MintPy enu2los coefficients (scenarios.schema.los_unit_vectors).
# Date:        2026-11-23
import argparse
import json
import os

import numpy as np

TRUTH_SCHEMA = 'pila-truth/1'
TRACKS = ('asc', 'desc')
PARAM_NAMES = {                       # FORWARD_MODELS order (scenarios/forward_numpy.py)
    'mogi':  ['xcen', 'ycen', 'd', 'dV'],
    'sun69': ['xcen', 'ycen', 'depth', 'radius', 'dV'],
    'okada': ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening'],
}
PARAM_UNITS = {
    'mogi':  ['m', 'm', 'm', 'm^3'],
    'sun69': ['m', 'm', 'm', 'm', 'm^3'],
    'okada': ['m', 'm', 'm', 'deg', 'deg', 'm', 'm', 'm'],
}
AMPLITUDE_PARAM = {'mogi': 'dV', 'sun69': 'dV', 'okada': 'opening'}   # the "how much" parameter per model
DEPTH_PARAM = {'mogi': 'd', 'sun69': 'depth', 'okada': 'depth'}


# ---------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------------------------
def los_unit_vectors(incidence_deg, heading_deg):
    """
    Ground->satellite LOS unit vector (e_E, e_N, e_U), identical to the PILA npz reader.

    Inputs:  incidence_deg (scalar or array, deg from vertical); heading_deg (scalar or array, deg CW from North).
    Output:  (los_E, los_N, los_U) arrays, unit length.
    """
    inc = np.deg2rad(np.asarray(incidence_deg, dtype=np.float64))
    az = np.deg2rad(90.0 - np.asarray(heading_deg, dtype=np.float64))   # MintPy azimuthAngle of a right-looking SAR
    los_e, los_n, los_u = -np.sin(inc) * np.sin(az), np.sin(inc) * np.cos(az), np.cos(inc)
    norm = np.sqrt(los_e ** 2 + los_n ** 2 + los_u ** 2)
    return los_e / norm, los_n / norm, los_u / norm


def source_los_m(model, params_si, x_m, y_m, incidence_deg, heading_deg):
    """
    LOS field (metres, + toward satellite) of ONE source on the full grid, with the independent NumPy forward model.

    Inputs:  model ('mogi'|'sun69'|'okada'); params_si (sequence in PARAM_NAMES order, SI);
             x_m [W], y_m [H] axes (m); incidence_deg / heading_deg scalar or [H, W].
    Output:  [H, W] float64 LOS displacement, metres.
    """
    from scenarios.forward_numpy import run_model
    east_m, north_m = np.meshgrid(np.asarray(x_m, float), np.asarray(y_m, float))
    ue_mm, un_mm, uu_mm = run_model(model, east_m.ravel(), north_m.ravel(), list(params_si))
    los_e, los_n, los_u = los_unit_vectors(incidence_deg, heading_deg)
    los_e, los_n, los_u = (np.broadcast_to(c, east_m.shape).ravel() for c in (los_e, los_n, los_u))
    return ((ue_mm * los_e + un_mm * los_n + uu_mm * los_u) / 1000.0).reshape(east_m.shape)


# ---------------------------------------------------------------------------------------------------------------
# Scenario npz
# ---------------------------------------------------------------------------------------------------------------
def write_scenario(path, x_m, y_m, los_m_by_track, incidence_deg, heading_deg_by_track, dates, mask=None,
                   height_m=None, clean_los_m_by_track=None, ref_date='', ref_pixel=(-1, -1), overwrite=False):
    """
    Write a synthetic scenario npz in the PILA-native schema (see header).

    Inputs:  path; x_m [W], y_m [H] (m); los_m_by_track {track: [T, H, W] m}; incidence_deg scalar/[H, W];
             heading_deg_by_track {track: scalar/[H, W] deg}; dates [T] 'YYYYMMDD'; optional mask [H, W] bool,
             height_m [H, W] m, clean_los_m_by_track {track: [T, H, W] m}, ref_date str, ref_pixel (row, col).
    Output:  path written. Refuses to overwrite an existing file unless overwrite=True.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; not overwriting")
    arrays = dict(x_m=np.asarray(x_m, np.float64), y_m=np.asarray(y_m, np.float64),
                  incidence_deg=np.asarray(incidence_deg, np.float64),
                  dates=np.asarray([str(d).replace('-', '') for d in dates], dtype='<U8'),
                  ref_date=np.asarray(str(ref_date).replace('-', '')), ref_pixel=np.asarray(ref_pixel, np.int64))
    for track, cube in los_m_by_track.items():
        arrays[f'los_{track}_m'] = np.asarray(cube, np.float32)
        arrays[f'heading_{track}_deg'] = np.asarray(heading_deg_by_track[track], np.float64)
    for track, cube in (clean_los_m_by_track or {}).items():
        arrays[f'clean_los_{track}_m'] = np.asarray(cube, np.float32)
    if mask is not None:
        arrays['mask'] = np.asarray(mask, bool)
    if height_m is not None:
        arrays['height_m'] = np.asarray(height_m, np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    problems = validate_scenario(path)
    if problems:
        raise ValueError(f"{path} written but invalid: {problems}")
    return path


def read_scenario(path):
    """Load a scenario npz into a plain dict (arrays) plus 'tracks' (list of tracks present)."""
    with np.load(path, allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}
    data['tracks'] = [t for t in TRACKS if f'los_{t}_m' in data]
    data['dates'] = [str(d) for d in data['dates']]
    return data


def validate_scenario(path):
    """Return a list of problems with a scenario npz (empty list = valid)."""
    problems = []
    try:
        data = read_scenario(path)
    except (OSError, ValueError, KeyError) as exc:
        return [f"cannot read: {exc}"]
    for key in ('x_m', 'y_m', 'incidence_deg', 'dates'):
        if key not in data:
            problems.append(f"missing {key}")
    if not data.get('tracks'):
        problems.append("no los_<track>_m array")
        return problems
    n_rows, n_cols = data['y_m'].size, data['x_m'].size
    n_dates = len(data['dates'])
    for track in data['tracks']:
        cube = data[f'los_{track}_m']
        if cube.shape != (n_dates, n_rows, n_cols):
            problems.append(f"los_{track}_m shape {cube.shape} != (T={n_dates}, H={n_rows}, W={n_cols})")
        if f'heading_{track}_deg' not in data:
            problems.append(f"missing heading_{track}_deg")
        clean = data.get(f'clean_los_{track}_m')
        if clean is not None and clean.shape != cube.shape:
            problems.append(f"clean_los_{track}_m shape {clean.shape} != {cube.shape}")
        valid = data['mask'] if 'mask' in data else np.ones((n_rows, n_cols), bool)
        if not np.all(np.isfinite(cube[:, valid])):
            problems.append(f"los_{track}_m has non-finite values inside the mask")
        if np.nanmax(np.abs(cube)) > 50.0:                    # > 50 m of LOS: almost surely mm stored as m
            problems.append(f"los_{track}_m max |LOS| {np.nanmax(np.abs(cube)):.1f} m -- units should be METRES")
    if 'mask' in data and data['mask'].shape != (n_rows, n_cols):
        problems.append(f"mask shape {data['mask'].shape} != {(n_rows, n_cols)}")
    if any(len(d) != 8 or not d.isdigit() for d in data['dates']):
        problems.append("dates must be 'YYYYMMDD'")
    return problems


# ---------------------------------------------------------------------------------------------------------------
# Truth json
# ---------------------------------------------------------------------------------------------------------------
def make_source(model, params_per_date, peak_los_mm=None):
    """Truth entry for one source: model name, SI params per date [T, P] (PARAM_NAMES order), own peak |LOS| [T]."""
    params = np.asarray(params_per_date, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != len(PARAM_NAMES[model]):
        raise ValueError(f"{model} params must be [T, {len(PARAM_NAMES[model])}], got {params.shape}")
    entry = dict(model=model, param_names=list(PARAM_NAMES[model]), param_units=list(PARAM_UNITS[model]),
                 params=params.tolist())
    if peak_los_mm is not None:
        entry['peak_los_mm'] = [float(v) for v in peak_los_mm]
    return entry


def write_truth(path, scenario_id, tracks, dates, sources, ref_date='', noise=None, notes='', overwrite=False):
    """Write <id>_truth.json in schema pila-truth/1 (see header). Refuses to overwrite unless overwrite=True."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; not overwriting")
    truth = dict(schema=TRUTH_SCHEMA, scenario_id=scenario_id, tracks=list(tracks),
                 dates=[str(d).replace('-', '') for d in dates], ref_date=str(ref_date).replace('-', ''),
                 frame='local ENU metres; origin = scene origin of x_m/y_m', sources=list(sources),
                 noise=dict(noise or {}), notes=notes)
    problems = validate_truth(truth)
    if problems:
        raise ValueError(f"invalid truth for {scenario_id}: {problems}")
    with open(path, 'w') as handle:
        json.dump(truth, handle, indent=1)
    return path


def read_truth(path):
    with open(path) as handle:
        truth = json.load(handle)
    problems = validate_truth(truth)
    if problems:
        raise ValueError(f"{path}: {problems}")
    return truth


def validate_truth(truth):
    """Return a list of problems with a truth dict (empty list = valid)."""
    problems = []
    if truth.get('schema') != TRUTH_SCHEMA:
        problems.append(f"schema {truth.get('schema')!r} != {TRUTH_SCHEMA!r}")
    n_dates = len(truth.get('dates', []))
    if not isinstance(truth.get('sources'), list):
        problems.append("'sources' must be a list (empty list = K=0 noise-only scenario)")
    for index, source in enumerate(truth.get('sources', [])):
        model = source.get('model')
        if model not in PARAM_NAMES:
            problems.append(f"source {index}: unknown model {model!r}")
            continue
        if source.get('param_names') != PARAM_NAMES[model]:
            problems.append(f"source {index}: param_names {source.get('param_names')} != {PARAM_NAMES[model]}")
        params = np.asarray(source.get('params', []), dtype=np.float64)
        if params.shape != (n_dates, len(PARAM_NAMES[model])):
            problems.append(f"source {index}: params shape {params.shape} != ({n_dates}, {len(PARAM_NAMES[model])})")
        if 'peak_los_mm' in source and len(source['peak_los_mm']) != n_dates:
            problems.append(f"source {index}: peak_los_mm length != {n_dates}")
    return problems


def truth_arrays(truth):
    """
    Truth as arrays for scoring.

    Output: dict models [K], param_names [K][P_k], params [T, K, P_max] float64 (NaN-padded, SI),
            peak_los_mm [T, K] (NaN if not stored).
    """
    sources = truth['sources']
    n_dates, n_sources = len(truth['dates']), len(sources)
    p_max = max([len(s['param_names']) for s in sources], default=0)
    params = np.full((n_dates, n_sources, p_max), np.nan)
    peaks = np.full((n_dates, n_sources), np.nan)
    for k, source in enumerate(sources):
        values = np.asarray(source['params'], dtype=np.float64)
        params[:, k, :values.shape[1]] = values
        if 'peak_los_mm' in source:
            peaks[:, k] = source['peak_los_mm']
    return dict(models=[s['model'] for s in sources], param_names=[s['param_names'] for s in sources],
                params=params, peak_los_mm=peaks)


def main():
    parser = argparse.ArgumentParser(description='Validate a scenario npz (and optionally its truth json)')
    parser.add_argument('--validate', required=True, help='scenario .npz')
    parser.add_argument('--truth', default=None, help='truth .json (pila-truth/1)')
    args = parser.parse_args()
    problems = validate_scenario(args.validate)
    if args.truth:
        try:
            read_truth(args.truth)
        except ValueError as exc:
            problems.append(str(exc))
    print("VALID" if not problems else "INVALID:\n  " + "\n  ".join(problems))
    raise SystemExit(0 if not problems else 1)


if __name__ == '__main__':
    main()
