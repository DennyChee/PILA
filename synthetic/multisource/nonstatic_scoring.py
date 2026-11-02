#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.nonstatic_scoring --self-test
#              (library: score_run(params_per_date, truth) -> (run summary dict, per source-date DataFrame))
# Description: Shared per-date scorer for the non-static K=2 albums (make_nonstatic_albums.py); used IDENTICALLY for
#              PILA and the classical per-date fit, so both are judged by the same rule. Truth is used only here.
#              Rules (fixed 2026-10-26 before any inversion was run):
#                * a true source is ACTIVE on a date if its own peak |LOS| >= max(2 x album noise RMS, 1 mm)
#                  (clean albums: >= 1 mm);
#                * per date the 2 fitted slots are matched to the 2 true sources by horizontal distance (Hungarian),
#                  so slot labels may swap between dates (per-date vs per-date, no tracking); if exactly ONE source
#                  is active that date, it is matched first to its nearest slot and the inactive one gets the other;
#                * non-finite fitted parameters on a date make that date's sources fail (no crash);
#                * an active source-date SUCCEEDS if location error < 1.5 km, |depth error| < 50 % and
#                  |dV - dV_true| / |dV_true| < 50 % (a dV sign flip therefore fails);
#                * a slot matched to an INACTIVE source is a FALSE DETECTION if its own peak |LOS| would be detectable
#                  (>= the same activity threshold) -- PILA / classical claim a visible source the data do not have;
#                * run success = >= 80 % of active source-dates succeed.
#              Per source-date SNR = own peak |LOS| / album noise RMS (inf for clean albums) for pooled SNR50 fits.
#              Also: per-date minimum separation / shallower depth (Pascal ratio) of the true pair.
# Notes:       Truth positions / dV describe the UNreferenced source fields; the albums are referenced to one pixel
#              (field there <= 5 % of each source's peak, checked by qa_nonstatic_albums.py), so a fitted Mogi with
#              an offset (classical) or PILA's residual absorbs a constant of that size. The 50 % dV tolerance is far
#              larger than this.
#              Units: km, m^3, LOS mm (+ toward satellite).
# Date:        2026-10-26

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from synthetic.forward_numpy import mogi
from synthetic.multisource.make_pila_stack import load_ta69_geometry

LOC_TOL_KM, DEPTH_TOL_FRAC, DV_TOL_FRAC = 1.5, 0.5, 0.5
ACTIVE_NOISE_FACTOR, ACTIVE_FLOOR_MM = 2.0, 1.0
RUN_SUCCESS_FRAC = 0.8

_GRID = {}


def _grid():
    """TA69 separable grid (km-free: metres) + LOS unit vectors, cached."""
    if not _GRID:
        grid = load_ta69_geometry()['grid']
        _GRID.update({k: np.asarray(grid[k], dtype=np.float64) for k in ('xE_m', 'yN_m', 'losE', 'losN', 'losU')})
    return _GRID


def own_peak_mm(x_km, y_km, depth_km, dV_m3):
    """Peak |LOS| (mm) of one Mogi source on the album grid (numpy forward model)."""
    grid = _grid()
    if dV_m3 == 0 or not np.isfinite(dV_m3) or depth_km <= 0:
        return 0.0 if dV_m3 == 0 else float('nan')
    uE, uN, uU = mogi(grid['xE_m'], grid['yN_m'], x_km * 1e3, y_km * 1e3, depth_km * 1e3, dV_m3)
    return float(np.abs(grid['losE'] * uE + grid['losN'] * uN + grid['losU'] * uU).max())


def truth_arrays(truth):
    """Per-date truth [T, 2, 4] (x km, y km, depth km, dV m^3) and own peaks [T, 2] (mm)."""
    sources = truth['sources_per_date']
    params = np.stack([np.stack([src['xcen_km'], src['ycen_km'], src['depth_km'], src['dV_m3']], axis=-1)
                       for src in sources], axis=1)
    peaks = np.stack([src['own_peak_mm'] for src in sources], axis=1)
    return params, peaks


def active_threshold_mm(truth):
    """Own-peak threshold (mm) above which a source counts as detectable on a date."""
    return max(ACTIVE_NOISE_FACTOR * truth['noise_rms_mm'], ACTIVE_FLOOR_MM)


def pascal_per_date(truth):
    """Per-date 3-D separation / shallower depth of the true pair [T]."""
    params, _ = truth_arrays(truth)
    dist_km = np.linalg.norm(params[:, 0, :3] - params[:, 1, :3], axis=-1)
    return dist_km / np.minimum(params[:, 0, 2], params[:, 1, 2])


def match_slots(true_date, fit_date, active_now):
    """
    One date's slot-to-source assignment (shared by the scorer and every figure, so they always agree).

    Inputs : true_date [2, 4], fit_date [2, 4] (finite), active_now [2] bool (own peak >= activity threshold).
    Output : (true_idx, slot_idx) arrays: true source true_idx[k] is matched to fitted slot slot_idx[k].
    """
    cost = np.linalg.norm(true_date[:, None, :2] - fit_date[None, :, :2], axis=-1)
    if np.sum(active_now) == 1:
        # one detectable source: give it its nearest slot first, so the quiet source cannot take the good slot
        src_active = int(np.argmax(active_now))
        slot_best = int(np.argmin(cost[src_active]))
        return np.array([src_active, 1 - src_active]), np.array([slot_best, 1 - slot_best])
    return linear_sum_assignment(cost)


def score_run(params_per_date, truth):
    """
    Score one run's per-date K=2 fit against the per-date truth.

    Inputs : params_per_date [T, 2, 4] array (x km, y km, depth km, dV m^3) -- fitted slots in any order per date.
             truth dict from <album>_truth.json.
    Output : (summary dict, DataFrame with one row per (date, true source)).
    """
    fit = np.asarray(params_per_date, dtype=np.float64).copy()
    bad_dates = ~np.isfinite(fit).all(axis=(1, 2))
    fit[bad_dates] = np.array([1e6, 1e6, 1e6, 0.0])      # far away + zero dV: those dates score as failures
    true_params, true_peaks = truth_arrays(truth)
    if fit.shape != true_params.shape:
        raise ValueError(f'fit shape {fit.shape} != truth shape {true_params.shape}')
    threshold_mm = active_threshold_mm(truth)
    noise_rms_mm = truth['noise_rms_mm']
    pascal = pascal_per_date(truth)
    rows = []
    for t in range(fit.shape[0]):
        true_idx, slot_idx = match_slots(true_params[t], fit[t], true_peaks[t] >= threshold_mm)
        for src, slot in zip(true_idx, slot_idx):
            truth_row, fit_row = true_params[t, src], fit[t, slot]
            active = bool(true_peaks[t, src] >= threshold_mm)
            loc_km = float(np.hypot(*(fit_row[:2] - truth_row[:2])))
            depth_err = float((fit_row[2] - truth_row[2]) / truth_row[2])
            dv_err = float((fit_row[3] - truth_row[3]) / abs(truth_row[3])) if truth_row[3] != 0 else float('inf')
            success = bool(loc_km < LOC_TOL_KM and abs(depth_err) < DEPTH_TOL_FRAC and abs(dv_err) < DV_TOL_FRAC)
            slot_peak = own_peak_mm(*fit_row) if not active else float('nan')
            rows.append(dict(date_index=t, date=truth['dates'][t], tau=truth['tau'][t], source=int(src), slot=int(slot),
                             active=active, true_peak_mm=float(true_peaks[t, src]),
                             snr=float('inf') if noise_rms_mm == 0 else float(true_peaks[t, src] / noise_rms_mm),
                             loc_err_km=loc_km, depth_err_pct=100 * depth_err, dV_err_pct=100 * dv_err,
                             success=success if active else None, slot_peak_mm=slot_peak,
                             false_detection=bool(slot_peak >= threshold_mm) if not active else None,
                             pascal=float(pascal[t]), nonfinite_fit=bool(bad_dates[t])))
    table = pd.DataFrame(rows)
    active_rows = table[table['active']]
    inactive_rows = table[~table['active']]
    frac = float(active_rows['success'].astype(bool).mean()) if len(active_rows) else float('nan')
    summary = dict(album=truth['album'], scenario=truth['scenario'], layout_set=truth['layout_set'],
                   noise_scale=truth['noise_scale'], album_snr=truth['snr_peak_over_rms'],
                   active_threshold_mm=threshold_mm, n_active=len(active_rows),
                   n_success=int(active_rows['success'].astype(bool).sum()), success_frac=frac,
                   run_success=bool(frac >= RUN_SUCCESS_FRAC) if len(active_rows) else False,
                   n_inactive=len(inactive_rows),
                   n_false_detection=int(inactive_rows['false_detection'].astype(bool).sum()) if len(inactive_rows) else 0,
                   median_loc_err_km=float(active_rows['loc_err_km'].median()) if len(active_rows) else float('nan'),
                   median_abs_depth_err_pct=float(active_rows['depth_err_pct'].abs().median()) if len(active_rows) else float('nan'),
                   median_abs_dV_err_pct=float(active_rows['dV_err_pct'].abs().median()) if len(active_rows) else float('nan'),
                   min_pascal=float(pascal.min()))
    return summary, table


def self_test(albums_dir):
    """Scorer checks on the real truth files: truth -> 100 %, swapped slots -> same, a 2 km shift -> 0 %."""
    import glob
    paths = sorted(glob.glob(os.path.join(albums_dir, '*_truth.json')))
    paths = [p for p in paths if '_nonoise' not in p]
    print(f"[1/2] Self-test on {len(paths)} truth files...")
    failures = []
    for path in paths:
        truth = json.load(open(path))
        params, _ = truth_arrays(truth)
        summary, table = score_run(params, truth)
        swapped, _ = score_run(params[:, ::-1], truth)
        shifted = params.copy(); shifted[..., 0] += 2.0
        shift_summary, _ = score_run(shifted, truth)
        # own_peak_mm must reproduce the generator's stored peaks
        peak_check = own_peak_mm(*params[-1, 0]) / truth['sources_per_date'][0]['own_peak_mm'][-1]
        broken = params.copy(); broken[5, 0, 0] = np.nan
        broken_summary, broken_table = score_run(broken, truth)
        ok_nan = broken_table.loc[broken_table['date_index'] == 5, 'success'].dropna().eq(False).all()
        ok = (ok_nan and summary['success_frac'] == 1.0 and summary['n_false_detection'] == 0 and swapped == summary
              and shift_summary['n_success'] == 0 and abs(peak_check - 1) < 1e-6 and summary['n_active'] > 0)
        if not ok:
            failures.append(truth['album'])
    print(f"[2/2] {len(paths) - len(failures)}/{len(paths)} pass" + (f"; FAIL {failures}" if failures else ""))
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--albums-dir', default='synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2')
    args = parser.parse_args()
    if args.self_test:
        if not self_test(args.albums_dir):
            raise SystemExit(1)
        print("Done. Self-test passed.")


if __name__ == '__main__':
    main()
