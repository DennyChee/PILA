#!/usr/bin/env python
# Usage:       python -m compare.score --run runs/<method>/<scenario>/<run_id> --scenario-id <id> \
#                  [--manifest scenarios/manifest.csv]
#              (library: score_run(truth, scenario, run_output) -> (summary dict, per source-date DataFrame))
# Description: The ONE scorer for every method (PILA, per-epoch MCMC, vbICA, later supervised): the same rule is
#              applied to every method's params.npz / recon.npz (methods/contract.py). Truth is used ONLY here.
#              Generalises the legacy non-static K=2 scorer (synthetic/multisource/nonstatic_scoring.py, fixed
#              2026-10-26) to any K and to Mogi / Sun69 / Okada; for K=2 Mogi it reproduces the legacy scores
#              exactly (tests/unit/test_score_vs_legacy.py).
# Rules (defaults = the legacy rule; final thresholds are a Phase 4 decision, all are arguments):
#   * a true source is ACTIVE on a date if its own peak |LOS| >= max(2 x noise RMS, 1 mm);
#   * per date, fitted slots are matched to true sources by horizontal distance: first the ACTIVE true sources
#     (Hungarian over all slots), then the inactive ones over the remaining slots -- so a quiet source never
#     takes the good slot; slot labels may swap between dates (per-date vs per-date, no tracking);
#   * finiteness is judged PER SLOT; non-finite slots are excluded from matching; a true source left without
#     a slot is MISSED and, if active, fails (so finding 1 of 2 sources can never score 100 %);
#   * an active source-date SUCCEEDS if location error < 1.5 km, |depth error| < 50 % and
#     |amplitude error| < 50 % (amplitude = dV for Mogi/Sun69, opening for Okada; a sign flip fails).
#     If the fitted model differs from the true model the amplitudes are not comparable: success then uses
#     location + depth only and the row is flagged model_mismatch;
#   * a slot matched to an INACTIVE source, or left unmatched (more slots than sources), is a FALSE DETECTION
#     if its own peak |LOS| would be detectable (>= the same threshold);
#   * run success = >= 80 % of active source-dates succeed; for K = 0 (noise only) = no false detection.
#   * true-source visibility = stored peak_los_mm, else computed from the truth parameters.
#   Reconstruction metrics: see recon_metrics(); rank on rms_phys_vs_clean (per-date-demeaned), not on recon-vs-data.
# Conventions: SI in files; this table reports km, %, mm. Okada depth = centroid depth.
# Date:        2026-11-23
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scenarios.schema import AMPLITUDE_PARAM, DEPTH_PARAM, PARAM_NAMES, read_scenario, read_truth, source_los_m  # noqa: E402
from methods.contract import read_method_output  # noqa: E402

DEFAULT_RULES = dict(loc_tol_km=1.5, depth_tol_frac=0.5, amp_tol_frac=0.5, active_noise_factor=2.0,
                     active_floor_mm=1.0, run_success_frac=0.8)


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------
def _horizontal_m(model, params):
    """(x, y) metres of a source from its parameter vector (first two entries for every model)."""
    return np.asarray(params[:2], dtype=np.float64)


def own_peak_mm(model, params, scenario, track):
    """Peak |LOS| (mm) of ONE fitted source on the scenario grid (independent NumPy forward model, unreferenced)."""
    values = np.asarray(params[:len(PARAM_NAMES[model])], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return float('nan')
    if values[PARAM_NAMES[model].index(AMPLITUDE_PARAM[model])] == 0:
        return 0.0
    try:
        field_m = source_los_m(model, values, scenario['x_m'], scenario['y_m'], scenario['incidence_deg'],
                               scenario[f'heading_{track}_deg'])
    except (ValueError, FloatingPointError):
        return float('nan')
    return float(np.nanmax(np.abs(field_m))) * 1e3


def match_slots(true_xy, fit_xy, active):
    """
    One date's assignment of fitted slots to true sources (active sources first).

    Inputs:  true_xy [K_true, 2], fit_xy [K_fit, 2] (finite), active [K_true] bool.
    Output:  list of (true index, slot index) pairs; len = min(K_true, K_fit).
    """
    pairs, free_slots = [], list(range(len(fit_xy)))
    for group in (np.flatnonzero(active), np.flatnonzero(~np.asarray(active))):
        if len(group) == 0 or not free_slots:
            continue
        cost = np.linalg.norm(true_xy[group][:, None, :] - fit_xy[free_slots][None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        pairs += [(int(group[r]), int(free_slots[c])) for r, c in zip(rows, cols)]
        taken = {free_slots[c] for c in cols}
        free_slots = [s for s in free_slots if s not in taken]
    return pairs


def align_to_scenario(cube, recon_x, recon_y, scenario):
    """Flip a recon cube [T, H, W] so its axes match the scenario's (x_m, y_m); error if grids differ."""
    out = cube
    if not np.allclose(recon_y, scenario['y_m']):
        if np.allclose(recon_y[::-1], scenario['y_m']):
            out = out[:, ::-1, :]
        else:
            raise ValueError("recon y axis does not match the scenario grid")
    if not np.allclose(recon_x, scenario['x_m']):
        if np.allclose(recon_x[::-1], scenario['x_m']):
            out = out[:, :, ::-1]
        else:
            raise ValueError("recon x axis does not match the scenario grid")
    return out


# ---------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------
def true_peaks_mm(truth, scenario, track):
    """
    Own peak |LOS| (mm) of every true source on every date [T, K]: stored peak_los_mm when present, otherwise
    computed from the truth parameters with the independent NumPy forward model (never silently NaN).
    """
    n_dates, sources = len(truth['dates']), truth['sources']
    peaks = np.full((n_dates, len(sources)), np.nan)
    for k, source in enumerate(sources):
        if 'peak_los_mm' in source:
            peaks[:, k] = source['peak_los_mm']
        else:
            peaks[:, k] = [own_peak_mm(source['model'], source['params'][t], scenario, track) for t in range(n_dates)]
    if not np.all(np.isfinite(peaks)):
        raise ValueError(f"{truth['scenario_id']}: non-finite true peak |LOS| -- cannot decide which sources are active")
    return peaks


def score_run(truth, scenario, run_output, rules=None):
    """
    Score one method run against the per-date truth.

    Inputs:  truth (pila-truth/1 dict); scenario (read_scenario dict); run_output (read_method_output dict);
             rules (dict overriding DEFAULT_RULES).
    Output:  (summary dict, DataFrame: one row per (date, true source) -- matched or MISSED -- plus one row per
             unmatched fitted slot).
    """
    rules = dict(DEFAULT_RULES, **(rules or {}))
    track = truth['tracks'][0]
    fit = np.asarray(run_output['params']['params'], dtype=np.float64)              # [T, K_fit, P_max]
    fit_models = run_output['params']['models']
    if run_output['params']['dates'] != truth['dates']:
        raise ValueError("method dates differ from truth dates")
    n_dates = len(truth['dates'])
    noise_rms_mm = float(truth['noise'].get('rms_mm', 0.0))
    threshold_mm = max(rules['active_noise_factor'] * noise_rms_mm, rules['active_floor_mm'])
    true_sources = truth['sources']
    peaks_all = true_peaks_mm(truth, scenario, track) if true_sources else np.zeros((n_dates, 0))
    rows = []
    for t in range(n_dates):
        true_params = [np.asarray(s['params'][t], dtype=np.float64) for s in true_sources]
        true_peaks = peaks_all[t]
        active = true_peaks >= threshold_mm
        # finiteness is judged PER SLOT: a NaN slot (e.g. a missing vbICA group) is excluded from matching
        slot_finite = np.array([bool(np.all(np.isfinite(fit[t, k, :len(PARAM_NAMES[m])])))
                                for k, m in enumerate(fit_models)], dtype=bool)
        usable = np.flatnonzero(slot_finite)
        true_xy = np.array([_horizontal_m(s['model'], p) for s, p in zip(true_sources, true_params)]).reshape(-1, 2)
        fit_xy = np.array([_horizontal_m(fit_models[k], fit[t, k]) for k in usable]).reshape(-1, 2)
        pairs = [(src, int(usable[j])) for src, j in match_slots(true_xy, fit_xy, active)] \
            if len(true_sources) and len(usable) else []
        matched_sources = {src for src, _ in pairs}
        matched_slots = {slot for _, slot in pairs}
        for src, slot in pairs:
            true_model, fit_model = true_sources[src]['model'], fit_models[slot]
            truth_row, fit_row = true_params[src], fit[t, slot]
            loc_km = float(np.hypot(*(fit_row[:2] - truth_row[:2]))) / 1e3
            true_depth = truth_row[PARAM_NAMES[true_model].index(DEPTH_PARAM[true_model])]
            fit_depth = fit_row[PARAM_NAMES[fit_model].index(DEPTH_PARAM[fit_model])]
            depth_err = float((fit_depth - true_depth) / true_depth)
            mismatch = true_model != fit_model
            if mismatch:
                amp_err = float('nan')
            else:
                true_amp = truth_row[PARAM_NAMES[true_model].index(AMPLITUDE_PARAM[true_model])]
                fit_amp = fit_row[PARAM_NAMES[fit_model].index(AMPLITUDE_PARAM[fit_model])]
                amp_err = float((fit_amp - true_amp) / abs(true_amp)) if true_amp != 0 else float('inf')
            amp_ok = True if mismatch else abs(amp_err) < rules['amp_tol_frac']
            success = bool(loc_km < rules['loc_tol_km'] and abs(depth_err) < rules['depth_tol_frac'] and amp_ok)
            slot_peak = own_peak_mm(fit_model, fit_row, scenario, track) if not active[src] else float('nan')
            rows.append(dict(date_index=t, date=truth['dates'][t], source=int(src), slot=int(slot),
                             true_model=true_model, fit_model=fit_model, model_mismatch=mismatch, missed=False,
                             active=bool(active[src]), true_peak_mm=float(true_peaks[src]),
                             snr=float('inf') if noise_rms_mm == 0 else float(true_peaks[src] / noise_rms_mm),
                             loc_err_km=loc_km, depth_err_pct=100 * depth_err, amp_err_pct=100 * amp_err,
                             success=success if active[src] else None, slot_peak_mm=slot_peak,
                             false_detection=bool(slot_peak >= threshold_mm) if not active[src] else None,
                             nonfinite_fit=False))
        for src in sorted(set(range(len(true_sources))) - matched_sources):
            # MISSED: no usable slot left for this true source (fewer / non-finite slots) -> fails if active
            rows.append(dict(date_index=t, date=truth['dates'][t], source=int(src), slot=-1,
                             true_model=true_sources[src]['model'], fit_model='', model_mismatch=False, missed=True,
                             active=bool(active[src]), true_peak_mm=float(true_peaks[src]),
                             snr=float('inf') if noise_rms_mm == 0 else float(true_peaks[src] / noise_rms_mm),
                             loc_err_km=np.nan, depth_err_pct=np.nan, amp_err_pct=np.nan,
                             success=False if active[src] else None, slot_peak_mm=np.nan, false_detection=None,
                             nonfinite_fit=bool(len(usable) < len(fit_models))))
        for slot in sorted(set(usable.tolist()) - matched_slots):          # more usable slots than true sources
            slot_peak = own_peak_mm(fit_models[slot], fit[t, slot], scenario, track)
            rows.append(dict(date_index=t, date=truth['dates'][t], source=-1, slot=int(slot), true_model='',
                             fit_model=fit_models[slot], model_mismatch=False, missed=False, active=False,
                             true_peak_mm=0.0, snr=0.0, loc_err_km=np.nan, depth_err_pct=np.nan, amp_err_pct=np.nan,
                             success=None, slot_peak_mm=slot_peak, false_detection=bool(slot_peak >= threshold_mm),
                             nonfinite_fit=False))
    table = pd.DataFrame(rows)
    active_rows = table[table['active']] if len(table) else table
    inactive_rows = table[(~table['active']) & (~table['missed'])] if len(table) else table
    frac = float(active_rows['success'].astype(bool).mean()) if len(active_rows) else float('nan')
    n_false = int(inactive_rows['false_detection'].astype(bool).sum()) if len(inactive_rows) else 0
    # Medians over matched (non-missed) active rows; missed / non-finite rows still count as failures in success_frac.
    finite_active = active_rows[~active_rows['missed'].astype(bool)] if len(active_rows) else active_rows
    if len(true_sources) == 0:
        run_success = n_false == 0                      # K = 0 (noise only): success = no false detection
    else:
        run_success = bool(frac >= rules['run_success_frac']) if len(active_rows) else False
    summary = dict(scenario_id=truth['scenario_id'], method=run_output['meta'].get('method'),
                   run_id=run_output['meta'].get('run_id'), n_dates=n_dates, n_true_sources=len(true_sources),
                   n_fit_slots=len(fit_models), k_fit_equals_k_true=len(fit_models) == len(true_sources),
                   active_threshold_mm=threshold_mm, noise_rms_mm=noise_rms_mm,
                   n_active=int(len(active_rows)),
                   n_success=int(active_rows['success'].astype(bool).sum()) if len(active_rows) else 0,
                   n_missed_active=int(active_rows['missed'].astype(bool).sum()) if len(active_rows) else 0,
                   success_frac=frac, run_success=run_success,
                   n_inactive=int(len(inactive_rows)), n_false_detection=n_false,
                   median_loc_err_km=float(finite_active['loc_err_km'].median()) if len(finite_active) else float('nan'),
                   median_abs_depth_err_pct=float(finite_active['depth_err_pct'].abs().median()) if len(finite_active) else float('nan'),
                   median_abs_amp_err_pct=float(finite_active['amp_err_pct'].abs().median()) if len(finite_active) else float('nan'),
                   n_nonfinite_dates=int(sum(not np.all(np.isfinite(fit[t])) for t in range(n_dates))),
                   rules=rules)
    summary.update(recon_metrics(scenario, run_output, track))
    summary['timing'] = run_output['timing']
    return summary, table


def recon_metrics(scenario, run_output, track):
    """
    Reconstruction metrics (mm) over valid pixels and FINITE dates; the counts are reported so methods scored on
    different subsets are visible:
      rms_recon_vs_data_mm   method's full reconstruction vs data (NOT comparable across methods: PILA includes
                             its low-rank residual, vbICA drops noise ICs) -- report, do not rank on it;
      rms_phys_vs_clean_mm   physics-only field vs the clean signal, both with their PER-DATE MEAN REMOVED, so a
                             method's offset convention (free offset / residual / vbICA centering) cannot help or hurt.
    """
    recon = run_output['recon']
    data = scenario[f'los_{track}_m']
    valid = scenario['mask'] if 'mask' in scenario else np.ones(data.shape[1:], bool)
    clean = scenario.get(f'clean_los_{track}_m')
    out = {}

    def demean(cube):
        return cube - np.nanmean(cube[:, valid], axis=1)[:, None, None]

    if f'los_{track}_m' in recon:
        cube = align_to_scenario(recon[f'los_{track}_m'], recon['x_m'], recon['y_m'], scenario)
        finite_dates = np.all(np.isfinite(cube[:, valid]), axis=1)
        out['n_finite_dates_recon'] = int(finite_dates.sum())
        out['rms_recon_vs_data_mm'] = float(np.sqrt(np.mean((cube - data)[finite_dates][:, valid] ** 2)) * 1e3) \
            if finite_dates.any() else float('nan')
    if f'phys_los_{track}_m' in recon and clean is not None:
        cube = align_to_scenario(recon[f'phys_los_{track}_m'], recon['x_m'], recon['y_m'], scenario)
        finite_dates = np.all(np.isfinite(cube[:, valid]), axis=1)
        out['n_finite_dates_phys'] = int(finite_dates.sum())
        out['rms_phys_vs_clean_mm'] = float(np.sqrt(np.mean(
            (demean(cube) - demean(clean))[finite_dates][:, valid] ** 2)) * 1e3) if finite_dates.any() else float('nan')
    return out


def score_from_paths(run_dir, truth_path, scenario_path, rules=None):
    """Convenience wrapper: read files, score, return (summary, table)."""
    return score_run(read_truth(truth_path), read_scenario(scenario_path), read_method_output(run_dir), rules)


def main():
    from scenarios.manifest import DEFAULT_MANIFEST, load_manifest, resolve_row
    parser = argparse.ArgumentParser(description='Score one method run against the scenario truth')
    parser.add_argument('--run', required=True, help='method run directory (methods/contract.py layout)')
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    rows = {row['scenario_id']: resolve_row(row) for row in load_manifest(args.manifest)}
    row = rows[args.scenario_id]
    if row['kind'] != 'synthetic':
        raise ValueError("real scenarios have no truth; only reconstruction metrics apply (compare.py)")
    print(f"[1/2] Scoring {args.run} against {args.scenario_id} ...")
    summary, table = score_from_paths(args.run, row['truth_path'], row['scenario_path'])
    print("[2/2] Writing score.json + score_source_dates.csv into the run directory ...")
    with open(os.path.join(args.run, 'score.json'), 'w') as handle:
        json.dump(summary, handle, indent=2, default=float)
    table.to_csv(os.path.join(args.run, 'score_source_dates.csv'), index=False)
    print(f"  success {summary['n_success']}/{summary['n_active']} active source-dates "
          f"({100 * summary['success_frac']:.1f} %), false detections {summary['n_false_detection']}, "
          f"recon-vs-data RMS {summary.get('rms_recon_vs_data_mm', float('nan')):.2f} mm")
    print(f"Done. Output saved to {args.run}")


if __name__ == '__main__':
    main()
