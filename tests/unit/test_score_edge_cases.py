#!/usr/bin/env python
# Usage:       python tests/unit/test_score_edge_cases.py [--scenario runs/_pilot_scenarios/G1_N4_moving_n0000.npz]
# Description: Edge cases of the shared scorer (compare/score.py) found in the 2026-11-23 review, on a noise-free
#              K=2 scenario whose truth is exact. Each check builds a fake method output from the TRUTH itself
#              (so the expected score is known) and prints PASS/FAIL:
#                1 perfect K=2 fit                         -> every active source-date succeeds
#                2 K_fit = 1 (finds one source only)       -> the other source is MISSED and fails (<= ~50 %)
#                3 one slot NaN on every date              -> the finite slot's source still succeeds; other missed
#                4 truth without peak_los_mm               -> peaks computed from params == stored peaks
#                5 K = 0 truth, zero-amplitude slot        -> run_success True, no false detection
#                6 K = 0 truth, strong slot                -> false detections, run_success False
#                7 swapped slot order                      -> identical score (matching is order-free)
# Date:        2026-11-23
import argparse
import copy
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from compare.score import score_run, true_peaks_mm  # noqa: E402
from scenarios.schema import read_scenario, read_truth, truth_arrays  # noqa: E402


def fake_output(params, models, dates):
    return dict(params=dict(params=np.asarray(params, float), models=list(models), dates=list(dates)),
                recon={}, timing={}, meta=dict(method='test', run_id='x'))


def main():
    parser = argparse.ArgumentParser(description='Scorer edge cases')
    parser.add_argument('--scenario', default='runs/_pilot_scenarios/G1_N4_moving_n0000.npz')
    args = parser.parse_args()
    scenario = read_scenario(args.scenario)
    truth = read_truth(args.scenario.replace('.npz', '_truth.json'))
    arrays = truth_arrays(truth)
    params, dates = arrays['params'], truth['dates']
    results = []

    summary, _ = score_run(truth, scenario, fake_output(params, ['mogi', 'mogi'], dates))
    results.append(('1 perfect K=2 fit succeeds everywhere', summary['success_frac'] == 1.0 and summary['n_missed_active'] == 0,
                    f"success {summary['n_success']}/{summary['n_active']}"))

    summary, table = score_run(truth, scenario, fake_output(params[:, :1], ['mogi'], dates))
    # every ACTIVE row of the absent source must fail: missed, or (on a date where only it is active) matched to
    # the one slot that sits at the other source's position (active sources are matched first)
    source1 = table[(table.source == 1) & table.active]
    source0 = table[(table.source == 0) & table.active]
    ok = (not source1['success'].astype(bool).any()) and bool(source0['success'].astype(bool).all()) \
        and summary['n_missed_active'] > 0 and not summary['run_success']
    results.append(('2 K_fit=1: other source missed and failing', ok,
                    f"success {summary['n_success']}/{summary['n_active']}, missed {summary['n_missed_active']}"))

    broken = params.copy()
    broken[:, 1, :] = np.nan
    summary, table = score_run(truth, scenario, fake_output(broken, ['mogi', 'mogi'], dates))
    source0 = table[(table.source == 0) & table.active]
    source1 = table[(table.source == 1) & table.active]
    ok = bool(source0['success'].astype(bool).all()) and not source1['success'].astype(bool).any()
    results.append(('3 NaN slot does not fail the finite slot', ok,
                    f"source-0 success {int(source0['success'].astype(bool).sum())}/{len(source0)}, missed {summary['n_missed_active']}"))

    no_peaks = copy.deepcopy(truth)
    for source in no_peaks['sources']:
        source.pop('peak_los_mm')
    computed = true_peaks_mm(no_peaks, scenario, truth['tracks'][0])
    ok = np.allclose(computed, arrays['peak_los_mm'], rtol=1e-9)
    results.append(('4 peaks computed from params == stored', ok, f"max |diff| {np.max(np.abs(computed - arrays['peak_los_mm'])):.2e} mm"))

    k0 = copy.deepcopy(truth)
    k0['sources'] = []
    quiet = params[:, :1].copy()
    quiet[:, 0, 3] = 0.0                                              # zero dV
    summary, _ = score_run(k0, scenario, fake_output(quiet, ['mogi'], dates))
    results.append(('5 K=0 + silent slot = success', summary['run_success'] and summary['n_false_detection'] == 0,
                    f"false detections {summary['n_false_detection']}"))

    loud = params[:, 1:2].copy()                                      # a real, visible source
    summary, _ = score_run(k0, scenario, fake_output(loud, ['mogi'], dates))
    results.append(('6 K=0 + strong slot = false detections', (not summary['run_success']) and summary['n_false_detection'] > 0,
                    f"false detections {summary['n_false_detection']}"))

    base, _ = score_run(truth, scenario, fake_output(params, ['mogi', 'mogi'], dates))
    swapped, _ = score_run(truth, scenario, fake_output(params[:, ::-1], ['mogi', 'mogi'], dates))
    keys = ('n_success', 'n_active', 'n_false_detection', 'median_loc_err_km')
    results.append(('7 slot order does not matter', all(base[k] == swapped[k] for k in keys), ''))

    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"Done. {len(results) - n_fail}/{len(results)} checks passed.")
    raise SystemExit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
