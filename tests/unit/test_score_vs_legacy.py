#!/usr/bin/env python
# Usage:       python tests/unit/test_score_vs_legacy.py [--scenarios-dir runs/_pilot_scenarios] [--max-runs 40]
# Description: Proves the new shared scorer (compare/score.py) reproduces the legacy non-static K=2 scorer
#              (synthetic/multisource/nonstatic_scoring.py) on REAL legacy results: every legacy PILA fit
#              (basin_best / lowest_loss / final fit_K2.npz) and every legacy classical per-date fit
#              (per_date_params.npy) of the converted albums is scored by both; per source-date success, activity,
#              false detection, location / depth / dV errors and the run summary must agree.
#              Reads legacy outputs from saved_legacy_20261123/ only (read-only). Prints PASS/FAIL per check.
# Date:        2026-11-23
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from compare.score import score_run  # noqa: E402
from scenarios.schema import read_scenario, read_truth  # noqa: E402
from synthetic.multisource.nonstatic_scoring import score_run as legacy_score_run  # noqa: E402

LEGACY_ALBUM_DIR = 'synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2'
LEGACY_PILA_GLOB = 'saved_legacy_20261123/multisource_k2_nonstatic/*_seed*/{album}/K2/*/*/models/{which}fit_K2.npz'
LEGACY_CLASSICAL = 'saved_legacy_20261123/multisource_k2_nonstatic_classical_v2/{album}/per_date_params.npy'


def fake_run_output(params_km, dates):
    """Wrap legacy [T, 2, 4] (km, km, km, m^3) params as a contract run output (SI) without recon."""
    params_si = np.asarray(params_km, dtype=np.float64).copy()
    params_si[..., :3] *= 1e3
    return dict(params=dict(params=params_si, models=['mogi', 'mogi'], dates=list(dates)),
                recon={}, timing={}, meta=dict(method='legacy', run_id='x'))


def compare_one(label, params_km, legacy_truth, truth, scenario):
    """Score with both scorers; return list of disagreements."""
    old_summary, old_table = legacy_score_run(params_km, legacy_truth)
    new_summary, new_table = score_run(truth, scenario, fake_run_output(params_km, truth['dates']))
    problems = []
    for key_old, key_new in (('n_active', 'n_active'), ('n_success', 'n_success'), ('n_inactive', 'n_inactive'),
                             ('n_false_detection', 'n_false_detection'), ('run_success', 'run_success')):
        if old_summary[key_old] != new_summary[key_new]:
            problems.append(f"{label}: summary {key_old} {old_summary[key_old]} vs {new_summary[key_new]}")
    old_table = old_table.sort_values(['date_index', 'source']).reset_index(drop=True)
    new_table = new_table[new_table['source'] >= 0].sort_values(['date_index', 'source']).reset_index(drop=True)
    if len(old_table) != len(new_table):
        return problems + [f"{label}: {len(old_table)} vs {len(new_table)} rows"]
    for column in ('slot', 'active', 'success', 'false_detection'):
        diff = old_table[column].astype(str) != new_table[column].astype(str)
        if diff.any():
            problems.append(f"{label}: column {column} differs on {int(diff.sum())} rows")
    for column_old, column_new in (('loc_err_km', 'loc_err_km'), ('depth_err_pct', 'depth_err_pct'),
                                   ('dV_err_pct', 'amp_err_pct')):
        a, b = old_table[column_old].to_numpy(float), new_table[column_new].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if (np.isfinite(a) != np.isfinite(b)).any() or (both.any() and np.max(np.abs(a[both] - b[both])) > 1e-6 * np.maximum(1, np.abs(a[both])).max()):
            problems.append(f"{label}: {column_old} differs (max {np.nanmax(np.abs(a - b)):.3e})")
    return problems


def main():
    parser = argparse.ArgumentParser(description='New scorer vs legacy non-static scorer on legacy results')
    parser.add_argument('--scenarios-dir', default='runs/_pilot_scenarios')
    parser.add_argument('--max-runs', type=int, default=40, help='max legacy PILA fits per album')
    args = parser.parse_args()
    n_checked, failures = 0, []
    for scenario_path in sorted(glob.glob(os.path.join(args.scenarios_dir, '*.npz'))):
        album = os.path.basename(scenario_path)[:-4]
        truth = read_truth(scenario_path.replace('.npz', '_truth.json'))
        scenario = read_scenario(scenario_path)
        with open(os.path.join(LEGACY_ALBUM_DIR, f'{album}_truth.json')) as handle:
            legacy_truth = json.load(handle)
        fits = []
        for which in ('basin_best/', 'lowest_loss/', ''):
            fits += sorted(glob.glob(LEGACY_PILA_GLOB.format(album=album, which=which)))
        fits = fits[:args.max_runs]
        classical = LEGACY_CLASSICAL.format(album=album)
        print(f"[{album}] {len(fits)} legacy PILA fits + {'1' if os.path.exists(classical) else '0'} classical fit")
        for fit_path in fits:
            failures += compare_one(os.path.relpath(fit_path, 'saved_legacy_20261123'),
                                    np.load(fit_path)['params_phys'], legacy_truth, truth, scenario)
            n_checked += 1
        if os.path.exists(classical):
            failures += compare_one(f'classical {album}', np.load(classical), legacy_truth, truth, scenario)
            n_checked += 1
    print(f"  [{'PASS' if not failures and n_checked else 'FAIL'}] new scorer == legacy scorer on {n_checked} legacy runs")
    for problem in failures[:20]:
        print(f"    - {problem}")
    print(f"Done. {n_checked} runs compared, {len(failures)} disagreements.")
    raise SystemExit(0 if not failures and n_checked else 1)


if __name__ == '__main__':
    main()
