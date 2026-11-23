#!/usr/bin/env python
# Usage:       python -m compare.compare --runs-root runs --out runs/_compare [--manifest scenarios/manifest.csv] \
#                  [--methods pila mcmc vbica] [--scenarios <id> ...] [--no-figures]
# Description: The ONE comparison entry point. Finds every method run directory
#              <runs-root>/<method>/<scenario_id>/<run_id>/ that satisfies the output contract (methods/contract.py),
#              scores the synthetic ones with the shared scorer (compare/score.py) and writes:
#                <out>/master_table.csv     one row per run: scores, reconstruction RMS, timing
#                <out>/summary_by_method.csv aggregated per (method, family)
#                <out>/figures/<scenario>_param_recovery.png, <scenario>_recon_maps.png   (every scenario)
#              Real scenarios (no truth) get reconstruction metrics vs data and the map figure only.
#              Checks: all methods of a scenario must share the pixel set (pixel_mask_sha1 / n_points in meta.json).
#              NOTE: every run directory is scored and listed; choosing ONE run per method among several seeds must
#              use a truth-free rule (the method's own loss) -- decided in Phase 4, not implemented here.
#              K is the ORACLE K in the main track (summary column k_fit_equals_k_true): no 'K detection' claims.
# Date:        2026-11-23
import argparse
import glob
import os
import sys
import time

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compare.figures import param_recovery_figure, recon_maps_figure  # noqa: E402
from compare.score import score_run  # noqa: E402
from methods.contract import read_method_output, validate_run_dir  # noqa: E402
from scenarios.manifest import DEFAULT_MANIFEST, load_manifest, resolve_row  # noqa: E402
from scenarios.schema import read_scenario, read_truth  # noqa: E402

METHODS = ('pila', 'mcmc', 'vbica', 'supervised')


def find_runs(runs_root, methods, scenario_ids):
    """List (method, scenario_id, run_dir) for every contract-valid run directory."""
    found, invalid = [], []
    for method in methods:
        for run_dir in sorted(glob.glob(os.path.join(runs_root, method, '*', '*'))):
            scenario_id = os.path.basename(os.path.dirname(run_dir))
            if scenario_ids and scenario_id not in scenario_ids:
                continue
            if not os.path.exists(os.path.join(run_dir, 'params.npz')):
                continue
            problems = validate_run_dir(run_dir)
            (invalid if problems else found).append((method, scenario_id, run_dir) if not problems else (run_dir, problems))
    return found, invalid


def main():
    parser = argparse.ArgumentParser(description='Score and compare all method runs')
    parser.add_argument('--runs-root', default=os.path.join(REPO_ROOT, 'runs'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--methods', nargs='*', default=list(METHODS))
    parser.add_argument('--scenarios', nargs='*', default=None)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    t0 = time.time()

    print(f"[1/4] Reading manifest {args.manifest} and finding runs under {args.runs_root} ...")
    manifest = {row['scenario_id']: resolve_row(row) for row in load_manifest(args.manifest)}
    runs, invalid = find_runs(args.runs_root, args.methods, args.scenarios)
    print(f"  {len(manifest)} scenarios, {len(runs)} valid runs, {len(invalid)} invalid run directories")
    for run_dir, problems in invalid:
        print(f"    INVALID {run_dir}: {problems[:2]}")

    print("[2/4] Scoring ...")
    rows, by_scenario = [], {}
    for method, scenario_id, run_dir in runs:
        if scenario_id not in manifest:
            print(f"    skip {run_dir}: scenario not in manifest")
            continue
        entry = manifest[scenario_id]
        run_output = read_method_output(run_dir)
        label = f"{method}:{os.path.basename(run_dir)}"
        record = dict(method=method, scenario_id=scenario_id, run_id=os.path.basename(run_dir), family=entry['family'],
                      kind=entry['kind'], run_dir=os.path.relpath(run_dir, REPO_ROOT),
                      pixel_mask_sha1=run_output['meta'].get('pixel_mask_sha1'), n_points=run_output['meta'].get('n_points'))
        if entry['kind'] == 'synthetic':
            truth, scenario = read_truth(entry['truth_path']), read_scenario(entry['scenario_path'])
            summary, _table = score_run(truth, scenario, run_output)
            record.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
            record.update({f'timing_{k}': v for k, v in run_output['timing'].items() if not isinstance(v, (dict, list))})
            by_scenario.setdefault(scenario_id, dict(truth=truth, scenario=scenario, runs={},
                                                     threshold=summary['active_threshold_mm']))['runs'][label] = run_output
        else:
            record.update({f'timing_{k}': v for k, v in run_output['timing'].items() if not isinstance(v, (dict, list))})
        rows.append(record)
    table = pd.DataFrame(rows)
    # Same-pixels guarantee: every method must have fitted the same pixel set of a scenario.
    if len(table):
        for scenario_id, group in table.groupby('scenario_id'):
            fingerprints = set(group['pixel_mask_sha1'].dropna())
            if len(fingerprints) > 1 or group['n_points'].dropna().nunique() > 1:
                print(f"  WARNING {scenario_id}: methods fitted DIFFERENT pixel sets {sorted(fingerprints)} / "
                      f"n_points {sorted(group['n_points'].dropna().unique())} -- not a like-for-like comparison")
    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, 'master_table.csv'), index=False)

    print("[3/4] Aggregating per (method, family) ...")
    if len(table) and 'success_frac' in table:
        summary = table.groupby(['method', 'family']).agg(
            n_runs=('run_id', 'count'), mean_success_frac=('success_frac', 'mean'),
            frac_run_success=('run_success', 'mean'), median_loc_err_km=('median_loc_err_km', 'median'),
            median_abs_depth_err_pct=('median_abs_depth_err_pct', 'median'),
            median_abs_amp_err_pct=('median_abs_amp_err_pct', 'median'),
            false_detections=('n_false_detection', 'sum'),
            median_rms_recon_vs_data_mm=('rms_recon_vs_data_mm', 'median'),
            median_total_s=('timing_total_s', 'median')).reset_index()
        summary.to_csv(os.path.join(args.out, 'summary_by_method.csv'), index=False)
        print(summary.to_string(index=False))

    if not args.no_figures:
        print(f"[4/4] Figures for {len(by_scenario)} scenarios ...")
        for scenario_id, item in by_scenario.items():
            param_recovery_figure(item['truth'], item['runs'], item['threshold'],
                                  os.path.join(args.out, 'figures', f'{scenario_id}_param_recovery.png'))
            recon_maps_figure(item['scenario'], item['runs'], os.path.join(args.out, 'figures', f'{scenario_id}_recon_maps.png'))
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"Done. Output saved to {args.out}/")


if __name__ == '__main__':
    main()
