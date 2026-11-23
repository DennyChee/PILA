#!/usr/bin/env python
# Usage:       python tests/integration/run_pipeline_matrix.py --manifest runs/_test_scenarios/manifest.csv \
#                  [--scenarios t1_mogi_k1_static_clean ...] [--methods pila mcmc vbica] [--out DIR --tag NAME]
#              python tests/integration/run_pipeline_matrix.py --aggregate --out DIR   (after parallel PBS jobs)
# Description: Pipeline test MATRIX: every method x every scenario of a test manifest with tiny settings
#              (plumbing, not science). For each cell it records PASS / FAIL / EXPECTED-REFUSAL and the error, then
#              runs compare.py and checks: contract validity, identical pixel sets per scenario, and a few known
#              answers on the run dates (noise-free K=1: MCMC must recover the source; every truth row must score).
#              Settings: PILA 3 epochs; MCMC 3 dates, 1 start, NBay 2000, offset fit; vbICA N = K + 2,
#              max_steps 400, oracle-K grouping, per-group MCMC on the same 3 dates.
#              Output: --out (default NEW runs/_test_matrix/<timestamp>/); per job matrix_<tag>.csv. Parallel PBS jobs
#              (one per scenario, same --out, different --tag) are combined by --aggregate (checks + compare.py).
# Date:        2026-11-23
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import traceback

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from compare.score import score_run  # noqa: E402
from methods.contract import read_method_output, validate_run_dir  # noqa: E402
from pila.paths import resolve_tokens  # noqa: E402
from scenarios.manifest import load_manifest, resolve_row  # noqa: E402
from scenarios.schema import read_scenario, read_truth  # noqa: E402

MATLAB = os.environ.get('MATLAB_BIN', '/usr/local/MATLAB/R2023b/bin/matlab')
RUN_NAME = 'test'                 # set from --attempt in main(): attempt 1 -> 'test', N -> 'test_a<N>'
ENV = dict(os.environ, MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='1')


def sh(cmd, log_path):
    """Run a command, append its output to log_path, raise with the log tail on failure."""
    with open(log_path, 'a') as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        code = subprocess.run(cmd, cwd=REPO_ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0:
        with open(log_path) as log:
            tail = log.read().splitlines()[-8:]
        raise RuntimeError(f"exit {code}: {' | '.join(tail)}")


def matlab_perdate(run_dir, date_idx, mcmc_repo, log_path):
    idx = '[' + ' '.join(str(i) for i in date_idx) + ']'
    sh([MATLAB, '-batch', f"addpath('methods/mcmc'); run_iclassic_perdate('{run_dir}', '{mcmc_repo}', {idx})"], log_path)


def run_pila(row, manifest, runs, out, log):
    sid = row['scenario_id']
    config = os.path.join(out, 'configs', f'{sid}_pila_{RUN_NAME}.json')
    epochs = '2' if row['kind'] == 'real' else '3'
    sh([sys.executable, 'methods/pila/make_config.py', '--scenario-id', sid, '--manifest', manifest, '--epochs', epochs,
        '--seed', '1', '--save-dir', os.path.join(out, 'pila_train'), '--out', config], log)
    sh([sys.executable, 'methods/pila/run_pila.py', '-c', config], log)
    ckpts = sorted(__import__('glob').glob(os.path.join(out, 'pila_train', sid, '*', 'models', 'model_best.pth')))
    if not ckpts:
        raise RuntimeError("no model_best.pth written")
    run_dir = os.path.join(runs, 'pila', sid, 'seed1' if RUN_NAME == 'test' else f'seed1_{RUN_NAME}')
    sh([sys.executable, 'methods/pila/predict.py', '--checkpoint', ckpts[-1], '--scenario-id', sid, '--out', run_dir], log)
    return run_dir


def run_mcmc(row, manifest, runs, date_idx, mcmc_repo, log):
    sid = row['scenario_id']
    run_dir = os.path.abspath(os.path.join(runs, 'mcmc', sid, RUN_NAME))
    sh([sys.executable, '-m', 'methods.mcmc.export', '--scenario-id', sid, '--manifest', manifest, '--offset', 'fit',
        '--n-starts', '1', '--n-bay', '2000', '--out', run_dir], log)
    matlab_perdate(run_dir, date_idx, mcmc_repo, log)
    sh([sys.executable, '-m', 'methods.mcmc.import_results', '--run', run_dir, '--manifest', manifest], log)
    return run_dir


def run_vbica(row, manifest, runs, date_idx, mcmc_repo, log):
    sid = row['scenario_id']
    run_dir = os.path.abspath(os.path.join(runs, 'vbica', sid, RUN_NAME))
    n_comp = int(row['n_sources']) + 2
    sh([sys.executable, '-m', 'methods.vbica.export', '--scenario-id', sid, '--manifest', manifest, '--out', run_dir,
        '--max-steps', '400'], log)
    try:
        sh(['methods/vbica/run_vbica.sh', run_dir, str(n_comp), '1'], log)
    except RuntimeError as exc:
        # Test policy only (the real N rule is a Phase 4 decision): if the data rank is below N, retry with N = rank.
        match = __import__('re').search(r'rerun with N <= (\d+)', str(exc))
        if not match:
            raise
        n_comp = int(match.group(1))
        with open(log, 'a') as handle:
            handle.write(f"\n[matrix] data rank below requested N -> retrying vbICA with N = {n_comp}\n")
        sh(['methods/vbica/run_vbica.sh', run_dir, str(n_comp), '1'], log)
    sh([sys.executable, '-m', 'methods.vbica.perepoch', 'export', '--run', run_dir, '--manifest', manifest,
        '--n-comp', str(n_comp), '--rule', 'overlap_cluster', '--offset', 'fit', '--n-starts', '1', '--n-bay', '2000'], log)
    for group_dir in sorted(__import__('glob').glob(os.path.join(run_dir, 'group_*'))):
        matlab_perdate(group_dir, date_idx, mcmc_repo, log)
    sh([sys.executable, '-m', 'methods.vbica.perepoch', 'import', '--run', run_dir, '--manifest', manifest], log)
    return run_dir


def expected_refusal(method, row, scenario):
    """Cells that MUST refuse (and the refusal text to look for), or None."""
    if method == 'vbica' and row['kind'] == 'real':
        return 'synthetic'                               # vbICA real-data branch not wired in the adapter (yet)
    if method == 'vbica' and scenario is not None and 'mask' in scenario and not scenario['mask'].all():
        return 'masked synthetic scenes'
    return None


def main():
    parser = argparse.ArgumentParser(description='Pipeline test matrix')
    parser.add_argument('--manifest', default='runs/_test_scenarios/manifest.csv')
    parser.add_argument('--scenarios', nargs='*', default=None)
    parser.add_argument('--methods', nargs='*', default=['pila', 'mcmc', 'vbica'])
    parser.add_argument('--out', default=None, help='shared output dir (default: new runs/_test_matrix/<timestamp>)')
    parser.add_argument('--tag', default='all', help='suffix of this job\'s matrix_<tag>.csv')
    parser.add_argument('--aggregate', action='store_true', help='combine matrix_*.csv in --out, check, compare')
    parser.add_argument('--attempt', type=int, default=1, help='re-run number: run dirs test_a<N>; latest attempt wins')
    args = parser.parse_args()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.abspath(args.out or os.path.join('runs', '_test_matrix', stamp))
    if args.aggregate:
        return aggregate(out, args.manifest)
    global RUN_NAME
    RUN_NAME = 'test' if args.attempt == 1 else f'test_a{args.attempt}'
    runs = os.path.join(out, 'runs')
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(out, 'configs'), exist_ok=True)
    mcmc_repo = resolve_tokens('${MCMC_REPO}')
    rows = [r for r in load_manifest(args.manifest) if not args.scenarios or r['scenario_id'] in args.scenarios]
    print(f"[1/3] {len(rows)} scenarios x {args.methods} -> {out}", flush=True)
    cells = []
    for row in rows:
        sid = row['scenario_id']
        resolved = resolve_row(row)
        scenario = read_scenario(resolved['scenario_path']) if row['kind'] == 'synthetic' else None
        n_dates = len(scenario['dates']) if scenario is not None else None
        for method in args.methods:
            log = os.path.join(out, 'logs', f'{sid}__{method}' + ('' if args.attempt == 1 else f'__a{args.attempt}') + '.log')
            refusal = expected_refusal(method, row, scenario)
            t0 = time.time()
            cell = dict(scenario_id=sid, method=method, status='', detail='', seconds=0.0, attempt=args.attempt)
            try:
                if n_dates is None:                       # real scene: number of dates from PILA's loader
                    from methods.mcmc.export import load_points
                    n_dates = len(load_points(row).dates)
                date_idx = sorted({1, (n_dates + 1) // 2, n_dates})
                cell['date_idx'] = date_idx
                if method == 'pila':
                    run_dir = run_pila(row, args.manifest, runs, out, log)
                elif method == 'mcmc':
                    run_dir = run_mcmc(row, args.manifest, runs, date_idx, mcmc_repo, log)
                else:
                    run_dir = run_vbica(row, args.manifest, runs, date_idx, mcmc_repo, log)
                problems = validate_run_dir(run_dir)
                if refusal:
                    cell.update(status='FAIL', detail=f'expected a refusal ({refusal}) but the run succeeded')
                elif problems:
                    cell.update(status='FAIL', detail=f'contract: {problems[:2]}')
                else:
                    cell.update(status='PASS', run_dir=run_dir)
            except (RuntimeError, ValueError, NotImplementedError, FileNotFoundError, KeyError) as exc:
                message = str(exc)
                if refusal and refusal in message:
                    cell.update(status='EXPECTED-REFUSAL', detail=message[-160:])
                else:
                    cell.update(status='FAIL', detail=message[-300:])
                    traceback.print_exc(file=open(log, 'a'))
            cell['seconds'] = round(time.time() - t0, 1)
            cells.append(cell)
            print(f"  {sid:28s} {method:6s} {cell['status']:16s} {cell['seconds']:7.1f}s  {cell['detail'][:110]}", flush=True)

    table = pd.DataFrame(cells)
    table['date_idx'] = table['date_idx'].apply(lambda v: json.dumps(v) if isinstance(v, list) else '')
    table.to_csv(os.path.join(out, f'matrix_{args.tag}.csv'), index=False)
    print(f"Done. {len(cells)} cells, {int((table['status'] == 'FAIL').sum())} FAIL. Output saved to {out}/matrix_{args.tag}.csv")
    if args.tag == 'all':
        aggregate(out, args.manifest)


def aggregate(out, manifest):
    """Combine every matrix_*.csv in out: science sanity + same-pixels checks, then compare.py over all runs."""
    import glob
    frames = [pd.read_csv(path, keep_default_na=False) for path in sorted(glob.glob(os.path.join(out, 'matrix_*.csv')))]
    if not frames:
        raise FileNotFoundError(f"no matrix_*.csv in {out}")
    table = pd.concat(frames, ignore_index=True)
    if 'attempt' not in table:
        table['attempt'] = 1
    table['attempt'] = pd.to_numeric(table['attempt'], errors='coerce').fillna(1).astype(int)
    n_before = len(table)
    table = table.sort_values('attempt').drop_duplicates(['scenario_id', 'method'], keep='last')  # latest attempt wins
    print(f"  {n_before} cells read, {len(table)} after keeping the latest attempt per scenario x method")
    cells = table.to_dict('records')
    for cell in cells:
        cell['date_idx'] = json.loads(cell['date_idx']) if cell.get('date_idx') else []
    runs = os.path.join(out, 'runs')
    print(f"[2/3] Science sanity on the run dates + same-pixels check ({len(cells)} cells) ...", flush=True)
    checks = []
    manifest_rows = {r['scenario_id']: resolve_row(r) for r in load_manifest(manifest)}
    for sid in sorted({c['scenario_id'] for c in cells}):
        passed = [c for c in cells if c['scenario_id'] == sid and c['status'] == 'PASS']
        metas = {c['method']: read_method_output(c['run_dir'])['meta'] for c in passed}
        fingerprints = {m.get('pixel_mask_sha1') for m in metas.values()}
        n_points = {m.get('n_points') for m in metas.values()}
        checks.append(dict(check=f'{sid}: same pixels across {sorted(metas)}', ok=len(fingerprints) <= 1 and len(n_points) <= 1,
                           detail=f'{fingerprints} n_points {n_points}'))
        entry = manifest_rows[sid]
        if entry['kind'] != 'synthetic':
            continue
        truth, scenario = read_truth(entry['truth_path']), read_scenario(entry['scenario_path'])
        for cell in passed:
            summary, scores = score_run(truth, scenario, read_method_output(cell['run_dir']))
            run_rows = scores[scores.date_index.isin([i - 1 for i in cell['date_idx']]) & scores.active]
            n_ok = int(run_rows['success'].astype(bool).sum()) if len(run_rows) else 0
            cell.update(run_date_success=f"{n_ok}/{len(run_rows)}", rms_phys_vs_clean_mm=summary.get('rms_phys_vs_clean_mm'))
            if cell['method'] == 'mcmc' and sid.startswith('t1_'):
                checks.append(dict(check=f'{sid}: MCMC recovers the noise-free source on every run date',
                                   ok=n_ok == len(run_rows) and len(run_rows) > 0, detail=f'{n_ok}/{len(run_rows)}'))
            checks.append(dict(check=f'{sid} {cell["method"]}: every true source-date has a score row',
                               ok=len(scores[scores.source >= 0]) == len(truth['dates']) * len(truth['sources']),
                               detail=f"{len(scores[scores.source >= 0])} rows"))
    for check in checks:
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']}  {check['detail']}")
    print("[3/3] compare.py over all runs ...", flush=True)
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    sh([sys.executable, '-m', 'compare.compare', '--runs-root', runs, '--out', os.path.join(out, 'compare'),
        '--manifest', manifest], os.path.join(out, 'logs', 'compare.log'))
    summary_table = pd.DataFrame(cells)
    summary_table.to_csv(os.path.join(out, 'matrix_combined.csv'), index=False)
    with open(os.path.join(out, 'checks.json'), 'w') as handle:
        json.dump(checks, handle, indent=1)
    print(summary_table[['scenario_id', 'method', 'status', 'seconds'] +
                        [c for c in ('run_date_success', 'rms_phys_vs_clean_mm') if c in summary_table]].to_string(index=False))
    n_fail = int((summary_table['status'] == 'FAIL').sum()) + sum(not c['ok'] for c in checks)
    print(f"Done. {len(cells)} cells, {n_fail} failure(s). Output saved to {out}/")
    raise SystemExit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
