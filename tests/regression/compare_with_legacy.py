#!/usr/bin/env python
# Usage:       python tests/regression/compare_with_legacy.py [--cases ...] [--skip-run]
# Description: Restructure regression test. For every case in legacy_cases.py, runs the NEW entry point
#              methods/pila/run_pila.py on the same shortened config (plus the config switches that replace the
#              legacy launcher) and compares the run directory with the LEGACY reference recorded by
#              run_legacy_reference.py. Strict by design (hardened after the 2026-11-23 code review):
#                * the SET of output files must match in both directions (only wall-clock telemetry files exempt:
#                  timing_epochs.csv, compute_usage.json, infer_usage*.json);
#                * *.pth: every state_dict tensor, every optimizer-state tensor/value, epoch, monitor_best and
#                  tau_r_values identical;
#                * *.csv: identical header AND every cell (no column exclusions);
#                * *.npz: every array identical (only the 'checkpoint' path string differs by design);
#                * *.json: identical, except config.json may differ in save_dir and the new-side switches, and
#                  keys ending in _sec / _s / _ms (wall-clock) are dropped;
#                * per-epoch metric lines of info.log identical (up to the compute-usage telemetry block);
#                * each case must produce at least EXPECTED_MIN outputs, so 'identical' can never mean 'both empty'.
#              Determinism: every REPEAT_CASES legacy run is also compared with its legacy repeat.
#              Exit code 0 only if everything is identical.
# Date:        2026-11-23
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legacy_cases import CASES, EXPECTED_MIN, REGRESSION_ROOT, REPEAT_CASES, REPO_ROOT, build_case_config  # noqa: E402
sys.path.insert(0, REPO_ROOT)   # legacy checkpoints pickle the legacy parse_config.ConfigParser

# Legacy launcher -> config switches that select the same behaviour in run_pila.py
LAUNCHER_FLAGS = {
    'train_pila.py': {},
    'train_pila_delayedpen.py': {'trainer.extensions.delayed_penalty': True},
    'train_pila_logsources.py': {'trainer.extensions.log_per_epoch_sources': True},
    'train_pila_keepbest.py': {'trainer.extensions.delayed_penalty': True, 'trainer.extensions.keep_lowest': True},
    'train_pila_biasinit.py': {'arch.phys_vae.init_sources_mode': 'bias_only'},
    'train_pila_basinhop.py': {'trainer.extensions.delayed_penalty': True, 'trainer.extensions.basin_hop': True},
}
TELEMETRY_FILES = re.compile(r'(^|/)(timing_epochs\.csv|compute_usage\.json|infer_usage[^/]*\.json)$')
WALLCLOCK_KEY = re.compile(r'(_sec|_s|_ms)$')                 # anchored: never matches e.g. '*_mse'
METRIC_LINE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?[0-9.]+(?:e[-+]?[0-9]+)?)\s*$')


# ---------------------------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------------------------
def _list_runs(side, case_name):
    return set(glob.glob(os.path.join(REGRESSION_ROOT, side, case_name, '*', '*')))


def run_new(case_name):
    """Write the new-side config (with launcher switches), run run_pila.py; return (code, s, log, run dir)."""
    config_path = build_case_config(case_name, 'new')
    with open(config_path) as handle:
        config = json.load(handle)
    for dotted, value in LAUNCHER_FLAGS[CASES[case_name][0]].items():
        node = config
        *parents, leaf = dotted.split('.')
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    with open(config_path, 'w') as handle:
        json.dump(config, handle, indent=2)
    log_dir = os.path.join(REGRESSION_ROOT, 'logs', 'new')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{case_name}.log')
    env = dict(os.environ, MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    before = _list_runs('new', case_name)
    t0 = time.time()
    with open(log_path, 'w') as log:
        code = subprocess.run([sys.executable, '-u', os.path.join(REPO_ROOT, 'methods', 'pila', 'run_pila.py'),
                               '-c', config_path], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    created = sorted(_list_runs('new', case_name) - before)
    return code, time.time() - t0, log_path, (created[-1] if len(created) == 1 else None)


def newest_run(side, case_name):
    """Newest <side>/<save_dir name>/<config name>/<timestamp> directory (repeat copies: case_name + '__repeat')."""
    candidates = sorted(_list_runs(side, case_name), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"no run directory for {side}/{case_name}")
    return candidates[-1]


# ---------------------------------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------------------------------
def _deep_equal(a, b, where, problems):
    """Recursive equality for checkpoint / JSON contents (tensors bitwise, containers elementwise)."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        same = torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype \
            and torch.equal(a, b)
        if not same:
            diff = float((a.double() - b.double()).abs().max()) if torch.is_tensor(a) and torch.is_tensor(b) \
                and a.shape == b.shape else float('nan')
            problems.append(f"{where}: tensors differ (max|diff|={diff:.3e})")
        return 1
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            problems.append(f"{where}: keys differ {sorted(map(str, set(a) ^ set(b)))[:5]}")
        return sum(_deep_equal(a[k], b[k], f"{where}.{k}", problems) for k in set(a) & set(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            problems.append(f"{where}: length {len(a)} vs {len(b)}")
        return sum(_deep_equal(x, y, f"{where}[{i}]", problems) for i, (x, y) in enumerate(zip(a, b)))
    both_nan = isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)
    if a != b and not both_nan:
        problems.append(f"{where}: {a!r} vs {b!r}")
    return 0


def compare_checkpoint(path_a, path_b):
    a = torch.load(path_a, map_location='cpu', weights_only=False)
    b = torch.load(path_b, map_location='cpu', weights_only=False)
    problems, n_tensors = [], 0
    for key in ('state_dict', 'optimizer', 'epoch', 'monitor_best', 'tau_r_values', 'arch'):
        if (key in a) != (key in b):
            problems.append(f"'{key}' present in only one checkpoint")
        elif key in a:
            n_tensors += _deep_equal(a[key], b[key], key, problems)
    return problems, n_tensors


def compare_csv(path_a, path_b):
    with open(path_a) as handle:
        rows_a = list(csv.reader(handle))
    with open(path_b) as handle:
        rows_b = list(csv.reader(handle))
    if not rows_a or not rows_b or rows_a[0] != rows_b[0]:
        return [f"header differs: {rows_a[:1]} vs {rows_b[:1]}"]
    if len(rows_a) != len(rows_b):
        return [f"{len(rows_a)} vs {len(rows_b)} rows"]
    return [f"row {i}: {ra} vs {rb}" for i, (ra, rb) in enumerate(zip(rows_a, rows_b)) if ra != rb][:10]


def compare_npz(path_a, path_b):
    a, b = np.load(path_a, allow_pickle=True), np.load(path_b, allow_pickle=True)
    problems = []
    if set(a.files) != set(b.files):
        problems.append(f"keys differ {sorted(set(a.files) ^ set(b.files))}")
    for key in sorted(set(a.files) & set(b.files)):
        if key == 'checkpoint':                       # path string: legacy/ vs new/ by design
            continue
        va, vb = a[key], b[key]
        if va.shape != vb.shape or va.dtype != vb.dtype or not np.array_equal(va, vb, equal_nan=va.dtype.kind == 'f'):
            problems.append(f"array {key} differs")
    return problems


def _drop_wallclock(obj):
    """Drop wall-clock keys (anchored *_sec / *_s / *_ms) at every level of a JSON object."""
    if isinstance(obj, dict):
        return {k: _drop_wallclock(v) for k, v in obj.items() if not WALLCLOCK_KEY.search(str(k))}
    if isinstance(obj, list):
        return [_drop_wallclock(v) for v in obj]
    return obj


def compare_json(path_a, path_b):
    with open(path_a) as handle:
        a = json.load(handle)
    with open(path_b) as handle:
        b = json.load(handle)
    if os.path.basename(path_a) == 'config.json':     # differs by design only in save_dir + new-side switches
        for cfg in (a, b):
            cfg['trainer'].pop('save_dir', None)
            cfg['trainer'].pop('extensions', None)
            cfg['arch'].get('phys_vae', {}).pop('init_sources_mode', None)
    problems = []
    _deep_equal(_drop_wallclock(a), _drop_wallclock(b), os.path.basename(path_a), problems)
    return problems


def epoch_metrics(log_path):
    """Per-epoch 'key : value' lines of the trainer's info.log, up to the compute-usage telemetry block."""
    out = []
    with open(log_path) as handle:
        for line in handle:
            if 'Compute usage' in line:              # memory / wall-clock telemetry varies run to run
                break
            match = METRIC_LINE.match(line.rsplit(' - ', 1)[-1])
            if match and not WALLCLOCK_KEY.search(match.group(1)):
                out.append((match.group(1), match.group(2)))
    return out


def _files(run_root):
    """Relative paths of all output files of a run (models/, figures/, ...); log/ is compared separately."""
    paths = set()
    for path in glob.glob(os.path.join(run_root, '**', '*'), recursive=True):
        rel = os.path.relpath(path, run_root)
        if os.path.isfile(path) and not rel.startswith('log' + os.sep) and not TELEMETRY_FILES.search(rel):
            paths.add(rel)
    return paths


def compare_runs(dir_a, dir_b, expected_min=None):
    """Compare two run directories; return (problems, counts)."""
    problems = []
    counts = dict(pth=0, tensors=0, csv=0, npz=0, json=0, other=0, metric_lines=0)
    files_a, files_b = _files(dir_a), _files(dir_b)
    problems += [f"only in reference: {rel}" for rel in sorted(files_a - files_b)]
    problems += [f"only in compared run: {rel}" for rel in sorted(files_b - files_a)]
    for rel in sorted(files_a & files_b):
        path_a, path_b = os.path.join(dir_a, rel), os.path.join(dir_b, rel)
        if rel.endswith('.pth'):
            found, n_tensors = compare_checkpoint(path_a, path_b)
            counts['pth'] += 1
            counts['tensors'] += n_tensors
        elif rel.endswith('.csv'):
            found = compare_csv(path_a, path_b)
            counts['csv'] += 1
        elif rel.endswith('.npz'):
            found = compare_npz(path_a, path_b)
            counts['npz'] += 1
        elif rel.endswith('.json'):
            found = compare_json(path_a, path_b)
            counts['json'] += os.path.basename(rel) != 'config.json'
        else:                                         # figures etc.: existence checked above, content not compared
            counts['other'] += 1
            continue
        problems += [f"{rel}: {p}" for p in found]
    metrics_a = epoch_metrics(os.path.join(dir_a, 'log', 'info.log'))
    metrics_b = epoch_metrics(os.path.join(dir_b, 'log', 'info.log'))
    counts['metric_lines'] = len(metrics_a)
    if metrics_a != metrics_b:
        first = next((i for i, (x, y) in enumerate(zip(metrics_a, metrics_b)) if x != y),
                     min(len(metrics_a), len(metrics_b)))
        problems.append(f"info.log metrics differ at line {first}: "
                        f"{metrics_a[first] if first < len(metrics_a) else None} vs "
                        f"{metrics_b[first] if first < len(metrics_b) else None}")
    if counts['metric_lines'] == 0:
        problems.append("no per-epoch metric lines found in info.log")
    for kind, minimum in (expected_min or {}).items():
        if counts[kind] < minimum:
            problems.append(f"expected >= {minimum} {kind} outputs, found {counts[kind]}")
    return problems, counts


def report(label, dir_a, dir_b, problems, counts):
    print(f"  {'IDENTICAL' if not problems else 'DIFFERENT'} [{label}] {counts}")
    print(f"    reference: {dir_a}\n    compared:  {dir_b}")
    for p in problems[:15]:
        print(f"    - {p}")
    return not problems


def main():
    parser = argparse.ArgumentParser(description='Compare run_pila.py against the legacy reference')
    parser.add_argument('--cases', nargs='*', default=list(CASES))
    parser.add_argument('--skip-run', action='store_true', help='only compare the newest existing new runs')
    args = parser.parse_args()
    all_ok = True
    print("[0] Determinism of the legacy reference (run vs repeat)...")
    for name in REPEAT_CASES:
        try:
            dir_a, dir_b = newest_run('legacy', name), newest_run('legacy', name + '__repeat')
        except FileNotFoundError as exc:
            print(f"  MISSING {name}: {exc}")
            all_ok = False
            continue
        problems, counts = compare_runs(dir_a, dir_b, EXPECTED_MIN[name])
        all_ok &= report(f"{name}: legacy vs legacy repeat", dir_a, dir_b, problems, counts)
    for index, case_name in enumerate(args.cases, 1):
        print(f"[{index}/{len(args.cases)}] {case_name} ({CASES[case_name][0]} -> run_pila.py "
              f"{LAUNCHER_FLAGS[CASES[case_name][0]] or ''})", flush=True)
        new_dir = None
        if not args.skip_run:
            code, seconds, log_path, new_dir = run_new(case_name)
            print(f"  new run exit={code} {seconds:.0f}s log={log_path}")
            if code != 0 or new_dir is None:
                print("  FAILED: new run did not finish or did not create exactly one run directory")
                all_ok = False
                continue
        try:
            ref_dir = newest_run('legacy', case_name)
            new_dir = new_dir or newest_run('new', case_name)
        except FileNotFoundError as exc:
            print(f"  MISSING: {exc}")
            all_ok = False
            continue
        problems, counts = compare_runs(ref_dir, new_dir, EXPECTED_MIN[case_name])
        all_ok &= report('legacy vs new', ref_dir, new_dir, problems, counts)
    print("ALL CASES IDENTICAL" if all_ok else "SOME CASES DIFFER")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
