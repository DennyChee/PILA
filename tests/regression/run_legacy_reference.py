#!/usr/bin/env python
# Usage:       python tests/regression/run_legacy_reference.py [--cases std_mogi_cube k2_basinhop ...] [--repeats]
# Description: Record the LEGACY reference outputs for the restructure regression test. Must be run from the
#              legacy tree (tag legacy-final-20261123 layout: launchers at the repo root). Each case runs its
#              legacy launcher on the shortened config from legacy_cases.py, CPU only, into
#              runs/_regression/legacy/<case>/. --repeat-first runs the first case twice (second copy under
#              <case>__repeat) to prove the reference is deterministic run-to-run.
# Date:        2026-11-23
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legacy_cases import CASES, REPEAT_CASES, REPO_ROOT, build_case_config  # noqa: E402


def run_case(case_name, side='legacy', suffix=''):
    """Run one legacy launcher; return (exit code, seconds). Log goes next to the run outputs."""
    launcher = CASES[case_name][0]
    config_path = build_case_config(case_name, side)
    if suffix:                                             # repeat copy -> separate save_dir
        import json
        with open(config_path) as handle:
            config = json.load(handle)
        config['trainer']['save_dir'] += suffix
        config_path = config_path.replace('.json', f'{suffix}.json')
        with open(config_path, 'w') as handle:
            json.dump(config, handle, indent=2)
    log_dir = os.path.join(REPO_ROOT, 'runs', '_regression', 'logs', side)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{case_name}{suffix}.log')
    env = dict(os.environ, MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    t0 = time.time()
    with open(log_path, 'w') as log:
        code = subprocess.run([sys.executable, '-u', launcher, '-c', config_path], cwd=REPO_ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT).returncode
    return code, time.time() - t0, log_path


def main():
    parser = argparse.ArgumentParser(description='Record legacy regression references')
    parser.add_argument('--cases', nargs='*', default=list(CASES))
    parser.add_argument('--repeat-first', action='store_true', help='(old) repeat the first case')
    parser.add_argument('--repeats', action='store_true', help='also run a repeat of every REPEAT_CASES case')
    args = parser.parse_args()
    jobs = [(name, '') for name in args.cases]
    if args.repeat_first:
        jobs.insert(1, (args.cases[0], '__repeat'))
    if args.repeats:
        jobs += [(name, '__repeat') for name in REPEAT_CASES if (name, '__repeat') not in jobs]
    for index, (name, suffix) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {name}{suffix} via {CASES[name][0]} ...", flush=True)
        code, seconds, log_path = run_case(name, suffix=suffix)
        print(f"  exit={code}  {seconds:.0f}s  log={log_path}", flush=True)
    print("Done. References in runs/_regression/legacy/")


if __name__ == '__main__':
    main()
