#!/usr/bin/env python
# Usage:       python generate_k1_amp_jobs.py [--seeds 123,1,2] [--out-dir jobs/multisource_k1_amp]
# Description: Writes PBS scripts (NOT submitted) for the amplitude-vs-SNR test: standard-recipe K=1 PILA with
#              per-epoch source logging (run_k1_logged.py) on
#                A  E1_S2_K1_weak_n012_amp333  weak SNR-7.9 album x 3.33 (signal = strong dV, SNR unchanged) -- the test
#                B  E1_S2_K1_weak_n012         unscaled weak SNR 7.9 (failed before; logging must reproduce that fit)
#                C  E1_S1_K1_strong_n050       strong SNR 6.6 (succeeded before; success-trajectory reference)
#              Tag k1snr_seed<S> under saved/multisource_k1_amp so summarize_k1_snr_runs.py --root works unchanged.
# Date:        2026-10-20

import argparse
import os

from generate_k2_tune_jobs import PBS_TEMPLATE, parse_seeds

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'
ARMS = {'A': 'E1_S2_K1_weak_n012_amp333', 'B': 'E1_S2_K1_weak_n012', 'C': 'E1_S1_K1_strong_n050'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='123,1,2')
    parser.add_argument('--out-dir', default='jobs/multisource_k1_amp')
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    # --- Validate everything before writing anything ---
    jobs = []
    for arm, album in ARMS.items():
        npz = os.path.join(SNR_DIR, f'{album}.npz')
        if not os.path.exists(os.path.join(PROJECT_ROOT, npz)):
            raise FileNotFoundError(npz)
        for seed in seeds:
            name = f'k1amp{arm}_s{seed}'
            pbs_rel = os.path.join(args.out_dir, f'{name}.pbs')
            if os.path.exists(os.path.join(PROJECT_ROOT, pbs_rel)):
                raise FileExistsError(pbs_rel)
            jobs.append((arm, album, npz, seed, name, pbs_rel))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[1/1] Writing {len(jobs)} PBS scripts to {args.out_dir}/...")
    submit = ['#!/bin/bash', f'# Submit from the project root: bash {args.out_dir}/submit_k1_amp.sh', 'set -e']
    for arm, album, npz, seed, name, pbs_rel in jobs:
        command = (f'python -u run_k1_logged.py --album {npz} --seed {seed} '
                   f'--save-root saved/multisource_k1_amp --tag k1snr_seed{seed}')
        text = PBS_TEMPLATE.format(pbs_rel=pbs_rel, what=f'arm {arm} {album}', seed=seed, job_name=name, command=command)
        text = (text.replace('Honest K=2-only tuning on K2_separated_static (no truth, penalties off)',
                             'Amplitude-vs-SNR K=1 test, standard recipe + per-epoch source log (no truth)')
                .replace('generate_k2_tune_jobs.py', 'generate_k1_amp_jobs.py').replace('2026-10-09', '2026-10-20'))
        with open(os.path.join(PROJECT_ROOT, pbs_rel), 'w') as handle:
            handle.write(text)
        submit.append(f'qsub {pbs_rel}')
    with open(os.path.join(out_dir, 'submit_k1_amp.sh'), 'w') as handle:
        handle.write('\n'.join(submit) + '\n')
    print(f"Done. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
