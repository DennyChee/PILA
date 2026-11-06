#!/usr/bin/env python
# Usage:       python generate_k1_dvbox_jobs.py [--seeds 123,1,2] [--out-dir jobs/multisource_k1_dvbox] [--set weak|strong]
# Description: Writes PBS scripts (NOT submitted) for the data-scaled dV prior box test (run_k1_logged.py, standard
#              recipe + per-epoch source log), weak single-source albums:
#                E  equivalence: E1_S2_K1_weak_n012 with box +-2e8/F, F = dV_strong/dV_weak = 3.3333 (from the truths).
#                   Mathematically identical to arm A of the amplitude test (album x F, standard box) -> must reproduce
#                   it (2/3 success), up to float round-off.
#                R  rule: weak n100/n050/n025/n012/n006 with box = 2e8 * data std / 12 mm (data only, no truth).
#              Save roots saved/multisource_k1_dvbox/{equiv,rule}/k1snr_seed<S> (summarize_k1_snr_runs.py --root works).
#              --set strong (2026-10-22): rule arm only on the STRONG album n100..n006 -> saved/multisource_k1_dvbox/rule_strong,
#              PBS k1dvRS<noise>_s<seed>.pbs + submit_k1_dvbox_strong.sh (same out dir, distinct names).
# Date:        2026-10-20

import argparse
import json
import os

from generate_k2_tune_jobs import PBS_TEMPLATE, parse_seeds

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR = 'synthetic/multisource/pila_stacks_v2/albums'
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='123,1,2')
    parser.add_argument('--out-dir', default='jobs/multisource_k1_dvbox')
    parser.add_argument('--set', choices=['weak', 'strong'], default='weak')
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    # --- Equivalence factor from the truths (weak set only; a diagnostic arm, never an honest inversion) ---
    if args.set == 'strong':
        dv_weak = dv_strong = 1.0
    else:
        dv_weak = json.load(open(os.path.join(PROJECT_ROOT, NOISY_DIR, 'E1_S2_K1_weak_truth.json')))['sources'][0]['dV_m3']
        dv_strong = json.load(open(os.path.join(PROJECT_ROOT, NOISY_DIR, 'E1_S1_K1_strong_truth.json')))['sources'][0]['dV_m3']
    equiv_box_m3 = 2.0e8 * dv_weak / dv_strong
    print(f"[1/2] Equivalence box +-{equiv_box_m3:.6e} m^3 (factor {dv_strong / dv_weak:.5f})")

    jobs = []   # (name, album npz, flags, save root)
    for seed in (seeds if args.set == 'strong' else []):
        for noise_tag in ('n100', 'n050', 'n025', 'n012', 'n006'):
            npz = f'{NOISY_DIR}/E1_S1_K1_strong.npz' if noise_tag == 'n100' else f'{SNR_DIR}/E1_S1_K1_strong_{noise_tag}.npz'
            jobs.append((f'k1dvRS{noise_tag[1:]}_s{seed}', npz, '--dv-scale-to-data', 'saved/multisource_k1_dvbox/rule_strong', seed))
    for seed in (seeds if args.set == 'weak' else []):
        jobs.append((f'k1dvE_s{seed}', f'{SNR_DIR}/E1_S2_K1_weak_n012.npz', f'--dv-max-m3 {equiv_box_m3:.6e}',
                     'saved/multisource_k1_dvbox/equiv', seed))
        for noise_tag in ('n100', 'n050', 'n025', 'n012', 'n006'):
            npz = f'{NOISY_DIR}/E1_S2_K1_weak.npz' if noise_tag == 'n100' else f'{SNR_DIR}/E1_S2_K1_weak_{noise_tag}.npz'
            jobs.append((f'k1dvR{noise_tag[1:]}_s{seed}', npz, '--dv-scale-to-data', 'saved/multisource_k1_dvbox/rule', seed))
    for name, npz, _, _, _ in jobs:            # validate everything before writing anything
        if not os.path.exists(os.path.join(PROJECT_ROOT, npz)):
            raise FileNotFoundError(npz)
        if os.path.exists(os.path.join(PROJECT_ROOT, args.out_dir, f'{name}.pbs')):
            raise FileExistsError(name)
    os.makedirs(os.path.join(PROJECT_ROOT, args.out_dir), exist_ok=True)
    print(f"[2/2] Writing {len(jobs)} PBS scripts to {args.out_dir}/...")
    submit = ['#!/bin/bash', f'# Submit from the project root: bash {args.out_dir}/submit_k1_dvbox.sh', 'set -e']
    for name, npz, flags, save_root, seed in jobs:
        pbs_rel = os.path.join(args.out_dir, f'{name}.pbs')
        command = (f'python -u run_k1_logged.py --album {npz} --seed {seed} {flags} '
                   f'--save-root {save_root} --tag k1snr_seed{seed}')
        text = PBS_TEMPLATE.format(pbs_rel=pbs_rel, what=f'{os.path.basename(npz)} {flags}', seed=seed,
                                   job_name=name, command=command)
        text = (text.replace('Honest K=2-only tuning on K2_separated_static (no truth, penalties off)',
                             'Data-scaled dV box K=1 test, standard recipe + per-epoch source log')
                .replace('generate_k2_tune_jobs.py', 'generate_k1_dvbox_jobs.py').replace('2026-10-09', '2026-10-20' if args.set == 'weak' else '2026-10-22'))
        with open(os.path.join(PROJECT_ROOT, pbs_rel), 'w') as handle:
            handle.write(text)
        submit.append(f'qsub {pbs_rel}')
    submit_name = 'submit_k1_dvbox.sh' if args.set == 'weak' else 'submit_k1_dvbox_strong.sh'
    submit[1] = f'# Submit from the project root: bash {args.out_dir}/{submit_name}'
    with open(os.path.join(PROJECT_ROOT, args.out_dir, submit_name), 'w') as handle:
        handle.write('\n'.join(submit) + '\n')
    print(f"Done. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
