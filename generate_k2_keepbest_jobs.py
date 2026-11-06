#!/usr/bin/env python
# Usage:       python generate_k2_keepbest_jobs.py [--seeds 1-16] [--anchor-mm 60] [--out-dir jobs/multisource_k2_keepbest]
# Description: Writes PBS scripts (NOT submitted) for the noisy two-source dV-range test: honest K=2 from scratch on
#              E1_S3_K2_separated at SNR 6 (n050) and SNR 12 (n025), penalties OFF all 400 epochs, warm restarts T_0=100
#              (= the 2026-10-19 "off400" arm), trained with train_pila_keepbest.py (also keeps the lowest-val_rec_loss
#              epoch), two dV ranges:
#                std    +-2e8 m^3 (standard; same config as the earlier off400 runs, rerun so both ranges share the same
#                       launcher and nodes)
#                box60  data-scaled +-2e8 * album std / 60 mm (~5-10x the true dV here; truth not used)
#              2 SNR x 2 ranges x 16 seeds = 64 jobs. Save root saved/multisource_k2_tune, tags
#              k2tune_snrE1<n050|n025>kb_<std|box60>_wr100_seed<S>.
# Date:        2026-10-22

import argparse
import os

from generate_k2_tune_jobs import PBS_TEMPLATE, parse_seeds

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'
ALBUMS = {'n050': 'SNR 6', 'n025': 'SNR 12'}
LR_SCHEDULER = '\'{"type":"CosineAnnealingWarmRestarts","args":{"T_0":100}}\''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='1-16')
    parser.add_argument('--anchor-mm', type=float, default=60.0)
    parser.add_argument('--out-dir', default='jobs/multisource_k2_keepbest')
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    print(f"[1/2] Building {len(ALBUMS) * 2 * len(seeds)} jobs...")
    jobs = []
    for noise_tag, snr_label in ALBUMS.items():
        npz = f'{SNR_DIR}/E1_S3_K2_separated_{noise_tag}.npz'
        if not os.path.exists(os.path.join(PROJECT_ROOT, npz)):
            raise FileNotFoundError(npz)
        for range_tag, flags, what in (('std', '', '+-2e8 dV range'),
                                       ('box60', f' --amp-box-anchor-mm {args.anchor_mm:g}',
                                        f'data-scaled dV range (anchor {args.anchor_mm:g} mm)')):
            for seed in seeds:
                name = f'k2kb{noise_tag[1:]}{range_tag[0]}_s{seed}'          # e.g. k2kb025b_s3 (qstat-friendly)
                tag = f'k2tune_snrE1{noise_tag}kb_{range_tag}_wr100_seed{seed}'
                command = (f'python -u run_k2_tune_keepbest.py --album {npz} --seed {seed} --epochs 400 --tag {tag} '
                           f'--save-root saved/multisource_k2_tune --lr-scheduler {LR_SCHEDULER} '
                           f'--off-epochs 400 --ramp-epochs 0{flags}')
                jobs.append((name, command, f'E1_S3_K2_separated_{noise_tag} ({snr_label}), penalties off, '
                                            f'warm restarts T_0=100, keep lowest-loss epoch, {what}', seed))
    os.makedirs(os.path.join(PROJECT_ROOT, args.out_dir), exist_ok=True)
    for name, *_ in jobs:
        if os.path.exists(os.path.join(PROJECT_ROOT, args.out_dir, f'{name}.pbs')):
            raise FileExistsError(name)

    print(f"[2/2] Writing PBS scripts to {args.out_dir}/...")
    submit = ['#!/bin/bash', f'# Submit from the project root: bash {args.out_dir}/submit_all.sh', 'set -e']
    for name, command, what, seed in jobs:
        pbs_rel = os.path.join(args.out_dir, f'{name}.pbs')
        text = PBS_TEMPLATE.format(pbs_rel=pbs_rel, what=what, seed=seed, job_name=name, command=command)
        text = (text.replace('Honest K=2-only tuning on K2_separated_static (no truth, penalties off)',
                             'Honest K=2-only noisy dV-range test (no truth)')
                .replace('generate_k2_tune_jobs.py', 'generate_k2_keepbest_jobs.py').replace('2026-10-09', '2026-10-22'))
        with open(os.path.join(PROJECT_ROOT, pbs_rel), 'w') as handle:
            handle.write(text)
        submit.append(f'qsub {pbs_rel}')
    with open(os.path.join(PROJECT_ROOT, args.out_dir, 'submit_all.sh'), 'w') as handle:
        handle.write('\n'.join(submit) + '\n')
    print(f"Done. Output saved to {args.out_dir}/ ({len(jobs)} PBS + submit_all.sh)")


if __name__ == '__main__':
    main()
