#!/usr/bin/env python
# Usage:       python generate_k2_polishext_jobs.py [--late-seeds 1,3,5,8,9,13] [--control-seeds 4,7,15]
#                  [--out-dir jobs/multisource_k2_polishext] [--extra-epochs 200]
# Description: Writes PBS scripts (NOT submitted) for the late-rescue test (extend_k2_polish.py): residual-guided basin-hop
#              runs at SNR 12 (k2tune_snrE1n025bh_slot_resid_seed<S>) continue from epoch 400 for --extra-epochs more at
#              the original polish's final lr (1.5e-5, constant), penalties off as before, no hopping; then every 20-epoch
#              checkpoint is scored. Late seeds separated at epoch >= 266; control seeds separated early (131-207) and
#              tell us how much ANY run gains from more epochs. Tags k2ext_slot_resid_seed<S>.
#              --chain (added 2026-10-26): continue the k2ext runs from their epoch-600 checkpoint instead (same lr, penalties
#              off), e.g. --chain --extra-epochs 800 --save-period 50 --walltime 03:00:00 -> tags k2ext2_slot_resid_seed<S>.
# Date:        2026-10-26

import argparse
import glob
import os

from generate_k2_tune_jobs import PBS_TEMPLATE, parse_seeds

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_GLOB = 'saved/multisource_k2_tune/k2tune_snrE1n025bh_slot_resid_seed{seed}/*/K2/*/*/models'
CHAIN_GLOB = 'saved/multisource_k2_polishext/k2ext_slot_resid_seed{seed}/train/*/*/models'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--late-seeds', default='1,3,5,8,9,13')
    parser.add_argument('--control-seeds', default='4,7,15')
    parser.add_argument('--out-dir', default='jobs/multisource_k2_polishext')
    parser.add_argument('--extra-epochs', type=int, default=200)
    parser.add_argument('--save-period', type=int, default=20)
    parser.add_argument('--walltime', default='01:00:00')
    parser.add_argument('--chain', action='store_true', help='continue the k2ext runs from epoch 600')
    args = parser.parse_args()
    groups = [('late rescue', parse_seeds(args.late_seeds)), ('early-separation control', parse_seeds(args.control_seeds))]

    print("[1/2] Resolving original run directories...")
    jobs = []
    for label, seeds in groups:
        for seed in seeds:
            models = glob.glob(os.path.join(PROJECT_ROOT, (CHAIN_GLOB if args.chain else RUN_GLOB).format(seed=seed)))
            if len(models) != 1:
                raise FileNotFoundError(f'seed {seed}: expected 1 run dir, found {len(models)}')
            models_rel = os.path.relpath(models[0], PROJECT_ROOT)
            start_epoch = 600 if args.chain else 400
            if not os.path.exists(os.path.join(models[0], f'checkpoint-epoch{start_epoch}.pth')):
                raise FileNotFoundError(f'{models_rel}/checkpoint-epoch{start_epoch}.pth')
            tag = f'k2ext2_slot_resid_seed{seed}' if args.chain else f'k2ext_slot_resid_seed{seed}'
            command = (f'python -u extend_k2_polish.py run --run-models-dir {models_rel} --tag {tag} '
                       f'--start-epoch {start_epoch} --extra-epochs {args.extra_epochs} --save-period {args.save_period}\n'
                       f'python -u extend_k2_polish.py score --tag {tag}')
            name = f'k2ext2_s{seed}' if args.chain else f'k2ext_s{seed}'
            jobs.append((name, command, f'{label}: epochs {start_epoch + 1}-{start_epoch + args.extra_epochs} at lr 1.5e-5', seed))
            print(f"  seed {seed} ({label}): {models_rel}")

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, *_ in jobs:
        if os.path.exists(os.path.join(out_dir, f'{name}.pbs')):
            raise FileExistsError(name)
    if os.path.exists(os.path.join(out_dir, 'submit_all.sh')):
        raise FileExistsError(os.path.join(args.out_dir, 'submit_all.sh'))

    print(f"[2/2] Writing {len(jobs)} PBS scripts to {args.out_dir}/...")
    submit = ['#!/bin/bash', f'# Submit from the project root: bash {args.out_dir}/submit_all.sh', 'set -e']
    for name, command, what, seed in jobs:
        pbs_rel = os.path.join(args.out_dir, f'{name}.pbs')
        text = PBS_TEMPLATE.format(pbs_rel=pbs_rel, what=what, seed=seed, job_name=name, command=command)
        text = (text.replace('Honest K=2-only tuning on K2_separated_static (no truth, penalties off)',
                             'K=2 late-rescue polish extension (no truth, penalties off)')
                .replace('generate_k2_tune_jobs.py', 'generate_k2_polishext_jobs.py').replace('2026-10-09', '2026-10-26')
                .replace('walltime=01:00:00', f'walltime={args.walltime}'))
        with open(os.path.join(PROJECT_ROOT, pbs_rel), 'w') as handle:
            handle.write(text)
        submit.append(f'qsub {pbs_rel}')
    with open(os.path.join(out_dir, 'submit_all.sh'), 'w') as handle:
        handle.write('\n'.join(submit) + '\n')
    print(f"Done. Output saved to {args.out_dir}/ ({len(jobs)} PBS + submit_all.sh)")


if __name__ == '__main__':
    main()
