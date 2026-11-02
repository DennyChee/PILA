#!/usr/bin/env python
# Usage:       python generate_k2_delayed_jobs.py [--snr-tags n050 n025] [--arms off delay iso] [--seeds 1-16] \
#                  [--epochs 400] [--off-epochs 200] [--ramp-epochs 50] [--out-dir jobs/multisource_k2_tune_snr_delayed400]
# Description: Writes PBS scripts (NOT submitted) for the honest K=2-only test on the SNR-scaled noisy twins
#              (albums_snr/E1_S3_K2_separated_<snr-tag>.npz, n050 = SNR 6, n025 = SNR 12), warm restarts T_0=100.
#              All arms run through run_k2_tune_delayedpen.py (same code path, same per_epoch_sources.csv log):
#                off   : penalties off for all --epochs (off_epochs = epochs)
#                delay : penalties off for --off-epochs, linear ramp over --ramp-epochs, then full standard
#                        penalties incl. KL + ramped sampling noise
#                iso   : as delay, but KL and sampling stay off throughout (isolates curvature/dV-L1 from the
#                        sampled-z noise)
#              Onset 200 is after the warm-restart cycle (epochs 101-200) in which the clean-data successes separated.
#              Reuses the PBS template and wr100 scheduler of generate_k2_tune_jobs.py (no existing file edited).
#              Tag = k2tune_snrE1<snr-tag><arm><epochs>_wr100_seed<S>; save root saved/multisource_k2_tune/.
#              One submit script submit_snr_delayed.sh. Never overwrites files.
# Date:        2026-10-19

import argparse
import os

from generate_k2_tune_jobs import ARMS, PBS_TEMPLATE, SAVE_ROOT, parse_seeds

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ALBUM_TEMPLATE = 'synthetic/multisource/pila_stacks_v2/albums_snr/E1_S3_K2_separated_{snr_tag}.npz'
SNR_LABEL = {'n050': 'SNR 6', 'n025': 'SNR 12'}


def arm_settings(arm, args):
    """Extra driver flags and a one-line description for one arm."""
    if arm == 'off':
        return (f' --off-epochs {args.epochs} --ramp-epochs 0',
                f'penalties OFF all {args.epochs} epochs')
    ramp = f' --off-epochs {args.off_epochs} --ramp-epochs {args.ramp_epochs}'
    if arm == 'delay':
        return ramp, (f'DELAYED standard penalties incl. KL (off {args.off_epochs} ep, ramp {args.ramp_epochs} ep), '
                      f'{args.epochs} epochs')
    return (ramp + ' --kl-stays-off',
            f'DELAYED curvature + dV-L1 only, KL/sampling off throughout (off {args.off_epochs} ep, '
            f'ramp {args.ramp_epochs} ep), {args.epochs} epochs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snr-tags', nargs='+', default=['n050', 'n025'])
    parser.add_argument('--arms', nargs='+', default=['off', 'delay', 'iso'], choices=['off', 'delay', 'iso'])
    parser.add_argument('--seeds', default='1-16')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--off-epochs', type=int, default=200)
    parser.add_argument('--ramp-epochs', type=int, default=50)
    parser.add_argument('--out-dir', default='jobs/multisource_k2_tune_snr_delayed400')
    args = parser.parse_args()
    if args.off_epochs + args.ramp_epochs >= args.epochs:
        raise ValueError("off + ramp epochs must leave some full-penalty epochs")

    # --- Warm-restart scheduler flags (T_0=100) from the tuning study; epoch count set here ---
    _group, _epochs_wr100, wr100_flags, _what = ARMS['wr100']
    seeds = parse_seeds(args.seeds)

    # --- Pre-check every target (album, job file, run dir) so a clash never leaves a half-written set ---
    jobs = []
    for snr_tag in args.snr_tags:
        album = ALBUM_TEMPLATE.format(snr_tag=snr_tag)
        if not os.path.exists(os.path.join(PROJECT_ROOT, album)):
            raise FileNotFoundError(album)
        for arm in args.arms:
            for seed in seeds:
                job_name = f'k2{arm[0]}{snr_tag[1:]}_s{seed}'          # e.g. k2d050_s3 (qstat truncates names)
                tag = f'k2tune_snrE1{snr_tag}{arm}{args.epochs}_wr100_seed{seed}'
                pbs_rel = os.path.join(args.out_dir, f'{job_name}.pbs')
                for path in (pbs_rel, os.path.join(SAVE_ROOT, tag)):
                    if os.path.exists(os.path.join(PROJECT_ROOT, path)):
                        raise FileExistsError(f"{path} exists; not overwriting")
                jobs.append((snr_tag, album, arm, seed, job_name, tag, pbs_rel))
    submit_rel = os.path.join(args.out_dir, 'submit_snr_delayed.sh')
    if os.path.exists(os.path.join(PROJECT_ROOT, submit_rel)):
        raise FileExistsError(f"{submit_rel} exists; not overwriting")

    # --- Write PBS scripts (patch the template's description; 300-epoch runs took ~5-7 min, 1 h is ample) ---
    os.makedirs(os.path.join(PROJECT_ROOT, args.out_dir), exist_ok=True)
    for snr_tag, album, arm, seed, job_name, tag, pbs_rel in jobs:
        extra, description = arm_settings(arm, args)
        what = f'{SNR_LABEL.get(snr_tag, snr_tag)}, {description}, warm restarts T_0=100'
        template = (PBS_TEMPLATE.replace('on K2_separated_static (no truth, penalties off)',
                                         f'on E1_S3_K2_separated_{snr_tag} (no truth)')
                    .replace('Generated by generate_k2_tune_jobs.py', 'Generated by generate_k2_delayed_jobs.py')
                    .replace('# Date:        2026-10-09', '# Date:        2026-10-19'))
        command = (f"python -u run_k2_tune_delayedpen.py --album {album} --seed {seed} --epochs {args.epochs} "
                   f"--tag {tag} --save-root {SAVE_ROOT} {wr100_flags}{extra}")
        with open(os.path.join(PROJECT_ROOT, pbs_rel), 'w') as handle:
            handle.write(template.format(pbs_rel=pbs_rel, what=what, seed=seed, job_name=job_name, command=command))
    with open(os.path.join(PROJECT_ROOT, submit_rel), 'w') as handle:
        handle.write('#!/bin/bash\nset -e\ncd "%s"\n' % PROJECT_ROOT + ''.join(f'qsub {job[6]}\n' for job in jobs))
    os.chmod(os.path.join(PROJECT_ROOT, submit_rel), 0o755)
    print(f"Done. Wrote {len(jobs)} PBS scripts + {submit_rel} (nothing submitted)")


if __name__ == '__main__':
    main()
