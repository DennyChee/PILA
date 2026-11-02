#!/usr/bin/env python
# Usage:       python generate_k2_basinhop_jobs.py [--seeds 1-16] [--out-dir jobs/multisource_k2_basinhop]
#                  [--noise-tag n025|n050] [--arms slot_resid observe]   (defaults: SNR 12, the 6 standard-box arms;
#                  the data-scaled-box arms slot_resid_box60 / observe_box60 only when listed in --arms)
# Description: Writes PBS scripts (NOT submitted) for the basin-hopping test: honest K=2 from scratch on
#              E1_S3_K2_separated_n025 (SNR 12), penalties OFF all 400 epochs, standard +-2e8 dV range, epochs 1-100 as
#              the 2026-10-22 keep-best std arm (warm-restart cycle 1), then train_pila_basinhop.py hops (revert to the
#              best deterministic album misfit, perturb, settle 40 epochs, extend while improving; 60-epoch polish).
#              Arms (default all 6 x 16 seeds = 96 jobs; --arms picks a subset):
#                slot       move the redundant slot to a random box position
#                slot_resid move the redundant slot to a residual-weighted position (PILA's own fit)
#                wnoise     Gaussian noise on the z_phy-mean head (RMS du ~ 1)
#                none       control: revert + 0.5x LR re-warm + same Adam reset, no perturbation
#                none_full  control at 1.0x LR peak (3e-4)
#                observe    baseline: the keep-best std recipe unchanged (warm restarts to 400), only logs the album
#                           misfit and keeps basin_best, so the baseline is selected by the same deterministic rule
#              --noise-tag n050 runs the same recipe on the SNR 6 album (added 2026-10-23).
#              --noise-set E2 --noise-tag n035 (added 2026-10-26): same sources, independent Marapi noise days
#              (E2_S3_K2_separated_n035, SNR 11.8, built by make_snr_scaled_albums --nonoise-album E1_S3_K2_separated).
#              Save root saved/multisource_k2_tune, tags k2tune_snr<noise-set><noise-tag>bh_<arm>_seed<S>.
# Date:        2026-10-23

import argparse
import os

from generate_k2_tune_jobs import PBS_TEMPLATE, parse_seeds

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ALBUM = 'synthetic/multisource/pila_stacks_v2/albums_snr/{noise_set}_S3_K2_separated_{noise}.npz'
SNR_LABEL = {'n025': 'SNR 12', 'n050': 'SNR 6', 'n035': 'SNR 12'}
LR_SCHEDULER = '\'{"type":"CosineAnnealingWarmRestarts","args":{"T_0":100}}\''
ARMS = {  # tag: (perturbation, settle LR peak fraction, description)
    'slot': ('slot', 0.5, 'move redundant slot to a random box position'),
    'slot_resid': ('slot_resid', 0.5, 'move redundant slot to a residual-weighted position'),
    'wnoise': ('weight_noise', 0.5, 'Gaussian noise on the z_phy-mean head (RMS du ~ 1)'),
    'none': ('none', 0.5, 'control: revert + 0.5x LR re-warm, no perturbation'),
    'none_full': ('none', 1.0, 'control: revert + 1.0x LR re-warm, no perturbation'),
    'observe': ('off', 0.5, 'baseline: warm restarts unchanged, deterministic-misfit logging + basin_best only'),
    # Data-scaled dV box (utils/amp_box.py, anchor 60 mm: +-2e8 m^3 * album LOS std / 60 mm), added 2026-10-23
    'slot_resid_box60': ('slot_resid', 0.5, 'residual-guided slot move, data-scaled dV box (60 mm anchor)'),
    'observe_box60': ('off', 0.5, 'baseline: warm restarts, data-scaled dV box (60 mm anchor)'),
}
ARMS.update({  # Curvature temporal smoothing ALONE (incl. dV; KL and dV-L1 off), on from epoch 1, added 2026-10-23
    'slot_resid_curv15000': ('slot_resid', 0.5, 'residual-guided slot move + temporal smoothing 15000'),
    'slot_resid_curv5000': ('slot_resid', 0.5, 'residual-guided slot move + temporal smoothing 5000'),
    'observe_curv15000': ('off', 0.5, 'baseline: warm restarts + temporal smoothing 15000'),
})
EXTRA_FLAGS = {'slot_resid_box60': ' --amp-box-anchor-mm 60', 'observe_box60': ' --amp-box-anchor-mm 60',
               'slot_resid_curv15000': ' --temporal-smoothness-weight 15000',
               'slot_resid_curv5000': ' --temporal-smoothness-weight 5000',
               'observe_curv15000': ' --temporal-smoothness-weight 15000'}
OFF_EPOCHS = {arm: 0 for arm in ('slot_resid_curv15000', 'slot_resid_curv5000', 'observe_curv15000')}   # default 400


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='1-16')
    parser.add_argument('--out-dir', default='jobs/multisource_k2_basinhop')
    parser.add_argument('--noise-tag', default='n025', choices=list(SNR_LABEL))
    parser.add_argument('--noise-set', default='E1', choices=['E1', 'E2'])
    parser.add_argument('--arms', nargs='*', default=[arm for arm in ARMS if arm not in EXTRA_FLAGS], choices=list(ARMS))
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    album = ALBUM.format(noise_set=args.noise_set, noise=args.noise_tag)
    if not os.path.exists(os.path.join(PROJECT_ROOT, album)):
        raise FileNotFoundError(album)

    print(f"[1/2] Building {len(args.arms) * len(seeds)} jobs ({SNR_LABEL[args.noise_tag]})...")
    jobs = []
    for arm in args.arms:
        perturbation, peak_frac, what = ARMS[arm]
        for seed in seeds:
            name = f'k2bh_{arm}_s{seed}' if args.noise_tag == 'n025' else f'k2bh{args.noise_tag[1:]}_{arm}_s{seed}'
            if args.noise_set != 'E1':
                name = f'k2bh{args.noise_set}{args.noise_tag[1:]}_{arm}_s{seed}'   # existing E2 n035 jobs predate the tag suffix
            tag = f'k2tune_snr{args.noise_set}{args.noise_tag}bh_{arm}_seed{seed}'
            command = (f'python -u run_k2_tune_basinhop.py --album {album} --seed {seed} --epochs 400 --tag {tag} '
                       f'--save-root saved/multisource_k2_tune --lr-scheduler {LR_SCHEDULER} '
                       f'--off-epochs {OFF_EPOCHS.get(arm, 400)} --ramp-epochs 0 --perturbation {perturbation} '
                       f'--settle-lr-peak-frac {peak_frac:g}{EXTRA_FLAGS.get(arm, "")}')
            penalties = 'smoothing only' if arm in OFF_EPOCHS else 'penalties off'
            jobs.append((name, command, f'{args.noise_set}_S3_K2_separated_{args.noise_tag} ({SNR_LABEL[args.noise_tag]}), {penalties}, '
                                        f'basin hopping: {what}', seed))
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, *_ in jobs:
        if os.path.exists(os.path.join(out_dir, f'{name}.pbs')):
            raise FileExistsError(name)

    print(f"[2/2] Writing PBS scripts to {args.out_dir}/...")
    submit = ['#!/bin/bash', f'# Submit from the project root: bash {args.out_dir}/submit_all.sh', 'set -e']
    for name, command, what, seed in jobs:
        pbs_rel = os.path.join(args.out_dir, f'{name}.pbs')
        text = PBS_TEMPLATE.format(pbs_rel=pbs_rel, what=what, seed=seed, job_name=name, command=command)
        text = (text.replace('Honest K=2-only tuning on K2_separated_static (no truth, penalties off)',
                             'Honest K=2-only basin-hopping test (no truth)')
                .replace('generate_k2_tune_jobs.py', 'generate_k2_basinhop_jobs.py').replace('2026-10-09', '2026-10-23'))
        with open(os.path.join(PROJECT_ROOT, pbs_rel), 'w') as handle:
            handle.write(text)
        submit.append(f'qsub {pbs_rel}')
    if os.path.exists(os.path.join(out_dir, 'submit_all.sh')):
        raise FileExistsError(os.path.join(args.out_dir, 'submit_all.sh'))
    with open(os.path.join(out_dir, 'submit_all.sh'), 'w') as handle:
        handle.write('\n'.join(submit) + '\n')
    print(f"Done. Output saved to {args.out_dir}/ ({len(jobs)} PBS + submit_all.sh)")


if __name__ == '__main__':
    main()
