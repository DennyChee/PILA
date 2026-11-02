#!/usr/bin/env python
# Usage:       python generate_k2_nonstatic_jobs.py [--seeds 1-8] [--out-dir jobs/multisource_k2_nonstatic]
#                  [--albums-dir synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2]
#                  [--arms slot_resid,slot_resid_dvl1u300] [--no-classical]
#              Smoothing follow-up (2026-10-27): --arms slot_resid_curv5000,slot_resid_curv15000_dvl1u300 --no-classical
#                  --out-dir jobs/multisource_k2_nonstatic_smooth
# Description: Writes PBS scripts (NOT submitted) for the non-static K=2 campaign
#              (plan ~/.claude/plans/synthetic-tests-for-pila-smooth-candle.md, Step 5):
#                classical  one job per album without a finished classical_k2_perdate.py result (cold per-date fits)
#                PILA       every album x 2 arms x seeds, honest K=2 from scratch, basin hopping (slot_resid), 400 epochs:
#                             slot_resid          the K=2-study recipe unchanged (penalties off, warm restarts T_0=100)
#                             slot_resid_dvl1u300 the same + dV-L1 sparsity (u-space, weight 300) alone from epoch 1
#                                                 (KL and z_phy sampling stay off, as in the baseline)
#              The 16 existing slot_resid runs (seeds 1-8) on E1_S3_K2_separated_n025/n050 are REUSED for
#              G1_S3_static_n0250/n0500 (identical albums, checked by make_nonstatic_albums.py regression) -> no new jobs;
#              the mapping is written to reused_runs.json.
#              Waves (submit_<wave>.sh): 01_classical, then one wave per G1 scenario, then the random layouts in waves
#              of <= 200 jobs. No job depends on another (no held jobs).
# Date:        2026-10-26

import argparse
import glob
import json
import os

from generate_k2_tune_jobs import PBS_TEMPLATE, parse_seeds

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PILA_SAVE_ROOT = 'saved/multisource_k2_nonstatic'
CLASSICAL_SAVE_ROOT = 'saved/multisource_k2_nonstatic_classical_v2'
LR_SCHEDULER = '\'{"type":"CosineAnnealingWarmRestarts","args":{"T_0":100}}\''
ARMS = {  # arm: (extra flags, description)
    'slot_resid': ('--off-epochs 400 --ramp-epochs 0', 'penalties off, residual-guided slot move'),
    'slot_resid_dvl1u300': ('--off-epochs 0 --ramp-epochs 0 --kl-stays-off --dv-l1-weight 300',
                            'residual-guided slot move + dV-L1 (u-space, 300) alone'),
    # 2026-10-27 temporal-smoothing follow-up (user: "full test"); KL / z_phy sampling off as in the arms above
    'slot_resid_curv5000': ('--off-epochs 0 --ramp-epochs 0 --kl-stays-off --temporal-smoothness-weight 5000',
                            'residual-guided slot move + curvature smoothing (5000, incl. dV) alone'),
    'slot_resid_curv15000_dvl1u300': ('--off-epochs 0 --ramp-epochs 0 --kl-stays-off --temporal-smoothness-weight 15000 '
                                      '--dv-l1-weight 300',
                                      'residual-guided slot move + standard penalties: curvature 15000 + dV-L1(u) 300'),
}
ARM_JOB_TAG = {'slot_resid': 'base', 'slot_resid_dvl1u300': 'dvl1', 'slot_resid_curv5000': 'curv',
               'slot_resid_curv15000_dvl1u300': 'cdvl'}   # short PBS job-name suffix per arm
REUSED = {  # G1 album -> existing tag template (same album bytes, slot_resid only)
    'G1_S3_static_n0250': 'saved/multisource_k2_tune/k2tune_snrE1n025bh_slot_resid_seed{seed}',
    'G1_S3_static_n0500': 'saved/multisource_k2_tune/k2tune_snrE1n050bh_slot_resid_seed{seed}',
}
SCENARIOS = ['S3_static', 'N1_staggered', 'N2_opposite', 'N3_pulse', 'N4_moving']
R_WAVE_SIZE = 200


def render(job_name, command, what, seed, pbs_rel, walltime='01:00:00'):
    """Fill the shared PBS template (project/queue/resources as all K=2 campaigns) with this campaign's header."""
    text = PBS_TEMPLATE.format(pbs_rel=pbs_rel, what=what, seed=seed, job_name=job_name, command=command)
    return (text.replace('Honest K=2-only tuning on K2_separated_static (no truth, penalties off)',
                         'Non-static K=2 campaign (no truth used in fitting)')
            .replace('generate_k2_tune_jobs.py', 'generate_k2_nonstatic_jobs.py').replace('2026-10-09', '2026-10-26')
            .replace('walltime=01:00:00', f'walltime={walltime}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='1-8')
    parser.add_argument('--out-dir', default='jobs/multisource_k2_nonstatic')
    parser.add_argument('--albums-dir', default='synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2')
    parser.add_argument('--arms', default='slot_resid,slot_resid_dvl1u300',
                        help='comma-separated PILA arms (keys of ARMS)')
    parser.add_argument('--no-classical', action='store_true', help='write no classical jobs (already run)')
    args = parser.parse_args()
    arms = args.arms.split(',')
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError(f'unknown arms {unknown}; choose from {list(ARMS)}')
    seeds = parse_seeds(args.seeds)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f'{args.out_dir} is not empty; not overwriting')
    os.makedirs(out_dir, exist_ok=True)

    print("[1/3] Listing albums...")
    albums = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(PROJECT_ROOT, args.albums_dir, '*.npz'))
                    if not p.endswith('_nonoise.npz'))
    if len(albums) != 90:
        raise ValueError(f'expected 90 albums, found {len(albums)}')
    print(f"  {len(albums)} albums")

    print("[2/3] Building jobs...")
    waves = {}
    # Classical: skip albums already scored (the clean albums were run interactively as the sanity check)
    for album in albums:
        if args.no_classical:
            break
        if os.path.exists(os.path.join(PROJECT_ROOT, CLASSICAL_SAVE_ROOT, album, 'score_summary.json')):
            continue
        command = f'python -u classical_k2_perdate.py --album {album} --albums-dir {args.albums_dir}'
        waves.setdefault('01_classical', []).append((f'k2nsC_{album}', command, f'classical per-date K=2 fit of {album}', 0))
    # PILA
    reused = {}
    r_jobs = []
    for album in albums:
        album_rel = os.path.join(args.albums_dir, f'{album}.npz')
        for arm in arms:
            flags, what = ARMS[arm]
            for seed in seeds:
                if arm == 'slot_resid' and album in REUSED:
                    run = REUSED[album].format(seed=seed)
                    if not glob.glob(os.path.join(PROJECT_ROOT, run, '*', 'K2', '*', '*', 'models', 'basin_best', 'fit_K2.npz')):
                        raise FileNotFoundError(f'reused run missing: {run}')
                    reused.setdefault(album, {})[seed] = run
                    continue
                tag = f'k2ns_{album}_{arm}_seed{seed}'
                if glob.glob(os.path.join(PROJECT_ROOT, PILA_SAVE_ROOT, tag)):
                    raise FileExistsError(f'{PILA_SAVE_ROOT}/{tag} exists')
                command = (f'python -u run_k2_tune_basinhop.py --album {album_rel} --seed {seed} --epochs 400 --tag {tag} '
                           f'--save-root {PILA_SAVE_ROOT} --lr-scheduler {LR_SCHEDULER} {flags} '
                           f'--perturbation slot_resid --settle-lr-peak-frac 0.5')
                job = (f'k2ns_{album}_{ARM_JOB_TAG[arm]}_s{seed}', command,
                       f'{album}, {what}', seed)
                if album.startswith('G1_'):
                    scenario = next(s for s in SCENARIOS if s in album)
                    waves.setdefault(f'{2 + SCENARIOS.index(scenario):02d}_G1_{scenario}', []).append(job)
                else:
                    r_jobs.append(job)
    for index in range(0, len(r_jobs), R_WAVE_SIZE):
        waves[f'{7 + index // R_WAVE_SIZE:02d}_R_part{index // R_WAVE_SIZE + 1}'] = r_jobs[index:index + R_WAVE_SIZE]

    print(f"[3/3] Writing PBS scripts to {args.out_dir}/...")
    os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
    total = 0
    for wave, jobs in waves.items():
        lines = ['#!/bin/bash', f'# Submit from the project root: bash {args.out_dir}/submit_{wave}.sh', 'set -e']
        for job_name, command, what, seed in jobs:
            pbs_rel = os.path.join(args.out_dir, f'{job_name}.pbs')
            with open(os.path.join(PROJECT_ROOT, pbs_rel), 'x') as handle:     # 'x' = never overwrite
                handle.write(render(job_name, command, what, seed, pbs_rel))
            lines.append(f'qsub {pbs_rel} | tee -a {args.out_dir}/submitted_job_ids.txt')
        with open(os.path.join(out_dir, f'submit_{wave}.sh'), 'x') as handle:
            handle.write('\n'.join(lines) + '\n')
        total += len(jobs)
        print(f"  {wave}: {len(jobs)} jobs")
    with open(os.path.join(out_dir, 'reused_runs.json'), 'x') as handle:
        json.dump(reused, handle, indent=1)
    print(f"Done. Output saved to {args.out_dir}/ ({total} PBS scripts, {sum(len(v) for v in reused.values())} reused runs)")


if __name__ == '__main__':
    main()
