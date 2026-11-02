#!/usr/bin/env python
# Usage:       python generate_k1_snr_jobs.py [--albums E1_S1_K1_strong E1_S2_K1_weak] \
#                  [--noise-tags n100 n050 n025 n012 n006] [--seeds 123,1,2] [--out-dir jobs/multisource_k1_snr]
# Description: Writes PBS scripts (NOT submitted) for the single-source SNR sweep: STANDARD-recipe K=1 PILA
#              (curvature 15000 incl. dV, dV-L1 300 u-space, KL beta 0.5, seq loader -- identical to the Step 3 `std`
#              K=1 stage) on the real-TA69-noise K=1 albums with the noise scaled down
#              (synthetic/multisource/make_snr_scaled_albums.py). Question: at what peak/RMS does single-source PILA
#              start recovering the source on real noise? n100 = the original album (scale 1.0).
#              One job per (album, noise scale, seed); one submit script submit_k1_snr.sh. Never overwrites files.
# Output:      saved/multisource_k1_snr/k1snr_seed<S>/<album>[_nXXX]/K1/... (score with summarize_k1_snr_runs.py)
# Date:        2026-10-20

import argparse
import os

from generate_k2_tune_jobs import PBS_TEMPLATE, parse_seeds

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR = 'synthetic/multisource/pila_stacks_v2/albums'
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'
SAVE_ROOT = 'saved/multisource_k1_snr'


def album_path(album, noise_tag):
    """Repo-relative npz path: n100 = original noisy album, otherwise the scaled twin <album>_<noise_tag>."""
    if noise_tag == 'n100':
        return os.path.join(NOISY_DIR, f'{album}.npz')
    return os.path.join(SNR_DIR, f'{album}_{noise_tag}.npz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--albums', nargs='+', default=['E1_S1_K1_strong', 'E1_S2_K1_weak'])
    parser.add_argument('--noise-tags', nargs='+', default=['n100', 'n050', 'n025', 'n012', 'n006'])
    parser.add_argument('--seeds', default='123,1,2', help='123 = the std default seed')
    parser.add_argument('--out-dir', default='jobs/multisource_k1_snr')
    args = parser.parse_args()

    # --- [1/2] Validate every album before writing anything ---
    print("[1/2] Validating albums...")
    seeds = parse_seeds(args.seeds)
    for album in args.albums:
        for noise_tag in args.noise_tags:
            path = album_path(album, noise_tag)
            if not os.path.exists(os.path.join(PROJECT_ROOT, path)):
                raise FileNotFoundError(path)

    # --- [2/2] One PBS script per (album, noise scale, seed) ---
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[2/2] Writing {len(args.albums) * len(args.noise_tags) * len(seeds)} PBS scripts to {args.out_dir}/...")
    submit_lines = ['#!/bin/bash', '# Submit from the project root: bash ' + os.path.join(args.out_dir, 'submit_k1_snr.sh'),
                    'set -e']
    for album in args.albums:
        short = 's' if 'strong' in album else 'w'          # PBS names are truncated, keep them short
        for noise_tag in args.noise_tags:
            for seed in seeds:
                job_name = f'k1{short}{noise_tag[1:]}_s{seed}'
                pbs_rel = os.path.join(args.out_dir, f'{job_name}.pbs')
                pbs_abs = os.path.join(PROJECT_ROOT, pbs_rel)
                if os.path.exists(pbs_abs):
                    raise FileExistsError(f"{pbs_rel} exists; not overwriting")
                # --kmax 1 with the standard recipe = exactly the std K=1 stage (K=1 always starts from scratch)
                command = (f'python -u run_sequential_album.py --album {album_path(album, noise_tag)} --kmax 1 '
                           f'--recipe standard --seed {seed} --save-root {SAVE_ROOT} --tag k1snr_seed{seed}')
                text = PBS_TEMPLATE.format(pbs_rel=pbs_rel, what=f'K=1 SNR sweep {album} {noise_tag}',
                                           seed=seed, job_name=job_name, command=command)
                # The shared template's description line is K=2-specific; correct it for this study.
                text = text.replace('Honest K=2-only tuning on K2_separated_static (no truth, penalties off)',
                                    'Single-source K=1 standard-recipe PILA (penalties ON, no truth)')
                text = text.replace('generate_k2_tune_jobs.py', 'generate_k1_snr_jobs.py').replace('2026-10-09', '2026-10-20')
                with open(pbs_abs, 'w') as handle:
                    handle.write(text)
                submit_lines.append(f'qsub {pbs_rel}')
    submit_path = os.path.join(out_dir, 'submit_k1_snr.sh')
    if os.path.exists(submit_path):
        raise FileExistsError(submit_path)
    with open(submit_path, 'w') as handle:
        handle.write('\n'.join(submit_lines) + '\n')
    print(f"Done. Output saved to {args.out_dir}/ (submit with: bash {args.out_dir}/submit_k1_snr.sh)")


if __name__ == '__main__':
    main()
