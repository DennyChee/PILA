#!/usr/bin/env python
# Usage:       python run_sequential_album.py --album synthetic/multisource/pila_stacks_v2/albums/E1_S3_K2_separated.npz \
#                  [--kmax 3] [--save-root saved/multisource_seq] [--seed 1] [--epochs 150] [--tag main]
#                  [--bootstrap-k N --bootstrap-block B --bootstrap-seed S] [--amp-box]
# Description: Sequential multi-source PILA on ONE album: train K = 1, then K = 2 warm-started from the
#              K = 1 fit, ..., up to K = kmax. Each training run auto-dumps fit_K<k>.npz (per-acquisition,
#              per-source parameters and fields) for the source-count selection (sequential_select_k.py).
#              All K are trained (the stopping decision is made afterwards, which is equivalent and is
#              needed for the noise-only calibration albums anyway).
# Output:      <save-root>/<tag>/<album>/K<k>/<album>_K<k>/<MMDD_HHMMSS>/models/{model_best.pth, fit_K<k>.npz,
#              compute_usage.json, timing_epochs.csv, infer_usage_K<k>.json}, plus
#              <save-root>/<tag>/<album>/sequence.json (checkpoints, wall time per K).
# Notes:       Resumable: a K whose fit_K<k>.npz already exists is skipped. Seeds per K = seed (same RNG
#              seed for every K of a member). Uses MKL_THREADING_LAYER=GNU like the PBS jobs.
#              --amp-box: data-scaled dV prior box (utils/amp_box.py, +-2e8 m^3 * album std / 12 mm), written once
#              to <album_root>/mogi_paras_symdV_ampbox.json and used by EVERY K so warm starts keep identical endpoints.
# Date:        2026-10-06 (--amp-box 2026-10-22)

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from sequential_configs import make_sequential_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def newest(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', required=True)
    parser.add_argument('--kmax', type=int, default=3)
    parser.add_argument('--save-root', default='saved/multisource_seq')
    parser.add_argument('--tag', default='main', help='sub-folder, e.g. main / ens_s3_b2')
    parser.add_argument('--seed', type=int, default=None, help='default: 123 (standard/original) / 1 (synth_v1)')
    parser.add_argument('--recipe', default='standard', choices=['standard', 'original', 'synth_v1'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--residual-rank', type=int, default=4, help='0 = physics-only PILA (no low-rank residual)')
    parser.add_argument('--nuisance-ramp', choices=['default', 'on', 'off'], default='default',
                        help="per-acquisition offset + ramp; default = recipe's choice (original: off)")
    parser.add_argument('--seed-depth-km', type=float, default=None,
                        help='opt-in test: start new sources at this depth (km); default = standard recipe')
    parser.add_argument('--seed-k1-at-peak', action='store_true',
                        help='with --seed-depth-km: also start source 1 at the stack-mean data peak')
    parser.add_argument('--init-sources', default=None,
                        help="JSON [[xcen_km, ycen_km, depth_km, dV_m3], ...]: train ONLY K=len(rows) from a fresh network started "
                             "at these sources (K>1 init test); skips the K-1 warm-start chain")
    parser.add_argument('--from-scratch-k', type=int, default=None,
                        help="train ONLY K=this (e.g. 2) from a fresh network with the default random start: no K-1 stage, "
                             "no warm start, no truth")
    parser.add_argument('--lr-scheduler', default=None,
                        help='JSON {"type": ..., "args": {...}} replacing the cosine scheduler, e.g. '
                             '\'{"type":"CosineAnnealingWarmRestarts","args":{"T_0":50}}\' (standard recipe only)')
    parser.add_argument('--no-penalties', action='store_true',
                        help="control run: switch OFF the KL term, dV-L1 and temporal curvature (pure reconstruction loss); "
                             "default = standard recipe incl. these penalties")
    parser.add_argument('--amp-box', action='store_true',
                        help='data-scaled dV prior box (utils/amp_box.py); same box for every K of the album')
    parser.add_argument('--bootstrap-k', type=int, default=None)
    parser.add_argument('--bootstrap-block', type=int, default=3)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    args = parser.parse_args()
    album_npz = os.path.abspath(args.album)
    if not os.path.exists(album_npz):
        raise FileNotFoundError(album_npz)
    album = os.path.basename(album_npz)[:-4]
    album_root = os.path.join(PROJECT_ROOT, args.save_root, args.tag, album)
    os.makedirs(album_root, exist_ok=True)
    insar_extra = None
    if args.bootstrap_k:
        insar_extra = dict(bootstrap_k=args.bootstrap_k, bootstrap_block=args.bootstrap_block,
                           bootstrap_seed=args.bootstrap_seed)
    seed_init = None
    if args.seed_depth_km is not None:
        seed_init = {'depth_km': args.seed_depth_km, 'at_peak': bool(args.seed_k1_at_peak)}
    init_sources = json.loads(args.init_sources) if args.init_sources else None
    lr_scheduler = json.loads(args.lr_scheduler) if args.lr_scheduler else None
    trainer_overrides = ({'use_kl_term_z_phy': False, 'dv_l1_weight': 0.0, 'temporal_smoothness_weight': 0.0}
                         if args.no_penalties else None)
    env = {**os.environ, 'MKL_THREADING_LAYER': 'GNU', 'PYTHONUNBUFFERED': '1'}

    sequence = {'album': album_npz, 'seed': args.seed, 'epochs': args.epochs, 'insar_extra': insar_extra,
                'residual_rank': args.residual_rank, 'recipe': args.recipe,
                'nuisance_ramp': args.nuisance_ramp, 'seed_init': seed_init, 'init_sources': init_sources, 'trainer_overrides': trainer_overrides, 'lr_scheduler': lr_scheduler, 'amp_box': None, 'runs': {}}
    previous_ckpt = None
    t_album0 = time.time()
    if init_sources and args.from_scratch_k:
        raise ValueError('--init-sources and --from-scratch-k are exclusive')
    k_values = ([len(init_sources)] if init_sources else [args.from_scratch_k] if args.from_scratch_k
                else range(1, args.kmax + 1))                      # explicit start / from scratch: only that K
    for k in k_values:
        k_dir = os.path.join(album_root, f'K{k}')
        done_fit = newest(os.path.join(k_dir, '*', '*', 'models', f'fit_K{k}.npz'))
        if done_fit:
            print(f"[K={k}] already done: {done_fit}")
            previous_ckpt = os.path.join(os.path.dirname(done_fit), 'model_best.pth')
            sequence['runs'][k] = {'checkpoint': previous_ckpt, 'fit': done_fit, 'skipped': True}
            continue
        print(f"[K={k}/{max(k_values)}] Training {album} with {k} source(s)"
              + (f", warm start from {previous_ckpt}" if previous_ckpt else '') + "...")
        cfg = make_sequential_config(album_npz, k, k_dir, init_from=previous_ckpt, seed=args.seed,
                                     epochs=args.epochs, insar_extra=insar_extra,
                                     residual_rank=args.residual_rank, recipe=args.recipe, seed_init=seed_init, init_sources=init_sources, trainer_overrides=trainer_overrides, from_scratch=bool(args.from_scratch_k), lr_scheduler=lr_scheduler,
                                     nuisance_ramp=None if args.nuisance_ramp == 'default' else args.nuisance_ramp == 'on')
        if args.amp_box:
            # One box per album (computed from the album's own LOS std, never the truth), shared by all K.
            if sequence['amp_box'] is None:
                from utils.amp_box import write_data_scaled_paras
                paras_path, box_info = write_data_scaled_paras(cfg, album_root, project_root=PROJECT_ROOT)
                sequence['amp_box'] = {**box_info, 'paras_path': paras_path}
                print(f"  dV box +-{box_info['box']:.4e} m^3 (album std {box_info['std_mm']:.3f} mm) -> {paras_path}")
            cfg['arch']['args']['mogi_paras'] = sequence['amp_box']['paras_path']
        os.makedirs(k_dir, exist_ok=True)
        cfg_path = os.path.join(k_dir, f'config_K{k}.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        t0 = time.time()
        proc = subprocess.run([sys.executable, 'train_pila.py', '-c', cfg_path], cwd=PROJECT_ROOT, env=env)
        wall_sec = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"K={k} training failed (exit {proc.returncode}); see {k_dir}")
        ckpt = newest(os.path.join(k_dir, '*', '*', 'models', 'model_best.pth'))
        fit = os.path.join(os.path.dirname(ckpt), f'fit_K{k}.npz')
        if not os.path.exists(fit):
            raise RuntimeError(f"K={k}: training finished but {fit} was not written")
        sequence['runs'][k] = {'checkpoint': ckpt, 'fit': fit, 'wall_sec': round(wall_sec, 1)}
        print(f"  K={k} done in {wall_sec:.0f}s -> {fit}")
        previous_ckpt = ckpt
    sequence['album_wall_sec'] = round(time.time() - t_album0, 1)
    with open(os.path.join(album_root, 'sequence.json'), 'w') as f:
        json.dump(sequence, f, indent=2)
    print(f"Done. Output saved to {album_root}/ ({sequence['album_wall_sec']:.0f}s total)")


if __name__ == '__main__':
    main()
