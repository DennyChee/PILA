#!/usr/bin/env python
# Usage:       python run_k2_tune.py --album synthetic/multisource/clean_scenarios/K2_separated_static.npz \
#                  --seed 3 --epochs 300 --tag k2tune_base_seed3 [--save-root saved/multisource_k2_tune] \
#                  [--optimizer-args '{"lr":1e-3}'] [--lr-scheduler '{"type":"CosineAnnealingWarmRestarts","args":{"T_0":100}}'] \
#                  [--physics-only] [--random-init] [--latent-sampling] [--dry-run]
# Description: Hyper-parameter tuning driver for the HONEST K=2-only from-scratch PILA test (no truth, penalties off).
#              Standalone wrapper: builds the config with sequential_configs.make_sequential_config EXACTLY like
#              run_sequential_album.py --from-scratch-k 2 --no-penalties, then mutates the returned dict
#              (optimizer args, scheduler, early_stop, physics-only, random start, latent sampling) and calls
#              train_pila.py. No existing code is modified. With no tuning flag the config equals the original
#              recipe except trainer.early_stop (set to the epoch count so warm restarts are never cut short).
# Output:      <save-root>/<tag>/<album>/K2/<album>_K2/<MMDD_HHMMSS>/models/{model_best.pth, fit_K2.npz, ...} and
#              <save-root>/<tag>/<album>/sequence.json -- same layout as run_sequential_album.py, so
#              summarize_k2_seed_runs.py (--root) and plot_multisource_report.py work unchanged.
# Assumptions: Units km / m^3 (init_sources rows = [xcen_km, ycen_km, depth_km, dV_m3]). --random-init draws the start
#              from a seed-keyed RNG with bounds that do NOT use the truth (x,y +-14 km, depth 0.5-12 km, |dV| log-uniform
#              1e7-1e8 m^3, random sign) -- needs the user's OK that this counts as an honest start.
# Date:        2026-10-09

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np

from sequential_configs import make_sequential_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NO_PENALTY_OVERRIDES = {'use_kl_term_z_phy': False, 'dv_l1_weight': 0.0, 'temporal_smoothness_weight': 0.0}
N_SOURCES = 2


def draw_random_init_sources(seed):
    """
    Honest random start for K=2 sources (never uses the truth or the scene; bounds span the generic prior box, not the
    truth: x,y +-14 km (prior +-15), depth 0.5-12 km (prior range), |dV| log-uniform 1e7-1e8 m^3 (prior +-2e8; floor 1e7 because the default head's random offset moves dV by up to ~2.5e7 and would swamp smaller draws), random sign).
    Applied BIAS-ONLY by train_pila_biasinit.py (weights and lnvar head keep the default random init), so the only
    difference from the baseline is the starting position.

    Input:  seed (int) -- RNG seed, same as the training seed so each seed gets a reproducible, different start.
    Output: list of 2 rows [xcen_km, ycen_km, depth_km, dV_m3].
    """
    rng = np.random.RandomState(int(seed))
    rows = []
    for _ in range(N_SOURCES):
        xcen_km = rng.uniform(-14.0, 14.0)
        ycen_km = rng.uniform(-14.0, 14.0)
        depth_km = rng.uniform(0.5, 12.0)
        dv_m3 = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(7.0, 8.0)   # log-uniform magnitude, floor 1e7 (smaller |dV| is swamped by the default head offset, up to ~2.5e7), random sign: do not assume inflation
        rows.append([float(xcen_km), float(ycen_km), float(depth_km), float(dv_m3)])
    return rows


def build_config(args, k_dir):
    """
    Original-recipe config for K=2 from scratch, then the requested tuning mutations.

    Inputs:  parsed args; k_dir (str) run output root. Output: config dict (not yet written).
    """
    init_sources = draw_random_init_sources(args.seed) if args.random_init else None
    lr_scheduler = json.loads(args.lr_scheduler) if args.lr_scheduler else None
    if lr_scheduler is not None and 'type' not in lr_scheduler:
        raise ValueError('--lr-scheduler JSON needs a "type" key')
    cfg = make_sequential_config(os.path.abspath(args.album), N_SOURCES, k_dir, init_from=None, seed=args.seed,
                                 epochs=args.epochs, recipe='standard', init_sources=init_sources,
                                 trainer_overrides=dict(NO_PENALTY_OVERRIDES), from_scratch=True,
                                 lr_scheduler=lr_scheduler)
    cfg = copy.deepcopy(cfg)
    # --- early_stop: template value 100 would stop a warm-restart run right after a restart (high LR) and save an
    #     unsettled model_best. Never early-stop in this tuning study. ---
    cfg['trainer']['early_stop'] = int(cfg['trainer']['epochs'])
    if args.optimizer_args:
        try:
            optimizer_args = json.loads(args.optimizer_args)
        except json.JSONDecodeError as error:
            raise ValueError(f"--optimizer-args is not valid JSON: {args.optimizer_args!r} ({error})")
        unknown = [key for key in optimizer_args if key not in cfg['optimizer']['args']]
        if unknown:
            raise ValueError(f"unknown optimizer args {unknown}; known {list(cfg['optimizer']['args'])}")
        cfg['optimizer']['args'].update(optimizer_args)
    if args.physics_only:       # no low-rank residual: physics must explain all coherent signal (recipe variant)
        cfg['arch']['phys_vae']['residual_rank'] = 0
        cfg['arch']['phys_vae']['dim_z_aux'] = 0
    if args.latent_sampling:    # sample z_phy (noise) without any KL pull toward the prior
        cfg['trainer']['phys_vae']['use_kl_term_z_phy'] = True
        cfg['trainer']['phys_vae']['beta_max_z_phy'] = 0.0
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--save-root', default='saved/multisource_k2_tune')
    parser.add_argument('--optimizer-args', default=None, help='JSON merged into optimizer.args, e.g. \'{"lr":1e-3}\'')
    parser.add_argument('--lr-scheduler', default=None, help='JSON {"type":..., "args":{...}} replacing the cosine')
    parser.add_argument('--physics-only', action='store_true', help='residual_rank=0, dim_z_aux=0')
    parser.add_argument('--random-init', action='store_true', help='honest per-seed random start via init_sources')
    parser.add_argument('--latent-sampling', action='store_true', help='z_phy sampling on, KL weight 0')
    parser.add_argument('--dry-run', action='store_true', help='write the config and stop (no training; use a scratch --save-root, the run dir is created)')
    args = parser.parse_args()

    # --- [1/4] Validate inputs and refuse to reuse an existing run directory ---
    print(f"[1/4] Validating inputs for tag {args.tag}...")
    album_npz = os.path.abspath(args.album)
    if not os.path.exists(album_npz):
        raise FileNotFoundError(album_npz)
    album = os.path.splitext(os.path.basename(album_npz))[0]
    album_root = os.path.join(PROJECT_ROOT, args.save_root, args.tag, album)
    if os.path.exists(album_root):
        raise FileExistsError(f"{album_root} exists; never reusing a run directory (pick a new --tag)")
    k_dir = os.path.join(album_root, f'K{N_SOURCES}')
    os.makedirs(k_dir)

    # --- [2/4] Build and write the config ---
    print("[2/4] Building config (original recipe + tuning mutations)...")
    cfg = build_config(args, k_dir)
    cfg_path = os.path.join(k_dir, f'config_K{N_SOURCES}.json')
    with open(cfg_path, 'w') as handle:
        json.dump(cfg, handle, indent=2)
    print(f"  optimizer={cfg['optimizer']}, lr_scheduler={cfg['lr_scheduler']}, early_stop={cfg['trainer']['early_stop']}, "
          f"residual_rank={cfg['arch']['phys_vae']['residual_rank']}, "
          f"init_sources={cfg['arch']['phys_vae'].get('init_sources')}")
    if args.dry_run:
        print(f"Done. Dry run, config saved to {cfg_path}")
        return

    # --- [3/4] Train (train_pila.py also dumps fit_K2.npz) ---
    print("[3/4] Training with train_pila.py...")
    env = {**os.environ, 'MKL_THREADING_LAYER': 'GNU', 'PYTHONUNBUFFERED': '1'}
    t0 = time.time()
    train_script = 'train_pila_biasinit.py' if args.random_init else 'train_pila.py'    # bias-only start, see its header
    proc = subprocess.run([sys.executable, train_script, '-c', cfg_path], cwd=PROJECT_ROOT, env=env)
    wall_sec = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"training failed (exit {proc.returncode}); see {k_dir}")
    print(f"  Done in {wall_sec:.1f}s")

    # --- [4/4] Record the run (same sequence.json layout as run_sequential_album.py) ---
    print("[4/4] Writing sequence.json...")
    sequence = {'album': album_npz, 'seed': args.seed, 'epochs': args.epochs, 'early_stop': cfg['trainer']['early_stop'],
                'optimizer': cfg['optimizer'], 'resolved_lr_scheduler': cfg['lr_scheduler'], 'tuning': {
        'optimizer_args': args.optimizer_args, 'lr_scheduler': args.lr_scheduler, 'physics_only': args.physics_only,
        'random_init': args.random_init, 'latent_sampling': args.latent_sampling}, 'wall_sec': round(wall_sec, 1)}
    with open(os.path.join(album_root, 'sequence.json'), 'w') as handle:
        json.dump(sequence, handle, indent=2)
    print(f"Done. Output saved to {album_root}/")


if __name__ == '__main__':
    main()
