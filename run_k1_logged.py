#!/usr/bin/env python
# Usage:       python run_k1_logged.py --album <npz> --seed 123 --tag k1amp_seed123 [--save-root saved/multisource_k1_amp]
# Description: Standard-recipe K=1 PILA, built EXACTLY like the K=1 stage of
#              `run_sequential_album.py --kmax 1 --recipe standard` (same make_sequential_config call and
#              arguments), but trained with train_pila_logsources.py so per_epoch_sources.csv records how
#              depth / dV / location evolve. Same output layout, so summarize_k1_snr_runs.py and
#              plot_multisource_report.py work unchanged.
#              --dv-max-m3 V      : symmetric dV prior box +-V (m^3) instead of the standard +-2e8.
#              --dv-scale-to-data : DATA-SCALED box, V = 2e8 * (album global LOS std / ANCHOR_STD_MM). Mogi LOS is linear
#                                   in dV and PILA already divides data AND decoder output by that std, so this makes
#                                   the whole inversion invariant to the album's absolute amplitude. Uses only the data
#                                   (never the truth). The anchor (12 mm = standard box) only sets the overall prior width.
#              The modified paras JSON is written into the run's K1 dir and referenced by absolute path.
# Output:      <save-root>/<tag>/<album>/K1/<album>_K1/<MMDD_HHMMSS>/models/{fit_K1.npz, per_epoch_sources.csv, ...}
# Date:        2026-10-20

import argparse
import json
import os
import subprocess
import sys
import time

from sequential_configs import make_sequential_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STANDARD_DV_MAX_M3 = 2.0e8      # configs/mogi_paras_symdV.json box
ANCHOR_STD_MM = 12.0            # data std at which the data-scaled box equals the standard box


def album_global_std_mm(album_npz):
    """Global LOS std (mm) over all finite cells and epochs = the loader's x_scale_global for these dense albums."""
    import numpy as np
    with np.load(album_npz) as f:
        los_mm = f['los_asc_m'].astype(np.float64) * 1e3
    return float(np.nanstd(los_mm[np.isfinite(los_mm)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--save-root', default='saved/multisource_k1_amp')
    parser.add_argument('--dv-max-m3', type=float, default=None, help='symmetric dV box +-V m^3')
    parser.add_argument('--dv-scale-to-data', action='store_true', help='box = 2e8 * data_std / 12 mm')
    args = parser.parse_args()

    # --- [1/3] Validate and refuse to reuse a run directory ---
    print(f"[1/3] Validating {args.album}...")
    album_npz = os.path.abspath(args.album)
    if not os.path.exists(album_npz):
        raise FileNotFoundError(album_npz)
    album = os.path.basename(album_npz)[:-4]
    album_root = os.path.join(PROJECT_ROOT, args.save_root, args.tag, album)
    if os.path.exists(album_root):
        raise FileExistsError(f"{album_root} exists; pick a new --tag")
    k_dir = os.path.join(album_root, 'K1')
    os.makedirs(k_dir)

    # --- [2/3] Config: identical call to run_sequential_album.py's K=1 stage with default flags ---
    cfg = make_sequential_config(album_npz, 1, k_dir, init_from=None, seed=args.seed, epochs=args.epochs,
                                 insar_extra=None, residual_rank=4, recipe='standard', seed_init=None,
                                 init_sources=None, trainer_overrides=None, from_scratch=False, lr_scheduler=None,
                                 nuisance_ramp=None)
    if args.dv_max_m3 is not None and args.dv_scale_to_data:
        raise ValueError('--dv-max-m3 and --dv-scale-to-data are exclusive')
    if args.dv_max_m3 is not None or args.dv_scale_to_data:
        data_std_mm = album_global_std_mm(album_npz)
        dv_max_m3 = args.dv_max_m3 if args.dv_max_m3 is not None else STANDARD_DV_MAX_M3 * data_std_mm / ANCHOR_STD_MM
        paras_src = os.path.join(PROJECT_ROOT, cfg['arch']['args']['mogi_paras'])
        paras = json.load(open(paras_src))
        if paras['dV'].get('scale', 1e5) != 1 or paras['dV'].get('shift', -1e7) != 0:
            raise ValueError(f'{paras_src}: expected the symmetric dV map (scale 1, shift 0)')
        paras['dV']['min'], paras['dV']['max'] = -dv_max_m3, dv_max_m3       # symmetric: dV=0 stays at z=0.5
        paras_path = os.path.join(k_dir, 'mogi_paras_dvbox.json')
        with open(paras_path, 'w') as handle:
            json.dump(paras, handle, indent=4)
        cfg['arch']['args']['mogi_paras'] = paras_path       # absolute; os.path.join(PARENT_DIR, abs) keeps it
        print(f"  dV box +-{dv_max_m3:.4e} m^3 (data std {data_std_mm:.3f} mm) -> {paras_path}")
    cfg_path = os.path.join(k_dir, 'config_K1.json')
    with open(cfg_path, 'w') as handle:
        json.dump(cfg, handle, indent=2)
    print(f"[2/3] Config written: {cfg_path}")

    # --- [3/3] Train with per-epoch source logging ---
    print("[3/3] Training with train_pila_logsources.py...")
    env = {**os.environ, 'MKL_THREADING_LAYER': 'GNU', 'PYTHONUNBUFFERED': '1'}
    t0 = time.time()
    proc = subprocess.run([sys.executable, 'train_pila_logsources.py', '-c', cfg_path], cwd=PROJECT_ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed (exit {proc.returncode}); see {k_dir}")
    with open(os.path.join(album_root, 'sequence.json'), 'w') as handle:
        json.dump({'album': album_npz, 'seed': args.seed, 'epochs': args.epochs, 'recipe': 'standard',
                   'driver': 'run_k1_logged.py', 'mogi_paras': cfg['arch']['args']['mogi_paras'], 'wall_sec': round(time.time() - t0, 1)}, handle, indent=2)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"Done. Output saved to {album_root}/")


if __name__ == '__main__':
    main()
