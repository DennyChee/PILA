#!/usr/bin/env python
# Usage:       python extend_k2_polish.py run   --run-models-dir saved/.../models --tag k2ext_slot_resid_seed1 [--extra-epochs 200]
#              python extend_k2_polish.py score --tag k2ext_slot_resid_seed1
# Description: Late-rescue test (2026-10-26). Hypothesis: basin-hop runs that separate the two sources late (e.g. seed 1 at
#              epoch ~274) end at a higher album misfit than early ones (seed 7, ~2111) only because they ran out of polish
#              epochs. "run" continues from the run's checkpoint-epoch400.pth (weights + Adam state; basin_best has no
#              optimizer state and a fresh Adam blew the fit up in the smoke test) and trains --extra-epochs more with a
#              second, longer polish: cosine LR from --lr-peak to --lr-floor (the original 60-epoch polish went
#              1.5e-4 -> 1.5e-5), NO hopping, and the SAME penalty regime as the original run (launched through
#              train_pila_delayedpen.py; a penalties-off run keeps them off for the whole extension).
#              "score" dumps every checkpoint (incl. the epoch-400 start) and scores it with the same
#              deterministic album misfit and truth rules as compare_k2_basinhop.py -> <out>/extension_scores.csv.
#              Truth is used only for scoring. The original run directory is never modified.
# Date:        2026-10-26

import argparse
import glob
import json
import os
import runpy
import shutil
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = 'saved/multisource_k2_polishext'
TRUTH = 'synthetic/multisource/pila_stacks_v2/albums_snr/E1_S3_K2_separated_n025_truth.json'
DEFAULT_START_EPOCH = 400       # the extension continues the original 400-epoch numbering (--start-epoch to chain)


def prepare(args, out_dir):
    """Write the extension config + starting checkpoint (copy of checkpoint-epoch400.pth) into out_dir/init/."""
    models_dir = os.path.join(PROJECT_ROOT, args.run_models_dir)
    best_path = os.path.join(models_dir, f'checkpoint-epoch{args.start_epoch}.pth')
    for path in (best_path, os.path.join(models_dir, 'config.json')):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    init_dir = os.path.join(out_dir, 'init')
    if os.path.exists(init_dir):
        raise FileExistsError(f'{init_dir} exists; use a new --tag (nothing is overwritten)')
    os.makedirs(init_dir)

    with open(os.path.join(models_dir, 'config.json')) as handle:
        config = json.load(handle)
    trainer_cfg = config['trainer']
    phys = trainer_cfg['phys_vae']
    schedule = phys.get('delayed_penalty')
    if not schedule:
        raise ValueError('expected a delayed_penalty block (basin-hop runs always have one)')
    if int(schedule['off_epochs']) >= int(trainer_cfg['epochs']):
        # Penalties were off for the whole original run: keep them off for the extension too
        schedule['off_epochs'] = 10 ** 6
    phys.pop('basin_hop', None)                         # no hopping in the extension
    total_epochs = args.start_epoch + args.extra_epochs
    trainer_cfg.update(epochs=total_epochs, early_stop=total_epochs, save_period=args.save_period,
                       save_dir=os.path.join(out_dir, 'train'))
    config['optimizer']['args']['lr'] = args.lr_peak
    config['lr_scheduler'] = {'type': 'CosineAnnealingLR', 'args': {'T_max': args.extra_epochs, 'eta_min': args.lr_floor}}
    config['extension'] = dict(source_models_dir=args.run_models_dir, start_checkpoint=best_path, start_epoch=args.start_epoch,
                               extra_epochs=args.extra_epochs, lr_peak=args.lr_peak, lr_floor=args.lr_floor,
                               created='2026-10-26')
    with open(os.path.join(init_dir, 'config.json'), 'w') as handle:
        json.dump(config, handle, indent=2)

    # -r reads config.json from the checkpoint's own folder, so the checkpoint is copied next to the new config
    init_ckpt = os.path.join(init_dir, f'checkpoint-epoch{args.start_epoch}.pth')
    shutil.copy(best_path, init_ckpt)
    return init_ckpt


def run(args):
    out_dir = os.path.join(PROJECT_ROOT, OUT_ROOT, args.tag)
    print(f"[1/2] Preparing {out_dir} from {args.run_models_dir}...")
    init_ckpt = prepare(args, out_dir)

    print(f"[2/2] Training epochs {args.start_epoch + 1}-{args.start_epoch + args.extra_epochs} (no hopping)...")
    t0 = time.time()
    # Resume loads weights + Adam state. Adam's load_state_dict also restores the OLD param-group lr, and the cosine
    # scheduler (built from the config before the resume, base_lrs = --lr-peak) steps from the group's current lr, so
    # re-impose --lr-peak on the groups afterwards (in memory only, this process only).
    from base.base_trainer import BaseTrainer
    original_resume = BaseTrainer._resume_checkpoint

    def resume_then_set_lr(self, resume_path):
        original_resume(self, resume_path)
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr'] = args.lr_peak
        print(f"  [extend] lr reset to {args.lr_peak:g} after resume")

    BaseTrainer._resume_checkpoint = resume_then_set_lr
    sys.argv = ['train_pila_delayedpen.py', '-c', os.path.join(os.path.dirname(init_ckpt), 'config.json'), '-r', init_ckpt]
    runpy.run_path(os.path.join(PROJECT_ROOT, 'train_pila_delayedpen.py'), run_name='__main__')
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"Done. Output saved to {out_dir}/")


def score(args):
    from compare_k2_basinhop import deterministic_misfit, run_x_scale
    from compare_k2_keepbest import score_fit
    import pandas as pd
    out_dir = os.path.join(PROJECT_ROOT, OUT_ROOT, args.tag)
    with open(os.path.join(PROJECT_ROOT, TRUTH)) as handle:
        truth = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km'], s['dV_m3']] for s in json.load(handle)['sources']])
    models = glob.glob(os.path.join(out_dir, 'train', '*', '*', 'models'))
    if len(models) != 1:
        raise FileNotFoundError(f'{out_dir}/train/*/*/models: expected 1 run dir, found {len(models)}')
    models_dir = models[0]
    with open(os.path.join(out_dir, 'init', 'config.json')) as handle:
        init_config = json.load(handle)
    album = init_config['arch']['args']['insar']['timeseries']
    start_epoch = int(init_config['extension']['start_epoch'])
    if 'E1_S3_K2_separated_n025' not in album:          # TRUTH above is the SNR 12 E1 album only
        raise ValueError(f'score only knows the E1 n025 truth, run album is {album}')
    x_scale = run_x_scale(os.path.join(out_dir, 'init'))
    checkpoints = sorted(glob.glob(os.path.join(models_dir, 'checkpoint-epoch*.pth')),
                         key=lambda path: int(path.split('epoch')[-1].split('.')[0]))
    print(f"[1/2] Scoring start fit + {len(checkpoints)} checkpoints (x_scale {x_scale:.3f} mm)...")
    rows = []
    fits = []
    for checkpoint in [os.path.join(out_dir, 'init', f'checkpoint-epoch{start_epoch}.pth')] + checkpoints:
        epoch = int(checkpoint.split('epoch')[-1].split('.')[0])
        fit_path = os.path.join(os.path.dirname(checkpoint), f'fit_K2_epoch{epoch}.npz')
        if not os.path.exists(fit_path):
            subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'dump_multisource_fit.py'), '--resume', checkpoint,
                            '--out', fit_path], check=True, cwd=PROJECT_ROOT)
        fits.append((epoch, fit_path))
    for epoch, fit_path in fits:
        misfit, phys_misfit = deterministic_misfit(fit_path, x_scale)
        rows.append(dict(tag=args.tag, epoch=epoch, misfit=misfit, phys_misfit=phys_misfit, **score_fit(fit_path, truth)))
        print(f"  epoch {epoch}: misfit {misfit:.1f}, loc {rows[-1]['loc_max_km']:.2f} km, "
              f"depth {rows[-1]['depth_pct_max']:.0f} %, dV {rows[-1]['dV_pct_max']:.0f} %")
    print("[2/2] Writing extension_scores.csv...")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'extension_scores.csv'), index=False, float_format='%.5g')
    print(f"Done. Output saved to {out_dir}/extension_scores.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--run-models-dir', required=True, help='original run models/ dir (relative to project)')
    run_parser.add_argument('--tag', required=True)
    run_parser.add_argument('--extra-epochs', type=int, default=200)
    run_parser.add_argument('--start-epoch', type=int, default=DEFAULT_START_EPOCH,
                            help='checkpoint-epoch<N>.pth to continue from (chain a previous extension: its train/.../models)')
    run_parser.add_argument('--lr-peak', type=float, default=1.5e-5, help='default = original polish END lr (constant continuation)')
    run_parser.add_argument('--lr-floor', type=float, default=1.5e-5, help='= original polish end')
    run_parser.add_argument('--save-period', type=int, default=20)
    score_parser = sub.add_parser('score')
    score_parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    run(args) if args.mode == 'run' else score(args)


if __name__ == '__main__':
    main()
