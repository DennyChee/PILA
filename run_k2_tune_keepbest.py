#!/usr/bin/env python
# Usage:       python run_k2_tune_keepbest.py --album <npz> --seed 1 --epochs 400 --tag k2tune_..._seed1 \
#                  --off-epochs 400 --ramp-epochs 0 [--amp-box-anchor-mm 60] [--lr-scheduler '{...}'] [--dry-run]
#              (all run_k2_tune_delayedpen.py / run_k2_tune.py flags are accepted)
# Description: run_k2_tune_delayedpen.py (honest K=2 from scratch, penalty schedule, per_epoch_sources.csv) with two
#              additions, both truth-free:
#                * training via train_pila_keepbest.py: the lowest-val_rec_loss epoch is also kept and dumped
#                  (models/lowest_loss/{model_lowest_loss.pth, fit_K2.npz, lowest_loss.json}); final outputs unchanged.
#                * --amp-box-anchor-mm A: data-scaled dV prior box = +-2e8 m^3 * (album LOS std / A mm)
#                  (utils/amp_box.py). A = 60 gives ~5-10x the source dV on the E1_S3 SNR 6/12 albums; omit for the
#                  standard +-2e8 box.
# Date:        2026-10-22

import argparse
import os
import subprocess
import sys

_box_parser = argparse.ArgumentParser(add_help=False)
_box_parser.add_argument('--amp-box-anchor-mm', type=float, default=None)
_box_args, _remaining = _box_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining       # run_k2_tune_delayedpen then run_k2_tune parse the rest

import run_k2_tune_delayedpen  # noqa: E402  (importing applies its config + launcher patches)
import run_k2_tune  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_delayed_build_config = run_k2_tune.build_config
_delayed_subprocess_run = run_k2_tune.subprocess.run


def build_config_keepbest(args, k_dir):
    """Delayed-penalty config, optionally with the data-scaled dV box written into the run dir."""
    cfg = _delayed_build_config(args, k_dir)
    if _box_args.amp_box_anchor_mm is not None:
        from utils.amp_box import write_data_scaled_paras
        paras_path, info = write_data_scaled_paras(cfg, k_dir, project_root=PROJECT_ROOT,
                                                   anchor_std_mm=_box_args.amp_box_anchor_mm)
        cfg['arch']['args']['mogi_paras'] = paras_path
        print(f"  dV box +-{info['box']:.4e} m^3 (album std {info['std_mm']:.3f} mm, anchor "
              f"{_box_args.amp_box_anchor_mm:g} mm) -> {paras_path}")
    return cfg


def subprocess_run_keepbest(command, *args, **kwargs):
    """Use the keep-lowest launcher (it imports the delayed-penalty patches) in place of train_pila.py.

    run_k2_tune calls train_pila.py; this wrapper runs BEFORE the delayed-penalty one, so it must swap train_pila.py
    itself (the delayed wrapper then finds nothing left to swap)."""
    command = ['train_pila_keepbest.py' if part in ('train_pila.py', 'train_pila_delayedpen.py') else part
               for part in command]
    return _delayed_subprocess_run(command, *args, **kwargs)


run_k2_tune.build_config = build_config_keepbest
run_k2_tune.subprocess.run = subprocess_run_keepbest

if __name__ == '__main__':
    print(f"[run_k2_tune_keepbest] keep lowest-val_rec_loss epoch; dV box "
          f"{'+-2e8 (standard)' if _box_args.amp_box_anchor_mm is None else f'data-scaled, anchor {_box_args.amp_box_anchor_mm:g} mm'}")
    run_k2_tune.main()
