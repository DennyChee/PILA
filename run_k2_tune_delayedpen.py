#!/usr/bin/env python
# Usage:       python run_k2_tune_delayedpen.py --album <npz> --seed 1 --epochs 300 --tag k2tune_..._seed1 \
#                  [--off-epochs 100] [--ramp-epochs 50] [--save-root saved/multisource_k2_tune] [--lr-scheduler '{...}'] [--dry-run]
#              (all run_k2_tune.py flags are accepted, except --random-init)
# Description: Honest K=2 from scratch (no truth) with the STANDARD penalties (curvature 15000, dV-L1 300 u, KL 0.5)
#              switched on only AFTER the sources have had time to separate: off for --off-epochs, then a linear ramp
#              over --ramp-epochs (train_pila_delayedpen.py). Same config as run_k2_tune_pen.py plus
#              trainer.phys_vae.delayed_penalty. run_k2_tune.py is reused unchanged (patched in memory only).
# Date:        2026-10-19

import argparse
import subprocess
import sys

import run_k2_tune

# --- Keep the standard-recipe penalties in the config; the launcher schedules them per epoch ---
run_k2_tune.NO_PENALTY_OVERRIDES = {}

_schedule_parser = argparse.ArgumentParser(add_help=False)
_schedule_parser.add_argument('--off-epochs', type=int, default=100)
_schedule_parser.add_argument('--ramp-epochs', type=int, default=50)
_schedule_parser.add_argument('--kl-stays-off', action='store_true',
                              help='isolating arm: curvature + dV-L1 follow the schedule, KL/sampling never switch on')
_schedule_args, _remaining = _schedule_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining         # run_k2_tune.main() parses the rest

_original_build_config = run_k2_tune.build_config
_original_subprocess_run = subprocess.run


def build_config_delayed(args, k_dir):
    """Standard-penalty config + the delayed_penalty schedule read by train_pila_delayedpen.py."""
    if args.random_init:
        raise ValueError("--random-init is not supported with the delayed-penalty launcher")
    cfg = _original_build_config(args, k_dir)
    # off_epochs >= epochs is allowed: that is the penalties-off arm run through the same launcher (same CSV log)
    if _schedule_args.off_epochs < cfg['trainer']['epochs'] <= _schedule_args.off_epochs + _schedule_args.ramp_epochs:
        raise ValueError("the ramp must finish before the last epoch (or set off_epochs >= epochs for no penalties)")
    cfg['trainer']['phys_vae']['delayed_penalty'] = {'off_epochs': _schedule_args.off_epochs,
                                                     'ramp_epochs': _schedule_args.ramp_epochs,
                                                     'kl_stays_off': bool(_schedule_args.kl_stays_off)}
    return cfg


def subprocess_run_delayed(command, *args, **kwargs):
    """Swap train_pila.py for the delayed-penalty launcher (same arguments)."""
    command = ['train_pila_delayedpen.py' if part == 'train_pila.py' else part for part in command]
    return _original_subprocess_run(command, *args, **kwargs)


run_k2_tune.build_config = build_config_delayed
run_k2_tune.subprocess.run = subprocess_run_delayed

if __name__ == '__main__':
    print(f"[run_k2_tune_delayedpen] standard penalties OFF for {_schedule_args.off_epochs} epochs, "
          f"ramp over {_schedule_args.ramp_epochs}, then full (curvature 15000, dV-L1 300 u, "
          f"{'KL/sampling OFF throughout' if _schedule_args.kl_stays_off else 'KL beta 0.5'})")
    run_k2_tune.main()
