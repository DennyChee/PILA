#!/usr/bin/env python
# Usage:       python run_k2_tune_basinhop.py --album <npz> --seed 1 --epochs 400 --tag k2tune_..._seed1 \
#                  --off-epochs 400 --ramp-epochs 0 --perturbation slot [--settle-lr-peak-frac 0.5] \
#                  [--basin-hop '{"initial_epochs":4,...}'] [--lr-scheduler '{...}'] [--dry-run]
#              (all run_k2_tune_keepbest.py / run_k2_tune_delayedpen.py / run_k2_tune.py flags are accepted)
# Description: run_k2_tune_keepbest.py (honest K=2 from scratch, penalty schedule, keep lowest val_rec_loss epoch) with
#              training via train_pila_basinhop.py: after the initial phase, revert to the best state (deterministic
#              album misfit), perturb (--perturbation slot | slot_resid | weight_noise | none | off = observe only),
#              settle, repeat; final polish from the best. Writes trainer.phys_vae.basin_hop into the config; total_epochs = --epochs and the
#              basin-hop seed = --seed. --basin-hop JSON overrides individual settings (used by the smoke test).
#              --temporal-smoothness-weight W adds curvature temporal smoothing alone (with --off-epochs 0 --ramp-epochs 0).
#              --dv-l1-weight W (added 2026-10-26, non-static K=2 test) adds the dV-L1 sparsity penalty alone, in u-space
#              (with --off-epochs 0 --ramp-epochs 0 --kl-stays-off, so KL and z_phy sampling stay off as in the baseline).
#              Both flags together (added 2026-10-27) = combined arm: smoothing + dV-L1 from epoch 1, needs the same
#              --off-epochs 0 --ramp-epochs 0 --kl-stays-off. Pass --kl-stays-off with smoothing alone too, or z_phy
#              sampling noise is on (the old K=2 curvature arms had it on).
# Date:        2026-10-23

import argparse
import json
import sys

_hop_parser = argparse.ArgumentParser(add_help=False)
_hop_parser.add_argument('--perturbation', required=True, choices=['slot', 'slot_resid', 'weight_noise', 'none', 'off'])
_hop_parser.add_argument('--settle-lr-peak-frac', type=float, default=0.5)
_hop_parser.add_argument('--basin-hop', default=None, help='JSON merged into the basin_hop block (overrides)')
_hop_parser.add_argument('--temporal-smoothness-weight', type=float, default=None,
                         help='turn ON curvature temporal smoothness (incl. dV) at this weight (with --dv-l1-weight too = combined arm); dV-L1 and KL stay '
                              'off; needs --off-epochs 0 --ramp-epochs 0 so the delayed-penalty factor is 1 from epoch 1')
_hop_parser.add_argument('--dv-l1-weight', type=float, default=None,
                         help='turn ON the dV-L1 sparsity penalty (u-space) at this weight (with --temporal-smoothness-weight too = combined arm); KL stays off; '
                              'needs --off-epochs 0 --ramp-epochs 0 --kl-stays-off')
_hop_args, _remaining = _hop_parser.parse_known_args()
# Both flags together = the combined arm (curvature smoothing + dV-L1, added 2026-10-27); see build_config_basinhop
sys.argv = [sys.argv[0]] + _remaining       # run_k2_tune_keepbest -> delayedpen -> run_k2_tune parse the rest

import run_k2_tune_keepbest  # noqa: E402,F401  (importing applies the keep-best + delayed-penalty patches)
import run_k2_tune  # noqa: E402

_keepbest_build_config = run_k2_tune.build_config
_keepbest_subprocess_run = run_k2_tune.subprocess.run


def build_config_basinhop(args, k_dir):
    """Keep-best config + the basin_hop block read by train_pila_basinhop.py."""
    cfg = _keepbest_build_config(args, k_dir)
    block = dict(perturbation=_hop_args.perturbation, settle_lr_peak_frac=_hop_args.settle_lr_peak_frac,
                 total_epochs=int(cfg['trainer']['epochs']), seed=int(args.seed))
    if _hop_args.basin_hop:
        block.update(json.loads(_hop_args.basin_hop))
    cfg['trainer']['phys_vae']['basin_hop'] = block
    if _hop_args.temporal_smoothness_weight is not None and _hop_args.dv_l1_weight is not None:
        # Combined (2026-10-27, non-static K=2 test of the standard PILA-vs-vbICA penalties): curvature smoothing
        # (x, y, depth AND dV, z-space) + dV-L1 (u-space), both at full weight from epoch 1. KL and z_phy sampling
        # stay off (kl_stays_off) as in every other arm of this campaign, so the arms differ by the penalty terms only.
        schedule = cfg['trainer']['phys_vae']['delayed_penalty']
        if schedule['off_epochs'] != 0 or schedule['ramp_epochs'] != 0 or not schedule.get('kl_stays_off', False):
            raise ValueError("combined smoothing + dV-L1 needs --off-epochs 0 --ramp-epochs 0 --kl-stays-off")
        cfg['trainer']['phys_vae'].update(temporal_smoothness_weight=float(_hop_args.temporal_smoothness_weight),
                                          temporal_smoothness_mode='curvature', temporal_smoothness_include_dv=True,
                                          dv_l1_weight=float(_hop_args.dv_l1_weight), dv_penalty_space='u',
                                          dv_l2_weight=0.0, use_kl_term_z_phy=False)
    elif _hop_args.temporal_smoothness_weight is not None:
        # Smoothing only (2026-10-23): curvature (2nd time difference) of every slot's x, y, depth AND dV over each
        # 7-scene training window, in z-space. The other standard penalties stay OFF, so this arm differs from the
        # penalties-off recipe by the smoothing term alone. The delayed-penalty wrapper multiplies the config weights
        # by its schedule factor, so it must be 1 from epoch 1 (off_epochs = ramp_epochs = 0).
        schedule = cfg['trainer']['phys_vae']['delayed_penalty']
        if schedule['off_epochs'] != 0 or schedule['ramp_epochs'] != 0:
            raise ValueError("--temporal-smoothness-weight needs --off-epochs 0 --ramp-epochs 0")
        cfg['trainer']['phys_vae'].update(temporal_smoothness_weight=float(_hop_args.temporal_smoothness_weight),
                                          temporal_smoothness_mode='curvature', temporal_smoothness_include_dv=True,
                                          dv_l1_weight=0.0, dv_l2_weight=0.0, use_kl_term_z_phy=False)
    elif _hop_args.dv_l1_weight is not None:
        # dV-L1 only (2026-10-26): |u_dV - u_dV(dV=0)| summed over both slots, pre-sigmoid u-space (the z-space gradient
        # vanishes at the rails). Everything else as the penalties-off recipe. The delayed-penalty wrapper scales the
        # weight by its schedule factor, so the factor must be 1 from epoch 1; kl_stays_off keeps KL weight AND z_phy
        # sampling noise at 0, which the penalties-off baseline also has (factor 0) -- the arms differ by the L1 term only.
        schedule = cfg['trainer']['phys_vae']['delayed_penalty']
        if schedule['off_epochs'] != 0 or schedule['ramp_epochs'] != 0 or not schedule.get('kl_stays_off', False):
            raise ValueError("--dv-l1-weight needs --off-epochs 0 --ramp-epochs 0 --kl-stays-off")
        cfg['trainer']['phys_vae'].update(dv_l1_weight=float(_hop_args.dv_l1_weight), dv_penalty_space='u',
                                          dv_l2_weight=0.0, temporal_smoothness_weight=0.0, use_kl_term_z_phy=False)
    if int(cfg['trainer']['early_stop']) < int(cfg['trainer']['epochs']):
        raise ValueError("basin hopping needs early_stop >= epochs (no early stopping)")
    return cfg


def subprocess_run_basinhop(command, *args, **kwargs):
    """Swap the keep-best launcher for the basin-hop launcher (it imports the keep-best patches itself)."""
    launchers = ('train_pila.py', 'train_pila_delayedpen.py', 'train_pila_keepbest.py')
    if not any(part in launchers for part in command):
        raise RuntimeError(f"no training launcher found to swap in {command}")
    command = ['train_pila_basinhop.py' if part in launchers else part for part in command]
    return _keepbest_subprocess_run(command, *args, **kwargs)


run_k2_tune.build_config = build_config_basinhop
run_k2_tune.subprocess.run = subprocess_run_basinhop

if __name__ == '__main__':
    print(f"[run_k2_tune_basinhop] perturbation={_hop_args.perturbation}, settle LR peak "
          f"{_hop_args.settle_lr_peak_frac:g} x base, overrides {_hop_args.basin_hop}")
    run_k2_tune.main()
