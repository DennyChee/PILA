#!/usr/bin/env python
# Usage:       from tests.regression.legacy_cases import CASES, build_case_config
# Description: The fixed set of short PILA runs used to prove that the 2026-11-23 restructure (pila/ package +
#              one run_pila.py with config flags) changes NOTHING numerically. Each case = one legacy launcher +
#              one legacy config, shortened to a few epochs, with save_dir redirected under runs/_regression/.
#              The same case configs are fed to the legacy launchers (reference, recorded BEFORE the move) and to
#              the new entry point (after the move); weights, per-epoch logs and dumped fits must match.
# Date:        2026-11-23
import copy
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGRESSION_ROOT = os.path.join(REPO_ROOT, 'runs', '_regression')

# K=2 non-static album run (residual-guided basin hop, standard curv15000 + dvL1(u)300 cell), copied verbatim
# from the legacy run so the reference exercises the exact production recipe.
K2_SOURCE_CONFIG = os.path.join(
    REPO_ROOT, 'saved_legacy_20261123', 'multisource_k2_nonstatic',
    'k2ns_G1_N1_staggered_n0000_slot_resid_curv15000_dvl1u300_seed1', 'G1_N1_staggered_n0000', 'K2', 'config_K2.json')


def _k2_without_extensions(config):
    """K=2 album config with basin hopping and the delayed-penalty schedule removed (plain trainer path)."""
    phys = config['trainer']['phys_vae']
    phys.pop('basin_hop', None)
    phys.pop('delayed_penalty', None)
    return config


def _edit_k2_basinhop(config):
    # Short but complete schedule: initial phase, >=1 hop window (with one possible extension), polish.
    config['trainer']['phys_vae']['basin_hop'] = dict(
        perturbation='slot_resid', settle_lr_peak_frac=0.5, total_epochs=10, seed=1, initial_epochs=3,
        settle_epochs=3, extend_if_best_in_last=1, extend_epochs=1, max_extensions=1, polish_epochs=2)
    return config


def _basinhop_arm(perturbation):
    """Short basin-hop schedule with the given perturbation arm (slot, slot_resid, weight_noise, none, off)."""
    def edit(config):
        config = _edit_k2_basinhop(config)
        config['trainer']['phys_vae']['basin_hop']['perturbation'] = perturbation
        return config
    return edit


def _edit_k2_delayed_only(config):
    config = _k2_without_extensions(config)
    config['trainer']['phys_vae']['delayed_penalty'] = dict(off_epochs=2, ramp_epochs=2, kl_stays_off=False)
    config['trainer']['phys_vae']['use_kl_term_z_phy'] = True
    return config


def _edit_k2_leftover_delayed(config):
    """Plain K=2 config that still CARRIES a delayed_penalty block (legacy logsources ignored the schedule)."""
    config = _k2_without_extensions(config)
    config['trainer']['phys_vae']['delayed_penalty'] = dict(off_epochs=2, ramp_epochs=2, kl_stays_off=False)
    return config


def _edit_k2_keepbest_delayed(config):
    config = _k2_without_extensions(config)
    # off 2 epochs, ramp over 2, KL allowed to switch on -> exercises the noise-scaled z_phy draw as well
    config['trainer']['phys_vae']['delayed_penalty'] = dict(off_epochs=2, ramp_epochs=2, kl_stays_off=False)
    config['trainer']['phys_vae']['use_kl_term_z_phy'] = True
    return config


def _edit_k2_init_sources(config):
    config = _k2_without_extensions(config)
    # Arbitrary honest start rows (x km, y km, depth km, dV m^3); NOT the album truth.
    config['arch']['phys_vae']['init_sources'] = [[-3.0, 2.0, 4.0, 5.0e6], [4.0, -1.0, 6.0, -5.0e6]]
    return config


# name -> (legacy launcher, source config, epochs, edit function or None)
CASES = {
    'std_mogi_cube':       ('train_pila.py', 'configs/synth/marapi_mogi_move_loc_curv15000_dvl1u300_pathfix_dvbox.json', 3, None),
    'okada_cube':          ('train_pila.py', 'configs/synth/marapi_okada_dike_combined_curv15000_dvl1u300_pathfix_dvbox.json', 2, None),
    'sun69_cube':          ('train_pila.py', 'configs/synth/marapi_sun69_sill_combined_curv15000_dvl1u300_pathfix_dvbox.json', 2, None),
    'k2_plain':            ('train_pila.py', K2_SOURCE_CONFIG, 3, _k2_without_extensions),
    'k2_logsources':       ('train_pila_logsources.py', K2_SOURCE_CONFIG, 3, _k2_without_extensions),
    'k2_keepbest_delayed': ('train_pila_keepbest.py', K2_SOURCE_CONFIG, 5, _edit_k2_keepbest_delayed),
    'k2_init_explicit':    ('train_pila.py', K2_SOURCE_CONFIG, 3, _edit_k2_init_sources),
    'k2_init_biasonly':    ('train_pila_biasinit.py', K2_SOURCE_CONFIG, 3, _edit_k2_init_sources),
    'k2_basinhop':         ('train_pila_basinhop.py', K2_SOURCE_CONFIG, 10, _edit_k2_basinhop),
    # --- added after the 2026-11-23 code review ---
    'k2_delayed_only':     ('train_pila_delayedpen.py', K2_SOURCE_CONFIG, 5, _edit_k2_delayed_only),
    # production config with delayed_penalty + basin_hop blocks, launched with PLAIN train_pila.py (blocks ignored)
    'k2_leftover_blocks':  ('train_pila.py', K2_SOURCE_CONFIG, 3, None),
    'k2_logsources_leftover_delayed': ('train_pila_logsources.py', K2_SOURCE_CONFIG, 3, _edit_k2_leftover_delayed),
    'k2_basinhop_slot':    ('train_pila_basinhop.py', K2_SOURCE_CONFIG, 10, _basinhop_arm('slot')),
    'k2_basinhop_weight_noise': ('train_pila_basinhop.py', K2_SOURCE_CONFIG, 10, _basinhop_arm('weight_noise')),
    'k2_basinhop_none':    ('train_pila_basinhop.py', K2_SOURCE_CONFIG, 10, _basinhop_arm('none')),
    'k2_basinhop_off':     ('train_pila_basinhop.py', K2_SOURCE_CONFIG, 10, _basinhop_arm('off')),
}

# Cases whose legacy run is repeated to prove the reference itself is bit-reproducible (incl. the stochastic
# noise-scaled draw and basin hopping).
REPEAT_CASES = ('std_mogi_cube', 'k2_keepbest_delayed', 'k2_basinhop')

# Minimum outputs each case must produce (both sides), so 'IDENTICAL' can never mean 'both empty'.
_K2_FIT = dict(pth=2, npz=1)                                           # model_best + checkpoint, auto fit_K2 dump
EXPECTED_MIN = {name: dict(pth=2) for name in CASES}
for _name in CASES:
    if _name.startswith('k2_'):
        EXPECTED_MIN[_name] = dict(_K2_FIT)
for _name in ('k2_logsources', 'k2_logsources_leftover_delayed', 'k2_delayed_only'):
    EXPECTED_MIN[_name].update(csv=1)                                   # per_epoch_sources.csv
EXPECTED_MIN['k2_keepbest_delayed'].update(pth=3, npz=2, csv=1, json=1) # + lowest_loss/ checkpoint, fit, json
for _name in [n for n in CASES if n.startswith('k2_basinhop')]:
    EXPECTED_MIN[_name].update(pth=4, npz=3, csv=3, json=3)            # + basin_best/, basin_*.csv, basin_hop.json
# 'off' = observe only: no hop windows are opened, so no basin_hops.csv (legacy behaviour too)
EXPECTED_MIN['k2_basinhop_off'].update(csv=2)


def build_case_config(case_name, side):
    """
    Write the shortened config for one case and return its path.

    Inputs:  case_name (key of CASES); side ('legacy' or 'new') -> separate save_dir trees.
    Output:  absolute path of runs/_regression/configs/<side>/<case>.json
    """
    launcher, source_path, epochs, edit = CASES[case_name]
    source_path = source_path if os.path.isabs(source_path) else os.path.join(REPO_ROOT, source_path)
    with open(source_path) as handle:
        config = json.load(handle)
    config = copy.deepcopy(config)
    if edit is not None:
        config = edit(config)
    config['name'] = case_name
    config['trainer']['epochs'] = epochs
    config['trainer']['save_period'] = epochs
    config['trainer']['early_stop'] = epochs
    config['trainer']['wandb'] = False
    config['trainer']['tensorboard'] = False
    config['n_gpu'] = 0                                   # CPU only: bitwise-reproducible reference
    config['trainer']['save_dir'] = os.path.join(REGRESSION_ROOT, side, case_name)
    out_dir = os.path.join(REGRESSION_ROOT, 'configs', side)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{case_name}.json')
    with open(out_path, 'w') as handle:
        json.dump(config, handle, indent=2)
    return out_path
