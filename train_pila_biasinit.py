#!/usr/bin/env python
# Usage:       python train_pila_biasinit.py -c <config.json>     (same arguments as train_pila.py)
# Description: Launcher for the random-start tuning arm. Runs train_pila.py UNCHANGED, except that the function it
#              calls for config arch.phys_vae.init_sources is swapped (in memory only) for a BIAS-ONLY start:
#              the z_phy-mean output bias is set to logit(z) of the requested [xcen_km, ycen_km, depth_km, dV_m3]
#              rows, but the output WEIGHTS and the lnvar head keep their seeded random default init. So only the
#              starting position differs from the baseline (no zeroed "compass", no tight variance).
#              No existing file is edited; utils/warm_start.py is only imported.
# Date:        2026-10-09

import os
import runpy
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import utils.warm_start as warm_start_module  # noqa: E402


def warm_start_bias_only(model, config, logger):
    """
    Set ONLY the z_phy-mean output bias to the requested sources (km, m^3); weights and lnvar head untouched.

    Input:  model (PHYS_VAE_SMPL), config (arch.phys_vae.init_sources = K rows [xcen_km, ycen_km, depth_km, dV_m3]).
    Output: info dict with the keys train_pila.log_compute_usage reads (same shape as warm_start_explicit_sources).
    """
    time_start_sec = time.perf_counter()
    sources = config['arch']['phys_vae']['init_sources']
    ranges = model.physics_model.z_phy_ranges
    n_sources = int(getattr(model.physics_model, 'n_sources', 1))
    parameter_names = ['xcen', 'ycen', 'd', 'dV']
    if len(sources) != n_sources or any(len(row) != 4 for row in sources):
        raise ValueError(f"init_sources must be {n_sources} rows of 4 values (xcen, ycen, d, dV)")
    # --- Physical start -> z in (0,1) -> clipped away from the sigmoid rails -> pre-sigmoid u ---
    z_requested = np.array([[warm_start_module._z_from_physical(ranges, name, float(value))
                             for name, value in zip(parameter_names, row)] for row in sources])
    clipped = bool(np.any((z_requested < warm_start_module.Z_CLIP) | (z_requested > 1.0 - warm_start_module.Z_CLIP)))
    u_start = np.array([[warm_start_module._logit(z) for z in row] for row in z_requested]).reshape(-1)
    head_weight_keys = [key for key in model.state_dict()
                        if key.startswith('enc.func_z_phy_mean.net.') and key.endswith('.weight')]
    last_layer = max(int(key.split('.')[3]) for key in head_weight_keys)
    bias_key = f'enc.func_z_phy_mean.net.{last_layer}.bias'
    with torch.no_grad():
        state = model.state_dict()
        state[bias_key].copy_(torch.tensor(u_start, dtype=state[bias_key].dtype))   # bias only: weights stay random
    z_back = (1.0 / (1.0 + np.exp(-u_start))).reshape(n_sources, 4)
    info = dict(requested=[list(map(float, row)) for row in sources], z=z_back.round(4).tolist(), clipped=clipped,
                lnvar_init=None, init_from='explicit_sources_bias_only',
                new_source_start=[list(map(float, row)) for row in sources],
                warm_start_sec=time.perf_counter() - time_start_sec)
    logger.info(f"init_sources BIAS-ONLY (K={n_sources}): {info}")
    print(f"[init_sources bias-only] K={n_sources} start: {info['requested']} -> z {info['z']}")
    return info


warm_start_module.warm_start_explicit_sources = warm_start_bias_only      # train_pila imports it at call time
runpy.run_path(os.path.join(PROJECT_ROOT, 'train_pila.py'), run_name='__main__')
