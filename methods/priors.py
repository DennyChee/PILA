#!/usr/bin/env python
# Usage:       from methods.priors import prior_bounds_si
# Description: ONE prior box for every method. PILA reads its paras JSON (configs/*_paras*.json: km for lengths,
#              degrees, metres of opening, dV with an affine scale/shift); every other method gets the SAME box
#              converted to SI here, so no method searches a wider or narrower space than PILA.
# Conventions: output order = scenarios.schema.PARAM_NAMES[model] (also iclassic_insar / iclassic_multi order);
#              units m, m^3, deg. Mirrors pila/model/model_phys_smpl.py rescale():
#                * names listed in KM_PARAMS are km in the JSON -> x 1000;
#                * dV (Mogi, Sun69): physical = value * scale + shift with defaults scale=1e5, shift=-1e7
#                  (legacy map); configs/mogi_paras_symdV.json sets scale=1, shift=0 (symmetric +-2e8 m^3).
# Date:        2026-11-23
import json
import os

import numpy as np

from pila.paths import REPO_ROOT
from scenarios.schema import PARAM_NAMES

KM_PARAMS = {'mogi': {'xcen', 'ycen', 'd'}, 'sun69': {'xcen', 'ycen', 'depth', 'radius'},
             'okada': {'xoff', 'yoff', 'depth', 'length', 'width'}}


def prior_bounds_si(model, paras_json):
    """
    2 x P array [lower; upper] in SI units for one source of `model`, from a PILA paras JSON.

    Inputs:  model ('mogi'|'sun69'|'okada'); paras_json (repo-relative or absolute path).
    Output:  np.ndarray [2, P] float64 in PARAM_NAMES[model] order.
    """
    path = paras_json if os.path.isabs(paras_json) else os.path.join(REPO_ROOT, paras_json)
    with open(path) as handle:
        paras = json.load(handle)
    missing = [name for name in PARAM_NAMES[model] if name not in paras]
    if missing:
        raise KeyError(f"{path} lacks {missing} for model {model}")
    bounds = np.zeros((2, len(PARAM_NAMES[model])))
    for j, name in enumerate(PARAM_NAMES[model]):
        low, high = float(paras[name]['min']), float(paras[name]['max'])
        if name in KM_PARAMS[model]:
            low, high = low * 1e3, high * 1e3
        if name == 'dV':
            scale = float(paras[name].get('scale', 1e5))
            shift = float(paras[name].get('shift', -1e7))
            low, high = sorted((low * scale + shift, high * scale + shift))
        bounds[:, j] = (low, high)
    return bounds
