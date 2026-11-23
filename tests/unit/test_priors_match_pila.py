#!/usr/bin/env python
# Usage:       python tests/unit/test_priors_match_pila.py
# Description: The shared prior box (methods/priors.py) must equal PILA's own decoder range: for every model's
#              default prior file, PILA's rescale() at z = 0 and z = 1 must give exactly the [lower, upper] SI bounds
#              that the MCMC / vbICA adapters use. Prints PASS/FAIL per model.
# Date:        2026-11-23
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from methods.pila.make_config import MODELS  # noqa: E402
from methods.priors import prior_bounds_si  # noqa: E402
from pila.model import model_phys_smpl as phys  # noqa: E402
from scenarios.schema import PARAM_NAMES  # noqa: E402

CLASSES = {'mogi': phys.Physics_Mogi, 'sun69': phys.Physics_Sun69, 'okada': phys.Physics_Okada}
KM = {'mogi': ['xcen', 'ycen', 'd'], 'sun69': ['xcen', 'ycen', 'depth', 'radius'],
      'okada': ['xoff', 'yoff', 'depth', 'length', 'width']}
failures = 0
for model, cls in CLASSES.items():
    prior_file = MODELS[model][3]
    shell = cls.__new__(cls)                         # rescale() only needs these two attributes
    torch.nn.Module.__init__(shell)
    shell.z_phy_ranges = json.load(open(prior_file))
    shell.km_params = KM[model]
    names = list(shell.z_phy_ranges.keys())
    z = torch.stack([torch.zeros(len(names)), torch.ones(len(names))]).double()
    out = shell.rescale(z)
    pila_bounds = np.array([[float(out[name][0]), float(out[name][1])] for name in PARAM_NAMES[model]]).T
    pila_bounds.sort(axis=0)
    ours = prior_bounds_si(model, prior_file)
    ok = np.allclose(pila_bounds, ours, rtol=1e-12, atol=0)
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {model} ({prior_file}): lower {ours[0].tolist()} upper {ours[1].tolist()}")
print(f"Done. {3 - failures}/3 models match PILA's decoder range.")
raise SystemExit(1 if failures else 0)
