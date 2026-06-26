# Usage:       python physics/okada/validate_okada_dike.py
# Description: Automated regression + correctness checks for the PILA Okada
#              opening-dike model (physics/okada/okada_dike.py, OkadaDike).
#              (1) matches the trusted NumPy okadaMod port to ~1e-12 m,
#              (2) reproduces a frozen golden fixture (no dependency),
#              (3) gradients flow through all 8 params,
#              (4) gradients stay finite at a degenerate (corner) geometry.
# Date:        2026-06-11

import os
import sys
import numpy as np
import torch

PILA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PILA_ROOT not in sys.path:
    sys.path.insert(0, PILA_ROOT)

from physics.okada.okada_dike import OkadaDike           # noqa: E402
from physics.okada.plot_okada_check import okada_mod     # trusted NumPy reference

torch.set_default_dtype(torch.float64)

# Frozen golden fixture: generated once from the validated OkadaDike.
# Geometry: stations E=[0,5000,-3000,8000], N=[0,-4000,6000,2000] (m);
# source xoff=500, yoff=-300, depth=6000, strike=35, dip=60, L=3000, W=1500,
# opening=8. Output [E(4) | N(4) | Up(4)] in mm.
GOLDEN_MM = [-4.78549, 114.310529, -6.70013, 64.790179, 3.080389, -91.811199,
             0.78912, 11.592861, 51.268349, 139.873239, -2.356495, 52.89991]


def _run(E, N, p):
    m = OkadaDike(torch.tensor(E), torch.tensor(N))
    return m.run(*[torch.tensor([v]) for v in p]).detach().numpy()[0]


def check_against_okadamod():
    print("[1] OkadaDike vs NumPy okadaMod, random geometries...")
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(10):
        p = [float(rng.uniform(-5000, 5000)), float(rng.uniform(-5000, 5000)),
             float(rng.uniform(2000, 12000)), float(rng.uniform(0, 360)),
             float(rng.uniform(2, 90)), float(rng.uniform(500, 8000)),
             float(rng.uniform(500, 6000)), float(rng.uniform(0.1, 15))]
        E = rng.uniform(-20000, 20000, 50)
        N = rng.uniform(-20000, 20000, 50)
        uE, uN, uZ = okada_mod(E, N, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        o = _run(E, N, p) / 1e3                      # mm -> m
        n = len(E)
        worst = max(worst, np.max(np.abs(o[:n]-uE)), np.max(np.abs(o[n:2*n]-uN)),
                    np.max(np.abs(o[2*n:]-uZ)))
    print(f"    worst |err| = {worst:.3e} m")
    assert worst < 1e-9, "OkadaDike disagrees with okadaMod"
    print("    PASS")


def check_golden():
    print("[2] Frozen golden fixture...")
    E = [0., 5000., -3000., 8000.]
    N = [0., -4000., 6000., 2000.]
    p = [500., -300., 6000., 35., 60., 3000., 1500., 8.]
    o = _run(E, N, p)
    err = float(np.max(np.abs(o - np.array(GOLDEN_MM))))
    print(f"    max |err| vs golden = {err:.3e} mm")
    assert err < 1e-4, "golden fixture mismatch"
    print("    PASS")


def check_gradients():
    print("[3] Gradient flow through all 8 params...")
    E = torch.tensor([0., 3000., -4000., 5000.])
    N = torch.tensor([0., -3000., 2000., 4000.])
    params = [torch.tensor([v], requires_grad=True) for v in
              (500., -300., 6000., 35., 60., 3000., 1500., 8.)]
    OkadaDike(E, N).run(*params).sum().backward()
    ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert ok, "gradient did not flow / non-finite"
    print("    PASS")


def check_degenerate_gradient():
    print("[4] Finite gradients at degenerate (station-at-corner) geometry...")
    # Place a station so that a Chinnery corner can coincide (xi=eta=q -> 0).
    E = torch.tensor([0.0, 1000.0])
    N = torch.tensor([0.0, 1000.0])
    params = [torch.tensor([v], requires_grad=True) for v in
              (0., 0., 0.0, 0., 90., 2000., 2000., 5.)]   # depth=0, vertical
    out = OkadaDike(E, N).run(*params)
    out.sum().backward()
    ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    finite_out = bool(torch.isfinite(out).all())
    print(f"    output finite={finite_out}, grads finite={ok}")
    assert finite_out and ok, "NaN/inf at degenerate geometry (sqrt eps guard?)"
    print("    PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("OkadaDike (PILA opening-dike) validation")
    print("=" * 60)
    check_against_okadamod()
    check_golden()
    check_gradients()
    check_degenerate_gradient()
    print("\nAll checks passed.")
