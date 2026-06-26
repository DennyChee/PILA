# Usage:       python physics/sun69/validate_sun69.py
# Description: Regression + gradient checks for the PyTorch Sun (1969) penny-shaped
#              crack kernel (physics/sun69/sun69.py). Compares it against an
#              independent NumPy port of the MATLAB reference `sun69.m` (eqs 16/17)
#              and confirms gradients flow to all five source parameters.
# Date:        2026-06-17

import os
import sys
import argparse
import numpy as np
import torch

# Make `physics` importable when run from the repo root or from this directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from physics.sun69.sun69 import Sun69


def sun69_numpy_golden(r, h, a, V):
    """
    Independent NumPy port of `sun69.m` (Sun 1969, eqs 16 & 17), volume form.

    Inputs (all in SI, scalars or broadcastable arrays):
        r (m): radial distance of observation point from crack centre
        h (m): depth H of the crack centre
        a (m): crack radius A
        V (m^3): injected volume
    Returns:
        ur (m): radial surface displacement
        uz (m): vertical surface displacement (Up positive)
    """
    # Maximum crack opening B = 3V / (2*pi*A^2)  (Sun 1969 eq 9/10a)
    B = 3.0 * V / (2.0 * np.pi * a**2)

    D = r**2 + h**2 - a**2
    theta = np.arctan2(2.0 * a * h, D)
    k = np.sqrt((D / a**2)**2 + (2.0 * h / a)**2)
    srk = np.sqrt(k)
    akcos2 = a * srk * np.cos(theta / 2.0)
    aksin2 = a * srk * np.sin(theta / 2.0)
    hkcos2 = h * srk * np.cos(theta / 2.0)
    ak2cos = a * k * np.cos(theta)

    uz = B * (srk * np.sin(theta / 2.0) - (h / (a * srk)) * np.cos(theta / 2.0))
    ur = (B * r * h / a) * (
        (a + aksin2) / ((h + akcos2)**2 + (a + aksin2)**2)
        - (hkcos2 - aksin2 + ak2cos)
        / ((hkcos2 - aksin2 + ak2cos)**2
           + (akcos2 + h * srk * np.sin(theta / 2.0) + a * k * np.sin(theta))**2))
    return ur, uz


def test_against_golden(tol_m=1e-6):
    """Compare PyTorch Sun69.run ENU output against the NumPy golden reference."""
    print("[1/3] Regression vs NumPy golden (sun69.m eqs 16/17)...")
    rng = np.random.default_rng(0)

    # --- Random station layout (km -> m), centred area around the source ---
    n_stations = 60
    x_m = rng.uniform(-12000.0, 12000.0, size=n_stations)   # East, m
    y_m = rng.uniform(-12000.0, 12000.0, size=n_stations)   # North, m

    # --- A few source scenarios spanning valid (H/A>2) and marginal regimes ---
    #     [xcen(m), ycen(m), depth H(m), radius A(m), volume V(m^3)]
    scenarios = np.array([
        [0.0,     0.0,    4000.0, 800.0,  5.0e7],   # H/A = 5.0  (excellent)
        [1500.0, -2000.0, 6000.0, 1200.0, 2.0e7],   # H/A = 5.0
        [-3000.0, 500.0,  3000.0, 1000.0, 1.0e8],   # H/A = 3.0  (good)
        [0.0,     0.0,    2500.0, 1000.0, -3.0e7],  # H/A = 2.5, deflation (V<0)
    ])

    max_abs_err_m = 0.0
    model = Sun69(torch.tensor(x_m), torch.tensor(y_m))
    for s in scenarios:
        xcen, ycen, H, A, V = s
        # PyTorch forward (batch of 1); output is [1, 3N] in mm -> back to m
        out_mm = model.run(
            torch.tensor([xcen]), torch.tensor([ycen]), torch.tensor([H]),
            torch.tensor([A]), torch.tensor([V])).detach().numpy()[0]
        ux_t = out_mm[:n_stations] / 1e3
        uy_t = out_mm[n_stations:2 * n_stations] / 1e3
        uz_t = out_mm[2 * n_stations:] / 1e3

        # NumPy golden, computed in polar then rotated to E/N
        dx = x_m - xcen
        dy = y_m - ycen
        r = np.sqrt(dx**2 + dy**2)
        ur_g, uz_g = sun69_numpy_golden(r, H, A, V)
        th = np.arctan2(dy, dx)
        ux_g = ur_g * np.cos(th)
        uy_g = ur_g * np.sin(th)

        err = max(np.max(np.abs(ux_t - ux_g)),
                  np.max(np.abs(uy_t - uy_g)),
                  np.max(np.abs(uz_t - uz_g)))
        max_abs_err_m = max(max_abs_err_m, err)

    print(f"  max abs error vs golden: {max_abs_err_m:.3e} m (tol {tol_m:.0e})")
    assert max_abs_err_m < tol_m, (
        f"Sun69 PyTorch kernel disagrees with golden by {max_abs_err_m:.3e} m")
    print("  PASS")


def test_gradients():
    """Confirm autograd produces finite gradients for all five parameters."""
    print("[2/3] Gradient-flow check (all 5 params require_grad)...")
    rng = np.random.default_rng(1)
    x_m = torch.tensor(rng.uniform(-10000.0, 10000.0, size=40))
    y_m = torch.tensor(rng.uniform(-10000.0, 10000.0, size=40))
    model = Sun69(x_m, y_m)

    xcen = torch.tensor([100.0], requires_grad=True)
    ycen = torch.tensor([-200.0], requires_grad=True)
    depth = torch.tensor([5000.0], requires_grad=True)
    radius = torch.tensor([900.0], requires_grad=True)
    dV = torch.tensor([4.0e7], requires_grad=True)

    out = model.run(xcen, ycen, depth, radius, dV)
    loss = (out**2).sum()
    loss.backward()

    for name, p in [('xcen', xcen), ('ycen', ycen), ('depth', depth),
                    ('radius', radius), ('dV', dV)]:
        g = p.grad
        assert g is not None and torch.isfinite(g).all(), f"bad grad for {name}: {g}"
        print(f"  d(loss)/d({name}) = {g.item():.6e}")
    print("  PASS")


def test_matlab_example():
    """Reproduce the sun69.m docstring example profile as a sanity sniff test."""
    print("[3/3] sun69.m docstring example (pressure form -> volume) ...")
    # MATLAB: sun69(rho, H=1, A=0.5, P=1e6, E=10e9, nu=0.25)
    #   B = 8*(1-nu^2)*P*A/(pi*E); convert to the equivalent volume V = B*2*pi*A^2/3
    H, A, P, E, nu = 1.0, 0.5, 1.0e6, 10.0e9, 0.25
    B_ref = 8.0 * (1.0 - nu**2) * P * A / (np.pi * E)
    V = B_ref * 2.0 * np.pi * A**2 / 3.0
    rho = np.linspace(0.0, 3.0, 7)
    ur, uz = sun69_numpy_golden(rho, H, A, V)
    print(f"  B (max opening) = {B_ref:.6e} m, equivalent V = {V:.6e} m^3")
    print(f"  uz(rho=0) = {uz[0]:.6e} m (peak uplift over crack centre)")
    assert uz[0] > 0, "expected uplift at centre for positive injected volume"
    assert np.isclose(ur[0], 0.0, atol=1e-12), "radial displacement must vanish at r=0"
    print("  PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Validate the PyTorch Sun (1969) penny-shaped crack kernel.")
    parser.add_argument('--tol', type=float, default=1e-6,
                        help="absolute tolerance (m) for the golden comparison")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)  # double precision for tight tolerance
    test_against_golden(tol_m=args.tol)
    test_gradients()
    test_matlab_example()
    print("Done. All Sun69 checks passed.")


if __name__ == '__main__':
    main()
