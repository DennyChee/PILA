#!/usr/bin/env python3
# Usage:       python check_amplitude_dim_resolution.py
# Description: Verify that the amplitude-parameter resolution used by the
#              pull-to-zero penalty (trainer_phys_smpl.py) lands on the right
#              latent dimension and the right pre-sigmoid target u0 for all three
#              physics families, and that the Mogi/Sun69 answers are UNCHANGED by
#              the Okada generalisation. Pure JSON arithmetic -- no torch, no h5,
#              no GPU -- so it is safe to run on the login node.
# Date:        2026-09-07
#
# Scientific note: the penalty is |u_amp - u0| with u = logit(z), so u0 must be the
# pre-sigmoid value that maps to PHYSICAL amplitude zero. If physical zero sits at
# a rail (z0 = 0 or 1) then u0 -> -+inf and the penalty becomes an enormous constant
# pull that swamps the reconstruction loss -- which is exactly why Okada needs a
# range symmetric about zero rather than the stock [0, 20] opening range.

import json
import math
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Expected answers, one row per physics family --------------------------
# (config, paras key, expected amplitude param, expected dim, expected u0)
CASES = [
    ("configs/synth/marapi_mogi_rising_combined_curv15000_dvl1u300.json",
     "mogi_paras",  "dV",      3, 0.0),
    ("configs/synth/marapi_sun69_sill_combined_curv15000_dvl1u300.json",
     "sun69_paras", "dV",      4, math.log(0.05 / 0.95)),
    ("configs/synth/marapi_okada_dike_combined_curv15000_dvl1u300.json",
     "okada_paras", "opening", 7, 0.0),
]


def resolve(paras):
    """Mirror trainer_phys_smpl.py's amplitude-dim resolution.

    Input:  paras -- dict parsed from a *_paras.json (key ORDER is significant;
            it is the physics decoder's latent ordering).
    Output: (amp_name, dim, z0, u0)  -- u0 is the pre-sigmoid penalty target.
    """
    names = list(paras.keys())
    amp_name = next((nm for nm in ('dV', 'opening') if nm in names), None)
    if amp_name is None:
        return None, None, None, None
    dim = names.index(amp_name)
    rng = paras[amp_name]
    lo, hi = float(rng['min']), float(rng['max'])
    if amp_name == 'dV':
        # Mogi/Sun69: physical dV = (rescaled dV)*scale + shift.
        scale = float(rng.get('scale', 1e5))
        shift = float(rng.get('shift', -1e7))
    else:
        # Okada 'opening' is a plain linear rescale -- identity transform.
        scale = float(rng.get('scale', 1.0))
        shift = float(rng.get('shift', 0.0))
    z0 = ((-shift / scale) - lo) / (hi - lo)
    zc = min(max(z0, 1e-6), 1.0 - 1e-6)
    return amp_name, dim, z0, math.log(zc / (1.0 - zc))


def main():
    print("[1/3] Asserting the trainer still carries the ('dV', 'opening') fallback...")
    trainer_src = open(os.path.join(REPO_ROOT, 'trainer/trainer_phys_smpl.py')).read()
    if not re.search(r"for nm in \('dV', 'opening'\)", trainer_src):
        print("  FAIL: trainer_phys_smpl.py no longer resolves the amplitude dim via "
              "('dV', 'opening'). This check is out of date with the code.")
        return 1
    print("  OK: fallback present.")

    print("[2/3] Resolving the amplitude dim for each physics family...")
    failures = []
    for cfg_rel, paras_key, exp_name, exp_dim, exp_u0 in CASES:
        cfg = json.load(open(os.path.join(REPO_ROOT, cfg_rel)))
        paras_rel = cfg['arch']['args'][paras_key]
        paras = json.load(open(os.path.join(REPO_ROOT, paras_rel)))
        amp_name, dim, z0, u0 = resolve(paras)
        dim_z_phy = cfg['arch']['phys_vae']['dim_z_phy']

        ok = (amp_name == exp_name and dim == exp_dim
              and abs(u0 - exp_u0) < 1e-9 and 0.0 <= z0 <= 1.0
              and dim_z_phy == len(paras))
        failures.append(None if ok else cfg_rel)
        print(f"  {'OK  ' if ok else 'FAIL'} {os.path.basename(cfg_rel)}")
        print(f"       paras={paras_rel}")
        print(f"       amplitude='{amp_name}' dim={dim} (expected {exp_name}/{exp_dim}), "
              f"z0={z0:.6f}, u0={u0:.6f} (expected {exp_u0:.6f})")
        print(f"       dim_z_phy={dim_z_phy} vs {len(paras)} paras keys")

    print("[3/3] Guarding against the stock Okada opening range...")
    # The stock file has opening min=0, which puts physical zero exactly at the
    # z=0 rail. Confirm that would have been pathological, so the symmetric file
    # is not mistaken for an arbitrary preference.
    stock = json.load(open(os.path.join(REPO_ROOT, 'configs/sierranegra_okada_paras.json')))
    _, _, z0_stock, u0_stock = resolve(stock)
    print(f"  stock sierranegra_okada_paras.json: z0={z0_stock:.6f}, u0={u0_stock:.4f}")
    if abs(u0_stock) < 5.0:
        print("  WARNING: the stock range is no longer pathological -- revisit whether "
              "the symmetric-opening file is still needed.")
    else:
        print("  OK: stock range puts the penalty target at a rail; symmetric file needed.")

    bad = [f for f in failures if f]
    if bad:
        print(f"\nFAILED for {len(bad)} config(s): {bad}")
        return 1
    print("\nAll amplitude-dim resolutions correct. Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
