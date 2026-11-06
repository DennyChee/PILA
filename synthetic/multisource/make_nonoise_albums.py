#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_nonoise_albums
# Description: Noise-free twins of the E1 evaluation albums (scenarios S1-S5): same Mogi sources, same
#              TA69 grid/geometry/dates/dV as the noisy albums, but NO APS/turbulence/ramp noise. Used to
#              test whether sequential PILA can reconstruct the sources when noise is not the obstacle.
# Notes:       dV is taken from the existing noisy-album truth JSON (identical truth). Each acquisition
#              is the same static field, referenced to the album's reference pixel (as in the noisy albums).
#              Units: LOS mm in memory, metres in npz (+ toward satellite). Existing files are never overwritten.
# Date:        2026-10-07
import json, os
import numpy as np
from synthetic.multisource.make_pila_stack import (SCENARIOS, load_ta69_geometry, source_field_mm,
                                                   write_album)

SRC_DIR = 'synthetic/multisource/pila_stacks_v2/albums'
OUT_DIR = 'synthetic/multisource/pila_stacks_v2/albums_nonoise'
NOISE_SET = 'E1'

print("[1/2] Loading TA69 geometry...")
geom = load_ta69_geometry()
for scenario in [s for s in SCENARIOS if s != 'S0_K0_null']:
    name = f'{NOISE_SET}_{scenario}'
    truth = json.load(open(os.path.join(SRC_DIR, f'{name}_truth.json')))
    with np.load(os.path.join(SRC_DIR, f'{name}.npz')) as f:
        dates = f['dates']
    # --- Re-render field with the TRUTH dV (not re-solved) so it equals the noisy album's signal ---
    field_mm, sources = source_field_mm(geom, truth['sources'], truth['sigma_ref_mm'])
    for new, old in zip(sources, truth['sources']):
        assert abs(new['dV_m3'] / old['dV_m3'] - 1) < 1e-6, 'dV mismatch vs noisy-album truth'
    obs_mm = np.repeat(field_mm[None], len(dates), axis=0)
    obs_mm = obs_mm - obs_mm.reshape(len(dates), -1)[:, truth['ref_pixel_flat']][:, None, None]
    print(f"[2/2] {name}: shape={obs_mm.shape}, peak={np.abs(obs_mm).max():.1f} mm")
    write_album(os.path.join(OUT_DIR, f'{name}_nonoise.npz'), obs_mm, geom, dates)
    truth['album'] = f'{name}_nonoise'; truth['noise_set'] = 'none'
    with open(os.path.join(OUT_DIR, f'{name}_nonoise_truth.json'), 'w') as f:
        json.dump(truth, f, indent=1)
print("Done. Output saved to", OUT_DIR)
