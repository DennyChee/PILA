#!/usr/bin/env python
# Usage:       python -m synthetic.regen_montages
# Description: Re-plot the time-series and per-epoch-normalised LOS montages for the
#              moving-source cases AFTER the top-margin fix in synthetic/plots.py
#              (the default fractional top margin left a big blank band below the
#              title on these tall, many-row montages). Reads ONLY the already-saved
#              cube .h5 files — no cube is regenerated, so checkpoints/cubes are
#              untouched. Writes into synthetic/cubes/<name>/ (canonical location)
#              and copies into synthetic/figures_moving_source/ (the curated set).
# Scientific note: the time-series montage uses the SAVED cube (signal+noise for the
#              'combined' runs). The normalised montage shows shape only and must use
#              the CLEAN signal; for a 'combined' run the clean cube is the sibling
#              '*_clean' cube's h5 (specs are identical apart from noise -> the clean
#              forward model is deterministic, verified by spec diff).
# Date:        2026-06-24
import json
import os
import shutil
import sys

import h5py
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic import plots as _plots

CUBES_DIR = os.path.join('synthetic', 'cubes')
FIG_DIR = os.path.join('synthetic', 'figures_moving_source')

# The six moving-source cube names (clean + combined for each physics).
CASES = [
    'marapi_mogi_rising_clean',  'marapi_mogi_rising_combined',
    'marapi_sun69_sill_clean',   'marapi_sun69_sill_combined',
    'marapi_okada_dike_clean',   'marapi_okada_dike_combined',
]

# Width-driving parameter overlaid/annotated on the normalised montage (matches
# generate_timeseries.py).
WIDTH_PARAM = {'mogi': 'd', 'sun69': 'radius', 'okada': 'length'}


def _load_cube_m(cube_h5):
    """Return (cube_m [n,Ny,Nx], dates list[str]) from a synthetic MintPy h5."""
    with h5py.File(cube_h5, 'r') as f:
        cube_m = np.array(f['timeseries'], dtype=np.float32)   # metres
        raw = list(f['date'])
    dates = [d.decode() if isinstance(d, (bytes, bytearray)) else str(d) for d in raw]
    return cube_m, dates[:cube_m.shape[0]]


def _read_mask(mask_h5, shape):
    if not mask_h5 or not os.path.exists(mask_h5):
        return None
    with h5py.File(mask_h5, 'r') as f:
        key = 'mask' if 'mask' in f else list(f.keys())[0]
        m = np.array(f[key], dtype=bool)
    return m if m.shape == tuple(shape) else None


def _copy_to_figdir(name, stem):
    for ext in ('png', 'pdf'):
        src = os.path.join(CUBES_DIR, name, f'{name}_{stem}.{ext}')
        dst = os.path.join(FIG_DIR, f'{name}_{stem}.{ext}')
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"  copied -> {dst}")


def main():
    os.chdir(ROOT)
    for name in CASES:
        out_dir = os.path.join(CUBES_DIR, name)
        cube_h5 = os.path.join(out_dir, f'{name}.h5')
        truth_json = os.path.join(out_dir, f'{name}_truth.json')
        if not (os.path.exists(cube_h5) and os.path.exists(truth_json)):
            print(f"SKIP {name}: missing cube or truth")
            continue
        print(f"\n=== {name} ===")
        with open(truth_json) as f:
            truth = json.load(f)
        mask_h5 = truth['base_scene'].get('mask')
        model = truth['forward_model']
        param_names = truth['param_names']
        params = np.asarray(truth['params_per_epoch'])          # [n_epoch, n_param]

        # --- Time-series montage: the cube PILA actually sees (saved h5) ---
        cube_m, dates = _load_cube_m(cube_h5)
        mask = _read_mask(mask_h5, cube_m.shape[1:])
        is_noisy = not name.endswith('_clean')
        _plots.plot_timeseries_montage(
            out_dir, name, cube_m, dates=dates, mask=mask,
            n_frames=cube_m.shape[0],
            title_suffix=(' (signal+noise)' if is_noisy else ' (clean)'))
        _copy_to_figdir(name, 'timeseries_montage')

        # --- Normalised montage: CLEAN cube (sibling *_clean h5 for combined runs) ---
        if is_noisy:
            clean_name = name.replace('_combined', '_clean')
            clean_h5 = os.path.join(CUBES_DIR, clean_name, f'{clean_name}.h5')
        else:
            clean_h5 = cube_h5
        if not os.path.exists(clean_h5):
            print(f"  WARNING: clean cube {clean_h5} missing -> skipping normalised montage")
            continue
        clean_cube_m, clean_dates = _load_cube_m(clean_h5)
        annot = None
        wkey = WIDTH_PARAM.get(model)
        if wkey in param_names:
            wj = param_names.index(wkey)
            annot = [f'{wkey}={v/1000.0:.1f}km' for v in params[:, wj]]   # m -> km
        _plots.plot_normalised_montage(
            out_dir, name, clean_cube_m, dates=clean_dates, mask=mask,
            annot=annot, title_suffix=' (clean)')
        _copy_to_figdir(name, 'normalised_montage')

    print("\nDone. Regenerated time-series + normalised montages (cubes/ + figures_moving_source/).")


if __name__ == '__main__':
    main()
