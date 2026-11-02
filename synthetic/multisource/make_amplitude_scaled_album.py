#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_amplitude_scaled_album \
#                  --album E1_S2_K1_weak_n012 --match-dv-of E1_S1_K1_strong [--tag amp333]
# Description: Amplitude-scaled twin of an album: the WHOLE observed stack (signal + noise) is multiplied by one
#              factor, so the SNR and the globally standardized PILA input are unchanged and only the absolute
#              amplitude (and hence the dV the Mogi decoder must find) changes. Factor = dV(match album) /
#              dV(album), so the scaled source has exactly the reference album's dV (Mogi LOS is linear in dV;
#              location and depth are unchanged). Tests whether PILA failure depends on absolute amplitude vs
#              the fixed prior box / start point, not on SNR. Existing files are never overwritten.
# Output:      synthetic/multisource/pila_stacks_v2/albums_snr/<album>_<tag>.npz + _truth.json
# Notes:       Units: LOS mm in memory, metres in npz (+ toward satellite); dV m^3.
# Date:        2026-10-20
import argparse
import json
import os

import numpy as np

from synthetic.multisource.make_pila_stack import write_album

NOISY_DIR = 'synthetic/multisource/pila_stacks_v2/albums'
SNR_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'


def album_paths(album):
    """(npz, truth json) for an album name; scaled twins (_nXXX) live in albums_snr/."""
    folder = SNR_DIR if '_n' in album.split('K1_')[-1] else NOISY_DIR
    return os.path.join(folder, f'{album}.npz'), os.path.join(folder, f'{album}_truth.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', default='E1_S2_K1_weak_n012')
    parser.add_argument('--match-dv-of', default='E1_S1_K1_strong')
    parser.add_argument('--tag', default='amp333')
    args = parser.parse_args()

    # --- [1/3] Load album + both truths, derive the factor ---
    npz_path, truth_path = album_paths(args.album)
    _, match_truth_path = album_paths(args.match_dv_of)
    for path in (npz_path, truth_path, match_truth_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    print(f"[1/3] Loading {npz_path}...")
    truth = json.load(open(truth_path))
    match_truth = json.load(open(match_truth_path))
    if len(truth['sources']) != 1 or len(match_truth['sources']) != 1:
        raise ValueError('single-source albums only')
    src, match_src = truth['sources'][0], match_truth['sources'][0]
    # Same geometry is required, otherwise matching dV does not give the same field.
    for key in ('xcen_km', 'ycen_km', 'depth_km'):
        if not np.isclose(src[key], match_src[key]):
            raise ValueError(f'{key} differs ({src[key]} vs {match_src[key]}); dV matching would not match the field')
    factor = match_src['dV_m3'] / src['dV_m3']
    with np.load(npz_path) as f:
        obs_mm = f['los_asc_m'] * 1e3
        dates = f['dates']
        geom = {'x_m': f['x_m'], 'y_m': f['y_m'],
                'inc_deg': float(f['incidence_deg']), 'heading_deg': float(f['heading_asc_deg'])}
    print(f"  Loaded: shape={obs_mm.shape}, dtype={obs_mm.dtype}; factor = {factor:.5f}")

    # --- [2/3] Write the scaled album ---
    name = f'{args.album}_{args.tag}'
    out_npz = os.path.join(SNR_DIR, f'{name}.npz')
    out_json = os.path.join(SNR_DIR, f'{name}_truth.json')
    if os.path.exists(out_npz) or os.path.exists(out_json):
        raise FileExistsError(f'{out_npz} exists; not overwriting')
    print(f"[2/3] Writing {out_npz}...")
    write_album(out_npz, obs_mm * factor, geom, dates)
    with np.load(out_npz) as f:                 # round trip: stored stack is exactly factor x input (float32 tol)
        assert np.allclose(f['los_asc_m'] * 1e3, obs_mm * factor, atol=1e-3 * factor), 'round-trip failed'
    scaled_src = dict(src, dV_m3=src['dV_m3'] * factor)
    truth_out = dict(truth, album=name, sources=[scaled_src], amplitude_factor=factor,
                     amplitude_matched_to=args.match_dv_of,
                     field_peak_mm=truth['field_peak_mm'] * factor,
                     field_at_ref_pixel_mm=truth['field_at_ref_pixel_mm'] * factor,
                     sigma_ref_mm=truth['sigma_ref_mm'] * factor)
    if 'noise_rms_mm' in truth:
        truth_out['noise_rms_mm'] = truth['noise_rms_mm'] * factor      # SNR (snr_peak_over_rms) is unchanged
    with open(out_json, 'w') as f:
        json.dump(truth_out, f, indent=1)

    # --- [3/3] Report ---
    print(f"[3/3] {name}: dV {src['dV_m3']:.3e} -> {scaled_src['dV_m3']:.3e} m^3, "
          f"peak {truth['field_peak_mm']:.1f} -> {truth_out['field_peak_mm']:.1f} mm, "
          f"SNR unchanged {truth.get('snr_peak_over_rms', float('nan')):.2f}")
    print(f"Done. Output saved to {SNR_DIR}/")


if __name__ == '__main__':
    main()
