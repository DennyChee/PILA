#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_snr_scaled_albums \
#                  [--album E1_S3_K2_separated] [--noise-scales 0.5 0.25] [--nonoise-album E1_S3_K2_separated]
# Description: Higher-SNR twins of a noisy evaluation album. The noisy album is exactly
#              (noise-free signal) + (noise field), so noise = noisy - nonoise; the new album is
#              signal + scale * noise. Sources, dV, geometry, dates and reference pixel are unchanged,
#              and the noise keeps its full spatial/temporal structure (ERA5 APS + turbulence + white +
#              ramp, already referenced) -- only its amplitude shrinks. SNR rises by 1/scale.
# Notes:       Units: LOS mm in memory, metres in npz (+ toward satellite). Existing files are never
#              overwritten. Output name: <album>_n<scale*100 as 3 digits>, e.g. E1_S3_K2_separated_n050.
#              --nonoise-album (added 2026-10-26): use another album's noise-free twin (e.g. E2 has no twin of its
#              own). Allowed only when that signal is STATIC (all frames identical), because the noise sets differ
#              in acquisition dates; the noisy album's own dates are kept.
# Date:        2026-10-13
import argparse
import json
import os

import numpy as np

from synthetic.multisource.make_pila_stack import write_album

NOISY_DIR = 'synthetic/multisource/pila_stacks_v2/albums'
NONOISE_DIR = 'synthetic/multisource/pila_stacks_v2/albums_nonoise'
OUT_DIR = 'synthetic/multisource/pila_stacks_v2/albums_snr'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', default='E1_S3_K2_separated')
    parser.add_argument('--noise-scales', type=float, nargs='+', default=[0.5, 0.25])
    parser.add_argument('--nonoise-album', default=None,
                        help='album whose noise-free twin to use (default: --album); signal must be static')
    args = parser.parse_args()

    # --- Load the noisy album and its noise-free twin ---
    noisy_path = os.path.join(NOISY_DIR, f'{args.album}.npz')
    nonoise_album = args.nonoise_album or args.album
    nonoise_path = os.path.join(NONOISE_DIR, f'{nonoise_album}_nonoise.npz')
    for path in (noisy_path, nonoise_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    print(f"[1/3] Loading {noisy_path} and {nonoise_path}...")
    with np.load(noisy_path) as f:
        noisy_mm = f['los_asc_m'] * 1e3
        dates = f['dates']
        geom = {'x_m': f['x_m'], 'y_m': f['y_m'],
                'inc_deg': float(f['incidence_deg']), 'heading_deg': float(f['heading_asc_deg'])}
    with np.load(nonoise_path) as f:
        signal_mm = f['los_asc_m'] * 1e3
        assert np.array_equal(f['x_m'], geom['x_m']) and np.array_equal(f['y_m'], geom['y_m'])
        if nonoise_album == args.album:
            # The twin must share dates, otherwise noisy - nonoise is not the noise field.
            assert np.array_equal(f['dates'], dates)
        else:
            # Borrowed twin: dates differ, so only valid if the signal does not change between frames.
            if np.max(np.abs(signal_mm - signal_mm[0])) > 1e-6:
                raise ValueError(f'{nonoise_path}: signal is not static, cannot borrow it for {args.album}')
            if signal_mm.shape != noisy_mm.shape:
                raise ValueError(f'shape mismatch {signal_mm.shape} vs {noisy_mm.shape}')
            print(f"  borrowing static noise-free signal from {nonoise_album}")
    print(f"  Loaded: shape={noisy_mm.shape}, dtype={noisy_mm.dtype}")
    noise_mm = noisy_mm - signal_mm
    peak_signal_mm = float(np.abs(signal_mm).max())
    noise_rms_mm = float(np.sqrt(np.mean(noise_mm ** 2)))
    print(f"  peak |signal| = {peak_signal_mm:.1f} mm, noise RMS = {noise_rms_mm:.2f} mm, "
          f"SNR (peak/RMS) = {peak_signal_mm / noise_rms_mm:.2f}")

    truth = json.load(open(os.path.join(NOISY_DIR, f'{args.album}_truth.json')))
    if nonoise_album != args.album:
        # Same sources required: the borrowed signal must be this album's signal.
        other = json.load(open(os.path.join(NOISY_DIR, f'{nonoise_album}_truth.json')))
        if other['sources'] != truth['sources']:
            raise ValueError(f'{args.album} and {nonoise_album} have different sources')
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Write one album per noise scale ---
    print(f"[2/3] Writing {len(args.noise_scales)} scaled album(s) to {OUT_DIR}/...")
    for noise_scale in args.noise_scales:
        name = f'{args.album}_n{int(round(noise_scale * 100)):03d}'
        out_npz = os.path.join(OUT_DIR, f'{name}.npz')
        out_json = os.path.join(OUT_DIR, f'{name}_truth.json')
        if os.path.exists(out_npz) or os.path.exists(out_json):
            print(f"  skip (exists, not overwritten): {out_npz}")
            continue
        obs_mm = signal_mm + noise_scale * noise_mm
        write_album(out_npz, obs_mm, geom, dates)
        # Sanity: round-trip recovers the scaled noise exactly (float32 storage tolerance).
        with np.load(out_npz) as f:
            back_noise_mm = f['los_asc_m'] * 1e3 - signal_mm
        assert np.allclose(back_noise_mm, noise_scale * noise_mm, atol=1e-3), 'noise round-trip failed'
        snr = peak_signal_mm / (noise_scale * noise_rms_mm)
        truth_out = dict(truth, album=name, noise_scale=noise_scale, noise_rms_mm=noise_scale * noise_rms_mm,
                         snr_peak_over_rms=snr)
        if nonoise_album != args.album:
            # Signal (and so its reference-pixel fields) comes from the borrowed twin; the noise stays referenced to
            # this album's own pixel, leaving a ~0.1 mm constant offset in the extracted noise (negligible vs RMS).
            twin = json.load(open(nonoise_path.replace('.npz', '_truth.json')))
            truth_out.update(ref_pixel_flat=twin['ref_pixel_flat'], field_at_ref_pixel_mm=twin['field_at_ref_pixel_mm'],
                             signal_from=f'{nonoise_album}_nonoise (static signal borrowed)')
        with open(out_json, 'w') as f:
            json.dump(truth_out, f, indent=1)
        print(f"  {name}: noise x{noise_scale} -> RMS {noise_scale * noise_rms_mm:.2f} mm, SNR {snr:.2f}")

    print("[3/3] Round-trip checks passed.")
    print(f"Done. Output saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
