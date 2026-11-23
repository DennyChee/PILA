#!/usr/bin/env python
# Usage:       python -m scenarios.convert_legacy_album --album synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2/G1_N4_moving_n0250.npz \
#                  --out-dir runs/_pilot_scenarios [--manifest runs/_pilot_scenarios/manifest.csv]
# Description: Convert one LEGACY multi-source album (npz + its legacy truth json) into the new scenario format,
#              for pipeline tests and pilots only -- the real comparison regenerates every scenario in Phase 4.
#              The npz is already in the PILA-native schema; it is re-written with the noise-free clean field
#              (recomputed from the truth with the independent NumPy forward model, referenced to the same pixel)
#              and the truth is translated to pila-truth/1 (km -> m, own peak LOS kept).
#              The legacy files are only READ.
# Checks:      the own peak |LOS| of every source recomputed here must match the legacy own_peak_mm within 1 %;
#              the RMS of (data - clean) is printed next to the legacy noise RMS (they must agree; for a noise-free
#              album it must be ~0 mm, which validates forward model, units, LOS geometry and referencing at once).
# Date:        2026-11-23
import argparse
import json
import os

import numpy as np

from scenarios.manifest import add_scenario
from scenarios.schema import make_source, source_los_m, write_scenario, write_truth


def convert(album_path, out_dir, manifest_path=None):
    """Convert one album; return (scenario path, truth path)."""
    truth_legacy_path = album_path.replace('.npz', '_truth.json')
    for path in (album_path, truth_legacy_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    print(f"[1/4] Reading legacy album {album_path} ...")
    with np.load(album_path) as handle:
        album = {key: handle[key] for key in handle.files}
    with open(truth_legacy_path) as handle:
        legacy = json.load(handle)
    track = 'asc' if 'los_asc_m' in album else 'desc'
    cube_m = album[f'los_{track}_m']
    print(f"  Loaded: los_{track}_m shape={cube_m.shape}, dtype={cube_m.dtype}; K={legacy['n_sources']}")

    print("[2/4] Translating truth (km -> m) and recomputing the clean field with the NumPy forward model ...")
    n_rows, n_cols = cube_m.shape[1:]
    ref_row, ref_col = divmod(int(legacy['ref_pixel_flat']), n_cols)
    clean_m = np.zeros(cube_m.shape, dtype=np.float64)
    sources = []
    for source in legacy['sources_per_date']:
        params = np.stack([np.asarray(source['xcen_km']) * 1e3, np.asarray(source['ycen_km']) * 1e3,
                           np.asarray(source['depth_km']) * 1e3, np.asarray(source['dV_m3'])], axis=1)
        peaks_mm = []
        for date_index, row in enumerate(params):
            field_m = source_los_m('mogi', row, album['x_m'], album['y_m'], album['incidence_deg'],
                                   album[f'heading_{track}_deg'])
            peaks_mm.append(float(np.abs(field_m).max()) * 1e3)          # own peak |LOS| BEFORE referencing
            clean_m[date_index] += field_m
        legacy_peaks = np.asarray(source['own_peak_mm'])
        mismatch = np.abs(np.asarray(peaks_mm) - legacy_peaks) / np.maximum(legacy_peaks, 1e-6)
        if np.nanmax(mismatch[legacy_peaks > 0.1]) > 0.01:
            raise ValueError(f"own peak LOS differs from legacy by {100 * np.nanmax(mismatch):.2f} % (source {source['index']})")
        sources.append(make_source('mogi', params, peaks_mm))
    clean_m -= clean_m[:, ref_row:ref_row + 1, ref_col:ref_col + 1]          # referenced like the data
    print(f"  own-peak check vs legacy: OK (<1 %); reference pixel ({ref_row}, {ref_col})")

    print("[3/4] Writing scenario npz + truth json ...")
    scenario_id = legacy['album']
    os.makedirs(out_dir, exist_ok=True)
    scenario_path = os.path.join(out_dir, f'{scenario_id}.npz')
    truth_path = os.path.join(out_dir, f'{scenario_id}_truth.json')
    write_scenario(scenario_path, album['x_m'], album['y_m'], {track: cube_m}, album['incidence_deg'],
                   {track: album[f'heading_{track}_deg']}, album['dates'], mask=album.get('mask'),
                   clean_los_m_by_track={track: clean_m}, ref_date=legacy['ref_date'], ref_pixel=(ref_row, ref_col))
    noise = dict(rms_mm=legacy['noise_rms_mm'], set=legacy['noise_set'], scale=legacy['noise_scale'],
                 snr_peak_over_rms=legacy['snr_peak_over_rms'])
    write_truth(truth_path, scenario_id, [track], album['dates'], sources, ref_date=legacy['ref_date'], noise=noise,
                notes=f"converted from legacy {os.path.relpath(album_path)} ({legacy['scenario']}); {legacy['convention']}")
    residual_rms_mm = float(np.sqrt(np.mean((cube_m - clean_m) ** 2)) * 1e3)
    print(f"  data - clean RMS = {residual_rms_mm:.2f} mm (legacy noise RMS {legacy['noise_rms_mm']:.2f} mm)")

    if manifest_path:
        print(f"[4/4] Registering in {manifest_path} ...")
        add_scenario(dict(scenario_id=scenario_id, kind='synthetic', family=f"legacy_{legacy['scenario']}",
                          models=';'.join(['mogi'] * len(sources)), n_sources=len(sources), tracks=track, split='test',
                          scenario_path=os.path.relpath(scenario_path, os.getcwd()) if not os.path.isabs(out_dir) else scenario_path,
                          truth_path=os.path.relpath(truth_path, os.getcwd()) if not os.path.isabs(out_dir) else truth_path,
                          geometry=track, notes='pilot-only conversion of a legacy album'), manifest_path)
    print(f"Done. Output saved to {scenario_path}")
    return scenario_path, truth_path


def main():
    parser = argparse.ArgumentParser(description='Convert a legacy multi-source album to the new scenario format')
    parser.add_argument('--album', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--manifest', default=None)
    args = parser.parse_args()
    convert(args.album, args.out_dir, args.manifest)


if __name__ == '__main__':
    main()
