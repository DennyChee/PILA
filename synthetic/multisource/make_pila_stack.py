#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_pila_stack --stage noise   # noise sets + sigma_ref
#              python -m synthetic.multisource.make_pila_stack --stage albums  # 57 PILA albums
#              [--out synthetic/multisource/pila_stacks] [--n-acq 30] [--n-cal-per-track 3]
# Description: Step 0 of the sequential multi-source PILA plan. Builds synthetic InSAR "albums"
#              (stacks of 30 acquisitions on the 128 x 128 TA69 grid) with KNOWN static Mogi
#              sources for testing whether PILA can choose the number of sources itself:
#                * 39 calibration albums C01-C39 (noise only): real ERA5 weather from 13 tracks at
#                  9 other volcanoes (3 albums per track: 1 volcano-centred + 2 random windows);
#                * 18 evaluation albums = 3 noise sets (E1, E2: Marapi TA69 window near Marapi;
#                  E3: NoiseArea TA69) x 6 scenarios (S0 null ... S5 K2-one-weak). Each noise set
#                  is pasted IDENTICALLY onto all 6 scenarios (paired design).
# Noise model (MintPy-style cumulative series, one album):
#   acquisition t = ERA5(day_t) - ERA5(ref_day)                      [real, other volcanoes]
#                 + turb_t - turb_ref                                [Marapi-cube spectrum, 6 mm expected RMS]
#                 + white_t - white_ref                              [sigma from Marapi cube]
#                 + ramp_t - ramp_ref                                [planar, sigma 0.3 mm/km per date]
#                 + static source field (evaluation albums only)
#   then minus its value at ONE reference pixel per album, chosen in the far field (>= 18 km
#   from the grid centre) so the source field there is only a few % of its peak (recorded).
#   ref_day = earliest of 31 distinct days of the track; acquisitions = the later 30 days.
# Units: LOS mm in memory, metres in the npz (los_asc_m), + toward satellite; sources km / m^3.
# Date:        2026-10-06

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from synthetic.multisource.era5_noise import (CALIBRATION_TRACKS, EVALUATION_TRACKS, Era5Track,
                                              GRID_SHAPE)
from synthetic.multisource.generate_multisource import load_grid, separation_dshallow
from synthetic.multisource.torch_generator import mogi_los_mm_torch
from synthetic.multisource.turbulence import fit_turbulence, sample_turbulence_mm

TA69_GRID_H5 = 'synthetic/marapi_ta69_grid/geo_geometryRadar.h5'
RAMP_SIGMA_MM_PER_KM = 0.3
REF_PIXEL_MIN_DIST_KM = 18.0

# --- Scenarios: static sources (East km, North km, depth km, peak/sigma); all inflation ---
SCENARIOS = {
    'S0_K0_null':       [],
    'S1_K1_strong':     [dict(xcen_km=0.0, ycen_km=0.0, depth_km=5.0, peak_ratio=5.0)],
    'S2_K1_weak':       [dict(xcen_km=0.0, ycen_km=0.0, depth_km=5.0, peak_ratio=1.5)],
    'S3_K2_separated':  [dict(xcen_km=-6.0, ycen_km=0.0, depth_km=4.0, peak_ratio=4.0),
                         dict(xcen_km=6.0, ycen_km=0.0, depth_km=5.0, peak_ratio=4.0)],
    'S4_K2_close':      [dict(xcen_km=-2.35, ycen_km=0.0, depth_km=4.0, peak_ratio=4.0),
                         dict(xcen_km=2.35, ycen_km=0.0, depth_km=5.0, peak_ratio=4.0)],
    'S5_K2_one_weak':   [dict(xcen_km=-6.0, ycen_km=0.0, depth_km=4.0, peak_ratio=4.0),
                         dict(xcen_km=6.0, ycen_km=0.0, depth_km=5.0, peak_ratio=1.5)],
}


# --- Geometry helpers -------------------------------------------------------------------------
def load_ta69_geometry():
    """TA69 grid: flattened ENU coords (m), LOS unit vectors, 1-D axes, scalar inc/heading (deg)."""
    import h5py
    grid = load_grid(TA69_GRID_H5)
    xE_2d = grid['xE_m'].reshape(grid['shape'])
    yN_2d = grid['yN_m'].reshape(grid['shape'])
    with h5py.File(TA69_GRID_H5, 'r') as f:
        inc_deg = float(np.median(f['incidenceAngle'][:]))
        az_deg = float(np.median(f['azimuthAngle'][:]))
    # The npz stores 1-D axes, and the PILA loader rebuilds a SEPARABLE meshgrid from them.
    # The geocoded TA69 grid is not exactly separable (up to ~0.6 m), so source fields are
    # rendered on the separable grid PILA will see; the truth is then exact for PILA.
    x_m, y_m = xE_2d.mean(axis=0), yN_2d.mean(axis=1)
    xE_sep, yN_sep = np.meshgrid(x_m, y_m)
    grid = {**grid, 'xE_m': xE_sep.ravel(), 'yN_m': yN_sep.ravel()}
    # npz loader convention: MintPy azimuthAngle = 90 - heading  ->  heading = 90 - az.
    return dict(grid=grid, x_m=x_m, y_m=y_m, inc_deg=inc_deg, heading_deg=90.0 - az_deg)


def source_field_mm(geom, sources, sigma_ref_mm):
    """
    Sum of static Mogi LOS fields [H, W] (mm, + toward satellite). dV of each source is solved
    so that its OWN peak |LOS| = peak_ratio * sigma_ref (LOS is linear in dV). Returns the
    field and the sources with dV_m3 filled in.
    """
    grid = geom['grid']
    xE = torch.tensor(grid['xE_m']); yN = torch.tensor(grid['yN_m'])
    los = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]))
    total_mm = np.zeros(grid['shape'])
    filled = []
    for src in sources:
        unit = mogi_los_mm_torch(xE, yN, los, torch.tensor([[src['xcen_km'] * 1e3]], dtype=torch.float64),
                                 torch.tensor([[src['ycen_km'] * 1e3]], dtype=torch.float64),
                                 torch.tensor([[src['depth_km'] * 1e3]], dtype=torch.float64),
                                 torch.tensor([[1.0]], dtype=torch.float64))[0, 0].numpy()   # mm per m^3
        dV_m3 = src['peak_ratio'] * sigma_ref_mm / np.abs(unit).max()
        total_mm += (unit * dV_m3).reshape(grid['shape'])
        filled.append({**src, 'dV_m3': float(dV_m3)})
    return total_mm, filled


# --- Noise sets -------------------------------------------------------------------------------
def pick_windows(track, rng, n_windows):
    """1 volcano-centred window (if available) + random valid windows, spread apart if possible."""
    chosen, kinds = [], []
    volcano = track.volcano_window()
    if volcano is not None:
        chosen.append((volcano[0], volcano[1])); kinds.append(f'volcano(offset {volcano[2]:.1f} km)')
    corners = track.valid_windows()
    rng.shuffle(corners)
    min_sep_px = track.win_rows // 2                    # prefer windows >= half a window apart
    for corner in corners:
        if len(chosen) == n_windows:
            break
        if all(max(abs(corner[0] - c[0]), abs(corner[1] - c[1])) >= min_sep_px for c in chosen):
            chosen.append(corner); kinds.append('random')
    while len(chosen) < n_windows:                       # small frames: allow overlap
        chosen.append(corners[len(chosen) % len(corners)]); kinds.append('random(overlap)')
    return chosen, kinds


def pick_days(track, rng, n_acq, avoid_days=None):
    """
    31 distinct days, sorted by date: index 0 = reference day, 1..n_acq = acquisitions.
    avoid_days: days to use only if the track has too few others (minimises overlap between
    two noise sets on the same track, e.g. E1 and E2).
    """
    avoid = set(avoid_days or [])
    fresh = np.array([d for d in range(track.n_dates) if d not in avoid])
    if len(fresh) >= n_acq + 1:
        days = rng.choice(fresh, size=n_acq + 1, replace=False)
    else:
        reused = rng.choice(np.array(sorted(avoid)), size=n_acq + 1 - len(fresh), replace=False)
        days = np.concatenate([fresh, reused])
    days = np.sort(days)
    return int(days[0]), [int(d) for d in days[1:]]


def pick_reference_pixel(geom, rng):
    """Random far-field pixel (>= REF_PIXEL_MIN_DIST_KM from the grid centre): flat index."""
    grid = geom['grid']
    dist_km = np.hypot(grid['xE_m'], grid['yN_m']) / 1e3
    candidates = np.where(dist_km >= REF_PIXEL_MIN_DIST_KM)[0]
    return int(rng.choice(candidates))


def build_noise_set(set_id, track, window, window_kind, rng, geom, turb_model, n_acq, avoid_days=None):
    """Noise for one album [n_acq, H, W] mm (un-referenced) + metadata dict."""
    ref_day, acq_days = pick_days(track, rng, n_acq, avoid_days)
    era5_mm = track.noise_stack_mm(window[0], window[1], ref_day, acq_days, geom['inc_deg'])
    turb_mm = sample_turbulence_mm(rng, n_acq + 1, turb_model)           # [ref, acq...]
    white_mm = rng.standard_normal((n_acq + 1,) + GRID_SHAPE) * turb_model['white_sigma_mm']
    grid = geom['grid']
    # Orbital/long-wavelength ramp per DATE (incl. the reference date), differenced like every
    # other term of a cumulative series: ramp_t - ramp_ref has a common-mode part.
    grad = rng.standard_normal((n_acq + 1, 2)) * RAMP_SIGMA_MM_PER_KM        # mm/km, E and N
    ramp_mm = (grad[:, 0, None] * grid['xE_m'][None] / 1e3 +
               grad[:, 1, None] * grid['yN_m'][None] / 1e3).reshape((n_acq + 1,) + GRID_SHAPE)
    noise_mm = (era5_mm + (turb_mm[1:] - turb_mm[:1]) + (white_mm[1:] - white_mm[:1])
                + (ramp_mm[1:] - ramp_mm[:1]))
    meta = dict(set_id=set_id, track=track.name, era5_h5=track.era5_h5, orbit=track.orbit,
                window_row0=int(window[0]), window_col0=int(window[1]),
                window_px=[track.win_rows, track.win_cols], window_kind=window_kind,
                ref_date=track.dates[ref_day], acq_dates=[track.dates[d] for d in acq_days],
                day_idx=[int(ref_day)] + [int(d) for d in acq_days],
                ref_pixel_flat=pick_reference_pixel(geom, rng),
                era5_rms_mm=float(np.mean(np.sqrt(np.mean((era5_mm - np.median(era5_mm, axis=(1, 2), keepdims=True)) ** 2, axis=(1, 2))))))
    return noise_mm, meta


def stage_noise(args, geom):
    """Build and save all noise sets C01-C39, E1-E3 plus sigma_ref."""
    noise_dir = os.path.join(args.out, 'noise_sets')
    os.makedirs(noise_dir, exist_ok=True)
    print("[1/4] Fitting turbulence model to the Marapi noise cube...")
    turb_model = fit_turbulence()

    print("[2/4] Opening ERA5 tracks (read-only)...")
    cal_tracks = [Era5Track(spec) for spec in CALIBRATION_TRACKS]
    eval_tracks = {spec['name']: Era5Track(spec) for spec in EVALUATION_TRACKS}

    plan = []                       # (set_id, track, window, kind, seed)
    set_number = 0
    for track_number, track in enumerate(cal_tracks):
        windows, kinds = pick_windows(track, np.random.default_rng(1000 + track_number), args.n_cal_per_track)
        for window, kind in zip(windows, kinds):
            set_number += 1
            plan.append((f'C{set_number:02d}', track, window, kind, set_number))
    marapi = eval_tracks['Marapi_TA69']
    volcano = marapi.volcano_window()
    marapi_window = (volcano[0], volcano[1])
    marapi_kind = f'volcano(offset {volcano[2]:.1f} km)'
    noise_area = eval_tracks['NoiseArea_TA69']
    noise_area_window, noise_area_kind = pick_windows(noise_area, np.random.default_rng(2003), 1)
    plan += [('E1', marapi, marapi_window, marapi_kind, 101),
             ('E2', marapi, marapi_window, marapi_kind, 102),
             ('E3', noise_area, noise_area_window[0], noise_area_kind[0], 103)]

    print(f"[3/4] Building {len(plan)} noise sets x {args.n_acq} acquisitions...")
    t0 = time.time()
    rms_cal = []
    rows = []
    used_days = {}                                    # track name -> days used so far (E1 -> E2)
    for set_id, track, window, kind, seed in plan:
        out_npz = os.path.join(noise_dir, f'{set_id}.npz')
        if os.path.exists(out_npz):
            raise FileExistsError(f"{out_npz} exists; not overwriting (use a new --out)")
        avoid = used_days.get(track.name) if set_id.startswith('E') else None
        noise_mm, meta = build_noise_set(set_id, track, window, kind, np.random.default_rng(seed),
                                         geom, turb_model, args.n_acq, avoid_days=avoid)
        used_days.setdefault(track.name, set()).update(meta['day_idx'])
        # Per-acquisition RMS about the spatial median (the reference offset is arbitrary).
        rms = np.sqrt(np.mean((noise_mm - np.median(noise_mm, axis=(1, 2), keepdims=True)) ** 2, axis=(1, 2)))
        meta['noise_rms_mm_mean'] = float(rms.mean())
        if set_id.startswith('C'):
            rms_cal.extend(rms.tolist())
        np.savez_compressed(out_npz, noise_mm=noise_mm.astype(np.float32))
        with open(os.path.join(noise_dir, f'{set_id}.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        rows.append({k: meta[k] for k in ('set_id', 'track', 'window_kind', 'ref_date', 'noise_rms_mm_mean', 'era5_rms_mm')})
        print(f"  {set_id:4s} {track.name:18s} {kind:22s} ref {meta['ref_date']} "
              f"RMS {meta['noise_rms_mm_mean']:5.1f} mm (ERA5 part {meta['era5_rms_mm']:5.1f} mm)")
    print(f"  Done in {time.time() - t0:.1f}s")

    print("[4/4] sigma_ref from calibration albums...")
    sigma_ref_mm = float(np.mean(rms_cal))
    summary = dict(sigma_ref_mm=sigma_ref_mm, n_acq=args.n_acq, turbulence_beta=turb_model['beta'],
                   turbulence_rms_mm=float(np.median(turb_model['rms_mm'])),
                   white_sigma_mm=turb_model['white_sigma_mm'], ramp_sigma_mm_per_km=RAMP_SIGMA_MM_PER_KM,
                   ref_pixel_min_dist_km=REF_PIXEL_MIN_DIST_KM, grid_h5=TA69_GRID_H5,
                   inc_deg=geom['inc_deg'], heading_deg=geom['heading_deg'])
    with open(os.path.join(noise_dir, 'noise_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(noise_dir, 'noise_sets.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"  sigma_ref = {sigma_ref_mm:.2f} mm (mean per-acquisition RMS over calibration albums)")


# --- Albums -----------------------------------------------------------------------------------
def write_album(path, obs_mm, geom, dates):
    """
    Write one album in the _read_npz_scene format (metres, + toward satellite).

    Albums are deterministic, so if the file already exists (e.g. a rerun after a crash) it is
    verified to hold the same data and left untouched; a mismatch raises instead of overwriting.
    """
    if os.path.exists(path):
        with np.load(path) as f:
            same = np.allclose(f['los_asc_m'], (obs_mm / 1e3).astype(np.float32), atol=1e-9)
        if not same:
            raise FileExistsError(f"{path} exists with DIFFERENT data; not overwriting")
        print(f"  {os.path.basename(path)} exists and matches; kept")
        return
    np.savez_compressed(path, x_m=geom['x_m'], y_m=geom['y_m'],
                        los_asc_m=(obs_mm / 1e3).astype(np.float32),
                        incidence_deg=np.float64(geom['inc_deg']),
                        heading_asc_deg=np.float64(geom['heading_deg']),
                        dates=np.array(dates))


def stage_albums(args, geom):
    """Combine saved noise sets with scenario sources into 57 albums + truth JSON + manifest."""
    noise_dir = os.path.join(args.out, 'noise_sets')
    with open(os.path.join(noise_dir, 'noise_summary.json')) as f:
        summary = json.load(f)
    sigma_ref_mm = summary['sigma_ref_mm']
    album_dir = os.path.join(args.out, args.album_subdir)
    os.makedirs(album_dir, exist_ok=True)
    set_ids = sorted(n[:-5] for n in os.listdir(noise_dir) if n.endswith('.json') and n[0] in 'CE'
                     and n != 'noise_summary.json')
    print(f"[1/2] {len(set_ids)} noise sets, sigma_ref = {sigma_ref_mm:.2f} mm")

    manifest = []
    t0 = time.time()
    print("[2/2] Writing albums...")
    for set_id in set_ids:
        with open(os.path.join(noise_dir, f'{set_id}.json')) as f:
            meta = json.load(f)
        with np.load(os.path.join(noise_dir, f'{set_id}.npz')) as f:
            noise_mm = f['noise_mm'].astype(np.float64)
        scenario_names = ['S0_K0_null'] if set_id.startswith('C') else list(SCENARIOS)
        for scenario in scenario_names:
            field_mm, sources = source_field_mm(geom, SCENARIOS[scenario], sigma_ref_mm)
            obs_mm = noise_mm + field_mm[None]                  # static: same field every acquisition
            ref_flat = meta['ref_pixel_flat']
            obs_mm = obs_mm - obs_mm.reshape(obs_mm.shape[0], -1)[:, ref_flat][:, None, None]
            album = f'{set_id}_{scenario}' if set_id.startswith('E') else f'{set_id}_cal'
            write_album(os.path.join(album_dir, f'{album}.npz'), obs_mm, geom, meta['acq_dates'])
            pascal = (separation_dshallow(sources[0], sources[1]) if len(sources) == 2 else None)
            truth = dict(album=album, scenario=scenario, noise_set=set_id, n_sources=len(sources),
                         sources=sources, sigma_ref_mm=sigma_ref_mm,
                         separation_dshallow=None if pascal is None else float(pascal),
                         pascal_flag=bool(pascal is not None and pascal < 1.6),
                         ref_pixel_flat=ref_flat,
                         field_at_ref_pixel_mm=float(field_mm.ravel()[ref_flat]),
                         field_peak_mm=float(np.abs(field_mm).max()),
                         noise_track=meta['track'], noise_window=meta['window_kind'],
                         ref_date=meta['ref_date'], units='km, m^3, LOS mm (+ toward satellite)')
            with open(os.path.join(album_dir, f'{album}_truth.json'), 'w') as f:
                json.dump(truth, f, indent=1)
            manifest.append(dict(album=album, role='calibration' if set_id.startswith('C') else 'evaluation',
                                 scenario=scenario, noise_set=set_id, n_sources=len(sources),
                                 noise_track=meta['track'], noise_rms_mm=meta['noise_rms_mm_mean'],
                                 field_peak_mm=truth['field_peak_mm'],
                                 field_at_ref_pixel_mm=truth['field_at_ref_pixel_mm']))
    with open(os.path.join(album_dir, 'manifest.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    print(f"  {len(manifest)} albums written in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['noise', 'albums'], required=True)
    # v2 (2026-10-06): natural turbulence RMS scatter, differenced ramp, E1/E2 minimal day overlap.
    parser.add_argument('--out', default='synthetic/multisource/pila_stacks_v2')
    parser.add_argument('--n-acq', type=int, default=30)
    parser.add_argument('--n-cal-per-track', type=int, default=3)
    # Source fields are rendered on the separable loader grid (see load_ta69_geometry).
    parser.add_argument('--album-subdir', default='albums')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    geom = load_ta69_geometry()
    print(f"TA69 grid: {GRID_SHAPE}, inc {geom['inc_deg']:.2f} deg, heading {geom['heading_deg']:.2f} deg")
    if args.stage == 'noise':
        stage_noise(args, geom)
    else:
        stage_albums(args, geom)
    print(f"Done. Output saved to {args.out}/")


if __name__ == '__main__':
    main()
