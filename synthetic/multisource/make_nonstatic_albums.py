#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_nonstatic_albums [--out synthetic/multisource/pila_stacks_v2/albums_nonstatic]
#                  [--seed 20261026] [--n-draws 2]
# Description: Non-static K=2 test albums for PILA (plan: ~/.claude/plans/synthetic-tests-for-pila-smooth-candle.md).
#              Two Mogi sources whose dV (and, for N4, position) change over time, on the TA69 grid with the E1 Marapi
#              noise set scaled by a noise factor. Time scenarios (second source = the one that does something unusual):
#                S3_static     both constant (control; G1 reproduces albums_snr/E1_S3_K2_separated_n025/n050)
#                N1_staggered  first grows with tau; second has a sigmoidal onset centred at tau = 0.5 (the profile_sigmoid
#                              of synthetic/trajectories.py: steepness 8 on 2*tau-1, renormalised to 0 at tau 0 and 1 at tau 1)
#                N2_opposite   first inflates with tau; second deflates with tau
#                N3_pulse      first grows with tau; second is a Gaussian pulse at tau = 0.5 (width 0.15)
#                N4_moving     first moves 4 km and shallows by 40 % (floored at 1.5 km depth) while growing; second grows
#              tau = (date - ref_date) / (last date - ref_date): every field is 0 on the reference day (MintPy convention).
#              Each source's largest own peak |LOS| over the 30 dates = 4 x sigma_ref (as S3).
#              Layout sets: G1 = fixed S3 positions x 8 noise scales; R = 2 random layouts per (scenario, scale) x 5 scales,
#              each album its own layout (seeded, recorded in the truth JSON).
# Notes:       N4 uses the "snapshot" convention (synthetic/trajectories.py): each date = one Mogi at the current position
#              with the current cumulative dV, not the sum of increments from earlier positions.
#              Truth fields are UNreferenced (score per date against the _nonoise twin); pascal_flag uses the start and
#              end dates only (the per-date minimum is computed in the scorer, not here).
#              Units: LOS mm in memory, metres in npz (+ toward satellite); km, m^3 in truth. Files are never overwritten
#              (write_album keeps an identical existing file, raises on a different one).
# Date:        2026-10-26 (v2: N1 second source sigmoid instead of a hard switch-on at tau 0.5)

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from synthetic.multisource.generate_multisource import separation_dshallow
from synthetic.multisource.make_pila_stack import load_ta69_geometry, write_album
from synthetic.multisource.torch_generator import mogi_los_mm_torch

STACKS = 'synthetic/multisource/pila_stacks_v2'
NOISE_SET = 'E1'
PEAK_RATIO = 4.0                                    # own peak |LOS| = 4 x sigma_ref, as S3
G1_SCALES = [0.0, 0.125, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]
R_SCALES = [0.0, 0.18, 0.25, 0.5, 1.0]
SCENARIOS = ['S3_static', 'N1_staggered', 'N2_opposite', 'N3_pulse', 'N4_moving']
G1_LAYOUT = [dict(xcen_km=-6.0, ycen_km=0.0, depth_km=4.0), dict(xcen_km=6.0, ycen_km=0.0, depth_km=5.0)]
PULSE_CENTRE_TAU, PULSE_WIDTH_TAU = 0.5, 0.15
MOVE_DISTANCE_KM, SHALLOW_FACTOR, MIN_DEPTH_KM = 4.0, 0.6, 1.5
G1_MOVE = dict(x_start_km=-8.0, x_end_km=-4.0, depth_start_km=5.0, depth_end_km=3.0)   # N4 first source on G1
# Random-layout draw rules (fixed in advance, not tuned to PILA)
R_BOX_KM, R_DEPTH_KM, R_SEP_KM, REF_LEAK_MAX = 10.0, (2.0, 8.0), (3.0, 15.0), 0.05


# --- Time profiles (dimensionless dV multipliers; 0 at tau = 0) ---------------------------------
def pulse(tau):
    """Gaussian pulse minus its tau = 0 value, so the field is exactly 0 on the reference day."""
    gauss = lambda t: np.exp(-0.5 * ((t - PULSE_CENTRE_TAU) / PULSE_WIDTH_TAU) ** 2)
    return gauss(tau) - gauss(0.0)


def sigmoid(tau, steepness=8.0):
    """synthetic/trajectories.profile_sigmoid evaluated at the real dates: logistic on 2*tau - 1, 0 at tau 0, 1 at tau 1."""
    logistic = lambda t: 1.0 / (1.0 + np.exp(-steepness * (2.0 * t - 1.0)))
    return (logistic(tau) - logistic(0.0)) / (logistic(1.0) - logistic(0.0))


PROFILES = {
    'sigmoid': sigmoid,
    'constant': lambda tau: np.ones_like(tau),
    'linear': lambda tau: tau,
    'staggered': lambda tau: np.clip((tau - 0.5) / 0.5, 0.0, None),
    'deflate_linear': lambda tau: -tau,
    'pulse': pulse,
}
SCENARIO_PROFILES = {   # (first source, second source)
    'S3_static': ('constant', 'constant'), 'N1_staggered': ('linear', 'sigmoid'),
    'N2_opposite': ('linear', 'deflate_linear'), 'N3_pulse': ('linear', 'pulse'), 'N4_moving': ('linear', 'linear'),
}


def unit_fields_mm(geom, x_km, y_km, depth_km):
    """LOS fields (mm per m^3) [T, N] of one source at per-date positions (arrays of length T, km)."""
    grid = geom['grid']
    xE = torch.tensor(grid['xE_m']); yN = torch.tensor(grid['yN_m'])
    los = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]))
    as_t = lambda v: torch.tensor(np.asarray(v, dtype=np.float64)[:, None] * 1e3)
    return mogi_los_mm_torch(xE, yN, los, as_t(x_km), as_t(y_km), as_t(depth_km),
                             torch.ones(len(x_km), 1, dtype=torch.float64))[:, 0].numpy()


def source_tracks(scenario, layout, move, n_dates):
    """Per-date (x, y, depth) km arrays [T] for both sources; only N4's first source moves."""
    tracks = []
    for index, src in enumerate(layout):
        if scenario == 'N4_moving' and index == 0:
            tracks.append(move)                                # dict of per-date arrays, filled by caller
        else:
            tracks.append(dict(x_km=np.full(n_dates, src['xcen_km']), y_km=np.full(n_dates, src['ycen_km']),
                               depth_km=np.full(n_dates, src['depth_km'])))
    return tracks


def render(geom, scenario, tracks, tau, sigma_ref_mm):
    """
    Signal [T, H, W] (mm) and per-source truth. dV(t) = A * profile(tau); A is set so the source's largest own peak
    |LOS| over the dates = PEAK_RATIO * sigma_ref (LOS is linear in dV, so this is exact).
    """
    shape = geom['grid']['shape']
    signal_mm = np.zeros((len(tau),) + tuple(shape))
    sources = []
    for index, (track, profile_name) in enumerate(zip(tracks, SCENARIO_PROFILES[scenario])):
        unit = unit_fields_mm(geom, track['x_km'], track['y_km'], track['depth_km'])        # [T, N]
        profile = PROFILES[profile_name](tau)
        unit_peak = np.abs(unit).max(axis=1)                                                   # mm per m^3, per date
        amplitude_m3 = PEAK_RATIO * sigma_ref_mm / np.max(np.abs(profile) * unit_peak)
        dV_m3 = amplitude_m3 * profile
        field = unit * dV_m3[:, None]
        signal_mm += field.reshape(signal_mm.shape)
        sources.append(dict(index=index, profile=profile_name, amplitude_m3=float(amplitude_m3),
                            xcen_km=track['x_km'].tolist(), ycen_km=track['y_km'].tolist(),
                            depth_km=track['depth_km'].tolist(), dV_m3=dV_m3.tolist(),
                            own_peak_mm=(np.abs(field).max(axis=1)).tolist(),
                            unit_field_at_ref_over_peak=None))
    return signal_mm, sources


def ref_leak(geom, track, ref_flat):
    """Largest |unit field at the reference pixel| / own peak over the dates (dimensionless)."""
    unit = unit_fields_mm(geom, track['x_km'], track['y_km'], track['depth_km'])
    return float(np.max(np.abs(unit[:, ref_flat]) / np.abs(unit).max(axis=1)))


def draw_layout(rng, geom, ref_flat, scenario, tau, max_tries=10000):
    """Random 2-source layout (+ N4 move) obeying the fixed draw rules. Returns (layout, move or None, tries)."""
    for tries in range(1, max_tries + 1):
        layout = [dict(xcen_km=float(rng.uniform(-R_BOX_KM, R_BOX_KM)), ycen_km=float(rng.uniform(-R_BOX_KM, R_BOX_KM)),
                       depth_km=float(rng.uniform(*R_DEPTH_KM))) for _ in range(2)]
        move = None
        if scenario == 'N4_moving':
            angle = rng.uniform(0, 2 * np.pi)
            end = (layout[0]['xcen_km'] + MOVE_DISTANCE_KM * np.cos(angle), layout[0]['ycen_km'] + MOVE_DISTANCE_KM * np.sin(angle))
            depth_end = max(SHALLOW_FACTOR * layout[0]['depth_km'], MIN_DEPTH_KM)
            if max(abs(end[0]), abs(end[1])) > R_BOX_KM:
                continue
            move = dict(x_end_km=float(end[0]), y_end_km=float(end[1]), depth_end_km=float(depth_end))
        tracks = source_tracks(scenario, layout, None, len(tau))
        if move is not None:
            tracks[0] = move_track(layout[0], move, tau)
        # separation on every date (horizontal centre distance)
        sep = np.hypot(tracks[0]['x_km'] - tracks[1]['x_km'], tracks[0]['y_km'] - tracks[1]['y_km'])
        if sep.min() < R_SEP_KM[0] or sep.max() > R_SEP_KM[1]:
            continue
        if max(ref_leak(geom, track, ref_flat) for track in tracks) > REF_LEAK_MAX:
            continue
        return layout, move, tries
    raise RuntimeError('could not draw a valid layout')


def move_track(start, move, tau):
    """Linear-in-tau move from the start position to the end position (km arrays [T])."""
    return dict(x_km=start['xcen_km'] + tau * (move['x_end_km'] - start['xcen_km']),
                y_km=start['ycen_km'] + tau * (move['y_end_km'] - start['ycen_km']),
                depth_km=start['depth_km'] + tau * (move['depth_end_km'] - start['depth_km']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=f'{STACKS}/albums_nonstatic_v2')   # v2 (2026-10-26): N1 sigmoid onset
    parser.add_argument('--seed', type=int, default=20261026, help='random-layout generator seed')
    parser.add_argument('--n-draws', type=int, default=2, help='random layouts per (scenario, scale)')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # --- Load geometry, noise set and reference pixel ---
    print("[1/4] Loading TA69 geometry and the E1 noise set...")
    geom = load_ta69_geometry()
    noise_dir = f'{STACKS}/noise_sets'
    for path in (f'{noise_dir}/{NOISE_SET}.npz', f'{noise_dir}/{NOISE_SET}.json', f'{noise_dir}/noise_summary.json'):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    with np.load(f'{noise_dir}/{NOISE_SET}.npz') as f:
        noise_mm = f['noise_mm'].astype(np.float64)            # [T, H, W], not yet referenced to the pixel
    meta = json.load(open(f'{noise_dir}/{NOISE_SET}.json'))
    sigma_ref_mm = json.load(open(f'{noise_dir}/noise_summary.json'))['sigma_ref_mm']
    ref_flat = meta['ref_pixel_flat']
    dates = meta['acq_dates']
    print(f"  Loaded noise: shape={noise_mm.shape}, dtype={noise_mm.dtype}; sigma_ref {sigma_ref_mm:.3f} mm, "
          f"ref pixel {ref_flat}, ref date {meta['ref_date']}")
    day = lambda s: datetime.strptime(s, '%Y%m%d')
    days = np.array([(day(d) - day(meta['ref_date'])).days for d in dates], dtype=np.float64)
    tau = days / days[-1]
    n_dates = len(dates)
    reference = lambda cube: cube - cube.reshape(n_dates, -1)[:, ref_flat][:, None, None]
    noise_ref_mm = reference(noise_mm)
    noise_rms_full_mm = float(np.sqrt(np.mean(noise_ref_mm ** 2)))
    print(f"  tau from {tau[0]:.3f} to {tau[-1]:.3f}; referenced noise RMS at scale 1 = {noise_rms_full_mm:.2f} mm")

    # --- Build the album list (G1 fixed, R random) ---
    print("[2/4] Building the album list...")
    rng = np.random.default_rng(args.seed)
    albums = []
    for scenario in SCENARIOS:
        g1_tracks = source_tracks(scenario, G1_LAYOUT, None, n_dates)
        if scenario == 'N4_moving':
            g1_tracks[0] = move_track(dict(xcen_km=G1_MOVE['x_start_km'], ycen_km=0.0, depth_km=G1_MOVE['depth_start_km']),
                                      dict(x_end_km=G1_MOVE['x_end_km'], y_end_km=0.0, depth_end_km=G1_MOVE['depth_end_km']),
                                      tau)
        for scale in G1_SCALES:
            albums.append(dict(name=f'G1_{scenario}_n{int(round(scale * 1000)):04d}', layout_set='G1', draw=None,
                               scenario=scenario, scale=scale, tracks=g1_tracks, layout=G1_LAYOUT, move=None, tries=0))
    for scenario in SCENARIOS:
        for scale in R_SCALES:
            for draw in range(1, args.n_draws + 1):
                layout, move, tries = draw_layout(rng, geom, ref_flat, scenario, tau)
                tracks = source_tracks(scenario, layout, None, n_dates)
                if move is not None:
                    tracks[0] = move_track(layout[0], move, tau)
                albums.append(dict(name=f'R{draw}_{scenario}_n{int(round(scale * 1000)):04d}', layout_set='R', draw=draw,
                                   scenario=scenario, scale=scale, tracks=tracks, layout=layout, move=move, tries=tries))
    names = [a['name'] for a in albums]
    assert len(set(names)) == len(names), 'duplicate album names'
    print(f"  {len(albums)} albums ({sum(a['layout_set'] == 'G1' for a in albums)} G1, "
          f"{sum(a['layout_set'] == 'R' for a in albums)} R)")

    # --- Render, write and check every album ---
    print(f"[3/4] Rendering and writing to {args.out}/...")
    for step, album in enumerate(albums, start=1):
        signal_mm, sources = render(geom, album['scenario'], album['tracks'], tau, sigma_ref_mm)
        for src, track in zip(sources, album['tracks']):
            src['unit_field_at_ref_over_peak'] = ref_leak(geom, track, ref_flat)
        signal_ref_mm = reference(signal_mm)
        obs_mm = reference(signal_mm + album['scale'] * noise_mm)
        nonoise_path = os.path.join(args.out, f"{album['name']}_nonoise.npz")
        album_path = os.path.join(args.out, f"{album['name']}.npz")
        write_album(album_path, obs_mm, geom, dates)
        write_album(nonoise_path, signal_ref_mm, geom, dates)

        # Built-in checks (independent QA comes later in qa_nonstatic_albums.py)
        with np.load(album_path) as f:
            back_obs_mm = f['los_asc_m'].astype(np.float64) * 1e3
        with np.load(nonoise_path) as f:
            back_sig_mm = f['los_asc_m'].astype(np.float64) * 1e3
        assert np.allclose(back_obs_mm - back_sig_mm, album['scale'] * noise_ref_mm, atol=1e-3), 'noise round trip'
        for src in sources:
            assert abs(max(src['own_peak_mm']) / (PEAK_RATIO * sigma_ref_mm) - 1) < 1e-9, 'peak normalisation'
            assert abs(PROFILES[src['profile']](np.array([0.0]))[0]) < 1e-12 or src['profile'] == 'constant'

        noise_rms_mm = album['scale'] * noise_rms_full_mm
        album_peak_mm = float(np.abs(signal_ref_mm).max())
        start = [dict(xcen_km=t['x_km'][0], ycen_km=t['y_km'][0], depth_km=t['depth_km'][0]) for t in album['tracks']]
        end = [dict(xcen_km=t['x_km'][-1], ycen_km=t['y_km'][-1], depth_km=t['depth_km'][-1]) for t in album['tracks']]
        pascal = [separation_dshallow(*start), separation_dshallow(*end)]
        truth = dict(album=album['name'], layout_set=album['layout_set'], draw=album['draw'], scenario=album['scenario'],
                     profiles=list(SCENARIO_PROFILES[album['scenario']]), noise_set=NOISE_SET,
                     noise_scale=album['scale'], noise_rms_mm=noise_rms_mm, album_peak_mm=album_peak_mm,
                     snr_peak_over_rms=None if noise_rms_mm == 0 else album_peak_mm / noise_rms_mm,
                     sigma_ref_mm=sigma_ref_mm, peak_ratio=PEAK_RATIO, ref_date=meta['ref_date'], dates=dates,
                     tau=tau.tolist(), ref_pixel_flat=ref_flat, n_sources=2, sources_per_date=sources,
                     separation_dshallow_start=float(pascal[0]), separation_dshallow_end=float(pascal[1]),
                     pascal_flag=bool(min(pascal) < 1.6), random_layout_seed=args.seed if album['layout_set'] == 'R' else None,
                     layout_draw_tries=album['tries'], move=album['move'],
                     convention='N4 snapshot: each date = one Mogi at the current position with the current cumulative dV',
                     units='km, m^3, LOS mm (+ toward satellite); npz in metres')
        for path, payload in ((album_path.replace('.npz', '_truth.json'), truth),
                              (nonoise_path.replace('.npz', '_truth.json'), dict(truth, album=album['name'] + '_nonoise',
                                                                                 noise_scale=0.0, noise_rms_mm=0.0,
                                                                                 snr_peak_over_rms=None))):
            text = json.dumps(payload, indent=1)
            if os.path.exists(path):
                if open(path).read() != text:
                    raise FileExistsError(f'{path} exists with DIFFERENT content; not overwriting')
            else:
                with open(path, 'w') as f:
                    f.write(text)
        snr = truth['snr_peak_over_rms']
        print(f"  ({step}/{len(albums)}) {album['name']}: peak {album_peak_mm:.1f} mm, noise RMS {noise_rms_mm:.2f} mm, "
              f"SNR {'inf' if snr is None else f'{snr:.1f}'}")

    # --- Regression: G1 static must reproduce the existing S3 SNR albums ---
    print("[4/4] Regression vs existing E1_S3_K2_separated albums...")
    for scale, tag in ((0.25, 'n025'), (0.5, 'n050')):
        old_path = f'{STACKS}/albums_snr/E1_S3_K2_separated_{tag}.npz'
        new_path = os.path.join(args.out, f'G1_S3_static_n{int(round(scale * 1000)):04d}.npz')
        with np.load(old_path) as f_old, np.load(new_path) as f_new:
            diff_m = float(np.max(np.abs(f_old['los_asc_m'].astype(np.float64) - f_new['los_asc_m'])))
            same_dates = np.array_equal(f_old['dates'], f_new['dates'])
        print(f"  {tag}: max |new - old| = {diff_m * 1e3:.2e} mm, dates equal {same_dates}")
        if diff_m > 1e-6 or not same_dates:
            raise RuntimeError(f'regression FAILED for {tag}')
    print(f"  REGRESSION PASS ({time.time() - t0:.1f}s total)")
    print(f"Done. Output saved to {args.out}/")


if __name__ == '__main__':
    main()
