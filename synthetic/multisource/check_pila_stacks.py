#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.check_pila_stacks [--root synthetic/multisource/pila_stacks]
#                  [--fig-dir figures/multisource_seq/step0_v2]
# Description: Step-0 sanity checks for the sequential multi-source PILA albums. Every check prints
#              PASS/FAIL with numbers, and the script exits non-zero if any check fails. Also writes
#              QC figures (PNG): noise RMS per set, example noise acquisitions, E1 scenario montage.
# Checks:
#   A  PILA loader round-trip (load_insar_mintpy on the npz): shape, mm units, finite, N = 16384,
#      LOS unit vectors and coordinates identical to the TA69 grid used to build the albums.
#   B  Reference pixel = 0 mm in every acquisition.
#   C  Paired design + physics: obs(scenario) - obs(S0) on the same noise set equals the Mogi field
#      re-rendered from the truth JSON with the LOADER's coordinates/LOS (minus its ref-pixel value).
#   D  Each source's own peak |LOS| / sigma_ref equals the specified peak ratio.
#   E  Pascal separation: S3/S5 = 3.0 x shallower depth (not flagged), S4 = 1.2 (flagged).
#   F  Dates: 30 unique, increasing, all after the reference date.
#   G  Calibration and evaluation noise come from disjoint tracks.
#   H  Evaluation noise RMS lies inside the calibration RMS range.
#   I  E1 and E2 (same Marapi window) share the minimum possible number of days.
#   J  Turbulence fields keep natural RMS scatter with expected RMS = 6 mm.
# Date:        2026-10-06

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from synthetic.multisource.make_pila_stack import SCENARIOS, load_ta69_geometry
from synthetic.multisource.torch_generator import mogi_los_mm_torch

RESULTS = []


def check(name, ok, detail):
    """Record and print one PASS/FAIL line."""
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load_album(path):
    """Load an album through the real PILA loader (no multilook, no mask)."""
    return load_insar_mintpy(path, 'asc', None, -0.38, 100.47, multilook=1, verbose=False)


def render_truth_mm(data, sources):
    """Mogi sum on the loader's grid [H, W] mm, using the truth sources (km, m^3)."""
    xE = torch.tensor(data.xE_grid.ravel() * 1e3); yN = torch.tensor(data.yN_grid.ravel() * 1e3)
    los = torch.tensor(np.stack([data.losE_grid.ravel(), data.losN_grid.ravel(), data.losU_grid.ravel()]))
    total = np.zeros(data.xE_grid.size)
    per_source = []
    for src in sources:
        field = mogi_los_mm_torch(xE, yN, los, *(torch.tensor([[v]], dtype=torch.float64) for v in
                                                 (src['xcen_km'] * 1e3, src['ycen_km'] * 1e3,
                                                  src['depth_km'] * 1e3, src['dV_m3'])))[0, 0].numpy()
        per_source.append(field.reshape(data.xE_grid.shape))
        total += field
    return total.reshape(data.xE_grid.shape), per_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='synthetic/multisource/pila_stacks_v2')
    parser.add_argument('--album-subdir', default='albums')
    parser.add_argument('--fig-dir', default='figures/multisource_seq/step0_v2')
    args = parser.parse_args()
    album_dir = os.path.join(args.root, args.album_subdir)
    noise_dir = os.path.join(args.root, 'noise_sets')
    if not os.path.isdir(album_dir):
        raise FileNotFoundError(album_dir)
    os.makedirs(args.fig_dir, exist_ok=True)
    geom = load_ta69_geometry()
    grid = geom['grid']
    with open(os.path.join(noise_dir, 'noise_summary.json')) as f:
        sigma_ref_mm = json.load(f)['sigma_ref_mm']
    album_paths = sorted(glob.glob(os.path.join(album_dir, '*.npz')))
    print(f"[1/3] {len(album_paths)} albums, sigma_ref = {sigma_ref_mm:.2f} mm")
    check('album count', len(album_paths) == 57, f'{len(album_paths)} (expected 39 + 18 = 57)')

    print("[2/3] Per-album checks...")
    loaded, truths = {}, {}
    worst = dict(los_vec=0.0, coord=0.0, ref=0.0, paired=0.0, peak=0.0)
    shape_ok = finite_ok = dates_ok = True
    for path in album_paths:
        album = os.path.basename(path)[:-4]
        with open(path[:-4] + '_truth.json') as f:
            truth = json.load(f)
        data = load_album(path)
        loaded[album], truths[album] = data, truth
        shape_ok &= data.los_mm.shape == (30, 128, 128) and data.n_points == 128 * 128
        finite_ok &= bool(np.isfinite(data.los_mm).all())
        worst['los_vec'] = max(worst['los_vec'], np.abs(data.losE_grid.ravel() - grid['losE']).max(),
                               np.abs(data.losN_grid.ravel() - grid['losN']).max(),
                               np.abs(data.losU_grid.ravel() - grid['losU']).max())
        worst['coord'] = max(worst['coord'], np.abs(data.xE_grid.ravel() * 1e3 - grid['xE_m']).max(),
                             np.abs(data.yN_grid.ravel() * 1e3 - grid['yN_m']).max())
        worst['ref'] = max(worst['ref'], np.abs(data.los_mm.reshape(30, -1)[:, truth['ref_pixel_flat']]).max())
        dates = data.dates
        dates_ok &= (len(set(dates)) == 30 and dates == sorted(dates)
                     and dates[0].replace('-', '') > truth['ref_date'])
        _, per_source = render_truth_mm(data, truth['sources'])
        for src, field in zip(truth['sources'], per_source):
            worst['peak'] = max(worst['peak'], abs(np.abs(field).max() / sigma_ref_mm - src['peak_ratio']))
    check('A loader shape/N', shape_ok, 'all albums (30, 128, 128), N = 16384')
    check('A finite', finite_ok, 'no NaN/inf after loading')
    check('A LOS vectors == build grid', worst['los_vec'] < 1e-6, f"max |diff| {worst['los_vec']:.2e}")
    check('A coordinates == build grid', worst['coord'] < 1e-3, f"max |diff| {worst['coord']:.2e} m")
    check('B reference pixel = 0', worst['ref'] < 1e-4, f"max |LOS at ref pixel| {worst['ref']:.2e} mm")
    check('D peak/sigma per source', worst['peak'] < 1e-6, f"max |error| {worst['peak']:.2e}")
    check('F dates', dates_ok, '30 unique increasing dates after the reference date in every album')

    # C: paired design + physics re-render
    for set_id in ['E1', 'E2', 'E3']:
        null = loaded[f'{set_id}_S0_K0_null']
        for scenario in SCENARIOS:
            if scenario == 'S0_K0_null':
                continue
            album = f'{set_id}_{scenario}'
            field, _ = render_truth_mm(loaded[album], truths[album]['sources'])
            expected = field - field.ravel()[truths[album]['ref_pixel_flat']]
            diff = loaded[album].los_mm - null.los_mm - expected[None]
            worst['paired'] = max(worst['paired'], np.abs(diff).max())
    check('C paired design + Mogi re-render', worst['paired'] < 1e-3,
          f"max |obs(S) - obs(S0) - field| {worst['paired']:.2e} mm over 15 albums x 30 acquisitions")

    # E: Pascal separation
    sep = {s: truths[f'E1_{s}']['separation_dshallow'] for s in ('S3_K2_separated', 'S4_K2_close', 'S5_K2_one_weak')}
    flags = {s: truths[f'E1_{s}']['pascal_flag'] for s in sep}
    check('E Pascal separation', abs(sep['S3_K2_separated'] - 3.0) < 0.02 and abs(sep['S4_K2_close'] - 1.2) < 0.02
          and flags['S4_K2_close'] and not flags['S3_K2_separated'],
          ', '.join(f"{s} {v:.2f}{' (flagged)' if flags[s] else ''}" for s, v in sep.items()))

    # G, H: noise pools
    meta = {}
    for path in sorted(glob.glob(os.path.join(noise_dir, '[CE]*.json'))):
        with open(path) as f:
            m = json.load(f)
        meta[m['set_id']] = m
    cal_tracks = {m['track'] for k, m in meta.items() if k.startswith('C')}
    eval_tracks = {m['track'] for k, m in meta.items() if k.startswith('E')}
    check('G disjoint noise pools', not (cal_tracks & eval_tracks),
          f"{len(cal_tracks)} calibration tracks, evaluation tracks {sorted(eval_tracks)}")
    cal_rms = np.array([m['noise_rms_mm_mean'] for k, m in meta.items() if k.startswith('C')])
    eval_rms = np.array([m['noise_rms_mm_mean'] for k, m in meta.items() if k.startswith('E')])
    overlap = len(set(meta['E1']['day_idx']) & set(meta['E2']['day_idx']))
    check('I E1/E2 day overlap minimal', overlap <= 2 * 31 - 58,
          f"{overlap} shared days (minimum possible with 58 Marapi dates: {2 * 31 - 58})")
    check('H evaluation RMS within calibration range',
          cal_rms.min() <= eval_rms.min() and eval_rms.max() <= cal_rms.max(),
          f"calibration {cal_rms.min():.1f}-{cal_rms.max():.1f} mm, evaluation {eval_rms.min():.1f}-{eval_rms.max():.1f} mm")

    from synthetic.multisource.turbulence import fit_turbulence, sample_turbulence_mm
    turb = sample_turbulence_mm(np.random.default_rng(7), 400, fit_turbulence(verbose=False))
    turb_rms = np.sqrt(np.mean(turb ** 2, axis=(1, 2)))
    check('J turbulence RMS', abs(np.sqrt(np.mean(turb_rms ** 2)) - 6.0) < 0.3 and turb_rms.std() > 0.3,
          f"RMS of RMS {np.sqrt(np.mean(turb_rms ** 2)):.2f} mm (target 6), scatter p5-p95 "
          f"{np.percentile(turb_rms, 5):.1f}-{np.percentile(turb_rms, 95):.1f} mm")

    print("[3/3] QC figures...")
    # Fig 1: noise RMS per set
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ids = sorted(meta)
    colors = ['tab:red' if i.startswith('E') else 'tab:blue' for i in ids]
    ax.bar(range(len(ids)), [meta[i]['noise_rms_mm_mean'] for i in ids], color=colors, label='total noise')
    ax.bar(range(len(ids)), [meta[i]['era5_rms_mm'] for i in ids], color='k', alpha=0.35, width=0.5, label='ERA5 part')
    ax.axhline(sigma_ref_mm, color='k', ls='--', label=f'σ_ref = {sigma_ref_mm:.1f} mm')
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([f"{i} {meta[i]['track']}" for i in ids], rotation=90, fontsize=7)
    ax.set_ylabel('Mean per-acquisition RMS (mm)')
    ax.set_title('Noise per set (blue = calibration, other volcanoes; red = evaluation, Marapi TA69 track)')
    ax.legend()
    plt.tight_layout(); fig.savefig(os.path.join(args.fig_dir, 'noise_rms_per_set.png'), dpi=150); plt.close(fig)

    # Fig 2: one noise acquisition per example set
    examples = ['C01', 'C10', 'C16', 'C34', 'E1', 'E3']
    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4))
    extent = [geom['x_m'][0] / 1e3, geom['x_m'][-1] / 1e3, geom['y_m'][-1] / 1e3, geom['y_m'][0] / 1e3]
    for ax, set_id in zip(axes, examples):
        album = f'{set_id}_cal' if set_id.startswith('C') else f'{set_id}_S0_K0_null'
        img = loaded[album].los_mm[10]
        vmax = np.percentile(np.abs(img), 99)
        im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(f"{set_id} {meta[set_id]['track']}\n{loaded[album].dates[10]}", fontsize=9)
        ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
        fig.colorbar(im, ax=ax, fraction=0.046).set_label('LOS (mm)')
    plt.tight_layout(); fig.savefig(os.path.join(args.fig_dir, 'noise_examples.png'), dpi=150); plt.close(fig)

    # Fig 3: E1 scenario montage: truth field | one acquisition | 30-acquisition mean
    scenarios = list(SCENARIOS)
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(13, 3.6 * len(scenarios)))
    for row, scenario in enumerate(scenarios):
        album = f'E1_{scenario}'
        data, truth = loaded[album], truths[album]
        field, _ = render_truth_mm(data, truth['sources'])
        panels = [(field, 'True source field'), (data.los_mm[0], f'Acquisition 1 ({data.dates[0]})'),
                  (data.los_mm.mean(0), 'Mean of 30 acquisitions')]
        vmax = max(np.percentile(np.abs(data.los_mm[0]), 99), np.abs(field).max(), 5.0)
        for col, (img, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent)
            for src in truth['sources']:
                ax.plot(src['xcen_km'], src['ycen_km'], '^', mfc='none', mec='k', ms=9, mew=1.5)
            ax.set_title(f"{scenario}: {title}" if col == 0 else title, fontsize=9)
            ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
            fig.colorbar(im, ax=ax, fraction=0.046).set_label('LOS (mm)')
    plt.tight_layout(); fig.savefig(os.path.join(args.fig_dir, 'E1_scenario_montage.png'), dpi=110); plt.close(fig)

    n_fail = RESULTS.count(False)
    print(f"Done. {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed. Figures in {args.fig_dir}/")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
