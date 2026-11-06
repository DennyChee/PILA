#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.diagnose_ta69_quality \
#                  --windows synthetic/multisource/real_ta69_quiet_windows
# Description: Read-only data-quality diagnostics of the real Marapi (W. Sumatra) S1 TA69
#              stack inside the "quiet" windows chosen by build_ta69_quiet_windows.py, to judge
#              whether the large patchy signal is deformation, residual troposphere or
#              processing artefacts:
#                (1) temporal / average spatial coherence maps and distributions;
#                (2) LOS velocity vs DEM height (stratified-troposphere residual);
#                (3) consecutive-epoch differences: peaks at multiples of lambda/2 = 27.75 mm
#                    (Sentinel-1, lambda = 0.0555 m) indicate phase-unwrapping errors;
#                (4) per-pixel time series: fraction of variance explained by a linear trend
#                    (steady deformation -> high) and RMS growth with time.
# Data: FULL-resolution geocoded files (no multilook), windows mapped back from x4 indices.
# Date:        2026-10-05

import argparse
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GEO_DIR = '/eos-rs/INSAR_processing/denny/Marapi/S1_TA69/mintpy/geo'
TS_FILE = 'geo_timeseries_SET_ERA5_ramp_demErr.h5'
HALF_WAVELENGTH_MM = 0.0555 / 2 * 1e3          # 27.75 mm: one 2*pi phase cycle in LOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', required=True, help='dir with windows.json')
    args = parser.parse_args()
    win_json = os.path.join(args.windows, 'windows.json')
    for path in [win_json, os.path.join(GEO_DIR, TS_FILE)]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    with open(win_json) as f:
        win_cfg = json.load(f)
    factor = win_cfg['multilook']

    print("[1/4] Reading coherence, height, velocity, mask (full resolution)...")
    with h5py.File(os.path.join(GEO_DIR, 'geo_temporalCoherence.h5'), 'r') as f:
        tcoh = f['temporalCoherence'][:]
    with h5py.File(os.path.join(GEO_DIR, 'geo_avgSpatialCoh.h5'), 'r') as f:
        scoh = f['coherence'][:]
    with h5py.File(os.path.join(GEO_DIR, 'geo_geometryRadar.h5'), 'r') as f:
        height_m = f['height'][:]
    with h5py.File(os.path.join(GEO_DIR, 'geo_velocity.h5'), 'r') as f:
        vel_mm_yr = f['velocity'][:] * 1e3
    with h5py.File(os.path.join(GEO_DIR, 'geo_maskTempCoh.h5'), 'r') as f:
        mask = f['mask'][:].astype(bool)
    print(f"  Loaded: shape={tcoh.shape}, dtype={tcoh.dtype}; mask valid {mask.mean():.1%}")

    with h5py.File(os.path.join(GEO_DIR, TS_FILE), 'r') as f:
        dates = [d.decode() for d in f['date'][:]]
    years = np.array([(np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:]}') - np.datetime64(f'{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}'))
                      .astype(int) / 365.25 for d in dates])

    report = {}
    fig, axes = plt.subplots(len(win_cfg['windows']), 4, figsize=(21, 4.6 * len(win_cfg['windows'])), squeeze=False)
    for w_idx, win in enumerate(win_cfg['windows']):
        r0, r1 = win['rows'][0] * factor, win['rows'][1] * factor
        c0, c1 = win['cols'][0] * factor, win['cols'][1] * factor
        print(f"[2/4] Window W{w_idx}: full-res rows {r0}-{r1}, cols {c0}-{c1}")
        m = mask[r0:r1, c0:c1]
        with h5py.File(os.path.join(GEO_DIR, TS_FILE), 'r') as f:
            ts_mm = f['timeseries'][:, r0:r1, c0:c1].astype(np.float32) * 1e3        # [T, h, w] window only
        ts_v = ts_mm[:, m]                                                           # [T, n_valid]
        ts_v = ts_v - np.median(ts_v, axis=1, keepdims=True)                         # per-epoch window reference

        print("[3/4]   coherence / topography / unwrapping / time-series diagnostics...")
        diffs = np.diff(ts_v, axis=0).ravel()                                        # consecutive-epoch changes
        cycles = diffs / HALF_WAVELENGTH_MM
        near_cycle = np.abs(cycles - np.round(cycles)) < 0.15
        frac_jump = float(np.mean(near_cycle & (np.abs(np.round(cycles)) >= 1)))
        # Expected fraction near a nonzero multiple if changes were smooth Gaussian with the same spread:
        rng = np.random.default_rng(0)
        sim = rng.normal(0, np.std(diffs), size=min(diffs.size, 2_000_000)) / HALF_WAVELENGTH_MM
        frac_jump_gauss = float(np.mean((np.abs(sim - np.round(sim)) < 0.15) & (np.abs(np.round(sim)) >= 1)))
        design = np.c_[np.ones_like(years), years]
        coef, *_ = np.linalg.lstsq(design, ts_v, rcond=None)
        resid = ts_v - design @ coef
        r2_linear = 1 - resid.var(axis=0) / np.maximum(ts_v.var(axis=0), 1e-6)
        vel_w, h_w = vel_mm_yr[r0:r1, c0:c1][m], height_m[r0:r1, c0:c1][m]
        corr_vh = float(np.corrcoef(vel_w, h_w)[0, 1])
        rms_t = np.sqrt((ts_v ** 2).mean(axis=1))
        report[f'W{w_idx}'] = {
            'temporal_coherence_median': float(np.median(tcoh[r0:r1, c0:c1][m])),
            'spatial_coherence_median': float(np.median(scoh[r0:r1, c0:c1][m])),
            'velocity_height_corr': corr_vh,
            'frac_epoch_changes_near_2pi_multiple': frac_jump,
            'same_for_gaussian_changes_of_equal_spread': frac_jump_gauss,
            'consecutive_change_std_mm': float(np.std(diffs)),
            'linear_trend_R2_median': float(np.median(r2_linear)),
            'rms_first_epoch_mm': float(rms_t[1]), 'rms_last_epoch_mm': float(rms_t[-1])}
        for key, val in report[f'W{w_idx}'].items():
            print(f"    {key:45s} {val:.3f}")

        ax = axes[w_idx, 0]
        ax.hist(tcoh[r0:r1, c0:c1][m], bins=50, range=(0, 1), histtype='step', density=True, label='temporal coh.')
        ax.hist(scoh[r0:r1, c0:c1][m], bins=50, range=(0, 1), histtype='step', density=True, label='avg spatial coh.')
        ax.set_xlabel('Coherence'); ax.set_ylabel('Density'); ax.legend(); ax.set_title(f'W{w_idx} coherence')
        ax = axes[w_idx, 1]
        sub = rng.choice(len(vel_w), size=min(20000, len(vel_w)), replace=False)
        ax.scatter(h_w[sub], vel_w[sub], s=1, alpha=0.3)
        ax.set_xlabel('DEM height (m)'); ax.set_ylabel('LOS velocity (mm/yr)')
        ax.set_title(f'Velocity vs height (r = {corr_vh:+.2f})')
        ax = axes[w_idx, 2]
        lim = 4 * HALF_WAVELENGTH_MM
        ax.hist(np.clip(diffs, -lim, lim), bins=400, density=True)
        for k in range(-3, 4):
            if k:
                ax.axvline(k * HALF_WAVELENGTH_MM, color='r', ls=':', lw=1)
        ax.set_yscale('log'); ax.set_xlabel('Consecutive-epoch LOS change (mm)'); ax.set_ylabel('Density (log)')
        ax.set_title(f'Unwrapping check: red = ±n·λ/2 ({frac_jump:.1%} near, Gaussian {frac_jump_gauss:.1%})')
        ax = axes[w_idx, 3]
        ax.plot(years, rms_t, 'o-')
        ax.set_xlabel('Years since 2022-01-01'); ax.set_ylabel('Window RMS (mm)')
        ax.set_title(f'RMS growth; median linear-trend R² {np.median(r2_linear):.2f}')

    print("[4/4] Saving figure and report...")
    plt.tight_layout()
    fig.savefig(os.path.join(args.windows, 'fig_quality_diagnostics.png'), dpi=130)
    plt.close(fig)
    with open(os.path.join(args.windows, 'quality_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Done. Output saved to {args.windows}/")


if __name__ == '__main__':
    main()
