#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_clean_scenarios \
#                  [--out synthetic/multisource/clean_scenarios] \
#                  [--fig-dir figures/multisource_clean_synth]
# Description: NOISE-FREE multi-source Mogi scenes on the TA69 128x128 grid for K = 0..3 sources, several
#              source-location scenarios per K. Each scenario is written twice: `_static` (same field every
#              epoch) and `_tv` (30 epochs, every source has its own dV(t) history). Also writes the figures
#              (per-source panels, all-epochs montage, per-K overview) and scripted sanity checks.
# Assumptions: Mogi point source; LOS in mm, + toward satellite; TA69 geometry/dates; NO noise and NO
#              reference-pixel subtraction (far field ~0 by construction); inflation = +dV. Each source's
#              PEAK |LOS| = peak_ratio * sigma_ref_mm (same yardstick as the noisy albums).
#              LOS is linear in dV, so epoch field = sum_k profile_k(t) * (field_k at peak dV).
# Date:        2026-10-07

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic.multisource.make_pila_stack import (load_ta69_geometry, source_field_mm, write_album)

NOISE_SUMMARY_JSON = 'synthetic/multisource/pila_stacks_v2/noise_sets/noise_summary.json'
DATE_SOURCE_NPZ = 'synthetic/multisource/pila_stacks_v2/albums/E1_S3_K2_separated.npz'
PEAK_RATIO = 4.0   # peak |LOS| of each source in units of sigma_ref (strong, clearly detectable)
COLORBAR_LABEL = 'LOS displacement (mm, + toward satellite)'


def src(xcen_km, ycen_km, depth_km):
    """Source spec in the make_pila_stack format (East km, North km, depth km, peak/sigma)."""
    return dict(xcen_km=xcen_km, ycen_km=ycen_km, depth_km=depth_km, peak_ratio=PEAK_RATIO)


# --- Scenarios: name -> list of sources (E km, N km, depth km) ---
SCENARIOS = {
    'K0_null':             [],
    'K1_centre':           [src(0, 0, 5)],
    'K1_offset_NE':        [src(7, 6, 4)],
    'K1_deep_SW':          [src(-8, -5, 8)],
    'K2_separated':        [src(-6, 0, 4), src(6, 0, 5)],
    'K2_close':            [src(-2.35, 0, 4), src(2.35, 0, 5)],      # overlapping lobes (Pascal-ambiguous)
    'K2_diag_mixed_depth': [src(-7, -7, 3), src(6, 5, 8)],
    'K2_stacked':          [src(0, 0, 3), src(1, 0, 9)],             # near-collinear in plan view
    'K3_line':             [src(-9, 0, 4), src(0, 0, 5), src(9, 0, 4)],
    'K3_triangle':         [src(-6, -5, 4), src(6, -5, 5), src(0, 7, 4)],
    'K3_cluster_plus_far': [src(-3, 0, 4), src(3, 0, 5), src(10, -8, 6)],
}


# --- dV(t) profiles, normalised so profile(t_end) is the reference strength (peak = 1 where monotonic) ---
def profile_rise(t):
    """Linear rise 0 -> 1."""
    return t


def profile_sigmoid(t):
    """Logistic onset around mid-series (steepness 12), renormalised to exactly 0 at t=0 and 1 at t=1."""
    raw = 1.0 / (1.0 + np.exp(-12.0 * (t - 0.5)))
    return (raw - raw[0]) / (raw[-1] - raw[0])


def profile_pulse(t):
    """Rise then fall back to 0 (half-sine)."""
    return np.sin(np.pi * t)


def profile_constant(t):
    """Constant at full strength."""
    return np.ones_like(t)


PROFILE_ORDER = [('rise', profile_rise), ('sigmoid', profile_sigmoid), ('pulse', profile_pulse)]


def build_scenario(geom, sigma_ref_mm, sources, n_epoch, time_varying):
    """
    Build the noise-free stack [n_epoch, H, W] (mm) for one scenario.
    Returns stack_mm, per-source unit fields at peak dV [K, H, W] (mm), per-source dV(t) [K, n_epoch] (m^3),
    per-source profile names, filled sources.
    """
    t_norm = np.linspace(0.0, 1.0, n_epoch)
    height_width = geom['grid']['shape']
    peak_fields_mm = np.zeros((len(sources),) + tuple(height_width))
    dV_t_m3 = np.zeros((len(sources), n_epoch))
    profiles = np.zeros((len(sources), n_epoch))   # dimensionless dV(t)/dV_peak, used directly as superposition weight
    profile_names = []
    filled_sources = []
    for source_index, source in enumerate(sources):
        field_mm, filled = source_field_mm(geom, [source], sigma_ref_mm)   # one source at a time -> own peak
        peak_fields_mm[source_index] = field_mm
        filled_sources.append(filled[0])
        if time_varying:
            name, function = PROFILE_ORDER[source_index % len(PROFILE_ORDER)]
            profile = function(t_norm)
        else:
            name, profile = 'static', profile_constant(t_norm)
        profile_names.append(name)
        dV_t_m3[source_index] = filled[0]['dV_m3'] * profile
        profiles[source_index] = profile
    # Linear superposition: epoch field = sum_k (dV_k(t) / dV_k_peak) * field_k_at_peak
    stack_mm = np.zeros((n_epoch,) + tuple(height_width))
    for source_index in range(len(filled_sources)):
        stack_mm += profiles[source_index][:, None, None] * peak_fields_mm[source_index][None]
    return stack_mm, peak_fields_mm, dV_t_m3, profile_names, filled_sources


# --- Figures -----------------------------------------------------------------------------------
def _extent_km(geom):
    x_km, y_km = geom['x_m'] / 1e3, geom['y_m'] / 1e3
    return [x_km.min(), x_km.max(), y_km.min(), y_km.max()]


def plot_sources(name, geom, peak_fields_mm, total_mm, sources, out_png):
    """Per-source fields + sum, true centres marked, depth/dV annotated. Common symmetric scale."""
    extent = _extent_km(geom)
    n_cols = len(sources) + 1
    vmax_mm = max(np.abs(total_mm).max(), 1.0)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols + 0.8, 3.6), squeeze=False)
    panels = list(peak_fields_mm) + [total_mm]
    titles = [f"S{i + 1}: d={s['depth_km']:.1f} km, dV={s['dV_m3']:.2e} m³" for i, s in enumerate(sources)] + ['Sum (all sources)']
    for axis, panel, title in zip(axes[0], panels, titles):
        image = axis.imshow(panel, cmap='RdBu_r', vmin=-vmax_mm, vmax=vmax_mm, extent=extent, origin='upper')
        axis.set_title(title, fontsize=8)
        axis.set_xlabel('East (km)'); axis.set_ylabel('North (km)')
        for s in sources:
            axis.plot(s['xcen_km'], s['ycen_km'], 'k+', ms=10, mew=1.5)
    if not sources:
        axes[0, 0].text(0.5, 0.5, 'K=0: no source\n(field = 0)', transform=axes[0, 0].transAxes, ha='center', va='center')
    fig.colorbar(image, ax=axes.ravel().tolist(), label=COLORBAR_LABEL, shrink=0.85).set_label(COLORBAR_LABEL, fontsize=7)
    fig.suptitle(name, fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close(fig)


def plot_all_epochs(name, geom, stack_mm, dates, dV_t_m3, profile_names, out_png, n_cols=6):
    """ONE figure: every epoch (montage, common symmetric scale) + dV(t) trace per source."""
    extent = _extent_km(geom)
    n_rows = int(np.ceil(len(dates) / n_cols))   # derived so any epoch count fits the grid
    vmax_mm = max(np.abs(stack_mm).max(), 1.0)
    fig = plt.figure(figsize=(2.3 * n_cols + 1.2, 2.3 * n_rows + 3.2))
    grid_spec = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [1.1], hspace=0.45, wspace=0.12)
    for epoch in range(len(dates)):
        axis = fig.add_subplot(grid_spec[epoch // n_cols, epoch % n_cols])
        image = axis.imshow(stack_mm[epoch], cmap='RdBu_r', vmin=-vmax_mm, vmax=vmax_mm, extent=extent, origin='upper')
        axis.set_title(f"{epoch + 1}: {dates[epoch]}", fontsize=7)
        axis.tick_params(labelsize=6)
        if epoch % n_cols == 0:
            axis.set_ylabel('North (km)', fontsize=7)
        if epoch // n_cols == n_rows - 1:
            axis.set_xlabel('East (km)', fontsize=7)
    trace_axis = fig.add_subplot(grid_spec[n_rows, :])
    epoch_axis = np.arange(1, len(dates) + 1)
    for source_index in range(dV_t_m3.shape[0]):
        trace_axis.plot(epoch_axis, dV_t_m3[source_index], marker='o', ms=3, label=f"S{source_index + 1} ({profile_names[source_index]})")
    trace_axis.set_xlabel('Epoch index'); trace_axis.set_ylabel('dV (m³)'); trace_axis.legend(fontsize=7)
    trace_axis.grid(alpha=0.3)
    fig.colorbar(image, ax=fig.axes[:n_rows * n_cols], label=COLORBAR_LABEL, shrink=0.6, pad=0.01)
    fig.suptitle(f"{name}: all {len(dates)} epochs", fontsize=11)
    fig.savefig(out_png, dpi=130, bbox_inches='tight'); plt.close(fig)


def plot_overview(k_value, names, static_fields_mm, geom, sources_by_name, out_png):
    """Contact sheet: all scenarios of one K (static field), one panel per scenario, shared scale."""
    extent = _extent_km(geom)
    vmax_mm = max(max(np.abs(static_fields_mm[n]).max() for n in names), 1.0)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names) + 0.8, 3.6), squeeze=False)
    for axis, name in zip(axes[0], names):
        image = axis.imshow(static_fields_mm[name], cmap='RdBu_r', vmin=-vmax_mm, vmax=vmax_mm, extent=extent, origin='upper')
        axis.set_title(name, fontsize=8); axis.set_xlabel('East (km)'); axis.set_ylabel('North (km)')
        for s in sources_by_name[name]:
            axis.plot(s['xcen_km'], s['ycen_km'], 'k+', ms=10, mew=1.5)
    fig.colorbar(image, ax=axes.ravel().tolist(), label=COLORBAR_LABEL, shrink=0.85).set_label(COLORBAR_LABEL, fontsize=7)
    fig.suptitle(f'K = {k_value} source(s), static noise-free fields (+ = true centre)', fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close(fig)


# --- Sanity checks -----------------------------------------------------------------------------
def run_checks(results, geom, sigma_ref_mm, out_dir):
    """Scripted pass/fail checks; returns (lines, n_fail). Caller prints and exits non-zero on failure."""
    lines, n_fail = [], 0

    def record(ok, message):
        nonlocal n_fail
        n_fail += (not ok)
        lines.append(f"{'PASS' if ok else 'FAIL'}  {message}")

    cell_km = float(np.mean(np.diff(geom['x_m'])) / 1e3)
    x_km, y_km = geom['x_m'] / 1e3, geom['y_m'] / 1e3
    for name, res in results.items():
        stack_mm = res['stack_mm']
        record(np.isfinite(stack_mm).all(), f"{name}: all finite, shape={stack_mm.shape}")
        if res['time_varying']:
            continue
        record(np.ptp(stack_mm, axis=0).max() == 0.0, f"{name}: static epochs identical (noise-free)")
        sources = res['filled_sources']
        if not sources:
            record(np.abs(stack_mm).max() == 0.0, f"{name}: K0 field identically 0")
            continue
        for source_index, source in enumerate(sources):
            peak_mm = np.abs(res['peak_fields_mm'][source_index]).max()
            record(abs(peak_mm / (PEAK_RATIO * sigma_ref_mm) - 1) < 1e-6, f"{name} S{source_index + 1}: peak {peak_mm:.2f} mm == {PEAK_RATIO}*sigma_ref")
        record(np.allclose(stack_mm[0], res['peak_fields_mm'].sum(axis=0), atol=1e-9), f"{name}: sum == sum of parts (linearity)")
        if len(sources) == 1:   # isolated source: Mogi LOS extremum lies near (not exactly at) the centre (LOS tilt) -> 2 km tolerance
            row, col = np.unravel_index(np.argmax(np.abs(res['peak_fields_mm'][0])), res['peak_fields_mm'][0].shape)
            offset_km = np.hypot(x_km[col] - sources[0]['xcen_km'], y_km[row] - sources[0]['ycen_km'])
            record(offset_km < 2.0, f"{name}: peak pixel {offset_km:.2f} km from true centre (cell {cell_km:.2f} km)")
    for name in [n for n in results if n.endswith('_tv')]:
        res = results[name]
        # Independent recomputation: epoch field = sum_k (dV_k(t)/dV_k_peak) * field_k_at_peak
        expected_mm = sum((res['dV_t_m3'][k] / res['filled_sources'][k]['dV_m3'])[:, None, None] * res['peak_fields_mm'][k][None]
                          for k in range(len(res['filled_sources'])))
        record(np.allclose(res['stack_mm'], expected_mm, atol=1e-9), f"{name}: stack == sum_k profile_k(t)*field_k")
        for k, profile_name in enumerate(res['profile_names']):
            profile = res['dV_t_m3'][k] / res['filled_sources'][k]['dV_m3']
            if profile_name == 'rise':
                ok = abs(profile[0]) < 1e-12 and abs(profile[-1] - 1) < 1e-12
            elif profile_name == 'sigmoid':
                ok = abs(profile[0]) < 1e-12 and abs(profile[-1] - 1) < 1e-12 and np.all(np.diff(profile) >= 0)
            else:   # pulse: 0 at both ends, max 1 mid-series
                ok = abs(profile[0]) < 1e-12 and abs(profile[-1]) < 1e-12 and abs(profile.max() - 1) < 1e-2
            record(ok, f"{name} S{k + 1}: {profile_name} profile endpoints/shape")
    record(n_fail == 0, f"TOTAL failures: {n_fail}")
    with open(os.path.join(out_dir, 'checks.txt'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    return lines, n_fail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='synthetic/multisource/clean_scenarios')
    parser.add_argument('--fig-dir', default='figures/multisource_clean_synth')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.fig_dir, exist_ok=True)

    print("[1/5] Loading TA69 geometry, sigma_ref, dates...")
    geom = load_ta69_geometry()
    with open(NOISE_SUMMARY_JSON) as handle:
        sigma_ref_mm = json.load(handle)['sigma_ref_mm']
    with np.load(DATE_SOURCE_NPZ) as handle:
        dates = [str(d) for d in handle['dates']]
    print(f"  grid={geom['grid']['shape']}, sigma_ref={sigma_ref_mm:.2f} mm, n_epoch={len(dates)}, inc={geom['inc_deg']:.2f} deg")

    print(f"[2/5] Building {len(SCENARIOS)} scenarios x (static, tv)...")
    t0 = time.time()
    results = {}
    for scenario_name, sources in SCENARIOS.items():
        for time_varying in (False, True):
            if time_varying and not sources:
                continue    # K0 has nothing to vary
            full_name = f"{scenario_name}_{'tv' if time_varying else 'static'}"
            stack_mm, peak_fields_mm, dV_t_m3, profile_names, filled = build_scenario(geom, sigma_ref_mm, sources, len(dates), time_varying)
            results[full_name] = dict(stack_mm=stack_mm, peak_fields_mm=peak_fields_mm, dV_t_m3=dV_t_m3, profile_names=profile_names,
                                      filled_sources=filled, time_varying=time_varying, scenario=scenario_name)
            print(f"  {full_name:32s} K={len(sources)} peak|LOS|={np.abs(stack_mm).max():6.1f} mm")
    print(f"  Done in {time.time() - t0:.1f}s")

    print("[3/5] Writing npz + truth JSON (never overwrites a differing file)...")
    for full_name, res in results.items():
        write_album(os.path.join(args.out, f'{full_name}.npz'), res['stack_mm'], geom, dates)
        truth = dict(album=full_name, scenario=res['scenario'], n_sources=len(res['filled_sources']),
                     time_varying=res['time_varying'], sources=res['filled_sources'], profile_names=res['profile_names'],
                     dV_t_m3=res['dV_t_m3'].tolist(), dates=dates, sigma_ref_mm=sigma_ref_mm, noise='none',
                     reference_pixel_subtraction=False, units='km, m^3, LOS mm (+ toward satellite)')
        truth_path = os.path.join(args.out, f'{full_name}_truth.json')
        if os.path.exists(truth_path):
            with open(truth_path) as handle:
                old_truth = json.load(handle)
            if old_truth != json.loads(json.dumps(truth)):
                raise FileExistsError(f"{truth_path} exists and DIFFERS from the new truth; not overwriting")
            print(f"  {truth_path} exists and matches; kept")
            continue
        with open(truth_path, 'w') as handle:
            json.dump(truth, handle, indent=1)

    print("[4/5] Sanity checks...")
    lines, n_fail = run_checks(results, geom, sigma_ref_mm, args.out)
    print('\n'.join('  ' + line for line in lines))

    print("[5/5] Figures...")
    for full_name, res in results.items():
        if not res['time_varying']:
            plot_sources(full_name, geom, res['peak_fields_mm'], res['stack_mm'][0], res['filled_sources'],
                         os.path.join(args.fig_dir, f'{full_name}_sources.png'))
        else:
            plot_all_epochs(full_name, geom, res['stack_mm'], dates, res['dV_t_m3'], res['profile_names'],
                            os.path.join(args.fig_dir, f'{full_name}_allepochs.png'))
    for k_value in range(4):
        names = [n for n in SCENARIOS if n.startswith(f'K{k_value}_')]
        plot_overview(k_value, names, {n: results[f'{n}_static']['stack_mm'][0] for n in names}, geom,
                      {n: results[f'{n}_static']['filled_sources'] for n in names},
                      os.path.join(args.fig_dir, f'overview_K{k_value}.png'))
    print(f"Done. Output saved to {args.out}/ and {args.fig_dir}/ ({n_fail} check failures)")
    if n_fail:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
