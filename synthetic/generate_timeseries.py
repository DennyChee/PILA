#!/usr/bin/env python
# Usage:       python -m synthetic.generate_timeseries --spec synthetic/specs/sierranegra_sun69_buildup_clean.json
#              (or)  from synthetic.generate_timeseries import generate; generate(spec)
# Description: Generate a synthetic InSAR LOS time-series with known ground-truth
#              source parameters and write it as a MintPy-format HDF5 the production
#              loader (datasets.preprocessing.insar_mintpy.load_insar_mintpy) reads
#              unchanged. Deformation is evaluated at FULL resolution on a real
#              scene's grid (so the loader's mask-aware multilook averages it exactly
#              as it would real data), projected to LOS with that scene's per-pixel
#              LOS unit vectors, optionally has noise added, and is saved in metres.
#              A JSON sidecar records the per-epoch ground-truth physical parameters.
#              Emits sanity-check plots (peak-epoch LOS map, signal-RMS per epoch).
# Scientific conventions:
#   * Reuses the real scene's geometry + mask files UNCHANGED (never modified).
#   * LOS sign: positive = motion toward satellite (MintPy enu2los), via
#     enu_to_los_unit_vectors -> d_los = losE*uE + losN*uN + losU*uU.
#   * Output units: metres (loader converts to mm). Source params stored in SI.
#   * Cumulative time-series: epoch 0 is the quiescent baseline; deformation
#     accumulates via the amplitude ramp in the trajectory.
# Date:        2026-06-22

import argparse
import json
import os
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.preprocessing.insar_mintpy import (
    enu_to_los_unit_vectors, read_lonlat_grid, lonlat_to_local_enu_km)
from synthetic.forward_numpy import run_model, FORWARD_MODELS
from synthetic import trajectories as traj
from synthetic import noise as noisemod
# Import at module top so the presentation-ready matplotlib rcParams set in synthetic.plots
# apply to the local sanity plots below too (not just the montage built via _plots).
from synthetic import plots as _plots


# --- Geometry: full-resolution ENU coords + per-pixel LOS unit vectors ------
def load_full_res_geometry(geometry_h5, lat0, lon0):
    """
    Read a MintPy geometry file and return full-resolution per-pixel quantities.

    Returns
    -------
    xE_m, yN_m : np.ndarray [L, W]   local-ENU East/North of each pixel, metres.
    losE, losN, losU : np.ndarray [L, W]   LOS unit-vector components.
    shape : (L, W)
    """
    with h5py.File(geometry_h5, 'r') as f:
        inc = np.array(f['incidenceAngle'], dtype=np.float64)   # deg
        az = np.array(f['azimuthAngle'], dtype=np.float64)      # deg
    n_rows, n_cols = inc.shape

    losE, losN, losU = enu_to_los_unit_vectors(inc, az)         # [L, W]

    lon_grid, lat_grid = read_lonlat_grid(geometry_h5, n_rows, n_cols)
    xE_km, yN_km = lonlat_to_local_enu_km(lon_grid, lat_grid, lat0, lon0)
    return xE_km * 1000.0, yN_km * 1000.0, losE, losN, losU, (n_rows, n_cols)


def _read_mask(mask_h5, shape):
    """Read a boolean mask [L,W] or return None if no file."""
    if not mask_h5 or not os.path.exists(mask_h5):
        return None
    with h5py.File(mask_h5, 'r') as f:
        key = 'mask' if 'mask' in f else list(f.keys())[0]
        m = np.array(f[key], dtype=bool)
    return m if m.shape == tuple(shape) else None


def _clean_vs_noisy_map(out_dir, name, clean_cube_m, noisy_cube_m, mask, peak, snr_db):
    """Side-by-side clean / noisy / noise-only LOS maps (mm) at the peak epoch."""
    clean = clean_cube_m[peak] * 1000.0
    noisy = noisy_cube_m[peak] * 1000.0
    noise = noisy - clean
    if mask is not None:
        clean = np.where(mask, clean, np.nan)
        noisy = np.where(mask, noisy, np.nan)
        noise = np.where(mask, noise, np.nan)
    vmax = np.nanmax(np.abs(clean)) + 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, d, ttl in [(axes[0], clean, 'Clean signal'),
                       (axes[1], noisy, f'Signal + noise (SNR {snr_db:.1f} dB)'),
                       (axes[2], noise, 'Noise only')]:
        im = ax.imshow(d, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
        ax.set_title(ttl); ax.set_xlabel('Column (px)'); ax.set_ylabel('Row (px)')
        cb = fig.colorbar(im, ax=ax); cb.set_label('LOS (mm), + toward sat')
    fig.suptitle(f'{_plots._pretty(name)}: peak epoch {peak}', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_clean_vs_noisy.png'), dpi=150)
    fig.savefig(os.path.join(out_dir, f'{name}_clean_vs_noisy.pdf'))
    plt.close(fig)


def read_base_dates_attrs(timeseries_h5):
    """Return (raw_date_bytes [n], bperp [n] or None, attrs dict) from a MintPy ts."""
    with h5py.File(timeseries_h5, 'r') as f:
        raw_dates = np.array(f['date'])                          # |S8 bytes
        bperp = np.array(f['bperp']) if 'bperp' in f else None
        attrs = dict(f.attrs)
    return raw_dates, bperp, attrs


# --- Core generation --------------------------------------------------------
def build_clean_cube(model_name, xE_m, yN_m, losE, losN, losU,
                     param_names, params_per_epoch):
    """
    Evaluate the forward model per epoch at full resolution and project to LOS.

    Returns
    -------
    los_cube_m : np.ndarray [n_epoch, L, W]  clean LOS displacement, metres.
    """
    n_epoch = params_per_epoch.shape[0]
    L, W = losE.shape
    xflat, yflat = xE_m.ravel(), yN_m.ravel()
    eflat, nflat, uflat = losE.ravel(), losN.ravel(), losU.ravel()

    los_cube_m = np.zeros((n_epoch, L, W), dtype=np.float32)
    for i in range(n_epoch):
        params_si = {name: params_per_epoch[i, j]
                     for j, name in enumerate(param_names)}
        uE_mm, uN_mm, uU_mm = run_model(model_name, xflat, yflat, params_si)
        d_los_mm = eflat * uE_mm + nflat * uN_mm + uflat * uU_mm   # toward-sat +, mm
        los_cube_m[i] = (d_los_mm / 1000.0).reshape(L, W).astype(np.float32)  # -> m
    return los_cube_m


def write_synthetic_h5(out_h5, los_cube_m, raw_dates, bperp, base_attrs, provenance):
    """Write a MintPy-format synthetic timeseries h5 (datasets + attrs + provenance)."""
    n_epoch = los_cube_m.shape[0]
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('timeseries', data=los_cube_m.astype(np.float32))
        f.create_dataset('date', data=raw_dates[:n_epoch])
        if bperp is not None:
            f.create_dataset('bperp', data=bperp[:n_epoch].astype(np.float32))
        else:
            f.create_dataset('bperp', data=np.zeros(n_epoch, dtype=np.float32))
        # Copy geocoding + identity attrs verbatim so any MintPy tool can open it.
        for k, v in base_attrs.items():
            f.attrs[k] = v
        f.attrs['FILE_TYPE'] = 'timeseries'
        f.attrs['UNIT'] = 'm'
        for k, v in provenance.items():
            f.attrs[k] = v


def sanity_plots(out_dir, name, los_cube_m, mask_h5, param_names, params_per_epoch):
    """Peak-epoch LOS map + signal-RMS-per-epoch curve (mm). Saves png + pdf."""
    los_mm = los_cube_m * 1000.0
    mask = None
    if mask_h5 and os.path.exists(mask_h5):
        with h5py.File(mask_h5, 'r') as f:
            key = 'mask' if 'mask' in f else list(f.keys())[0]
            mask = np.array(f[key], dtype=bool)

    # Signal RMS and peak |LOS| per epoch over coherent pixels (mm). Peak is the largest
    # absolute surface deformation present that epoch (more physical than RMS, which is
    # diluted by the quiet area around a compact source).
    rms = np.zeros(los_mm.shape[0])
    peak_mm_per_epoch = np.zeros(los_mm.shape[0])
    for i in range(los_mm.shape[0]):
        vals = los_mm[i][mask] if mask is not None else los_mm[i].ravel()
        vals = vals[np.isfinite(vals)]
        rms[i] = np.sqrt(np.mean(vals ** 2)) if vals.size else 0.0
        peak_mm_per_epoch[i] = np.max(np.abs(vals)) if vals.size else 0.0
    peak = int(np.argmax(rms))

    # --- Peak-epoch LOS map ---
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = np.where(mask, los_mm[peak], np.nan) if mask is not None else los_mm[peak]
    vmax = np.nanmax(np.abs(disp))
    im = ax.imshow(disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
    ax.set_title(f'{_plots._pretty(name)}\nclean LOS, peak epoch {peak} '
                 f'(signal RMS {rms[peak]:.1f} mm)')
    ax.set_xlabel('Column (pixel)'); ax.set_ylabel('Row (pixel)')
    cb = fig.colorbar(im, ax=ax); cb.set_label('LOS displacement (mm), + toward satellite')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_peak_los_map.png'), dpi=150)
    fig.savefig(os.path.join(out_dir, f'{name}_peak_los_map.pdf'))
    plt.close(fig)

    # --- Signal RMS + amplitude per epoch ---
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(rms, 'o-', color='C0', label='signal RMS')
    ax1.set_xlabel('Epoch index'); ax1.set_ylabel('Signal RMS (mm)', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    amp_key = 'opening' if 'opening' in param_names else 'dV'
    aj = param_names.index(amp_key)
    ax2 = ax1.twinx()
    ax2.plot(params_per_epoch[:, aj], 's--', color='C1', alpha=0.7, label=amp_key)
    unit = 'm' if amp_key == 'opening' else 'm³'
    ax2.set_ylabel(f'{amp_key} ({unit})', color='C1'); ax2.tick_params(axis='y', labelcolor='C1')
    ax1.set_title(f'{_plots._pretty(name)}: signal buildup over epochs')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_signal_rms.png'), dpi=150)
    fig.savefig(os.path.join(out_dir, f'{name}_signal_rms.pdf'))
    plt.close(fig)
    return rms, peak, peak_mm_per_epoch


# --- Spec-driven entry point ------------------------------------------------
def generate(spec):
    """
    Generate a synthetic cube from a spec dict (see synthetic/specs/*.json).

    Returns the output h5 path and the truth-json path.
    """
    t0 = time.time()
    name = spec['name']
    model_name = spec['model']
    base = spec['base_scene']
    out_dir = spec.get('out_dir', os.path.join('synthetic', 'cubes', name))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/6] Loading base-scene geometry: {base['geometry']}")
    xE_m, yN_m, losE, losN, losU, (L, W) = load_full_res_geometry(
        base['geometry'], base['lat0'], base['lon0'])
    print(f"  Grid: {L} x {W}; LOS |U| median = {np.nanmedian(losU):.3f}")

    raw_dates, bperp, base_attrs = read_base_dates_attrs(base['timeseries'])
    n_epoch_base = len(raw_dates)
    print(f"  Base scene has {n_epoch_base} epochs")

    print(f"[2/6] Building trajectory ({model_name}, {spec['trajectory']['type']})")
    tr = spec['trajectory']
    n_epochs = tr.get('n_epochs', n_epoch_base)
    if tr['type'] == 'static_buildup':
        param_names, params = traj.static_buildup(
            model_name, n_epochs, tr['geometry'], tr['amplitude_key'],
            tr['amp_start'], tr['amp_end'], tr.get('profile', 'sigmoid'))
    elif tr['type'] == 'moving':
        # `profiles` (dict, per-parameter) takes precedence over `profile` (str, global),
        # so geometry can migrate linearly while amplitude ramps sigmoidally.
        moving_profile = tr.get('profiles', tr.get('profile', 'linear'))
        param_names, params = traj.build_trajectory(
            model_name, n_epochs, tr['start'], tr['end'], moving_profile)
    else:
        raise ValueError(f"Unknown trajectory type '{tr['type']}'")

    # --- Bounds check against the PILA paras json (recoverability) ---
    with open(spec['paras'], 'r') as f:
        paras_ranges = json.load(f)
    ok, violations = traj.check_within_bounds(model_name, param_names, params, paras_ranges)
    if not ok:
        print("  WARNING: ground truth outside PILA paras bounds (unrecoverable):")
        for v in violations:
            print("    -", v)
    else:
        print("  Ground-truth trajectory within PILA paras bounds.")

    print(f"[3/6] Evaluating forward model at full resolution ({L*W} px x {n_epochs} ep)")
    clean_cube_m = build_clean_cube(model_name, xE_m, yN_m, losE, losN, losU,
                                    param_names, params)

    # --- Noise injection ---------------------------------------------------
    # Plan: FIXED noise, signal amplitude drives the SNR (sweep). The realized
    # SNR is computed (MATLAB convention) and recorded. Optionally scale noise to
    # hit a target SNR instead.
    noise_spec = spec.get('noise', {'type': 'none'})
    ntype = noise_spec.get('type', 'none')
    mask_arr = _read_mask(base.get('mask'), (L, W))
    realized = {'type': ntype}
    if ntype == 'none':
        los_cube_m = clean_cube_m
    elif ntype == 'marapi':
        comps = tuple(noise_spec.get('components', ['combined']))
        noise_cube = noisemod.load_marapi_noise(comps).astype(np.float32)
        if noise_cube.shape != clean_cube_m.shape:
            raise ValueError(
                f"Marapi noise cube {noise_cube.shape} != signal cube "
                f"{clean_cube_m.shape}. Mode B requires the Marapi 128x128/76-epoch "
                f"grid (see synthetic/make_marapi_grid.py).")
        if noise_spec.get('reference_epoch0', True):
            noise_cube = noise_cube - noise_cube[0]            # cumulative referencing
        # Either scale to a target SNR or apply a fixed global scale (default 1).
        if 'target_snr_db' in noise_spec:
            scale, info = noisemod.scale_to_target_snr(
                clean_cube_m, noise_cube, noise_spec['target_snr_db'], mask_arr)
        else:
            scale = float(noise_spec.get('scale', 1.0)); info = {}
        noise_cube = (noise_cube * scale).astype(np.float32)
        snr_db, s_rms, n_rms, ep = noisemod.compute_snr_db(clean_cube_m, noise_cube, mask_arr)
        # Peak |LOS| deformation at the SNR epoch (mm) and the linear signal/noise ratio
        # (= 10^(SNR_dB/20)) — more physically readable companions to the dB value.
        s_peak = noisemod._valid_peak(clean_cube_m[ep], mask_arr)
        snr_ratio = (s_rms / n_rms) if n_rms > 0 else float('nan')
        los_cube_m = (clean_cube_m + noise_cube).astype(np.float32)
        realized.update({'components': list(comps), 'scale': scale,
                         'realized_snr_db': snr_db, 'snr_ratio': snr_ratio,
                         'signal_rms_mm': s_rms * 1000, 'signal_peak_mm': s_peak * 1000,
                         'noise_rms_mm': n_rms * 1000, 'snr_epoch': ep, **info})
        print(f"  Marapi noise {comps}: realized SNR = {snr_db:.2f} dB "
              f"(signal {s_rms*1000:.1f} mm / noise {n_rms*1000:.1f} mm @ epoch {ep})")
    else:
        raise ValueError(f"Unknown noise type '{ntype}'")

    print(f"[4/6] Writing synthetic h5")
    out_h5 = os.path.join(out_dir, f'{name}.h5')
    provenance = {
        'SYNTHETIC': 1,
        'SYNTH_FORWARD_MODEL': model_name,
        'SYNTH_NOISE_TYPE': ntype,
        'SYNTH_TARGET_SNR_DB': float(realized.get('realized_snr_db', np.nan)),
    }
    write_synthetic_h5(out_h5, los_cube_m, raw_dates, bperp, base_attrs, provenance)
    print(f"  Wrote {out_h5}  shape={los_cube_m.shape} dtype={los_cube_m.dtype}")

    print(f"[5/6] Sanity plots")
    # Signal RMS / peak come from the CLEAN cube (signal, not noise). When noisy,
    # also save a clean-vs-noisy peak-epoch comparison map.
    rms, peak, peak_mm_per_epoch = sanity_plots(out_dir, name, clean_cube_m, base.get('mask'),
                                                param_names, params)
    if ntype != 'none':
        _clean_vs_noisy_map(out_dir, name, clean_cube_m, los_cube_m, mask_arr, peak,
                            realized.get('realized_snr_db', float('nan')))
    # Multi-epoch montage of the data PILA actually sees (noisy if noise added).
    dates_lbl = [d.decode() if isinstance(d, (bytes, bytearray)) else str(d)
                 for d in raw_dates[:los_cube_m.shape[0]]]
    # Plot EVERY input epoch (not a subset) so a moving source can be sanity-checked
    # frame-by-frame; n_frames = number of epochs in the cube PILA actually sees.
    _plots.plot_timeseries_montage(out_dir, name, los_cube_m, dates=dates_lbl,
                                   mask=mask_arr, n_frames=los_cube_m.shape[0],
                                   title_suffix=(' (signal+noise)' if ntype != 'none'
                                                 else ' (clean)'))

    # --- Amplitude-independent footprint diagnostics (CLEAN cube) ---------------
    # These decouple spatial extent from amplitude growth, so a depth/size-changing
    # source can be checked frame-by-frame. The width-driving parameter (depth for a
    # rising Mogi, radius for a growing sill, length for a propagating dike) is overlaid
    # and annotated on each normalised panel. Computed on the CLEAN signal (noise would
    # bias the half-max region and per-epoch normalisation).
    WIDTH_PARAM = {'mogi': 'd', 'sun69': 'radius', 'okada': 'length'}
    wkey = WIDTH_PARAM.get(model_name)
    overlay_km, annot = None, None
    if wkey in param_names:
        wj = param_names.index(wkey)
        overlay_km = params[:, wj] / 1000.0            # km_params are metres -> km
        annot = [f'{wkey}={v:.1f}km' for v in overlay_km]
    px_km = float(np.nanmedian(np.abs(np.diff(xE_m, axis=1)))) / 1000.0  # grid spacing
    _plots.plot_normalised_montage(out_dir, name, clean_cube_m, dates=dates_lbl,
                                   mask=mask_arr, annot=annot, title_suffix=' (clean)')
    _plots.plot_footprint_width(out_dir, name, clean_cube_m, px_km, mask=mask_arr,
                                overlay=overlay_km,
                                overlay_label=(f'{wkey} (km)' if wkey else None))

    print(f"[6/6] Writing ground-truth sidecar")
    dates_iso = [d.decode() if isinstance(d, (bytes, bytearray)) else str(d)
                 for d in raw_dates[:n_epochs]]
    dates_iso = [f"{s[0:4]}-{s[4:6]}-{s[6:8]}" for s in dates_iso]
    truth = {
        'name': name,
        'forward_model': model_name,
        'param_names': param_names,
        'param_units': _param_units(model_name, param_names),
        'epochs': dates_iso,
        'params_per_epoch': params.tolist(),
        'peak_epoch_index': peak,
        'signal_rms_mm_per_epoch': rms.tolist(),
        'signal_peak_mm_per_epoch': peak_mm_per_epoch.tolist(),
        'noise': realized,
        'base_scene': base,
        'paras': spec['paras'],
        'timeseries_h5': out_h5,
    }
    truth_json = os.path.join(out_dir, f'{name}_truth.json')
    with open(truth_json, 'w') as f:
        json.dump(truth, f, indent=2)
    print(f"  Wrote {truth_json}")
    print(f"Done in {time.time()-t0:.1f}s. Output in {out_dir}")
    return out_h5, truth_json


def _param_units(model_name, param_names):
    """Return a units string per parameter (for the truth sidecar)."""
    u = {'xcen': 'm', 'ycen': 'm', 'd': 'm', 'depth': 'm', 'radius': 'm',
         'xoff': 'm', 'yoff': 'm', 'length': 'm', 'width': 'm',
         'strike': 'deg', 'dip': 'deg', 'opening': 'm', 'dV': 'm^3'}
    return [u.get(n, '?') for n in param_names]


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic InSAR LOS time-series for PILA testing.")
    ap.add_argument('--spec', required=True, help='Path to a scenario spec JSON.')
    args = ap.parse_args()
    if not os.path.exists(args.spec):
        raise FileNotFoundError(args.spec)
    with open(args.spec, 'r') as f:
        spec = json.load(f)
    generate(spec)


if __name__ == '__main__':
    main()
