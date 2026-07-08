#!/usr/bin/env python
# Usage:       python plot_insar_timeseries.py \
#                  --resume saved/sierranegra_mogi_insar_A/<ts>/models/model_best.pth \
#                  [--config <config.json>] [--n_map_epochs 6] [--output_dir DIR] [--gif]
# Description: Full-time-series verification plots for a PILA InSAR (Mogi_LOS / Okada_LOS /
#              Sun69_LOS, MLP) checkpoint. This is the InSAR analog of plot_mogi_results.py
#              for GNSS: where GNSS reconstructs the whole displacement time series as 2-D
#              line graphs (displacement vs date, per ENU component/station), an InSAR epoch
#              is a SPATIAL LOS field, so the time series is the temporal evolution of the
#              spatial pattern. The model is run over EVERY epoch (one batched inference) and
#              three deliverables are produced:
#                1. maps-over-time   — actual / reconstruction / residual LOS maps across a
#                                      set of epochs (small-multiples grid; optional GIF),
#                                      on a shared colour scale so growth is visible.
#                2. per-pixel LOS(t) — target vs predicted LOS time series at a few
#                                      representative points (direct analog of the GNSS
#                                      per-station timeseries_uz figure).
#                3. parameters/fit   — inferred source parameters over time (dV(t), depth(t),
#                                      ...), pooled predicted-vs-target scatter, and per-epoch
#                                      RMSE/R^2 over time.
# Date:        2026-06-18
#
# Scientific conventions (READ — these affect interpretation):
#   * LOS sign: positive = motion TOWARD the satellite (range decrease), MintPy convention.
#     Displacement maps use a diverging colormap (RdBu_r) centred at 0.
#   * The dataset z-scores LOS with a single GLOBAL (scene-wide) scaler. Model outputs
#     (x_PB physics+residual, x_P pure physics) and targets are de-standardized to mm with
#     mm = std*x_scale_global + x_mean_global before any plotting / RMSE.
#   * PILA inverts PER EPOCH (amortized): each epoch gets its OWN source-parameter estimate,
#     so dV(t)/depth(t) are genuine time series. A fixed source location is only SOFTLY
#     enforced (Stage B temporal-smoothness reg), not guaranteed — epoch-to-epoch geometry
#     wobble is therefore expected and is visible in the maps-over-time / parameter plots.
#   * lat0/lon0 in the config is the local-ENU peg, NOT the inverted source location.

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # headless (HPC / no display)
import matplotlib.pyplot as plt

from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from datasets.displacementGPS import time_feats
from utils import read_json
# Reuse the single-epoch script's helpers so the two tools stay consistent.
from plot_insar_results import (build_model, hard_z_flags, load_hillshade,
                                points_to_grid, _r2_rmse)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Inferred-parameter display metadata ------------------------------------
# Maps a rescaled physics-parameter name to (human label, display unit, scale factor
# applied to the SI value from rescale()). Units mirror plot_insar_results.format_param_text.
def param_label_unit_scale(name):
    """Return (label, unit, scale) for a rescaled physics parameter `name`."""
    km_params = {'xcen', 'ycen', 'd', 'xoff', 'yoff', 'depth', 'length', 'width', 'radius'}
    if name == 'dV':
        return ('Volume change', '×10⁶ m³', 1e-6)   # m³ -> 10^6 m³
    if name in km_params:
        return (name, 'km', 1e-3)                    # m -> km
    if name in ('strike', 'dip'):
        return (name, 'deg', 1.0)
    if name == 'opening':
        return ('opening', 'm', 1.0)
    return (name, '', 1.0)


def run_inference_all_epochs(model, d, config, device):
    """
    Run the trained PILA model over EVERY epoch in one batched forward pass.

    Inputs:
        model  : PHYS_VAE_SMPL in eval() mode.
        d      : InSARData (the multilooked MintPy field; cumulative LOS in mm).
        config : run config dict.
        device : torch.device.

    Returns (all mm / physical units, NumPy):
        actual_mm : [n_epoch, N]   observed cumulative LOS (= d.los_points_mm).
        full_mm   : [n_epoch, N]   physics + low-rank residual reconstruction (x_PB).
        phys_mm   : [n_epoch, N]   pure-physics reconstruction (x_P, denoised).
        params    : dict name -> [n_epoch]  inferred source parameters (SI units), or {}.
    """
    # GLOBAL standardization (must match dataset + decoder); shape [n_epoch, N].
    std_series = (d.los_points_mm - d.x_mean_global) / d.x_scale_global
    x_in = torch.tensor(std_series.astype(np.float32), device=device)         # [n_epoch, N]

    # 4-dim time features per epoch; the decoder uses only the first time_feat_dim.
    tfeat_dim = config['arch']['args'].get('time_feat_dim', 4)
    tfeat = np.stack([time_feats(dt, time_feat_dim=tfeat_dim) for dt in d.dates])
    tfeat = torch.tensor(tfeat.astype(np.float32), device=device)             # [n_epoch, T]

    hz_phy, hz_aux = hard_z_flags(config)
    with torch.no_grad():
        latent_phy, _latent_aux, x_PB, x_P = model(
            x_in, t=tfeat, inference=True, hard_z_phy=hz_phy, hard_z_aux=hz_aux, const=None)

    # De-standardize to mm with the same global scaler.
    full_mm = x_PB.cpu().numpy() * d.x_scale_global + d.x_mean_global         # [n_epoch, N]
    phys_mm = x_P.cpu().numpy() * d.x_scale_global + d.x_mean_global          # [n_epoch, N]
    actual_mm = d.los_points_mm.astype(np.float64)                           # [n_epoch, N]

    params = {}
    if not config['arch']['phys_vae']['no_phy']:
        # rescale() is vectorized over the batch (z_phy[:, i]); each value is [n_epoch].
        rescaled = model.physics_model.rescale(latent_phy)
        for k, v in rescaled.items():
            arr = v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)
            params[k] = arr.reshape(-1)                                       # [n_epoch]
    return actual_mm, full_mm, phys_mm, params


def plot_maps_over_time(actual_mm, full_mm, d, dates, idxs, hs, hs_extent, disp_extent,
                        out_base):
    """
    Deliverable #1 — temporal evolution of the spatial LOS pattern.

    Small-multiples grid: one row per selected epoch, columns =
        [Actual LOS | Reconstruction (x_PB) | Residual (actual − x_PB)].
    ALL three columns share ONE colour scale (the displacement vmax) and a single colorbar,
    so the pattern's growth over time is directly comparable AND the residual is shown on the
    same scale as the signal — a small residual correctly appears faint/near-white (good fit)
    rather than being stretched to full saturation by its own auto-scaled colorbar.
    """
    print(f"[plot 1/3] maps-over-time for {len(idxs)} epochs ...")
    t0 = time.time()

    # Single shared scale across the selected epochs AND across all three columns, so the
    # residual is read against the same magnitude as the deformation signal.
    disp_vals = np.concatenate([actual_mm[idxs].ravel(), full_mm[idxs].ravel()])
    vmax = float(np.nanmax(np.abs(disp_vals)))

    n_rows = len(idxs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows), squeeze=False)
    im = None
    col_titles = ['Actual LOS', 'Reconstruction (x_PB)', 'Residual (actual − x_PB)']

    for row, epoch_idx in enumerate(idxs):
        actual_grid = points_to_grid(actual_mm[epoch_idx], d.mask_d)
        full_grid = points_to_grid(full_mm[epoch_idx], d.mask_d)
        resid_grid = actual_grid - full_grid
        grids = [actual_grid, full_grid, resid_grid]

        for col, grid in enumerate(grids):
            ax = axes[row][col]
            ax.imshow(hs, extent=hs_extent, cmap='gray', origin='upper',
                      vmin=0.0, vmax=1.0, zorder=0)
            # Same vmin/vmax for every panel => one common colour scale.
            im = ax.imshow(np.ma.masked_invalid(grid), extent=disp_extent, origin='upper',
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.8, zorder=1)
            ax.set_xlim(disp_extent[0], disp_extent[1])
            ax.set_ylim(disp_extent[2], disp_extent[3])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(f'{dates[epoch_idx]}\nLatitude (deg)', fontsize=9)
            else:
                ax.set_ylabel('Latitude (deg)', fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel('Longitude (deg)', fontsize=9)

    # One shared colorbar for all columns (actual, reconstruction AND residual on one scale).
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                        location='right', pad=0.02)
    cbar.set_label('LOS displacement (mm)')

    # Pixel count for QC: number of valid (multilooked) InSAR pixels fed to the model,
    # plus the coarse rasterization grid these are scattered onto.
    n_rows_grid, n_cols_grid = d.mask_d.shape
    # Place the (2-line) suptitle just ABOVE the figure box so it never collides with the
    # per-column titles — important for the single-row (single-date) layout where the figure
    # is short. bbox_inches='tight' on save keeps it in-frame.
    fig.suptitle('PILA InSAR — temporal evolution of the LOS field (actual vs reconstruction)\n'
                 f'N = {d.n_points} valid pixels  (coarse grid {n_rows_grid}×{n_cols_grid})',
                 fontsize=14, y=1.06 if n_rows == 1 else 1.0)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}_maps_over_time.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_base}_maps_over_time.png / .pdf  (done in {time.time() - t0:.1f}s)")


def animate_maps(full_mm, actual_mm, d, dates, hs, hs_extent, disp_extent, out_base, fps=4):
    """
    Optional 3-panel GIF across ALL epochs — a continuous view of the temporal evolution
    of the spatial pattern, with the input data, the reconstruction, and the residual side
    by side in one figure:
        Input LOS (actual) | Reconstruction (x_PB) | Residual (actual − x_PB).
    All three panels share ONE colour scale (the displacement vmax), so the fit is directly
    comparable and the residual is shown against the same magnitude as the signal (a small
    residual correctly looks faint rather than fully saturated). Requires Pillow
    (matplotlib PillowWriter).
    """
    print("[plot 1b/3] writing 3-panel maps animation (GIF) ...")
    t0 = time.time()
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        print(f"  Skipping GIF (animation backend unavailable: {exc})")
        return

    # Single shared scale over ALL epochs and all panels so the animation does not "flicker"
    # between frames and the residual is read on the same scale as the signal.
    vmax = float(np.nanmax(np.abs(np.concatenate([actual_mm.ravel(), full_mm.ravel()]))))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    panel_meta = [('Input LOS (actual)', actual_mm, vmax, 'LOS displacement (mm)'),
                  ('Reconstruction (x_PB)', full_mm, vmax, 'LOS displacement (mm)'),
                  ('Residual (actual − x_PB)', actual_mm - full_mm, vmax, 'LOS displacement (mm)')]

    images = []
    for ax, (title, series, scale, cbar_label) in zip(axes, panel_meta):
        ax.imshow(hs, extent=hs_extent, cmap='gray', origin='upper',
                  vmin=0.0, vmax=1.0, zorder=0)
        grid0 = points_to_grid(series[0], d.mask_d)
        im = ax.imshow(np.ma.masked_invalid(grid0), extent=disp_extent, origin='upper',
                       cmap='RdBu_r', vmin=-scale, vmax=scale, alpha=0.8, zorder=1)
        ax.set_xlim(disp_extent[0], disp_extent[1])
        ax.set_ylim(disp_extent[2], disp_extent[3])
        ax.set_title(title)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(cbar_label)
        images.append(im)

    suptitle = fig.suptitle('')

    def update(frame):
        # Recompute each panel's grid for this epoch (input, reconstruction, residual).
        series_frame = [actual_mm[frame], full_mm[frame], actual_mm[frame] - full_mm[frame]]
        for im, series_vals in zip(images, series_frame):
            im.set_array(np.ma.masked_invalid(points_to_grid(series_vals, d.mask_d)))
        suptitle.set_text(f'PILA InSAR — {dates[frame]}  (epoch {frame + 1}/{len(dates)})  '
                          f'|  N = {d.n_points} valid pixels')
        return (*images, suptitle)

    anim = FuncAnimation(fig, update, frames=len(dates), blit=False)
    gif_path = f'{out_base}_maps.gif'
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved {gif_path}  (done in {time.time() - t0:.1f}s)")


def select_representative_points(actual_mm):
    """
    Pick representative points by PEAK-over-time |LOS|:
        peak (max), mid (closest to the median), far-field (min).
    Returns list of (index, label).

    Uses the per-point maximum |LOS| over ALL epochs (not just the final epoch) so the
    'peak deformation' point is correct even for non-monotonic series (e.g. a transient or
    deflation-then-inflation sequence where the last epoch is not the largest signal).
    """
    cum_abs = np.max(np.abs(actual_mm), axis=0)          # [N] peak magnitude over time
    ix_peak = int(np.argmax(cum_abs))
    ix_far = int(np.argmin(cum_abs))
    ix_mid = int(np.argmin(np.abs(cum_abs - np.median(cum_abs))))
    return [(ix_peak, 'peak deformation'),
            (ix_mid, 'intermediate'),
            (ix_far, 'far-field')]


def plot_pixel_timeseries(actual_mm, full_mm, phys_mm, d, date_objs, out_base):
    """
    Deliverable #2 — per-pixel LOS(t): target vs predicted at representative points.
    Direct analog of the GNSS per-station timeseries_uz figure (one panel per point).
    """
    print("[plot 2/3] per-pixel LOS time series ...")
    t0 = time.time()
    selected = select_representative_points(actual_mm)

    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 3.2 * len(selected)),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax, (ix, label) in zip(axes, selected):
        y_true = actual_mm[:, ix]
        y_full = full_mm[:, ix]
        y_phys = phys_mm[:, ix]
        rmse_full = float(np.sqrt(np.mean((y_full - y_true) ** 2)))
        ax.plot(date_objs, y_true, color='black', lw=1.0, label='Actual')
        ax.plot(date_objs, y_full, color='tomato', lw=1.0, ls='--', label='Reconstruction (x_PB)')
        ax.plot(date_objs, y_phys, color='steelblue', lw=1.0, ls=':', label='Physics only (x_P)')
        # Point coordinates (km in the local-ENU frame) for context.
        ax.set_title(f'{label}  —  point {ix}  (E={d.xE_pts[ix]:.2f} km, N={d.yN_pts[ix]:.2f} km)  '
                     f'RMSE={rmse_full:.2f} mm', fontsize=10)
        ax.set_ylabel('LOS displacement (mm)', fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.suptitle('PILA InSAR — per-pixel LOS time series (actual vs reconstruction)\n'
                 f'N = {d.n_points} valid pixels total (3 representative shown)',
                 fontsize=13)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}_pixel_timeseries.{ext}', dpi=150)
    plt.close(fig)
    print(f"  Saved {out_base}_pixel_timeseries.png / .pdf  (done in {time.time() - t0:.1f}s)")


def plot_params_and_fit(actual_mm, full_mm, phys_mm, params, date_objs, out_base):
    """
    Deliverable #3 — inferred source parameters over time, pooled scatter, and per-epoch fit.

    Three figures:
      *_parameters : each inferred parameter (dV, depth, ...) vs date, with median line.
      *_scatter    : pooled (all epochs x all points) predicted-vs-target LOS, R^2/RMSE.
      *_fit_over_time : per-epoch RMSE (mm) and R^2 over date, physics-only vs +residual.
    """
    print("[plot 3/3] parameters over time + scatter + per-epoch fit ...")
    t0 = time.time()

    # --- (a) Inferred parameters over time ---
    if params:
        keys = list(params.keys())
        n = len(keys)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), squeeze=False)
        axes_flat = axes.ravel()
        for ax, key in zip(axes_flat, keys):
            label, unit, scale = param_label_unit_scale(key)
            vals = params[key] * scale
            ax.plot(date_objs, vals, lw=0.9, color='steelblue', alpha=0.85)
            median_val = float(np.median(vals))
            ax.axhline(median_val, color='tomato', lw=1.1, ls='--',
                       label=f'median: {median_val:.3g} {unit}')
            ax.set_ylabel(f'{label}\n({unit})' if unit else label, fontsize=9)
            ax.set_xlabel('Date', fontsize=9)
            ax.set_title(key, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        # Hide any unused axes.
        for ax in axes_flat[len(keys):]:
            ax.set_visible(False)
        # NOTE: position params (xcen/ycen, xoff/yoff) are local-ENU OFFSETS from the
        # lat0/lon0 peg in km, not absolute coordinates.
        fig.suptitle('PILA InSAR — inferred source parameters over time '
                     '(per-epoch / amortized inversion; position params are km offsets '
                     'from the peg)', fontsize=12)
        plt.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(f'{out_base}_parameters.{ext}', dpi=150)
        plt.close(fig)
        print(f"  Saved {out_base}_parameters.png / .pdf")

    # --- (b) Pooled predicted-vs-target scatter (all epochs, all points) ---
    y_true = actual_mm.ravel()
    y_full = full_mm.ravel()
    r2, rmse = _r2_rmse(y_full, y_true)
    lim = float(np.nanmax(np.abs(np.concatenate([y_true, y_full]))))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_full, s=3, alpha=0.1, color='steelblue', rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='1:1')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Actual LOS displacement (mm)')
    ax.set_ylabel('Reconstructed LOS displacement (mm)')
    ax.set_title(f'Pooled reconstruction vs actual (all epochs x points)\n'
                 f'R²={r2:.3f}, RMSE={rmse:.2f} mm, N={y_true.size}')
    ax.legend(loc='best')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}_scatter.{ext}', dpi=150)
    plt.close(fig)
    print(f"  Saved {out_base}_scatter.png / .pdf  (pooled R²={r2:.3f}, RMSE={rmse:.2f} mm)")

    # --- (c) Per-epoch fit over time (RMSE mm + R^2), physics-only vs +residual ---
    n_epoch = actual_mm.shape[0]
    rmse_phys = np.empty(n_epoch)
    rmse_full = np.empty(n_epoch)
    r2_phys = np.empty(n_epoch)
    r2_full = np.empty(n_epoch)
    for e in range(n_epoch):
        r2_phys[e], rmse_phys[e] = _r2_rmse(phys_mm[e], actual_mm[e])
        r2_full[e], rmse_full[e] = _r2_rmse(full_mm[e], actual_mm[e])

    fig, (ax_rmse, ax_r2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax_rmse.plot(date_objs, rmse_phys, color='steelblue', lw=1.0, label='physics only (x_P)')
    ax_rmse.plot(date_objs, rmse_full, color='tomato', lw=1.0, ls='--', label='physics + residual (x_PB)')
    ax_rmse.set_ylabel('RMSE (mm)', fontsize=10)
    ax_rmse.set_title('Per-epoch reconstruction RMSE', fontsize=11)
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(alpha=0.3)

    ax_r2.plot(date_objs, r2_phys, color='steelblue', lw=1.0, label='physics only (x_P)')
    ax_r2.plot(date_objs, r2_full, color='tomato', lw=1.0, ls='--', label='physics + residual (x_PB)')
    ax_r2.set_ylabel('R²', fontsize=10)
    ax_r2.set_xlabel('Date', fontsize=11)
    ax_r2.set_title('Per-epoch reconstruction R²', fontsize=11)
    ax_r2.legend(fontsize=8)
    ax_r2.grid(alpha=0.3)

    fig.suptitle('PILA InSAR — reconstruction quality over time', fontsize=13)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}_fit_over_time.{ext}', dpi=150)
    plt.close(fig)
    print(f"  Saved {out_base}_fit_over_time.png / .pdf  (done in {time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description='PILA InSAR full-time-series reconstruction plots (GNSS-analog).')
    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='path to model checkpoint (.pth)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config.json (default: <checkpoint_dir>/config.json)')
    parser.add_argument('--n_map_epochs', default=6, type=int,
                        help='number of epochs to show in the maps-over-time grid (default 6)')
    parser.add_argument('--date', default=None, type=str,
                        help='restrict the maps-over-time grid to a SINGLE epoch: the epoch whose '
                             'acquisition date is nearest this YYYY-MM-DD (one row: actual | recon '
                             '| residual). Overrides --n_map_epochs. Each epoch is the CUMULATIVE '
                             'LOS field at that date relative to the stack reference date.')
    parser.add_argument('--maps_only', action='store_true',
                        help='write ONLY the maps-over-time deliverable (skip pixel-timeseries, '
                             'parameters, scatter and fit-over-time). Useful for a single-date '
                             'presentation figure.')
    parser.add_argument('--gif', action='store_true',
                        help='also write an animated GIF over ALL epochs (needs Pillow)')
    parser.add_argument('--gif_fps', default=4, type=int, help='GIF frames per second')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='figure output directory (default: <experiment_dir>/figures)')
    parser.add_argument('--hs_azimuth', default=315.0, type=float, help='hillshade azimuth (deg)')
    parser.add_argument('--hs_altitude', default=45.0, type=float, help='hillshade altitude (deg)')
    args = parser.parse_args()

    t_start = time.time()
    resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(CURRENT_DIR, args.resume)
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    # --- [1/5] Config ---
    cfg_path = args.config or os.path.join(os.path.dirname(resume_path), 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    print(f"[1/5] Loading config from {cfg_path} ...")
    config = read_json(cfg_path)
    if 'insar' not in config['arch']['args']:
        raise ValueError("This plotter is h5-native: config['arch']['args'] must contain an "
                         "'insar' block (timeseries/geometry/mask/lat0/lon0/multilook).")
    insar_args = config['arch']['args']['insar']
    geometry_h5 = insar_args['geometry']
    multilook = int(insar_args.get('multilook', 20))

    # --- [2/5] Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[2/5] Building model and loading checkpoint on {device} ...")
    model = build_model(config, resume_path, device)

    # --- [3/5] MintPy LOS field (same memoized points the model used) ---
    print("[3/5] Loading multilooked InSAR field ...")
    d = load_insar_mintpy(
        insar_args['timeseries'], insar_args['geometry'], insar_args.get('mask'),
        insar_args['lat0'], insar_args['lon0'],
        multilook=multilook, coh_valid_frac=insar_args.get('coh_valid_frac', 0.5),
        bbox=insar_args.get('bbox'), verbose=False,
        ref_date=insar_args.get('ref_date'))
    n_epoch = len(d.dates)
    print(f"  Loaded: N points={d.n_points}, coarse grid={d.mask_d.shape}, epochs={n_epoch}")

    # --- [4/5] Inference over ALL epochs (one batched pass) ---
    print(f"[4/5] Running inference over all {n_epoch} epochs ...")
    actual_mm, full_mm, phys_mm, params = run_inference_all_epochs(model, d, config, device)
    print(f"  Reconstructed: actual={actual_mm.shape}, x_PB={full_mm.shape}, x_P={phys_mm.shape}")

    # --- [5/5] Plot all three deliverables ---
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(resume_path)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"{config['name']}_timeseries")
    print(f"[5/5] Writing figures to {out_dir} ...")

    date_objs = pd.to_datetime(d.dates)                       # x-axis for time plots
    hs, hs_extent = load_hillshade(geometry_h5, args.hs_azimuth, args.hs_altitude)
    disp_extent = list(d.extent_lonlat)

    # Epochs shown in the maps grid.
    if args.date is not None:
        # Single-date mode: pick the ONE epoch whose acquisition date is nearest --date.
        target = pd.to_datetime(args.date)
        nearest = int(np.argmin(np.abs((date_objs - target).values.astype('timedelta64[D]'))))
        delta_days = abs((date_objs[nearest] - target).days)
        if delta_days > 0:
            print(f"  NOTE: no exact epoch on {args.date}; using nearest epoch "
                  f"{d.dates[nearest]} ({delta_days} day(s) away).")
        else:
            print(f"  Single-date mode: epoch {d.dates[nearest]} (index {nearest}).")
        idxs = np.array([nearest])
    else:
        # Evenly spaced epochs for the maps grid (unique, in order).
        k = max(1, min(args.n_map_epochs, n_epoch))
        idxs = np.unique(np.linspace(0, n_epoch - 1, k).astype(int))

    plot_maps_over_time(actual_mm, full_mm, d, d.dates, idxs, hs, hs_extent, disp_extent, out_base)
    if args.gif:
        animate_maps(full_mm, actual_mm, d, d.dates, hs, hs_extent, disp_extent, out_base,
                     fps=args.gif_fps)
    if not args.maps_only:
        plot_pixel_timeseries(actual_mm, full_mm, phys_mm, d, date_objs, out_base)
        plot_params_and_fit(actual_mm, full_mm, phys_mm, params, date_objs, out_base)

    print(f"Done in {time.time() - t_start:.1f}s. Output saved to {out_dir}/")


if __name__ == '__main__':
    main()
