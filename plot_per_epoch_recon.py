#!/usr/bin/env python
# Usage:       python plot_per_epoch_recon.py \
#                  -r saved/<EXP>/<ts>/models/model_best.pth \
#                  [-c config.json] [--output_dir DIR] [--truth path_truth.json] \
#                  [--spike_frac 0.2] [--quiescent_frac 0.1] [--ncol 8]
# Description: Per-epoch reconstruction diagnostic for a PILA InSAR (Mogi_LOS) checkpoint.
#              For EVERY acquisition epoch it renders the observed input LOS field, the
#              PHYSICS-ONLY reconstruction (x_P, the pure source field driven by the
#              recovered parameters), and the residual (input - x_P). The goal is to check
#              whether the SOURCE (dV) is fitting APS noise at quiescent epochs: if so, a
#              quiescent epoch's x_P shows a spurious Mogi bullseye and the residual carries
#              structured (source-shaped) leftover. Epochs where the recovered dV spikes away
#              from the (near-baseline) truth are flagged and highlighted on every figure.
# Date:        2026-07-30
#
# Scientific conventions (READ — these affect interpretation):
#   * LOS sign: positive = motion TOWARD the satellite (range decrease), MintPy convention.
#     Displacement maps use a diverging colormap (RdBu_r) centred at 0.
#   * The dataset z-scores LOS with a single GLOBAL (scene-wide) scaler. Model outputs
#     (x_P pure physics) and targets are de-standardized to mm with
#     mm = std*x_scale_global + x_mean_global before any plotting / RMSE.
#   * We deliberately plot the PHYSICS-ONLY field x_P (not the physics+residual x_PB): x_PB's
#     low-rank residual term absorbs noise and would mask the diagnostic. x_P is exactly what
#     the recovered source parameters (dV, depth, location) explain, so a dV spike is visible.
#   * dV is displayed in x10^6 m^3. Position params are local-ENU offsets from the lat0/lon0
#     peg (km), NOT the source location.

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # headless (HPC / no display)
import matplotlib.pyplot as plt

from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from utils import read_json
# Reuse the existing single-epoch and time-series tools so all three stay consistent.
from plot_insar_results import (build_model, load_hillshade, points_to_grid, _r2_rmse)
from plot_insar_timeseries import run_inference_all_epochs

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DV_SCALE = 1e-6          # m^3 -> x10^6 m^3 for display
DV_UNIT = '×10⁶ m³'


def try_load_hillshade(geometry_h5, azimuth_deg, altitude_deg):
    """
    Load a DEM hillshade if the geometry file has a 'height' dataset; otherwise return
    (None, None). Synthetic grids (marapi_grid) carry only incidence/azimuth angles and no
    DEM, so the diagnostic must still work without a hillshade backdrop.
    """
    try:
        return load_hillshade(geometry_h5, azimuth_deg, altitude_deg)
    except KeyError:
        print("  No 'height' DEM in geometry file — plotting without a hillshade background.")
        return None, None


# --- Ground-truth dV loading ------------------------------------------------
def load_truth_dv(truth_path, dates):
    """
    Load the per-epoch TRUE dV (m^3) from a synthetic cube's *_truth.json.

    Inputs:
        truth_path : path to <name>_truth.json (or None).
        dates      : list of the loaded epoch date strings, for a length sanity check.
    Output:
        truth_dV_m3 : [n_epoch] float array (m^3), or None if unavailable/mismatched.
    """
    if truth_path is None or not os.path.exists(truth_path):
        return None
    with open(truth_path, 'r') as f:
        truth = json.load(f)
    if 'params_per_epoch' not in truth or 'param_names' not in truth:
        print(f"  Truth file {truth_path} lacks params_per_epoch/param_names; skipping overlay.")
        return None
    if 'dV' not in truth['param_names']:
        print("  Truth file has no 'dV' parameter; skipping overlay.")
        return None
    dv_idx = truth['param_names'].index('dV')
    params_per_epoch = np.asarray(truth['params_per_epoch'], dtype=np.float64)  # [n_epoch, n_par]
    truth_dV_m3 = params_per_epoch[:, dv_idx]                                   # [n_epoch]
    if truth_dV_m3.shape[0] != len(dates):
        # Length mismatch (e.g. ref_date re-referencing dropped an epoch) -> do not risk a
        # silent misalignment between truth and recovered traces.
        print(f"  Truth has {truth_dV_m3.shape[0]} epochs but data has {len(dates)}; "
              f"skipping truth overlay to avoid misalignment.")
        return None
    return truth_dV_m3


# --- Comparison-variant dV series -------------------------------------------
def load_compare_dv_series(compare_args, n_epoch):
    """
    Parse repeated 'LABEL=path/to/uspace_series.csv' entries and return a list of
    (label, dV_m3 [n_epoch]) for overlay on the dV(t) trace. The CSV is a
    dump_uspace_series.py output; its physical 'dV' column is read (m^3). Series with a
    row count that disagrees with the primary run are truncated to the shorter length and
    a warning is printed (so a mismatched dump is never silently misaligned).
    """
    series = []
    if not compare_args:
        return series
    for entry in compare_args:
        if '=' not in entry:
            raise ValueError(f"--compare entry must be LABEL=path.csv, got: {entry!r}")
        label, path = entry.split('=', 1)
        label, path = label.strip(), path.strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"comparison series not found for '{label}': {path}")
        df = pd.read_csv(path)
        if 'dV' not in df.columns:
            raise KeyError(f"comparison CSV for '{label}' has no 'dV' column: {path}")
        dV = df['dV'].values.astype(np.float64)
        if dV.shape[0] != n_epoch:
            n = min(dV.shape[0], n_epoch)
            print(f"  WARNING [{label}]: {dV.shape[0]} rows vs {n_epoch} epochs; using first {n}.")
            dV = np.concatenate([dV[:n], np.full(n_epoch - n, np.nan)])[:n_epoch]
        series.append((label, dV))
        print(f"  Comparison series '{label}' loaded from {path}")
    return series


# --- Spike detection --------------------------------------------------------
def detect_spike_epochs(rec_dV_m3, truth_dV_m3, spike_frac, quiescent_frac,
                        flag_all_misfit=False):
    """
    Flag epochs where the recovered dV spikes even though the TRUTH says the source is at rest.

    Definition (truth available):
      max_abs = max|truth_dV|
      quiescent : |truth_dV| < quiescent_frac * max_abs   (source ~ at rest / "dV supposed 0")
      spike     : quiescent AND |rec_dV| > spike_frac*max_abs   -> a source bullseye appears
                                                                   where there should be none.
      (--flag_all_misfit additionally flags ANY epoch with
       |rec_dV - truth_dV| > spike_frac*max_abs, incl. noisy but genuinely-active epochs.)
    Fallback (no truth): robust outlier flag |rec_dV - median| > 3 * (1.4826 * MAD).

    Threshold rationale / ASSUMPTION: both thresholds are fractions of the GLOBAL peak
    max|truth_dV| (matching the repo's "SNR relative to global peak" framing). Defaults 0.1/0.2
    are heuristics — on the synthetic rising cube the recovered-dV noise floor at quiescent
    epochs is ~10-20% of the peak, so 0.2 catches the clear spurious spikes without flagging
    baseline wander. CAVEAT: because thresholds scale with the global peak, the tool is less
    sensitive to spikes riding on a fast-rising tail (moving-source configs) than on a
    static-then-active step; tighten spike_frac there. Tune via --spike_frac/--quiescent_frac.

    Returns (spike_idxs sorted list, quiescent_mask or None).
    """
    rec_dV_m3 = np.asarray(rec_dV_m3, dtype=np.float64)
    if truth_dV_m3 is not None:
        max_abs = float(np.max(np.abs(truth_dV_m3)))
        if max_abs < 1e-30:
            max_abs = float(np.max(np.abs(rec_dV_m3))) or 1.0
        quiescent_mask = np.abs(truth_dV_m3) < quiescent_frac * max_abs
        # Primary target: quiescent epochs whose recovered dV nevertheless spikes.
        spike_mask = quiescent_mask & (np.abs(rec_dV_m3) > spike_frac * max_abs)
        if flag_all_misfit:
            spike_mask = spike_mask | (np.abs(rec_dV_m3 - truth_dV_m3) > spike_frac * max_abs)
    else:
        med = float(np.median(rec_dV_m3))
        mad = 1.4826 * float(np.median(np.abs(rec_dV_m3 - med)))
        if mad < 1e-30:
            mad = float(np.std(rec_dV_m3)) or 1.0
        spike_mask = np.abs(rec_dV_m3 - med) > 3.0 * mad
        quiescent_mask = None
    spike_idxs = sorted(np.where(spike_mask)[0].tolist())
    return spike_idxs, quiescent_mask


# --- All-epochs montage -----------------------------------------------------
def plot_montage(field_mm, d, dates, rec_dV_m3, truth_dV_m3, spike_set, vmax, cbar_label,
                 suptitle, out_path, ncol, title_fontsize=9):
    """
    One figure = ALL epochs as a grid of small LOS maps on a shared symmetric colour scale.

    Inputs:
        field_mm    : [n_epoch, N] per-point values (mm) to render (input / x_P / residual).
        d           : InSARData (provides mask_d for scatter-to-grid).
        dates       : [n_epoch] date strings (cell titles).
        rec_dV_m3   : [n_epoch] MODELLED (recovered) dV (m^3), annotated per cell.
        truth_dV_m3 : [n_epoch] TRUE dV (m^3) or None; annotated alongside the modelled dV.
        spike_set   : set of flagged epoch indices (red border).
        vmax        : symmetric colour limit (mm).
        cbar_label  : colorbar label (mm).
        suptitle    : figure title.
        out_path    : output path stem (".png"/".pdf" appended).
        ncol        : number of columns in the grid.
        title_fontsize : font size for the per-cell titles (date + dV values).
    """
    n_epoch = field_mm.shape[0]
    nrow = int(np.ceil(n_epoch / ncol))
    fig_w_in = 2.0 * ncol
    fig_h_in = 2.5 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w_in, fig_h_in), squeeze=False)
    im = None
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= n_epoch:
            ax.axis('off')
            continue
        grid = points_to_grid(field_mm[k], d.mask_d)
        im = ax.imshow(np.ma.masked_invalid(grid), cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, origin='upper')
        ax.set_xticks([])
        ax.set_yticks([])
        # Per-cell title: date, then MODELLED (recovered) and TRUE dV in x10^6 m^3.
        if truth_dV_m3 is not None:
            cell_title = (f'{dates[k]}\n'
                          f'mod={rec_dV_m3[k] * DV_SCALE:.1f}  true={truth_dV_m3[k] * DV_SCALE:.1f}')
        else:
            cell_title = f'{dates[k]}\nmod dV={rec_dV_m3[k] * DV_SCALE:.1f}'
        ax.set_title(cell_title, fontsize=title_fontsize)
        # Red border on spike epochs so they stand out at a glance.
        if k in spike_set:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2.5)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor('0.7')
                spine.set_linewidth(0.5)

    # Reserve absolute-inch top margin for the two-line suptitle (pattern from
    # synthetic/plots.py::plot_timeseries_montage).
    top_frac = 1.0 - 0.9 / fig_h_in
    # Extra hspace so the larger two-line (date + mod/true dV) titles clear the cell above.
    fig.subplots_adjust(top=top_frac, bottom=0.02, left=0.02, right=0.9, hspace=0.6, wspace=0.05)
    fig.suptitle(suptitle, y=1.0 - 0.15 / fig_h_in, va='top', fontsize=15)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_path}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}.png / .pdf")


# --- dV(t) trace + residual RMS ---------------------------------------------
# Distinct colours for comparison variants (primary is steelblue, truth is black).
COMPARE_COLORS = ['0.45', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']


def _quiescent_dv_rmse(dV_m3, truth_dV_m3, quiescent_mask):
    """Quiescent-window dV RMSE (×10⁶ m³): sqrt(mean((rec-truth)^2)) over quiescent epochs.
    This is the "how much does it spike where the source is at rest" score — lower is better."""
    if truth_dV_m3 is None or quiescent_mask is None or not quiescent_mask.any():
        return None
    err = (dV_m3[quiescent_mask] - truth_dV_m3[quiescent_mask]) * DV_SCALE
    return float(np.sqrt(np.mean(err ** 2)))


def plot_dv_trace(rec_dV_m3, truth_dV_m3, date_objs, spike_idxs, quiescent_mask,
                  primary_label, compare_series, out_path, dv_ylim=None):
    """
    Recovered dV(t) vs truth dV(t) for the PRIMARY run, overlaid with comparison variants
    (e.g. no-loss baseline, curvature-only, dV-L1-only) so it is visible that the combined
    prior spikes least where the true source is at rest.

    Inputs:
        rec_dV_m3      : [n_epoch] primary run recovered dV (m^3), from live inference.
        truth_dV_m3    : [n_epoch] true dV (m^3) or None.
        date_objs      : [n_epoch] datetimes (x-axis).
        spike_idxs     : flagged spike epochs of the primary run (red markers).
        quiescent_mask : [n_epoch] bool (true dV ≈ baseline) or None; shaded grey.
        primary_label  : legend label for the primary (current/best) run.
        compare_series : list of (label, dV_m3 [n_epoch]) for other runs (from their CSVs).
        out_path       : output stem.

    A text box reports the quiescent-window dV RMSE for every series; the smallest is the
    least noise-fitting and is tagged "best".
    """
    fig, ax_dv = plt.subplots(figsize=(13, 5.5))
    # Grey band over the quiescent epochs = where the true source is at rest.
    if quiescent_mask is not None:
        quiescent_shaded = False
        for e in np.where(quiescent_mask)[0]:
            ax_dv.axvspan(date_objs[max(e - 1, 0)], date_objs[min(e + 1, len(date_objs) - 1)],
                          color='0.6', alpha=0.15, zorder=0,
                          label=('quiescent (true dV≈baseline)' if not quiescent_shaded else None))
            quiescent_shaded = True
    if truth_dV_m3 is not None:
        ax_dv.plot(date_objs, truth_dV_m3 * DV_SCALE, color='black', lw=2.2, ls='--',
                   label='true dV', zorder=6)
    # Comparison variants first (thinner, behind the primary).
    for k, (label, dV_m3) in enumerate(compare_series):
        ax_dv.plot(date_objs, dV_m3 * DV_SCALE, lw=1.1, alpha=0.8, marker='.', ms=2,
                   color=COMPARE_COLORS[k % len(COMPARE_COLORS)], label=label, zorder=3)
    # Primary run on top, bold.
    ax_dv.plot(date_objs, rec_dV_m3 * DV_SCALE, color='steelblue', lw=2.2,
               marker='o', ms=3, label=primary_label, zorder=5)
    if spike_idxs:
        ax_dv.scatter([date_objs[e] for e in spike_idxs],
                      [rec_dV_m3[e] * DV_SCALE for e in spike_idxs],
                      color='red', s=40, zorder=7,
                      label=f'{primary_label} spike ({len(spike_idxs)})')
    ax_dv.set_xlabel('Date')
    ax_dv.set_ylabel(f'Volume change dV ({DV_UNIT})')
    ax_dv.grid(alpha=0.3)
    ax_dv.legend(loc='upper left', fontsize=9, ncol=2)

    # Lock the y-axis to caller-supplied limits (for a shared scale across runs) or report
    # the autoscaled limits so they can be reused on another figure. matplotlib only resolves
    # the auto limits once the artists are drawn, so trigger autoscale explicitly before read.
    if dv_ylim is not None:
        ax_dv.set_ylim(float(dv_ylim[0]), float(dv_ylim[1]))
    else:
        ax_dv.autoscale_view()
    _ylo, _yhi = ax_dv.get_ylim()
    print(f"  dV trace y-limits ({DV_UNIT}): [{_ylo:.2f}, {_yhi:.2f}]"
          + ("  (fixed via --dv_ylim)" if dv_ylim is not None else "  (auto)"))

    # --- Quiescent-window RMSE box (lower right): quantifies "spikes least where at rest" ---
    scored = []
    r_primary = _quiescent_dv_rmse(rec_dV_m3, truth_dV_m3, quiescent_mask)
    if r_primary is not None:
        scored.append((primary_label, r_primary))
        for label, dV_m3 in compare_series:
            r = _quiescent_dv_rmse(dV_m3, truth_dV_m3, quiescent_mask)
            if r is not None:
                scored.append((label, r))
        best = min(scored, key=lambda t: t[1])[0]
        lines = ['quiescent dV RMSE (↓ better):']
        for label, r in sorted(scored, key=lambda t: t[1]):
            tag = '  ← best' if label == best else ''
            lines.append(f'  {label}: {r:.1f}{tag}')
        ax_dv.text(0.985, 0.03, '\n'.join(lines), transform=ax_dv.transAxes,
                   ha='right', va='bottom', fontsize=8.5, family='monospace',
                   bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))

    fig.suptitle('PILA InSAR — recovered dV(t) vs truth: prior-variant comparison\n'
                 'grey = quiescent (true dV≈baseline); the least-spiking variant tracks the '
                 'flat truth there', fontsize=12)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_path}.{ext}', dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}.png / .pdf")


# --- Detailed spike triptych ------------------------------------------------
def plot_spike_triptych(actual_mm, phys_mm, d, dates, rec_dV_m3, truth_dV_m3, spike_idxs,
                        hs, hs_extent, disp_extent, out_path, local_km=False):
    """
    Zoom-in for the flagged spike epochs only: rows = spike epochs, columns =
        [Input LOS | Physics-only (x_P) | Residual (input - x_P)],
    over a hillshade background on ONE shared colour scale so a small residual reads faint.
    local_km: True for local-frame npz scenes (extent in km, not lon/lat degrees).
    """
    x_label = 'East (km)' if local_km else 'Longitude (deg)'
    y_label = 'North (km)' if local_km else 'Latitude (deg)'
    if not spike_idxs:
        print("  No spike epochs flagged — skipping spike triptych.")
        return
    idxs = spike_idxs
    # Shared scale across input and x_P (residual read on the same magnitude as the signal).
    disp_vals = np.concatenate([actual_mm[idxs].ravel(), phys_mm[idxs].ravel()])
    vmax = float(np.nanmax(np.abs(disp_vals)))

    n_rows = len(idxs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows), squeeze=False)
    im = None
    col_titles = ['Input LOS', 'Physics only (x_P)', 'Residual (input − x_P)']
    for row, epoch_idx in enumerate(idxs):
        actual_grid = points_to_grid(actual_mm[epoch_idx], d.mask_d)
        phys_grid = points_to_grid(phys_mm[epoch_idx], d.mask_d)
        resid_grid = actual_grid - phys_grid
        grids = [actual_grid, phys_grid, resid_grid]
        r2, rmse = _r2_rmse(phys_mm[epoch_idx], actual_mm[epoch_idx])
        for col, grid in enumerate(grids):
            ax = axes[row][col]
            if hs is not None:
                ax.imshow(hs, extent=hs_extent, cmap='gray', origin='upper',
                          vmin=0.0, vmax=1.0, zorder=0)
            im = ax.imshow(np.ma.masked_invalid(grid), extent=disp_extent, origin='upper',
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.8, zorder=1)
            ax.set_xlim(disp_extent[0], disp_extent[1])
            ax.set_ylim(disp_extent[2], disp_extent[3])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)
            if col == 0:
                truth_txt = (f', true={truth_dV_m3[epoch_idx] * DV_SCALE:.1f}'
                             if truth_dV_m3 is not None else '')
                ax.set_ylabel(f'{dates[epoch_idx]}\n'
                              f'dV={rec_dV_m3[epoch_idx] * DV_SCALE:.1f}{truth_txt} {DV_UNIT}\n'
                              f'x_P R²={r2:.2f}, RMSE={rmse:.1f} mm\n{y_label}', fontsize=8)
            else:
                ax.set_ylabel(y_label, fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, location='right', pad=0.02)
    cbar.set_label('LOS displacement (mm)')
    fig.suptitle('PILA InSAR — flagged dV-spike epochs: input / physics-only / residual\n'
                 f'{n_rows} epoch(s); a spurious x_P bullseye + structured residual '
                 f'indicates the source is fitting noise',
                 fontsize=13, y=1.0)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_path}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}.png / .pdf")


def main():
    parser = argparse.ArgumentParser(
        description='PILA InSAR per-epoch reconstruction diagnostic (input / physics-only / '
                    'residual), with dV-spike flagging.')
    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='path to model checkpoint (.pth)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config.json (default: <checkpoint_dir>/config.json)')
    parser.add_argument('--truth', default=None, type=str,
                        help='synthetic ground-truth json (default: auto-detect '
                             '<cube_dir>/<cube_stem>_truth.json)')
    parser.add_argument('--spike_frac', default=0.2, type=float,
                        help='|dV| deviation (fraction of max|truth dV|) to flag a spike (0.2)')
    parser.add_argument('--quiescent_frac', default=0.1, type=float,
                        help='|truth dV| below this fraction of max|truth dV| = quiescent (0.1)')
    parser.add_argument('--flag_all_misfit', action='store_true',
                        help='also flag ANY epoch with |rec_dV - truth_dV| > spike_frac*max|truth| '
                             '(includes noisy but genuinely-active epochs, not just quiescent)')
    parser.add_argument('--label', default=None, type=str,
                        help='legend label for the primary (current) run on the dV trace '
                             '(default: the config name)')
    parser.add_argument('--compare', nargs='*', default=None,
                        help="overlay other runs' dV(t) on the dV-trace figure. Each entry is "
                             "LABEL=path/to/uspace_series.csv (dump_uspace_series.py output). "
                             "e.g. --compare 'no prior=saved/baseline_.../run/uspace_series.csv' "
                             "'curv only=saved/tscurv_.../lam15000/uspace_series.csv'")
    parser.add_argument('--ncol', default=8, type=int,
                        help='columns in the all-epochs montages (default 8)')
    parser.add_argument('--vmax_pct', default=99.0, type=float,
                        help='percentile of |LOS| used for the montage colour limit, so one '
                             'extreme-amplitude epoch does not wash out the rest (default 99; '
                             'use 100 for the true max). A few peaked cells may saturate.')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='figure output directory (default: <experiment_dir>/figures)')
    parser.add_argument('--dv_ylim', nargs=2, type=float, default=None,
                        metavar=('YMIN', 'YMAX'),
                        help='fix the dV(t) trace y-axis limits (in 1e6 m^3) instead of '
                             'autoscaling, so this figure can be placed on the SAME vertical '
                             'scale as another run for side-by-side comparison. The script '
                             'prints the auto y-limits when this is omitted, so you can read '
                             'them off one run and pass them to another.')
    parser.add_argument('--hs_azimuth', default=315.0, type=float, help='hillshade azimuth (deg)')
    parser.add_argument('--hs_altitude', default=45.0, type=float, help='hillshade altitude (deg)')
    parser.add_argument('--seed', default=0, type=int,
                        help='RNG seed for reproducible inference. If KL is enabled the latents '
                             'are drawn stochastically, so the flagged-spike list would otherwise '
                             'vary slightly run to run (default 0).')
    args = parser.parse_args()

    t_start = time.time()
    # Deterministic inference so the flagged-spike list is reproducible (config has KL on ->
    # model.draw() samples the latents).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(CURRENT_DIR, args.resume)
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    # --- [1/6] Config ---
    cfg_path = args.config or os.path.join(os.path.dirname(resume_path), 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    print(f"[1/6] Loading config from {cfg_path} ...")
    config = read_json(cfg_path)
    if 'insar' not in config['arch']['args']:
        raise ValueError("This diagnostic is h5-native: config['arch']['args'] must contain an "
                         "'insar' block (timeseries/geometry/mask/lat0/lon0/multilook).")
    insar_args = config['arch']['args']['insar']
    geometry_h5 = insar_args['geometry']
    multilook = int(insar_args.get('multilook', 20))

    # --- [2/6] Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[2/6] Building model and loading checkpoint on {device} ...")
    model = build_model(config, resume_path, device)

    # --- [3/6] MintPy LOS field (same memoized points the model used) ---
    print("[3/6] Loading multilooked InSAR field ...")
    d = load_insar_mintpy(
        insar_args['timeseries'], insar_args['geometry'], insar_args.get('mask'),
        insar_args['lat0'], insar_args['lon0'],
        multilook=multilook, coh_valid_frac=insar_args.get('coh_valid_frac', 0.5),
        bbox=insar_args.get('bbox'), verbose=False,
        ref_date=insar_args.get('ref_date'))
    n_epoch = len(d.dates)
    print(f"  Loaded: N points={d.n_points}, coarse grid={d.mask_d.shape}, epochs={n_epoch}")

    # --- [4/6] Inference over ALL epochs (one batched pass) ---
    print(f"[4/6] Running inference over all {n_epoch} epochs ...")
    actual_mm, full_mm, phys_mm, params = run_inference_all_epochs(model, d, config, device)
    print(f"  Reconstructed: actual={actual_mm.shape}, x_PB={full_mm.shape}, x_P={phys_mm.shape}")
    # Diagnostic target is the PHYSICS-ONLY field and its residual.
    residual_mm = actual_mm - phys_mm                                        # [n_epoch, N]
    if 'dV' not in params:
        raise KeyError("No 'dV' in recovered params — is this a Mogi/Sun69 (dV-bearing) model?")
    rec_dV_m3 = np.asarray(params['dV'], dtype=np.float64)                    # [n_epoch]

    # --- [5/6] Ground truth + spike detection ---
    print("[5/6] Loading truth dV and detecting spike epochs ...")
    truth_path = args.truth
    if truth_path is None:
        ts_h5 = insar_args['timeseries']
        stem = os.path.splitext(os.path.basename(ts_h5))[0]
        cand = os.path.join(os.path.dirname(ts_h5), f'{stem}_truth.json')
        truth_path = cand if os.path.exists(cand) else None
    truth_dV_m3 = load_truth_dv(truth_path, d.dates)
    if truth_dV_m3 is not None:
        print(f"  Truth dV loaded from {truth_path}")
    else:
        print("  No usable truth dV; spike detection falls back to robust (MAD) outliers.")
    spike_idxs, quiescent_mask = detect_spike_epochs(
        rec_dV_m3, truth_dV_m3, args.spike_frac, args.quiescent_frac,
        flag_all_misfit=args.flag_all_misfit)
    spike_set = set(spike_idxs)
    print(f"  Flagged {len(spike_idxs)} spike epoch(s):")
    for e in spike_idxs:
        truth_txt = (f', true={truth_dV_m3[e] * DV_SCALE:.2f}' if truth_dV_m3 is not None else '')
        print(f"    epoch {e:3d}  {d.dates[e]}  rec_dV={rec_dV_m3[e] * DV_SCALE:.2f}{truth_txt} {DV_UNIT}")

    # --- [6/6] Figures ---
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(resume_path)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{config['name']}_perepoch")
    print(f"[6/6] Writing figures to {out_dir} ...")

    date_objs = pd.to_datetime(d.dates)
    # Local-frame npz scenes (e.g. COMSOL): 'geometry' is a track token and the extent is km,
    # so there is no DEM to hillshade and the axes are East/North (km), not lon/lat.
    is_npz_scene = str(insar_args['timeseries']).lower().endswith('.npz')
    if is_npz_scene:
        hs, hs_extent = None, None
    else:
        hs, hs_extent = try_load_hillshade(geometry_h5, args.hs_azimuth, args.hs_altitude)
    disp_extent = list(d.extent_lonlat)

    # Shared symmetric scale for the signal montages (input and physics-only), so the two are
    # directly comparable. The residual montage gets its OWN (smaller) scale so structured
    # leftover is visible rather than washed out by the full-signal scale. Both use a robust
    # percentile (--vmax_pct) rather than the raw max, so a single extreme-amplitude epoch
    # (e.g. a shallow-depth noise-fit) does not saturate the whole montage to near-white.
    # TRADEOFF: a spike epoch's peak may itself sit above the percentile and be colour-clipped
    # in the montage — the montage is an OVERVIEW (use the red border + dV annotation to spot
    # spikes); the spike_triptych below auto-scales to the flagged epochs for an undistorted view.
    vmax_signal = float(np.nanpercentile(
        np.abs(np.concatenate([actual_mm.ravel(), phys_mm.ravel()])), args.vmax_pct))
    vmax_resid = float(np.nanpercentile(np.abs(residual_mm), args.vmax_pct))

    # Comparison variants' dV(t) for the trace overlay (from their dumped uspace_series.csv).
    compare_series = load_compare_dv_series(args.compare, n_epoch)

    t_maps = time.time()
    dv_note = 'per-cell: mod/true dV in ×10⁶ m³; red border = flagged dV spike'
    print("[fig 1/4] input LOS montage (all epochs) ...")
    plot_montage(actual_mm, d, d.dates, rec_dV_m3, truth_dV_m3, spike_set, vmax_signal,
                 'LOS displacement (mm)',
                 f'PILA InSAR — INPUT LOS, all epochs\n({dv_note})',
                 f'{base}_montage_input', args.ncol)
    print("[fig 2/4] physics-only (x_P) montage (all epochs) ...")
    plot_montage(phys_mm, d, d.dates, rec_dV_m3, truth_dV_m3, spike_set, vmax_signal,
                 'LOS displacement (mm)',
                 f'PILA InSAR — PHYSICS-ONLY reconstruction x_P, all epochs\n({dv_note})',
                 f'{base}_montage_physonly', args.ncol)
    print("[fig 3/4] residual (input − x_P) montage (all epochs) ...")
    plot_montage(residual_mm, d, d.dates, rec_dV_m3, truth_dV_m3, spike_set, vmax_resid,
                 'Residual LOS (mm)',
                 f'PILA InSAR — RESIDUAL (input − x_P), all epochs '
                 f'(own scale ±{vmax_resid:.0f} mm)\n({dv_note})',
                 f'{base}_montage_residual', args.ncol)
    print(f"  montages done in {time.time() - t_maps:.1f}s")

    print("[fig 4/4] dV(t) trace + spike triptych ...")
    plot_dv_trace(rec_dV_m3, truth_dV_m3, date_objs, spike_idxs, quiescent_mask,
                  args.label or config['name'], compare_series, f'{base}_dv_trace',
                  dv_ylim=args.dv_ylim)
    plot_spike_triptych(actual_mm, phys_mm, d, d.dates, rec_dV_m3, truth_dV_m3, spike_idxs,
                        hs, hs_extent, disp_extent, f'{base}_spike_triptych',
                        local_km=is_npz_scene)

    print(f"Done in {time.time() - t_start:.1f}s. Output saved to {out_dir}/")


if __name__ == '__main__':
    main()
