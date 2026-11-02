#!/usr/bin/env python
# Usage:       python plot_multisource_report.py \
#                  --album-root saved/multisource_seq/clean/K2_separated_tv \
#                  [--truth synthetic/multisource/clean_scenarios/K2_separated_tv_truth.json] \
#                  [--fit-k 2 3] [--out-dir figures/multisource_clean_pila/K2_separated_tv]
#              python plot_multisource_report.py --summarise --summary-root figures/multisource_clean_pila
# Description: PILA-vs-truth report for the sequential multi-source runs on NOISE-FREE clean scenes
#              (vbICA-report style: truth black, PILA orange). Per fitted K it writes
#                1. <album>_K<k>_maps_truth_vs_recon.png   truth | physics recon | physics+residual | difference
#                2. <album>_K<k>_maps_per_source.png       each recovered source vs its truth source
#                3. <album>_K<k>_param_recovery.png        xcen, ycen, depth, dV vs epoch vs truth
#                4. <album>_K<k>_combined_timeseries.png   combined LOS vs truth through time
#                and <album>_summary.csv (one row per truth source per K). --summarise builds the
#                cross-album summary_matrix.png + summary_all.csv from all those CSVs.
# Assumptions: Units km, m^3, LOS mm (+ toward satellite). Truth = noise-free Mogi fields scaled per epoch
#              by truth['dV_t_m3'] (static truths use the constant dV_m3). NO offset/ramp removal (clean
#              scenes are noise-free and PILA fits no ramp). "Recon" headline = PHYSICS-ONLY sum of source
#              fields (xP_src_mm); physics + low-rank residual (xPB_mm) is shown separately because the
#              residual can absorb signal. Matching is PER EPOCH on field-space L2 cost, only on epochs
#              where the truth source |dV| > 25% of its own peak (elsewhere location/depth are
#              unidentifiable); slot order is arbitrary, so swaps are reported.
# Date:        2026-10-08

import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment

from plot_sequential_album import load_fits
from synthetic.multisource.make_pila_stack import load_ta69_geometry
from synthetic.multisource.torch_generator import mogi_los_mm_torch

MATCH_DV_FRACTION = 0.25     # epochs used for matching/metrics: truth |dV| > 25% of the source's peak
MASK_DV_FRACTION = 0.05      # vbICA-report convention: hide recovered geometry where truth |dV| < 5% of peak
DEPTH_BOUNDS_KM = (0.5, 12.0)   # configs/mogi_paras_symdV.json depth prior; "railed" if within 0.1 km
RAIL_TOL_KM = 0.1
HERE = os.path.dirname(os.path.abspath(__file__))     # anchor default paths so the script runs from any cwd
ACTIVE_PEAK_LOS_MM = 1.0     # a slot counts as an ACTIVE source only if its own peak |LOS| exceeds this (absolute; K0 false positives)
PASS_LOC_KM, PASS_DEPTH_PCT, PASS_DV_PCT, PASS_R2 = 1.0, 20.0, 20.0, 0.95   # proposed noise-free success criteria
COLORBAR_LABEL = 'LOS displacement (mm, + toward satellite)'
PARAM_NAMES = ['xcen (km)', 'ycen (km)', 'depth (km)', 'dV (m³)']
TRUTH_COLOR, PILA_COLOR, EXTRA_COLOR = 'k', 'tab:orange', '0.6'


# --- Truth rendering --------------------------------------------------------------------------
def truth_unit_fields_mm_per_m3(truth, geometry):
    """Unit LOS field [n_true, H, W] (mm per m^3 of dV) for each truth source on the TA69 grid."""
    grid = geometry['grid']
    xE = torch.tensor(grid['xE_m']); yN = torch.tensor(grid['yN_m'])
    los = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]))
    fields = []
    for source in truth['sources']:
        unit_field = mogi_los_mm_torch(
            xE, yN, los, torch.tensor([[source['xcen_km'] * 1e3]], dtype=torch.float64),
            torch.tensor([[source['ycen_km'] * 1e3]], dtype=torch.float64),
            torch.tensor([[source['depth_km'] * 1e3]], dtype=torch.float64),
            torch.tensor([[1.0]], dtype=torch.float64))[0, 0].numpy()
        fields.append(unit_field.reshape(grid['shape']))
    return np.stack(fields) if fields else np.zeros((0,) + tuple(grid['shape']))


def truth_dV_series_m3(truth, n_epoch):
    """dV(t) [n_true, T] in m^3; static truths (no dV_t_m3 key) use the constant dV_m3."""
    if truth.get('dV_t_m3'):
        series = np.array(truth['dV_t_m3'], dtype=float)
    else:
        series = np.array([[s['dV_m3']] * n_epoch for s in truth['sources']], dtype=float)
    if series.size and series.shape[1] != n_epoch:
        raise ValueError(f"truth has {series.shape[1]} epochs, fit has {n_epoch}")
    return series.reshape(len(truth['sources']), n_epoch)


def truth_fields_mm(unit_fields, dV_series):
    """Per-source truth fields [n_true, T, H, W] (mm) = unit field x dV(t)."""
    return unit_fields[:, None] * dV_series[:, :, None, None]


# --- Matching ---------------------------------------------------------------------------------
def match_per_epoch(truth_src_fields_mm, dV_series, rec_src_fields_mm, mask):
    """
    Per-epoch Hungarian matching in FIELD space. Only truth sources with |dV(t)| > 25% of their peak
    take part at epoch t. Returns assign [n_true, T] (recovered slot index, -1 = unmatched / not valid),
    valid [n_true, T] (bool), modal [n_true] (most frequent slot over valid epochs, -1 if none),
    assign_raw [n_true, T] (UNFILLED: -1 wherever the source was not actually matched at that epoch; all
    metrics must use valid & (assign_raw >= 0), the filled `assign` is only for drawing positions),
    cost_rms_mm [n_true, T] (field RMS of the chosen pair; NaN where not matched).
    """
    n_true, n_epoch = dV_series.shape
    n_rec = rec_src_fields_mm.shape[1]
    peak = np.abs(dV_series).max(axis=1, keepdims=True) if n_true else np.zeros((0, 1))
    valid = np.abs(dV_series) > MATCH_DV_FRACTION * np.maximum(peak, 1e-30)
    assign = -np.ones((n_true, n_epoch), dtype=int)
    cost_rms_mm = np.full((n_true, n_epoch), np.nan)
    for epoch in range(n_epoch):
        true_indices = np.where(valid[:, epoch])[0]
        if len(true_indices) == 0 or n_rec == 0:
            continue
        # RMS field difference over coherent pixels between every (truth j, recovered k) pair
        diff = (truth_src_fields_mm[true_indices, epoch][:, None, :, :]
                - rec_src_fields_mm[epoch][None, :, :, :])
        cost = np.sqrt(np.nanmean(np.where(mask, diff, np.nan) ** 2, axis=(2, 3)))
        cost = np.nan_to_num(cost, nan=1e9)      # empty mask -> NaN cost; keep the solver defined
        row_indices, col_indices = linear_sum_assignment(cost)
        assign[true_indices[row_indices], epoch] = col_indices
        cost_rms_mm[true_indices[row_indices], epoch] = cost[row_indices, col_indices]
    assign_raw = assign.copy()
    modal = -np.ones(n_true, dtype=int)
    for j in range(n_true):
        slots = assign[j][valid[j] & (assign[j] >= 0)]
        if len(slots):
            modal[j] = np.bincount(slots).argmax()
    # Fill invalid epochs with the modal slot so maps/params stay defined (params are masked in plots)
    for j in range(n_true):
        assign[j, assign[j] < 0] = modal[j]
    return assign, valid, modal, assign_raw, cost_rms_mm


# --- Metrics ----------------------------------------------------------------------------------
def source_metrics(truth, dV_series, params_phys, assign_raw, valid, modal, cost_rms_mm):
    """Per truth source (valid epochs where it was truly matched): median/p90 loc (km), depth (%), dV (%) errors,
    sign-flip, swap and rail FRACTIONS, unmatched-epoch fraction, median match cost (mm)."""
    rows = []
    for j, source in enumerate(truth['sources']):
        row = dict(source=j + 1, truth_depth_km=source['depth_km'])
        use = valid[j] & (assign_raw[j] >= 0)
        row['unmatched_epoch_frac'] = float(1 - use.sum() / max(valid[j].sum(), 1))
        if not use.any():
            rows.append({**row, 'matched': False}); continue
        epochs = np.where(use)[0]
        rec = params_phys[epochs, assign_raw[j, epochs], :]
        true_dV = dV_series[j, epochs]
        row.update(
            matched=True, n_valid_epochs=int(len(epochs)),
            loc_err_km=float(np.median(np.hypot(rec[:, 0] - source['xcen_km'], rec[:, 1] - source['ycen_km']))),
            loc_err_p90_km=float(np.percentile(np.hypot(rec[:, 0] - source['xcen_km'], rec[:, 1] - source['ycen_km']), 90)),
            depth_err_pct=float(np.median(np.abs(rec[:, 2] - source['depth_km']) / source['depth_km'] * 100)),
            depth_err_p90_pct=float(np.percentile(np.abs(rec[:, 2] - source['depth_km']) / source['depth_km'] * 100, 90)),
            dV_err_pct=float(np.median(np.abs(rec[:, 3] - true_dV) / np.abs(true_dV) * 100)),
            dV_err_p90_pct=float(np.percentile(np.abs(rec[:, 3] - true_dV) / np.abs(true_dV) * 100, 90)),
            match_cost_median_mm=float(np.nanmedian(cost_rms_mm[j, epochs])),
            sign_flip_frac=float(np.mean(np.sign(rec[:, 3]) != np.sign(true_dV))),
            swap_frac=float(np.mean(assign_raw[j, epochs] != modal[j])),
            rail_frac=float(np.mean((rec[:, 2] <= DEPTH_BOUNDS_KM[0] + RAIL_TOL_KM) | (rec[:, 2] >= DEPTH_BOUNDS_KM[1] - RAIL_TOL_KM))))
        rows.append(row)
    return rows


def physics_r2(truth_total_mm, rec_total_mm, mask, active_only=False):
    """R² (about ZERO, no mean removal) of physics recon vs truth over coherent pixels. active_only=True keeps only
    epochs whose truth peak |LOS| > 5% of the series peak (stricter; not dominated by quiescent epochs)."""
    if active_only:
        peak_per_epoch = np.abs(truth_total_mm).reshape(truth_total_mm.shape[0], -1).max(axis=1)
        keep = peak_per_epoch > 0.05 * max(peak_per_epoch.max(), 1e-12)
        truth_total_mm, rec_total_mm = truth_total_mm[keep], rec_total_mm[keep]
        if len(truth_total_mm) == 0:
            return float('nan')
    sum_squares_total = np.nansum(np.where(mask, truth_total_mm, np.nan) ** 2)
    if sum_squares_total == 0:
        return float('nan')
    return float(1 - np.nansum(np.where(mask, truth_total_mm - rec_total_mm, np.nan) ** 2) / sum_squares_total)


# --- Plot helpers -----------------------------------------------------------------------------
def pick_epochs(truth_total_mm, dV_series, n_show=6):
    """Epochs to show: each source's peak-|dV| epoch plus evenly spaced fillers (greedy farthest-point)."""
    n_epoch = truth_total_mm.shape[0]
    chosen = sorted({int(np.abs(series).argmax()) for series in dV_series})[:n_show]
    for candidate in np.linspace(0, n_epoch - 1, n_show).round().astype(int):
        if len(chosen) >= n_show:
            break
        if candidate not in chosen:
            chosen.append(int(candidate))
    return sorted(chosen)


def show_map(axis, field_mm, extent, vmax_mm, title):
    """One map panel on the shared symmetric RdBu_r scale; returns the image for the colourbar."""
    image = axis.imshow(field_mm, cmap='RdBu_r', vmin=-vmax_mm, vmax=vmax_mm, extent=extent, origin='upper')
    axis.set_title(title, fontsize=7); axis.tick_params(labelsize=6)
    return image


def mark_centres(axis, truth_sources, rec_xy_list):
    """Truth centres '+' (black) and recovered centres 'o' (orange)."""
    for source in truth_sources:
        axis.plot(source['xcen_km'], source['ycen_km'], '+', color=TRUTH_COLOR, ms=9, mew=1.5)
    for rec_x_km, rec_y_km in rec_xy_list:
        axis.plot(rec_x_km, rec_y_km, 'o', mfc='none', mec=PILA_COLOR, ms=8, mew=1.5)


def rms_mm(array_mm, mask):
    """RMS over coherent pixels (mm)."""
    return float(np.sqrt(np.nanmean(np.where(mask, array_mm, np.nan) ** 2)))


# --- Figure 1 ---------------------------------------------------------------------------------
def fig_maps_truth_vs_recon(path, title, fit, truth, truth_src_mm, dV_series, extent, time_varying):
    """Truth | physics recon | physics+residual | truth-physics difference (rows = types, cols = epochs)."""
    mask = fit['mask'].astype(bool)
    truth_total = truth_src_mm.sum(axis=0) if len(truth_src_mm) else np.zeros_like(fit['xPB_mm'])
    physics = fit['xP_src_mm'].sum(axis=1)
    both = fit['xPB_mm']
    vmax_mm = max(float(np.nanmax(np.abs([truth_total.max(), truth_total.min(), np.nanmax(np.abs(physics)),
                                          np.nanmax(np.abs(both))]))), 1.0)
    epochs = pick_epochs(truth_total, dV_series) if time_varying else [0]
    row_names = ['Truth', 'PILA physics-only', 'PILA physics + residual', 'Truth − physics']
    # Static albums (one epoch): lay the 4 map types out as ONE ROW so titles never collide with tick labels above;
    # time-varying albums: rows = map types, columns = epochs.
    transpose = len(epochs) == 1
    if transpose:
        fig, axes = plt.subplots(1, len(row_names), figsize=(4.2 * len(row_names) + 1.5, 4.6), squeeze=False,
                                 gridspec_kw={'wspace': 0.35})
        axis_at = lambda row, col: axes[0, row]
    else:
        fig, axes = plt.subplots(len(row_names), len(epochs), figsize=(2.6 * len(epochs) + 1.3, 3.1 * len(row_names)), squeeze=False,
                                 gridspec_kw={'hspace': 0.45, 'wspace': 0.15})   # room for 2-line titles under tick labels above
        axis_at = lambda row, col: axes[row, col]
    diff_vmax_mm = vmax_mm      # residual on the SAME scale as the deformation: shows how (in)significant the misfit is
    peak_dV = np.abs(fit['params_phys'][:, :, 3]).max(axis=0)
    for col, epoch in enumerate(epochs):
        panels = [truth_total[epoch], physics[epoch], both[epoch], truth_total[epoch] - physics[epoch]]
        rec_xy = [(q[0], q[1]) for q, pk in zip(fit['params_phys'][epoch], peak_dV)
                  if abs(q[3]) > MASK_DV_FRACTION * max(pk, 1e-30) and pk > 0]   # hide ghost slots with ~0 dV
        for row, panel in enumerate(panels):
            axis = axis_at(row, col)
            label = row_names[row] + (f"  RMS {rms_mm(panel, mask):.1f} mm" if row == 3 else '')
            image = show_map(axis, panel, extent, vmax_mm if row < 3 else diff_vmax_mm, f"{label}\nepoch {epoch + 1} ({fit['dates'][epoch]})")
            if row == 2:
                main_image = image          # colourbar for rows 0-2 must come from a full-scale panel, not the difference
            if row == 3:
                diff_image = image
            mark_centres(axis, truth['sources'], rec_xy if row > 0 else [])
            if col == 0 or transpose:
                axis.set_ylabel('North (km)', fontsize=7)
            if row == len(row_names) - 1 or transpose:
                axis.set_xlabel('East (km)', fontsize=7)
    colorbar = fig.colorbar(main_image, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    colorbar.set_label(COLORBAR_LABEL, fontsize=8)      # small font: the long label was clipped on the 1-row static layout
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"{title}\n+ = truth centre, o = PILA centre. Raw LOS (no offset/ramp removal). All panels incl. the difference share ONE colour scale.", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


# --- Figure 2 ---------------------------------------------------------------------------------
def fig_maps_per_source(path, title, fit, truth, truth_src_mm, dV_series, assign, extent):
    """Rows = truth sources (+ extra recovered slots): [truth source | matched recovered | difference]."""
    mask = fit['mask'].astype(bool)
    n_true, n_rec = len(truth['sources']), fit['xP_src_mm'].shape[1]
    row_specs = []                         # (truth idx or None, rec idx or None, epoch)
    for j in range(n_true):
        epoch = int(np.abs(dV_series[j]).argmax())
        row_specs.append((j, int(assign[j, epoch]) if assign[j, epoch] >= 0 else None, epoch))
    for k in range(n_rec):
        if not any(r == k for _, r, _ in row_specs):
            row_specs.append((None, k, int(np.abs(fit['params_phys'][:, k, 3]).argmax())))
    if not row_specs:
        return
    stacks = [truth_src_mm[j, e] for j, _, e in row_specs if j is not None] + \
             [fit['xP_src_mm'][e, k] for _, k, e in row_specs if k is not None]
    vmax_mm = max(max(float(np.nanmax(np.abs(s))) for s in stacks), 1.0)
    diff_vmax_mm = vmax_mm     # difference on the SAME scale as the deformation (shows significance of the misfit)
    fig, axes = plt.subplots(len(row_specs), 3, figsize=(10.5, 3.0 * len(row_specs)), squeeze=False)
    image = None
    for row, (j, k, epoch) in enumerate(row_specs):
        truth_map = truth_src_mm[j, epoch] if j is not None else np.zeros(mask.shape)
        rec_map = fit['xP_src_mm'][epoch, k] if k is not None else np.zeros(mask.shape)
        tag = f"S{j + 1}" if j is not None else 'EXTRA'
        info = f"epoch {epoch + 1} ({fit['dates'][epoch]})"
        show_map(axes[row, 0], truth_map, extent, vmax_mm, f"Truth {tag}  {info}" if j is not None else 'no truth source (extra)')
        if k is not None:
            p = fit['params_phys'][epoch, k]
            image = show_map(axes[row, 1], rec_map, extent, vmax_mm,
                             f"PILA slot {k + 1}: d={p[2]:.1f} km, dV={p[3]:.2e} m³")
            mark_centres(axes[row, 1], [], [(p[0], p[1])])
        else:
            image = show_map(axes[row, 1], rec_map, extent, vmax_mm, 'unmatched (no recovered slot)')
        show_map(axes[row, 2], truth_map - rec_map, extent, diff_vmax_mm, f"Truth − PILA  RMS {rms_mm(truth_map - rec_map, mask):.1f} mm")
        if j is not None:
            mark_centres(axes[row, 0], [truth['sources'][j]], [])
        for col in range(3):
            axes[row, col].set_xlabel('East (km)', fontsize=7)
            if col == 0:      # y-label only on the left column: repeated labels collided with neighbouring tick labels
                axes[row, col].set_ylabel('North (km)', fontsize=7)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
        colorbar.set_label(COLORBAR_LABEL, fontsize=8); colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"{title}: each recovered source vs its truth source (peak-|dV| epoch); all panels share the ±{vmax_mm:.0f} mm scale", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


# --- Figure 3 ---------------------------------------------------------------------------------
def fig_param_recovery(path, title, fit, truth, dV_series, assign, valid, assign_raw):
    """Rows = truth sources (or recovered slots for K0); cols = xcen, ycen, depth, dV vs epoch."""
    params = fit['params_phys']
    n_epoch, n_rec = params.shape[0], params.shape[1]
    epoch_axis = np.arange(1, n_epoch + 1)
    n_true = len(truth['sources'])
    n_rows = max(n_true, 1)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.3 * n_rows), squeeze=False)
    matched_slots = {int(s) for s in assign_raw.ravel() if s >= 0}
    extra_labelled = False
    for row in range(n_rows):
        for col in range(4):
            axis = axes[row, col]
            for k in range(n_rec):               # unmatched recovered slots in grey (context)
                if k not in matched_slots or n_true == 0:
                    axis.plot(epoch_axis, params[:, k, col], '-', color=EXTRA_COLOR, lw=0.8, label='extra/other slot' if not extra_labelled else None)
                    extra_labelled = extra_labelled or col == 0 and row == 0
            if n_true:
                source = truth['sources'][row]
                truth_values = [source['xcen_km'], source['ycen_km'], source['depth_km']]
                truth_line = np.full(n_epoch, truth_values[col]) if col < 3 else dV_series[row]
                axis.plot(epoch_axis, truth_line, '-', color=TRUTH_COLOR, lw=2, label='Truth')
                rec_line = params[np.arange(n_epoch), np.maximum(assign[row], 0), col]
                show = (np.abs(dV_series[row]) > MASK_DV_FRACTION * np.abs(dV_series[row]).max()) & (assign_raw[row] >= 0)
                axis.plot(epoch_axis, np.where(show, rec_line, np.nan), 's--', color=PILA_COLOR, ms=3, lw=1.2, label='PILA (matched)')
                use = valid[row] & (assign_raw[row] >= 0)
                if use.any():
                    error = rec_line[use] - truth_line[use]
                    if col < 2:
                        text = f"RMSE {np.sqrt(np.mean(error ** 2)):.2f} km"
                    else:
                        text = f"RMSE {np.sqrt(np.mean((error / truth_line[use]) ** 2)) * 100:.0f} %"
                    if col == 2 and np.any((rec_line[use] <= DEPTH_BOUNDS_KM[0] + RAIL_TOL_KM) | (rec_line[use] >= DEPTH_BOUNDS_KM[1] - RAIL_TOL_KM)):
                        text += '\nRAILED on depth bound'
                    if col == 3 and np.any(np.sign(rec_line[use]) != np.sign(truth_line[use])):
                        text += '\nSIGN FLIP'
                    axis.text(0.03, 0.97, text, transform=axis.transAxes, va='top', fontsize=7,
                              bbox=dict(boxstyle='round', fc='w', ec='0.6', alpha=0.9))
                axis.set_title(f"S{row + 1} {PARAM_NAMES[col]}", fontsize=8)
            else:
                axis.set_title(f"K0 truth: no source; slot params {PARAM_NAMES[col]}", fontsize=8)
            axis.set_xlabel('Epoch index', fontsize=7); axis.set_ylabel(PARAM_NAMES[col], fontsize=7)
            axis.tick_params(labelsize=6); axis.grid(alpha=0.3)
            if col == 2:
                axis.axhline(DEPTH_BOUNDS_KM[1], color='r', ls=':', lw=0.8); axis.axhline(DEPTH_BOUNDS_KM[0], color='r', ls=':', lw=0.8)
    axes[0, 0].legend(fontsize=6, loc='lower right')
    fig.suptitle(f"{title}: parameter recovery vs epoch [noise-free]. PILA masked where truth |dV| < 5% of peak; "
                 f"RMSE on epochs with |dV| > 25% of peak; red dotted = depth prior bounds.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


# --- Figure 4 ---------------------------------------------------------------------------------
def fig_combined_timeseries(path, title, fit, truth, truth_src_mm, geometry):
    """Combined deformation vs truth through time: LOS at max-|truth| pixel, at source centres, RMS and peak."""
    mask = fit['mask'].astype(bool)
    n_epoch = fit['xPB_mm'].shape[0]
    epoch_axis = np.arange(1, n_epoch + 1)
    truth_total = truth_src_mm.sum(axis=0) if len(truth_src_mm) else np.zeros_like(fit['xPB_mm'])
    physics = fit['xP_src_mm'].sum(axis=1)
    both = fit['xPB_mm']
    reference = truth_total if np.abs(truth_total).max() > 0 else physics
    # ONE fixed pixel (max |truth| over the whole series): re-picking per epoch jumps between pixels at quiescent epochs
    flat_index = int(np.abs(np.where(mask, reference, 0)).reshape(n_epoch, -1).max(axis=0).argmax())
    pick = lambda stack: stack.reshape(n_epoch, -1)[:, flat_index]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(epoch_axis, pick(truth_total), 'k-', lw=2, label='Truth'); ax.plot(epoch_axis, pick(physics), 's--', color=PILA_COLOR, ms=3, label='PILA physics-only')
    ax.plot(epoch_axis, pick(both), '^:', color='tab:green', ms=3, label='PILA physics + residual')
    ax.set_title('LOS at the fixed pixel of max |truth| over the series', fontsize=9); ax.set_ylabel('LOS displacement (mm)')
    ax = axes[0, 1]
    x_km, y_km = geometry['x_m'] / 1e3, geometry['y_m'] / 1e3
    for j, source in enumerate(truth['sources']):
        col_i, row_i = int(np.abs(x_km - source['xcen_km']).argmin()), int(np.abs(y_km - source['ycen_km']).argmin())
        color = f'C{j}'
        ax.plot(epoch_axis, truth_total[:, row_i, col_i], '-', color=color, lw=2, label=f'Truth @ S{j + 1} centre')
        ax.plot(epoch_axis, physics[:, row_i, col_i], '--', color=color, lw=1.2, label=f'PILA @ S{j + 1} centre')
    ax.set_title('Combined LOS at each truth-source centre pixel', fontsize=9); ax.set_ylabel('LOS displacement (mm)')
    ax = axes[1, 0]
    ax.plot(epoch_axis, [rms_mm(truth_total[t] - physics[t], mask) for t in range(n_epoch)], 's-', color=PILA_COLOR, ms=3, label='RMS(truth − physics-only)')
    ax.plot(epoch_axis, [rms_mm(truth_total[t] - both[t], mask) for t in range(n_epoch)], '^-', color='tab:green', ms=3, label='RMS(truth − physics + residual)')
    ax.set_title('Whole-scene misfit vs epoch', fontsize=9); ax.set_ylabel('RMS (mm)')
    ax = axes[1, 1]
    peak = lambda stack: np.nanmax(np.abs(np.where(mask, stack, np.nan)).reshape(n_epoch, -1), axis=1)
    ax.plot(epoch_axis, peak(truth_total), 'k-', lw=2, label='Truth'); ax.plot(epoch_axis, peak(physics), 's--', color=PILA_COLOR, ms=3, label='PILA physics-only')
    ax.plot(epoch_axis, peak(both), '^:', color='tab:green', ms=3, label='PILA physics + residual')
    ax.set_title('Peak |LOS| vs epoch', fontsize=9); ax.set_ylabel('Peak |LOS| (mm)')
    for axis in axes.ravel():
        axis.set_xlabel('Epoch index'); axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    fig.suptitle(f"{title}: combined deformation vs truth through time", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


# --- Sanity checks ----------------------------------------------------------------------------
def run_checks(album, k, fit, truth_total_mm, assign_raw, valid, tol_mm, geometry, truth):
    """Scripted pass/fail; returns list of (ok, message). Mask-not-all-True is only a note."""
    mask = fit['mask'].astype(bool)
    results = []
    data_diff = np.nanmax(np.abs(np.where(mask, truth_total_mm - fit['data_mm'], np.nan)))
    results.append((bool(data_diff < tol_mm), f"{album} K{k}: rendered truth == fit data_mm over mask (max |diff| {data_diff:.4f} mm, tol {tol_mm})"))
    results.append((True, f"{album} K{k}: mask {int(mask.sum())}/{mask.size} px coherent (note only)"))
    # Injective on epochs where every truth source was matched (a slot credited to two truth sources = bug)
    duplicates = sum(1 for t in range(assign_raw.shape[1])
                     if len(set(assign_raw[:, t][assign_raw[:, t] >= 0])) != int((assign_raw[:, t] >= 0).sum()))
    results.append((duplicates == 0, f"{album} K{k}: no slot matched to two truth sources ({duplicates} bad epochs)"))
    # Units/sign check: re-render the recovered parameters (km -> m) and compare with the decoder's own per-source field
    grid = geometry['grid']
    xE = torch.tensor(grid['xE_m']); yN = torch.tensor(grid['yN_m'])
    los = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]))
    worst_mm = 0.0
    n_epoch, n_rec = fit['params_phys'].shape[:2]
    for epoch in sorted({0, n_epoch // 2, n_epoch - 1}):
        for slot in range(n_rec):
            x_km, y_km, depth_km, dV_m3 = fit['params_phys'][epoch, slot]
            field = mogi_los_mm_torch(xE, yN, los, torch.tensor([[x_km * 1e3]], dtype=torch.float64),
                                      torch.tensor([[y_km * 1e3]], dtype=torch.float64),
                                      torch.tensor([[depth_km * 1e3]], dtype=torch.float64),
                                      torch.tensor([[dV_m3]], dtype=torch.float64))[0, 0].numpy().reshape(mask.shape)
            worst_mm = max(worst_mm, float(np.nanmax(np.abs(np.where(mask, field - fit['xP_src_mm'][epoch, slot], np.nan)))))
    results.append((worst_mm < 0.05,
                    f"{album} K{k}: params_phys re-rendered == xP_src_mm (units/sign/geometry; worst {worst_mm:.4f} mm)"))
    # Residual contribution identity: (physics+residual) - physics must be finite and data - physics+residual consistent
    residual_part = fit['xPB_mm'] - fit['xP_src_mm'].sum(axis=1)
    results.append((bool(np.isfinite(np.where(mask, residual_part, 0)).all()), f"{album} K{k}: residual contribution finite, RMS {rms_mm(residual_part, mask):.2f} mm"))
    if truth.get('dates'):
        fit_dates = [str(d).replace('-', '') for d in fit['dates']]
        results.append((fit_dates == [str(d).replace('-', '') for d in truth['dates']], f"{album} K{k}: fit dates == truth dates"))
    return results


# --- Per-album driver ---------------------------------------------------------------------------
def main_album(args):
    album = os.path.basename(os.path.normpath(args.album_root))
    truth_path = args.truth or os.path.join(HERE, 'synthetic/multisource/clean_scenarios', f'{album}_truth.json')
    if not os.path.exists(truth_path):
        raise FileNotFoundError(truth_path)
    out_dir = args.out_dir or os.path.join(HERE, 'figures/multisource_clean_pila', album)
    if not os.path.isdir(args.album_root):
        raise FileNotFoundError(args.album_root)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[1/5] Loading truth {truth_path} and fits under {args.album_root}...")
    with open(truth_path) as handle:
        truth = json.load(handle)
    fits = load_fits(args.album_root)
    n_true = len(truth['sources'])
    fit_ks = args.fit_k or sorted({k for k in ([n_true] if n_true else []) + [1, 3] if k in fits})
    missing_ks = [k for k in fit_ks if k not in fits]
    if missing_ks:
        print(f"  WARNING: requested K={missing_ks} has no fit under {args.album_root}")
    fit_ks = [k for k in fit_ks if k in fits]
    if not fit_ks:
        raise FileNotFoundError(f"no reportable fit K in {args.album_root} (have {sorted(fits)})")
    print(f"  truth K={n_true}, dates check below, time_varying={truth.get('time_varying')}, fits available K={sorted(fits)}, reporting K={fit_ks}")
    geometry = load_ta69_geometry()
    unit_fields = truth_unit_fields_mm_per_m3(truth, geometry)
    all_checks, summary_rows = [], []
    for k_index, k in enumerate(fit_ks, start=1):
        fit, usage = fits[k]
        usage = usage or {}
        n_epoch = fit['xPB_mm'].shape[0]
        print(f"[2/5] ({k_index}/{len(fit_ks)}) K={k}: shapes params={fit['params_phys'].shape} xP_src={fit['xP_src_mm'].shape} dtype={fit['xP_src_mm'].dtype}")
        dV_series = truth_dV_series_m3(truth, n_epoch)
        truth_src_mm = truth_fields_mm(unit_fields, dV_series)
        truth_total = truth_src_mm.sum(axis=0) if n_true else np.zeros_like(fit['xPB_mm'])
        mask = fit['mask'].astype(bool)
        print(f"[3/5] K={k}: per-epoch field-space matching...")
        empty = (np.zeros((0, n_epoch), int), np.zeros((0, n_epoch), bool), np.zeros(0, int), np.zeros((0, n_epoch), int), np.zeros((0, n_epoch)))
        assign, valid, modal, assign_raw, cost_rms_mm = match_per_epoch(truth_src_mm, dV_series, fit['xP_src_mm'], mask) if n_true else empty
        all_checks += run_checks(album, k, fit, truth_total, assign_raw, valid, args.truth_tol_mm, geometry, truth)
        metrics = source_metrics(truth, dV_series, fit['params_phys'], assign_raw, valid, modal, cost_rms_mm)
        r2 = physics_r2(truth_total, fit['xP_src_mm'].sum(axis=1), mask)
        r2_active = physics_r2(truth_total, fit['xP_src_mm'].sum(axis=1), mask, active_only=True)
        abs_dV_peaks = np.abs(fit['params_phys'][:, :, 3]).max(axis=0)
        slot_peak_los_mm = np.nanmax(np.abs(np.where(mask, fit['xP_src_mm'], np.nan)).reshape(n_epoch, fit['xP_src_mm'].shape[1], -1), axis=(0, 2))
        n_active = int(np.sum(slot_peak_los_mm > ACTIVE_PEAK_LOS_MM))      # ABSOLUTE threshold: K0 false positives are visible
        for row in metrics:
            summary_rows.append(dict(album=album, scenario=truth.get('scenario'), time_varying=bool(truth.get('time_varying')), true_K=n_true,
                                     fit_K=k, physics_R2_about_zero=r2, physics_R2_active_epochs=r2_active, n_active_sources=n_active, n_true_sources_unmatched=int(sum(not r.get('matched') for r in metrics)),
                                     extra_slots_peak_dV_m3=float(
                                         np.max([abs_dV_peaks[slot] for slot in range(len(abs_dV_peaks)) if slot not in set(modal)] or [0])),
                                     train_sec=usage.get('train_sec'), **row))
        if not metrics:    # K0 truth: record amplitude of what PILA invented
            summary_rows.append(dict(album=album, scenario=truth.get('scenario'), time_varying=False, true_K=0, fit_K=k, physics_R2_about_zero=r2,
                                     n_active_sources=n_active, extra_slots_peak_dV_m3=float(abs_dV_peaks.max()), n_true_sources_unmatched=0,
                                     peak_LOS_mm=float(np.nanmax(np.abs(fit['xP_src_mm'].sum(axis=1))))))
        extent = [fit['xE_km'].min(), fit['xE_km'].max(), fit['yN_km'].min(), fit['yN_km'].max()]
        time_varying = bool(truth.get('time_varying'))
        title = f"{album}, PILA K={k} fit (physics R² about zero {r2:.3f}, active-epoch R² {r2_active:.3f}, {n_active} active source(s) >{ACTIVE_PEAK_LOS_MM:.0f} mm)"
        print(f"[4/5] ({k_index}/{len(fit_ks)}) K={k}: figures -> {out_dir}")
        stem = os.path.join(out_dir, f'{album}_K{k}')
        fig_maps_truth_vs_recon(stem + '_maps_truth_vs_recon.png', title, fit, truth, truth_src_mm, dV_series, extent, time_varying)
        fig_maps_per_source(stem + '_maps_per_source.png', title, fit, truth, truth_src_mm, dV_series, assign, extent)
        fig_param_recovery(stem + '_param_recovery.png', title, fit, truth, dV_series, assign, valid, assign_raw)
        fig_combined_timeseries(stem + '_combined_timeseries.png', title, fit, truth, truth_src_mm, geometry)
    print("[5/5] Sanity checks + summary CSV...")
    n_fail = 0
    for ok, message in all_checks:
        n_fail += (not ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    fieldnames = sorted({key for row in summary_rows for key in row})
    with open(os.path.join(out_dir, f'{album}_summary.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(summary_rows)
    print(f"Done. Output saved to {out_dir}/ ({n_fail} check failures)")
    if n_fail:
        raise SystemExit(1)


# --- Cross-album summary ------------------------------------------------------------------------
def main_summarise(args):
    paths = sorted(glob.glob(os.path.join(args.summary_root, '*', '*_summary.csv')))
    if not paths:
        raise FileNotFoundError(f"no *_summary.csv under {args.summary_root}")
    rows = []
    for path in paths:
        with open(path) as handle:
            rows += list(csv.DictReader(handle))
    fieldnames = sorted({key for row in rows for key in row})
    out_csv = os.path.join(args.summary_root, 'summary_all.csv')
    with open(out_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    # True-K fit (or K=1 for K0 albums), one line per album. Unmatched truth sources count as FAILURES (never dropped).
    per_album = {}
    for row in rows:
        true_k, fit_k = int(row['true_K']), int(row['fit_K'])
        if fit_k == max(true_k, 1):
            per_album.setdefault(row['album'], []).append(row)
    if not per_album:
        raise ValueError("no true-K fits found in the summary CSVs")
    albums = sorted(per_album, key=lambda a: (int(per_album[a][0]['true_K']), a))
    metric_names = ['physics_R2_active_epochs', 'n_active_sources', 'n_true_sources_unmatched', 'loc_err_km', 'depth_err_pct', 'dV_err_pct', 'rail_frac', 'sign_flip_frac']
    to_float = lambda text: float(text) if text not in (None, '') else float('nan')
    matrix = np.full((len(albums), len(metric_names)), np.nan)
    for i, album in enumerate(albums):
        album_rows = per_album[album]
        for m, name in enumerate(metric_names):
            values = [to_float(r.get(name)) for r in album_rows]
            values = [v for v in values if not np.isnan(v)]
            if not values:
                continue
            # R2 / counts: first row; per-source errors: WORST source (max) so a failed source cannot hide in a mean
            matrix[i, m] = values[0] if name in ('physics_R2_active_epochs', 'n_active_sources', 'n_true_sources_unmatched') else max(values)
        if matrix[i, metric_names.index('n_true_sources_unmatched')] > 0:      # an unmatched source is a failure
            for name in ('loc_err_km', 'depth_err_pct', 'dV_err_pct'):
                matrix[i, metric_names.index(name)] = np.inf
    passing = np.full(matrix.shape, np.nan)     # 1 = pass, 0 = fail, NaN = not applicable
    thresholds = {'physics_R2_active_epochs': (PASS_R2, 'ge'), 'loc_err_km': (PASS_LOC_KM, 'le'), 'depth_err_pct': (PASS_DEPTH_PCT, 'le'),
                  'dV_err_pct': (PASS_DV_PCT, 'le'), 'rail_frac': (0.0, 'le'), 'sign_flip_frac': (0.0, 'le'), 'n_true_sources_unmatched': (0, 'le')}
    for m, name in enumerate(metric_names):
        if name in thresholds:
            limit, kind = thresholds[name]
            ok = np.where(np.isnan(matrix[:, m]), np.nan, (matrix[:, m] >= limit) if kind == 'ge' else (matrix[:, m] <= limit))
            passing[:, m] = ok
    fig, axis = plt.subplots(figsize=(1.6 * len(metric_names) + 3, 0.35 * len(albums) + 2.2))
    colour_grid = np.where(np.isnan(passing), 0.5, passing)          # grey = informational, green = pass, red = fail
    axis.imshow(colour_grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, alpha=0.75)
    for i in range(len(albums)):
        for m in range(len(metric_names)):
            axis.text(m, i, '–' if np.isnan(matrix[i, m]) else ('FAIL' if np.isinf(matrix[i, m]) else f"{matrix[i, m]:.2g}"), ha='center', va='center', fontsize=7)
    axis.set_xticks(range(len(metric_names))); axis.set_xticklabels(metric_names, rotation=30, ha='right', fontsize=8)
    axis.set_yticks(range(len(albums))); axis.set_yticklabels(albums, fontsize=7)
    axis.set_title(f"True-K fit per album (K0 albums: K=1 fit); worst source shown. Green = pass: loc < {PASS_LOC_KM} km, depth < {PASS_DEPTH_PCT:.0f}%, "
                   f"dV < {PASS_DV_PCT:.0f}%, active-epoch R² > {PASS_R2}, no rail/sign flip/unmatched; grey = informational", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.summary_root, 'summary_matrix.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Done. Output saved to {args.summary_root}/ (summary_all.csv, summary_matrix.png)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album-root')
    parser.add_argument('--truth')
    parser.add_argument('--out-dir')
    parser.add_argument('--fit-k', type=int, nargs='*')
    parser.add_argument('--truth-tol-mm', type=float, default=0.05,
                        help='max |rendered truth - fit data_mm| allowed (clean npz are noise-free; float32 -> ~1e-3 mm)')
    parser.add_argument('--summarise', action='store_true')
    parser.add_argument('--summary-root', default=os.path.join(HERE, 'figures/multisource_clean_pila'))
    args = parser.parse_args()
    if args.summarise:
        main_summarise(args)
    else:
        if not args.album_root:
            parser.error('--album-root is required (or use --summarise)')
        main_album(args)


if __name__ == '__main__':
    main()
