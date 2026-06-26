#!/usr/bin/env python
# Usage:       imported by synthetic/evaluate.py
# Description: Plot helpers for the PILA synthetic-test framework, following the
#              project plotting standards: RdBu_r for displacement, colorbars,
#              axis labels with units, tight_layout, save png(dpi=150)+pdf.
# Date:        2026-06-22

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Presentation-ready defaults (applied to every figure built in this module) ----
# Larger fonts/line widths and crisper DPI for slides; grids are added per-function
# (only on line plots, never on image/scatter maps). Importing this module sets these
# globally for the process, so generate_timeseries' sanity plots inherit them too.
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'figure.titlesize': 18,
    'axes.linewidth': 1.1,
    'lines.linewidth': 2.2,
    'savefig.dpi': 200,
})


# Friendly labels for the trailing noise-component token in an experiment slug, so a
# presentation title reads "... (realistic noise)" instead of the internal "Combined".
_NOISE_LABELS = {
    'combined': 'realistic noise',        # sum of all components (the headline case)
    'trop': 'tropospheric noise',         # stratified / topo-correlated APS
    'turbulent': 'turbulent noise',       # turbulent atmosphere
    'orbit': 'orbital ramp',              # orbital ramp only
    'white': 'white noise',               # white / decorrelation only
}


def _pretty(name):
    """Human-readable title from an experiment slug: drop the internal 'perepoch' tag,
    render a trailing noise token (e.g. 'combined') as a friendly parenthetical, the rest
    underscores -> spaces, title-cased (presentation-friendly)."""
    tokens = [t for t in name.split('_') if t.lower() != 'perepoch']
    noise = None
    if tokens and tokens[-1].lower() in _NOISE_LABELS:       # peel trailing noise token
        noise = _NOISE_LABELS[tokens[-1].lower()]
        tokens = tokens[:-1]
    base = ' '.join(tokens).strip().title()
    return f'{base} ({noise})' if noise else base


def _save(fig, out_dir, stem):
    fig.tight_layout()
    # bbox_inches='tight' so a figure-level suptitle (placed above the axes) is not
    # clipped by tight_layout, which ignores suptitles when computing the bounding box.
    fig.savefig(os.path.join(out_dir, stem + '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_param_recovery_vs_epoch(out_dir, name, param_names, param_units,
                                 truth, inferred, signal_rms):
    """
    One subplot per parameter: truth vs inferred across epochs.

    truth, inferred : [n_epoch, n_param] physical units.
    signal_rms      : [n_epoch] mm. Accepted for call-signature compatibility but
                      no longer drawn (the strong-signal shading was removed).
    """
    n_param = len(param_names)
    n_epoch = truth.shape[0]
    ncol = min(3, n_param)
    nrow = int(np.ceil(n_param / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow), squeeze=False)
    ep = np.arange(n_epoch)
    for j, (pname, punit) in enumerate(zip(param_names, param_units)):
        ax = axes[j // ncol][j % ncol]
        ax.plot(ep, truth[:, j], 'k-', lw=2, label='truth')
        ax.plot(ep, inferred[:, j], 'o--', color='C3', ms=4, label='PILA')
        ax.set_xlabel('Epoch index')
        ax.set_ylabel(f'{pname} ({punit})')
        ax.set_title(pname)
        ax.grid(alpha=0.3)
    # Single shared legend OUTSIDE the panels: drop it into the first empty grid cell
    # if there is one (cleaner than crowding a data panel); otherwise put a figure-level
    # legend below the grid. bbox_inches='tight' in _save keeps it from being clipped.
    handles, labels = axes[0][0].get_legend_handles_labels()
    empty = list(range(n_param, nrow * ncol))
    for j in empty:
        axes[j // ncol][j % ncol].axis('off')
    if empty:
        slot = axes[empty[0] // ncol][empty[0] % ncol]
        slot.legend(handles, labels, loc='center', frameon=True)
    else:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{_pretty(name)}: parameter recovery vs epoch', y=1.0)
    _save(fig, out_dir, f'{name}_param_recovery')


def plot_los_maps(out_dir, name, xE_km, yN_km, obs_mm, pred_mm, epoch_label):
    """Observed / predicted / residual LOS scatter maps (mm) at one epoch."""
    resid = obs_mm - pred_mm
    vmax = np.nanmax(np.abs(np.concatenate([obs_mm, pred_mm])))
    rmax = np.nanmax(np.abs(resid)) + 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, data, ttl, vm in [
            (axes[0], obs_mm, 'Observed (synthetic)', vmax),
            (axes[1], pred_mm, 'PILA physics prediction', vmax),
            (axes[2], resid, 'Residual (obs − pred)', rmax)]:
        sc = ax.scatter(xE_km, yN_km, c=data, cmap='RdBu_r', vmin=-vm, vmax=vm, s=6)
        ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
        ax.set_title(ttl); ax.set_aspect('equal')
        cb = fig.colorbar(sc, ax=ax); cb.set_label('LOS (mm), + toward sat')
    fig.suptitle(f'{_pretty(name)}: LOS maps, epoch {epoch_label}', y=1.02)
    _save(fig, out_dir, f'{name}_los_maps')


def plot_trajectory_map(out_dir, name, truth_xy_km, inferred_xy_km, signal_rms):
    """Map view of the source horizontal trajectory: truth vs inferred path."""
    fig, ax = plt.subplots(figsize=(6, 6))
    s = 20 + 120 * (signal_rms / (signal_rms.max() + 1e-12))
    ax.plot(truth_xy_km[:, 0], truth_xy_km[:, 1], 'k-', lw=2, label='truth path')
    ax.scatter(truth_xy_km[:, 0], truth_xy_km[:, 1], c='k', s=s, alpha=0.5)
    ax.plot(inferred_xy_km[:, 0], inferred_xy_km[:, 1], '--', color='C3', label='PILA path')
    ax.scatter(inferred_xy_km[:, 0], inferred_xy_km[:, 1], c='C3', s=s, alpha=0.6)
    ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
    ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f'{_pretty(name)}: source trajectory')
    _save(fig, out_dir, f'{name}_trajectory')


def plot_trajectory_compare(out_dir, name, truth_xy_km, raw_xy_km, blend_xy_km,
                            signal_rms, alpha_label=''):
    """Map view of the source horizontal trajectory: truth vs RAW per-epoch vs
    TEMPORALLY-BLENDED inference. Marker size scales with per-epoch signal RMS so
    well-constrained (strong-signal) epochs stand out.

    truth_xy_km, raw_xy_km, blend_xy_km : [n_epoch, 2] East/North in km.
    For a genuinely MOVING source the blended path will visibly LAG the truth path
    when the location alpha is large (temporal-smoothing bias) -- that lag is the
    quantity this comparison exists to expose.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    s = 20 + 120 * (signal_rms / (signal_rms.max() + 1e-12))
    # truth path (black), raw inference (orange), blended inference (blue)
    ax.plot(truth_xy_km[:, 0], truth_xy_km[:, 1], 'k-', lw=2, label='truth path')
    ax.scatter(truth_xy_km[:, 0], truth_xy_km[:, 1], c='k', s=s, alpha=0.4)
    ax.plot(raw_xy_km[:, 0], raw_xy_km[:, 1], '--', color='C1', label='PILA raw (per-epoch)')
    ax.scatter(raw_xy_km[:, 0], raw_xy_km[:, 1], c='C1', s=s, alpha=0.5)
    ax.plot(blend_xy_km[:, 0], blend_xy_km[:, 1], '--', color='C0',
            label='PILA temporal-blended')
    ax.scatter(blend_xy_km[:, 0], blend_xy_km[:, 1], c='C0', s=s, alpha=0.6)
    ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
    ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=0.3)
    ttl = f'{_pretty(name)}: source trajectory (raw vs temporal)'
    if alpha_label:
        ttl += f'\n{alpha_label}'
    ax.set_title(ttl)
    _save(fig, out_dir, f'{name}_trajectory_compare')


def plot_param_recovery_compare(out_dir, name, param_names, param_units,
                                truth, raw, blend, signal_rms, alpha_label=''):
    """One subplot per parameter: truth vs RAW vs TEMPORALLY-BLENDED across epochs.

    truth, raw, blend : [n_epoch, n_param] physical units.
    signal_rms        : [n_epoch] mm (accepted for signature symmetry; not drawn).
    """
    n_param = len(param_names)
    n_epoch = truth.shape[0]
    ncol = min(3, n_param)
    nrow = int(np.ceil(n_param / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow), squeeze=False)
    ep = np.arange(n_epoch)
    for j, (pname, punit) in enumerate(zip(param_names, param_units)):
        ax = axes[j // ncol][j % ncol]
        ax.plot(ep, truth[:, j], 'k-', lw=2, label='truth')
        ax.plot(ep, raw[:, j], 'o--', color='C1', ms=3, alpha=0.7, label='PILA raw')
        ax.plot(ep, blend[:, j], 's--', color='C0', ms=3, alpha=0.8, label='PILA blended')
        ax.set_xlabel('Epoch index')
        ax.set_ylabel(pname if not punit else f'{pname} ({punit})')
        ax.set_title(pname)
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    empty = list(range(n_param, nrow * ncol))
    for j in empty:
        axes[j // ncol][j % ncol].axis('off')
    if empty:
        slot = axes[empty[0] // ncol][empty[0] % ncol]
        slot.legend(handles, labels, loc='center', frameon=True)
    else:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02))
    suptitle = f'{_pretty(name)}: parameter recovery vs epoch (raw vs temporal)'
    if alpha_label:
        suptitle += f'  [{alpha_label}]'
    fig.suptitle(suptitle, y=1.0)
    _save(fig, out_dir, f'{name}_param_recovery_compare')


def plot_timeseries_montage(out_dir, name, cube_m, dates=None, mask=None,
                            n_frames=12, title_suffix=''):
    """
    Montage of LOS displacement maps across epochs (mm) for a visual sanity check
    of a synthetic time-series. Picks up to n_frames evenly spaced epochs and shares
    one symmetric color scale so the buildup is visible across panels.

    cube_m : [n_epoch, Ny, Nx] LOS displacement, metres.
    """
    n_epoch = cube_m.shape[0]
    idx = np.linspace(0, n_epoch - 1, min(n_frames, n_epoch)).round().astype(int)
    idx = sorted(set(idx.tolist()))
    cube_mm = cube_m * 1000.0
    if mask is not None:
        cube_mm = np.where(mask[None], cube_mm, np.nan)
    vmax = np.nanmax(np.abs(cube_mm[idx])) + 1e-9

    ncol = 4
    nrow = int(np.ceil(len(idx) / ncol))
    fig_h_in = 3.0 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, fig_h_in), squeeze=False)
    im = None
    for k, ep in enumerate(idx):
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(cube_mm[ep], cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
        lbl = dates[ep] if (dates is not None and ep < len(dates)) else f'epoch {ep}'
        ax.set_title(lbl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(idx), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    # Margins are set in ABSOLUTE inches (converted to figure fraction) rather than the
    # default fractional margins: for a tall montage the default top=0.88 fraction would
    # leave several inches of blank space below the title. ~0.7 in reserves room for the
    # two-line suptitle; the title sits just above the first row regardless of nrow.
    top_frac = 1.0 - 0.7 / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.2 / fig_h_in, right=0.9, hspace=0.25)
    fig.suptitle(f'{_pretty(name)}: LOS time-series{title_suffix}\n(mm, + toward satellite)',
                 y=1.0 - 0.1 / fig_h_in, va='top')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='LOS (mm)')
    fig.savefig(os.path.join(out_dir, f'{name}_timeseries_montage.png'), dpi=150,
                bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{name}_timeseries_montage.pdf'),
                bbox_inches='tight')
    plt.close(fig)


def plot_normalised_montage(out_dir, name, cube_m, dates=None, mask=None,
                            annot=None, title_suffix=''):
    """
    Montage where EACH epoch is scaled to its OWN |max|, so footprint SHAPE/width is
    comparable across epochs regardless of amplitude. This decouples spatial extent
    from amplitude growth — the confound that makes a shared-scale montage of a
    depth-changing source look like its footprint never changes.

    cube_m : [n_epoch, Ny, Nx] LOS displacement, metres.
    annot  : optional list[str] of per-epoch annotations (e.g. 'd=3.5km') appended to
             each panel title.
    """
    n_epoch = cube_m.shape[0]
    cube_mm = cube_m * 1000.0
    if mask is not None:
        cube_mm = np.where(mask[None], cube_mm, np.nan)

    ncol = 8
    nrow = int(np.ceil(n_epoch / ncol))
    fig_h_in = 2.0 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, fig_h_in), squeeze=False)
    for ep in range(n_epoch):
        ax = axes[ep // ncol][ep % ncol]
        vmax = np.nanmax(np.abs(cube_mm[ep])) + 1e-9   # per-epoch scale -> shape only
        ax.imshow(cube_mm[ep], cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
        lbl = dates[ep] if (dates is not None and ep < len(dates)) else f'ep{ep}'
        if annot is not None and ep < len(annot):
            lbl = f'{lbl}  {annot[ep]}'
        ax.set_title(lbl, fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n_epoch, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    # Absolute-inch top margin so the two-line title hugs the first row (see the
    # time-series montage above): the default fractional top margin leaves a large
    # blank band below the title on a tall, many-row montage.
    fig.subplots_adjust(top=1.0 - 0.6 / fig_h_in, bottom=0.1 / fig_h_in, hspace=0.3)
    fig.suptitle(f'{_pretty(name)}: per-epoch-NORMALISED LOS{title_suffix}\n'
                 '(shape only; each panel scaled to its own |max|)',
                 y=1.0 - 0.1 / fig_h_in, va='top')
    fig.savefig(os.path.join(out_dir, f'{name}_normalised_montage.png'), dpi=130,
                bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{name}_normalised_montage.pdf'),
                bbox_inches='tight')
    plt.close(fig)


def plot_footprint_width(out_dir, name, cube_m, px_km, mask=None,
                         overlay=None, overlay_label=None):
    """
    Footprint half-max equivalent radius (km) per epoch — a quantitative, amplitude-
    independent measure of spatial extent. For each epoch: count pixels where
    |LOS| >= 0.5*peak (the half-max region), convert that area to an equivalent-circle
    radius. Translation-invariant, so a laterally migrating source does not bias it.

    cube_m        : [n_epoch, Ny, Nx] LOS displacement, metres.
    px_km         : float   pixel size in km (for area -> radius).
    overlay       : optional [n_epoch] truth parameter to plot on a twin axis
                    (e.g. source depth for Mogi) to compare against the footprint.
    overlay_label : str     y-label for the overlay (e.g. 'source depth (km)').

    Returns
    -------
    eq_radius_km : np.ndarray [n_epoch]   half-max equivalent radius per epoch.
    """
    n_epoch = cube_m.shape[0]
    los_mm = cube_m * 1000.0
    eq_radius_km = np.full(n_epoch, np.nan)
    for ep in range(n_epoch):
        a = np.abs(los_mm[ep])
        if mask is not None:
            a = np.where(mask, a, np.nan)
        pk = np.nanmax(a)
        if not np.isfinite(pk) or pk < 1e-6:
            continue
        area_px = int(np.nansum(a >= 0.5 * pk))        # pixels above half-max
        eq_radius_km[ep] = np.sqrt(area_px / np.pi) * px_km

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(eq_radius_km, 'o-', color='C0', label='footprint half-max radius')
    ax1.set_xlabel('Epoch index')
    ax1.set_ylabel('Half-max equiv. radius (km)', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    if overlay is not None:
        ax2 = ax1.twinx()
        ax2.plot(np.asarray(overlay), 's--', color='C3', alpha=0.8)
        ax2.set_ylabel(overlay_label or 'overlay', color='C3')
        ax2.tick_params(axis='y', labelcolor='C3')
    ax1.grid(alpha=0.3)
    ax1.set_title(f'{_pretty(name)}: footprint width per epoch')
    _save(fig, out_dir, f'{name}_footprint_width')
    return eq_radius_km


def plot_breakdown_curve(out_dir, name, xvals, loc_err_m, xlabel='SNR (dB)',
                         stem='breakdown_snr', secondary=None,
                         breakdown_threshold_m=None):
    """
    Breakdown curve vs a generic predictor. Location error (m) on the left axis (the
    headline metric); optional secondary metrics (e.g. {'dV %err': [...],
    'LOS RMSE (mm)': [...]}) on a twin right axis so they are visible despite the very
    different scale.

    xvals, loc_err_m : sequences over the sweep (any order; sorted by xvals internally).
    xlabel           : x-axis label, e.g. 'SNR (dB)', 'signal/noise ratio',
                       'noise RMS (mm)'. Drives both the axis label and the title.
    stem             : output filename stem suffix (file = '<name>_<stem>').
    breakdown_threshold_m : if given, draw a horizontal line marking the failure
                            threshold (e.g. one multilook cell).
    """
    order = np.argsort(xvals)
    x = np.asarray(xvals)[order]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    l1, = ax.plot(x, np.asarray(loc_err_m)[order], 'o-', color='C0',
                  label='location error (m)')
    ax.set_xlabel(xlabel); ax.set_ylabel('Source location error (m)', color='C0')
    ax.tick_params(axis='y', labelcolor='C0'); ax.grid(alpha=0.3)
    if breakdown_threshold_m is not None:
        ax.axhline(breakdown_threshold_m, ls=':', color='0.4',
                   label=f'breakdown ({breakdown_threshold_m:.0f} m)')
    lines = [l1]
    if breakdown_threshold_m is not None:
        lines.append(ax.lines[-1])
    if secondary:
        ax2 = ax.twinx()
        for c, (label, vals) in zip(['C1', 'C2', 'C3'], secondary.items()):
            ln, = ax2.plot(x, np.asarray(vals)[order], 's--', color=c, alpha=0.8,
                           label=label)
            lines.append(ln)
        ax2.set_ylabel('parameter / LOS error', color='0.3')
    ax.set_title(f'{_pretty(name)}: PILA recovery vs {xlabel}')
    ax.legend(lines, [l.get_label() for l in lines], loc='center right')
    _save(fig, out_dir, f'{name}_{stem}')


def binned_median(xvals, yvals, edges, min_count=3):
    """Median + IQR of yvals within each [edge, edge) bin of xvals.

    Returns (centres, medians, q25, q75) arrays, one entry per bin that holds at least
    `min_count` points. Shared by the per-epoch breakdown and the R^2-vs-SNR plots so a
    single binning definition is used everywhere.
    """
    xvals = np.asarray(xvals); yvals = np.asarray(yvals)
    cx, cy, clo, chi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (xvals >= a) & (xvals < b)
        if m.sum() >= min_count:
            cx.append(0.5 * (a + b)); cy.append(np.median(yvals[m]))
            clo.append(np.percentile(yvals[m], 25)); chi.append(np.percentile(yvals[m], 75))
    return np.array(cx), np.array(cy), np.array(clo), np.array(chi)


def plot_recon_r2_vs_snr(out_dir, name, snr_x, recon_r2, recovered, bin_w=None,
                         min_count=8, xlabel='per-epoch signal/noise ratio (linear)',
                         stem='recon_r2_vs_snr'):
    """
    Reconstruction quality vs SNR: squared Pearson correlation r^2 between the clean LOS
    field forward-modelled from PILA's RECOVERED parameters and the clean field from the
    TRUTH parameters, plotted against the per-epoch signal-to-noise predictor. Bounded in
    [0, 1] (1 = same spatial pattern, 0 = no correlation) — a continuous accuracy axis and
    one of the three definitive SNR diagnostics (alongside dV and peak-deformation
    detectability).

    SNR predictor is the LINEAR signal/noise ratio by default (set xlabel accordingly);
    pass a dB array + xlabel='per-epoch SNR (dB)' for the log version.

    Caveat: r^2 is amplitude-insensitive (a pattern-correct but wrong-amplitude recovery
    still scores ~1), so it measures pattern match only and complements the per-parameter
    error.

    Parameters
    ----------
    snr_x, recon_r2 : per-epoch arrays (SNR predictor — linear ratio or dB; r^2 in [0,1]).
    recovered       : per-epoch boolean/0-1 array (loc-error < 1 multilook cell).
    bin_w           : bin width (x-units) for the binned-median curve. Default None ->
                      ~30 bins across the data range, so it adapts to linear ratio vs dB.
    min_count       : minimum epochs per bin for a median point to be drawn (sparse-tail
                      jitter guard).
    xlabel          : x-axis label; drives only the label, not the data.
    """
    snr = np.asarray(snr_x, float)
    r2 = np.asarray(recon_r2, float)
    rec = np.asarray(recovered).astype(bool)
    if bin_w is None:                                        # ~30 bins across the range
        bin_w = max((snr.max() - snr.min()) / 30.0, 1e-9)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(snr[rec], r2[rec], s=12, c='C0', alpha=0.45, label='recovered (<1 cell)')
    ax.scatter(snr[~rec], r2[~rec], s=12, c='C3', alpha=0.45, label='failed')
    edges = np.arange(snr.min(), snr.max() + bin_w, bin_w)
    cx, cy, clo, chi = binned_median(snr, r2, edges, min_count=min_count)
    if len(cx):
        ax.plot(cx, cy, 'k-', lw=2, label='median')
        ax.fill_between(cx, clo, chi, color='0.6', alpha=0.3, label='IQR')
    ax.set_ylim(-0.02, 1.02)                                  # r^2 is bounded in [0,1]
    ax.set_xlabel(xlabel)
    ax.set_ylabel('reconstruction r² (recovered vs truth LOS pattern)')
    ax.set_title(f'{_pretty(name)}: reconstruction r² vs signal/noise ratio')
    ax.grid(alpha=0.3); ax.legend(loc='center right')
    _save(fig, out_dir, f'{name}_{stem}')


def plot_deformation_recovery_curve(out_dir, name, peak_def_mm, noise_rms_mm, recovered,
                                    k50, k90, k50_ci=None, k90_ci=None, min_count=8,
                                    stem='deformation_recovery_curve'):
    """
    Logistic recovery-probability curve in DEFORMATION terms: P(recover) vs the
    deformation-to-noise ratio x = peak|LOS| / noise_rms. This is the same logistic that
    sets the SNR50/SNR90 detection lines on the deformation-vs-noise plot, re-expressed so
    its 50%-recovery point sits exactly at x = k50 and its 90% point at x = k90.

    The fitted sigmoid is P(x) = 1 / (1 + exp(-(x - k50)/sigma)) with sigma = (k90-k50)/ln9
    (so P(k50)=0.5 and P(k90)=0.9 by construction). Empirical binned recovery fractions are
    overlaid to show the fit quality; vertical markers (+ optional 95% CI spans from the
    bootstrap of k50/k90) flag the two detection thresholds.

    Parameters
    ----------
    peak_def_mm, noise_rms_mm : per-epoch arrays (mm).
    recovered                 : per-epoch boolean/0-1 array.
    k50, k90                  : detection-limit slopes (= deformation/noise at P=0.5, 0.9)
                                from detection_limit_lines().
    k50_ci, k90_ci            : optional (lo, hi) bootstrap 95% CIs for the markers.
    """
    peak = np.asarray(peak_def_mm, float)
    noise = np.asarray(noise_rms_mm, float)
    rec = np.asarray(recovered).astype(bool)
    good = noise > 0
    x = peak[good] / noise[good]                              # deformation-to-noise ratio
    r = rec[good]
    sigma = max((k90 - k50) / np.log(9.0), 1e-9)              # logistic width

    fig, ax = plt.subplots(figsize=(8, 4.8))
    # Empirical binned recovery fraction vs x.
    edges = np.linspace(x.min(), np.percentile(x, 99), 26)
    fx, fy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= min_count:
            fx.append(0.5 * (a + b)); fy.append(float(r[m].mean()))
    ax.plot(fx, fy, 'o', color='C0', alpha=0.7, label='empirical recovery fraction')
    # Fitted logistic curve.
    grid = np.linspace(0.0, np.percentile(x, 99), 300)
    ax.plot(grid, 1.0 / (1.0 + np.exp(-(grid - k50) / sigma)), 'k-', lw=2,
            label='logistic fit')
    # Threshold markers + 95% CI spans.
    for kk, cc, lab in [(k50, 'k', f'SNR50 (P=0.5): {k50:.1f}'),
                        (k90, 'C2', f'SNR90 (P=0.9): {k90:.1f}')]:
        ax.axvline(kk, ls='--', color=cc, lw=1.5, label=lab)
    for ci, cc in [(k50_ci, 'k'), (k90_ci, 'C2')]:
        if ci is not None and np.isfinite(ci[0]) and np.isfinite(ci[1]):
            ax.axvspan(ci[0], ci[1], color=cc, alpha=0.12)
    ax.axhline(0.5, ls=':', color='0.6'); ax.axhline(0.9, ls=':', color='0.6')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('peak |LOS| deformation / noise RMS')
    ax.set_ylabel('fraction of epochs recovered')
    ax.set_title(f'{_pretty(name)}: recovery probability vs deformation/noise')
    ax.grid(alpha=0.3); ax.legend(loc='lower right')
    _save(fig, out_dir, f'{name}_{stem}')


def plot_deformation_vs_noise(out_dir, name, noise_rms_mm, peak_def_mm, recovered,
                              k_det=None, rms_def_mm=None, det_lines=None,
                              stem='deformation_vs_noise'):
    """
    Detectability plot: surface deformation (y) against noise level (x), so one can
    read off "with this much noise, how much surface deformation do we need / can we
    expect to actually detect?".

    x = noise RMS (mm); y = peak |LOS| deformation (mm). Each point is one (per-epoch)
    inversion, coloured by whether PILA recovered the source. Detection limits are drawn
    as deformation = k * noise lines through the origin: recovered points sit above,
    failed points below.

    Parameters
    ----------
    noise_rms_mm, peak_def_mm : per-point arrays (mm).
    recovered                 : per-point boolean/0-1 array (True = source recovered).
    det_lines                 : optional list of dicts {label, k, k_lo, k_hi, color} —
                                detection-limit slopes (e.g. SNR50/SNR90) each drawn as
                                peak = k*noise with a shaded 95% CI band between k_lo and
                                k_hi (see detection_limit_lines() in snr_failure_analysis).
                                Preferred over the single k_det line.
    k_det                     : legacy single-slope fallback used only when det_lines is
                                not given (the SNR50 boundary).
    rms_def_mm                : optional secondary y series (scene RMS deformation, mm),
                                drawn faintly for cross-checking peak vs RMS.
    """
    noise = np.asarray(noise_rms_mm, float)
    peak = np.asarray(peak_def_mm, float)
    rec = np.asarray(recovered).astype(bool)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.scatter(noise[rec], peak[rec], s=14, c='C0', alpha=0.5,
               label='recovered (<1 cell)')
    ax.scatter(noise[~rec], peak[~rec], s=14, c='C3', alpha=0.5, label='failed')
    if rms_def_mm is not None:
        rms = np.asarray(rms_def_mm, float)
        ax.scatter(noise, rms, s=8, c='0.6', alpha=0.35, marker='x',
                   label='scene RMS deformation')
    xline = np.array([0.0, float(np.nanmax(noise)) if noise.size else 1.0])
    if det_lines:
        # Each detection-limit slope as peak = k*noise, with a shaded 95% CI band.
        for d in det_lines:
            k = d['k']; c = d.get('color', 'k')
            if k is None or not np.isfinite(k):
                continue
            ax.plot(xline, k * xline, '--', color=c, lw=2,
                    label=d.get('label', f'peak = {k:.1f}×noise'))
            klo, khi = d.get('k_lo'), d.get('k_hi')
            if klo is not None and khi is not None and np.isfinite(klo) and np.isfinite(khi):
                ax.fill_between(xline, klo * xline, khi * xline, color=c, alpha=0.15,
                                label='95% CI')
    elif k_det is not None and np.isfinite(k_det):
        ax.plot(xline, k_det * xline, 'k--', lw=2,
                label=f'min detectable: peak = {k_det:.1f}×noise')
    ax.set_xlabel('Noise RMS (mm)')
    ax.set_ylabel('Surface deformation, peak |LOS| (mm)')
    ax.set_title(f'{_pretty(name)}: detectable deformation vs noise')
    ax.grid(alpha=0.3)
    # Legend outside the axes (upper-right) so it never sits on the scatter/lines.
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    _save(fig, out_dir, f'{name}_{stem}')
