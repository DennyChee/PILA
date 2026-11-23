#!/usr/bin/env python
# Usage:       from compare.figures import param_recovery_figure, recon_maps_figure
# Description: Standard comparison figures, made automatically for every scenario by compare.py:
#                * parameter recovery (vbICA layout): one panel per (source, parameter), date on x;
#                  truth = black line, each method = coloured markers with its posterior band if it has one;
#                  fitted slots are matched to true sources per date with the SAME rule as the scorer;
#                * reconstruction maps at a few dates: data, clean signal (synthetic), each method's recon and
#                  residual (recon - data), shared colour scale per row.
# Conventions: x / y / depth / length / width in km, dV in 1e6 m^3, opening in m, angles in deg, LOS in mm.
#              Displacement colormap RdBu_r with a colorbar; PNG at 150 dpi.
# Date:        2026-11-23
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compare.score import align_to_scenario, match_slots  # noqa: E402
from scenarios.schema import PARAM_NAMES  # noqa: E402

METHOD_COLOURS = {'pila': '#1f77b4', 'mcmc': '#d62728', 'vbica': '#2ca02c', 'supervised': '#9467bd'}
DISPLAY = {   # parameter -> (label, scale applied to the SI value)
    'xcen': ('x (km)', 1e-3), 'ycen': ('y (km)', 1e-3), 'xoff': ('x (km)', 1e-3), 'yoff': ('y (km)', 1e-3),
    'd': ('depth (km)', 1e-3), 'depth': ('depth (km)', 1e-3), 'radius': ('radius (km)', 1e-3),
    'length': ('length (km)', 1e-3), 'width': ('width (km)', 1e-3), 'dV': ('dV (1e6 m$^3$)', 1e-6),
    'opening': ('opening (m)', 1.0), 'strike': ('strike (deg)', 1.0), 'dip': ('dip (deg)', 1.0),
}


def _dates_axis(dates):
    return np.array([np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}") for d in dates])


def matched_params(truth, run_output, threshold_mm):
    """
    Per date, the fitted slot (and band) matched to every true source with the scorer's rule.

    Output: dict source index -> dict(params [T, P], lo [T, P], hi [T, P]) (NaN where unmatched / model differs).
    """
    fit = run_output['params']['params']
    models = run_output['params']['models']
    lo, hi = run_output['params'].get('params_lo'), run_output['params'].get('params_hi')
    n_dates = len(truth['dates'])
    out = {}
    for k, source in enumerate(truth['sources']):
        n_par = len(source['param_names'])
        out[k] = {key: np.full((n_dates, n_par), np.nan) for key in ('params', 'lo', 'hi')}
    for t in range(n_dates):
        true_xy = np.array([source['params'][t][:2] for source in truth['sources']], dtype=float).reshape(-1, 2)
        if not np.all(np.isfinite(fit[t, :, :2])):
            continue
        peaks = np.array([s.get('peak_los_mm', [np.nan] * n_dates)[t] for s in truth['sources']], dtype=float)
        for src, slot in match_slots(true_xy, fit[t, :, :2], peaks >= threshold_mm):
            if models[slot] != truth['sources'][src]['model']:
                continue
            n_par = len(PARAM_NAMES[models[slot]])
            out[src]['params'][t] = fit[t, slot, :n_par]
            if lo is not None and hi is not None:
                out[src]['lo'][t], out[src]['hi'][t] = lo[t, slot, :n_par], hi[t, slot, :n_par]
    return out


def param_recovery_figure(truth, runs, threshold_mm, out_path):
    """
    Truth vs every method, per source and parameter, over the dates.

    Inputs:  truth (pila-truth/1); runs {label: run_output}; threshold_mm (activity threshold, for matching);
             out_path (.png).
    """
    sources = truth['sources']
    if not sources:
        return None
    n_rows = max(len(s['param_names']) for s in sources)
    fig, axes = plt.subplots(n_rows, len(sources), figsize=(5.2 * len(sources), 2.1 * n_rows), sharex=True, squeeze=False)
    dates = _dates_axis(truth['dates'])
    matched = {label: matched_params(truth, run, threshold_mm) for label, run in runs.items()}
    for k, source in enumerate(sources):
        truth_params = np.asarray(source['params'], dtype=float)
        peaks = np.asarray(source.get('peak_los_mm', [np.nan] * len(dates)), dtype=float)
        inactive = peaks < threshold_mm
        for j, name in enumerate(source['param_names']):
            ax = axes[j, k]
            label, scale = DISPLAY.get(name, (name, 1.0))
            ax.plot(dates, truth_params[:, j] * scale, 'k-', lw=1.6, label='truth')
            if inactive.any():
                ax.plot(dates[inactive], truth_params[inactive, j] * scale, 'o', mfc='none', mec='0.5', ms=4,
                        label='truth (below detection)')
            for run_label, per_source in matched.items():
                method = run_label.split(':')[0]
                colour = METHOD_COLOURS.get(method, None)
                values = per_source[k]
                if np.isfinite(values['lo'][:, j]).any():
                    ax.fill_between(dates, values['lo'][:, j] * scale, values['hi'][:, j] * scale, color=colour,
                                    alpha=0.18, lw=0)
                ax.plot(dates, values['params'][:, j] * scale, 'o', ms=3.5, color=colour, label=run_label)
            ax.set_ylabel(label)
            if j == 0:
                ax.set_title(f"source {k + 1} ({source['model']})")
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel('Date')
        ax.tick_params(axis='x', rotation=30)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle(f"Parameter recovery: {truth['scenario_id']}", y=0.995)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=min(len(labels), 6),
               fontsize=8, frameon=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def recon_maps_figure(scenario, runs, out_path, n_dates_shown=3):
    """
    Data / clean / each method's reconstruction and residual at a few evenly spaced dates.

    Inputs:  scenario (read_scenario dict); runs {label: run_output}; out_path (.png); n_dates_shown.
    """
    track = scenario['tracks'][0]
    data_mm = scenario[f'los_{track}_m'] * 1e3
    clean = scenario.get(f'clean_los_{track}_m')
    n_dates = data_mm.shape[0]
    picks = np.unique(np.linspace(0, n_dates - 1, n_dates_shown).round().astype(int))
    columns = [('data', data_mm)] + ([('clean', clean * 1e3)] if clean is not None else [])
    for run_label, run in runs.items():
        cube = run['recon'].get(f'los_{track}_m')
        if cube is None:
            continue
        recon_mm = align_to_scenario(cube, run['recon']['x_m'], run['recon']['y_m'], scenario) * 1e3
        columns += [(f'{run_label}\nrecon', recon_mm), (f'{run_label}\nresidual', recon_mm - data_mm)]
    extent = [scenario['x_m'].min() / 1e3, scenario['x_m'].max() / 1e3, scenario['y_m'].min() / 1e3,
              scenario['y_m'].max() / 1e3]
    origin = 'lower' if scenario['y_m'][1] > scenario['y_m'][0] else 'upper'
    fig, axes = plt.subplots(len(picks), len(columns), figsize=(2.6 * len(columns), 2.6 * len(picks)), squeeze=False)
    for r, t in enumerate(picks):
        vmax = np.nanmax(np.abs(data_mm[t])) or 1.0
        for c, (title, cube) in enumerate(columns):
            ax = axes[r, c]
            image = ax.imshow(cube[t], cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent, origin=origin)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{scenario['dates'][t]}\nNorth (km)", fontsize=8)
            if r == len(picks) - 1:
                ax.set_xlabel('East (km)', fontsize=8)
        cbar = fig.colorbar(image, ax=axes[r, :].tolist(), shrink=0.8, pad=0.01)
        cbar.set_label('LOS (mm, + toward satellite)', fontsize=8)
    fig.suptitle('Reconstructions (shared colour scale per date = data max |LOS|)', fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path
