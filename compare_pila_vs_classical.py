#!/usr/bin/env python
# Usage:       conda activate pila
#              python compare_pila_vs_classical.py \
#                  [--exported exported] [--classic classic_out] \
#                  [--saved saved] [--out comparison_out]
# Description: Benchmark PILA (deep-learning) vs the classical analytical inversion
#              (Simulated Annealing best-fit + Bayesian MCMC posterior) on the SAME
#              multilooked LOS points. All three parameter sets are evaluated with
#              PILA's OWN physics forward models (Mogi / Sun69 / Okada), so the only
#              thing that differs is the inverted parameters. Outputs a CSV table,
#              an RMSE bar chart, and per-config source-location overlay maps.
# Date:        2026-06-18
#
# Scientific conventions (READ — these affect interpretation):
#   * LOS sign: positive = motion TOWARD the satellite (MintPy). Predicted LOS is the
#     forward physics output projected onto the per-point LOS unit vectors (mm).
#   * Both PILA and classical models are compared DIRECTLY to the observed LOS (mm) with
#     no demeaning: PILA's global-z-score standardization is affine and cancels in the
#     residual, so it too fits the absolute physics LOS — the comparison is consistent.
#   * Forward models assume a flat half-space (no topographic correction), the same
#     assumption on both sides.
#   * Parameter units: positions/depth/length/width/radius in metres, strike/dip in
#     degrees, opening in metres, volume (dV) in m^3 — the units PILA's *.run() expects.

import argparse
import glob
import os
import re

import numpy as np
import h5py
import torch
import pyproj
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

from physics.mogi.mogi import Mogi
from physics.okada.okada_dike import OkadaDike
from physics.sun69.sun69 import Sun69

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# run() keyword order per model (matches the classical param vector order).
RUN_KEYS = {
    'mogi':  ['xcen', 'ycen', 'd', 'dV'],
    'sun69': ['xcen', 'ycen', 'depth', 'radius', 'dV'],
    'okada': ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening'],
}


# ----------------------------------------------------------------------------
def evaluate_los_mm(model, params_si, x_m, y_m, losE, losN, losU):
    """
    Predicted LOS (mm) at the observation points for one parameter set.

    params_si : dict keyed by RUN_KEYS[model], SI units (m / deg / m / m^3).
    x_m, y_m  : [N] point coordinates, metres.
    losE/N/U  : [N] per-point LOS unit-vector components.
    Returns   : [N] predicted LOS in mm (positive = toward satellite).
    """
    xt = torch.tensor(np.asarray(x_m, dtype=np.float64))
    yt = torch.tensor(np.asarray(y_m, dtype=np.float64))
    if model == 'mogi':
        fwd = Mogi(xt, yt)
    elif model == 'okada':
        fwd = OkadaDike(xt, yt)
    elif model == 'sun69':
        fwd = Sun69(xt, yt)
    else:
        raise ValueError(f"Unknown model {model}")

    kwargs = {k: torch.tensor([float(params_si[k])], dtype=torch.float64)
              for k in RUN_KEYS[model]}
    enu = fwd.run(**kwargs)                       # [1, 3N] mm
    enu = enu.detach().cpu().numpy().reshape(-1)
    n = x_m.size
    uE, uN, uU = enu[:n], enu[n:2 * n], enu[2 * n:]
    return losE * uE + losN * uN + losU * uU      # [N] mm


def r2_rmse(pred_mm, obs_mm):
    """R^2 and RMSE (mm) of a prediction against the observed LOS (mm)."""
    rmse = float(np.sqrt(np.mean((pred_mm - obs_mm) ** 2)))
    ss_res = np.sum((obs_mm - pred_mm) ** 2)
    ss_tot = np.sum((obs_mm - np.mean(obs_mm)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return r2, rmse


# ----------------------------------------------------------------------------
def parse_pila_params(params_txt, model, lat0, lon0):
    """
    Parse a PILA figures/*_params.txt into an SI param dict for RUN_KEYS[model].

    Horizontal position is given as source_lon/source_lat, converted to local-ENU
    metres about (lat0, lon0) with the same aeqd projection PILA uses.
    Returns (params_si, source_lon, source_lat) or (None, None, None) if unreadable.
    """
    # Parse ONLY the header block (above the "# prior bounds" marker). The header
    # reports each parameter once with explicit, consistent units (km / m^3 / m / deg),
    # which is what the unit conversions below assume. The bounds section that follows
    # repeats the same keys but in SI metres for Mogi (km for Okada/Sun69) — a writer
    # inconsistency. Letting those lines overwrite the header values silently inflated
    # the Mogi depth by 1000x (2.33 km -> 2335 km), collapsing the Mogi fit. Stop at the
    # marker so only the unambiguous header values are used.
    txt = {}
    with open(params_txt, 'r') as fh:
        for line in fh:
            if line.lstrip().startswith('# prior bounds'):
                break
            m = re.match(r'\s*([A-Za-z_]+)\s*=\s*([-+0-9.eE]+)', line)
            if m:
                txt[m.group(1)] = float(m.group(2))
    if 'source_lon' not in txt or 'source_lat' not in txt:
        return None, None, None
    source_lon, source_lat = txt['source_lon'], txt['source_lat']

    aeqd = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    to_enu = pyproj.Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    x_off_m, y_off_m = to_enu.transform(source_lon, source_lat)

    # params.txt reports: depth/length/width/radius in km, dV in m^3, opening in m,
    # strike/dip in deg, d (Mogi depth) in km.
    p = {}
    if model == 'mogi':
        p = {'xcen': x_off_m, 'ycen': y_off_m,
             'd': txt['d'] * 1000.0, 'dV': txt['dV']}
    elif model == 'sun69':
        p = {'xcen': x_off_m, 'ycen': y_off_m,
             'depth': txt['depth'] * 1000.0, 'radius': txt['radius'] * 1000.0,
             'dV': txt['dV']}
    elif model == 'okada':
        p = {'xoff': x_off_m, 'yoff': y_off_m, 'depth': txt['depth'] * 1000.0,
             'strike': txt['strike'], 'dip': txt['dip'],
             'length': txt['length'] * 1000.0, 'width': txt['width'] * 1000.0,
             'opening': txt['opening']}
    return p, source_lon, source_lat


def find_pila_params_txt(config_name, saved_root):
    """Most recent PILA figures/*_params.txt for this config, or None."""
    pattern = os.path.join(saved_root, '*', config_name, '*', 'figures', '*_params.txt')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def vec_to_si(model, vec, names):
    """Classical param vector (SI) -> RUN_KEYS dict (positional: MATLAB names share run order)."""
    assert len(vec) == len(RUN_KEYS[model]), \
        f"{model}: {len(vec)} params != {len(RUN_KEYS[model])} run keys ({names})"
    return {k: float(v) for k, v in zip(RUN_KEYS[model], vec)}


def load_classical(classic_dir, config_name, model):
    """
    Load the classical result .mat for a config; return dict or None.

    iclassic_insar saves with MATLAB '-v7.3' (HDF5), which scipy.loadmat cannot read,
    so use h5py. v7.3 stores arrays column-major (transposed vs MATLAB), but these are
    1-D parameter vectors, so .reshape(-1) recovers them in param order. The param order
    is fixed by the model (RUN_KEYS), so config.param_names is not needed here.
    """
    hits = glob.glob(os.path.join(classic_dir, config_name, f"{model}_{config_name}_*_result.mat"))
    if not hits:
        hits = glob.glob(os.path.join(classic_dir, config_name, "*_result.mat"))
    if not hits:
        return None
    with h5py.File(sorted(hits)[-1], 'r') as R:
        return {
            'm_best_sa': np.array(R['m_best_sa']).reshape(-1),
            'post_med': np.array(R['post_med']).reshape(-1),
            'm_best': np.array(R['m_best']).reshape(-1),
            'param_names': RUN_KEYS[model],
            'source_lonlat': np.array(R['source_lonlat']).reshape(-1),
        }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Compare PILA vs classical (SA/Bayesian) inversions.')
    ap.add_argument('--exported', default=os.path.join(CURRENT_DIR, 'exported'))
    ap.add_argument('--classic', default=os.path.join(CURRENT_DIR, 'classic_out'))
    ap.add_argument('--saved', default=os.path.join(CURRENT_DIR, 'saved'))
    ap.add_argument('--out', default=os.path.join(CURRENT_DIR, 'comparison_out'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    exported = sorted(glob.glob(os.path.join(args.exported, '*.mat')))
    if not exported:
        raise FileNotFoundError(f"No exported .mat files in {args.exported}")

    rows = []
    header = ['config', 'volcano', 'model', 'track', 'epoch', 'N',
              'method', 'rmse_mm', 'r2', 'source_lon', 'source_lat']
    print(f"[0/{len(exported)}] Comparing {len(exported)} config(s)...")

    for i, exp_path in enumerate(exported, 1):
        config_name = os.path.splitext(os.path.basename(exp_path))[0]
        E = loadmat(exp_path)
        model = str(E['model'][0])
        volcano = str(E['volcano'][0])
        track = str(E['track'][0]) if E['track'].size else ''
        epoch = str(E['epoch_yyyymmdd'][0])
        lat0, lon0 = float(E['lat0'][0, 0]), float(E['lon0'][0, 0])
        x_m = np.asarray(E['x_east_m']).reshape(-1)
        y_m = np.asarray(E['y_north_m']).reshape(-1)
        obs_mm = np.asarray(E['los_obs_m']).reshape(-1) * 1000.0
        losE = np.asarray(E['losE']).reshape(-1)
        losN = np.asarray(E['losN']).reshape(-1)
        losU = np.asarray(E['losU']).reshape(-1)
        N = x_m.size
        print(f"\n[{i}/{len(exported)}] {config_name}  (model={model}, N={N})")

        methods = {}   # label -> (params_si, source_lon, source_lat)

        # --- PILA ---
        ptxt = find_pila_params_txt(config_name, args.saved)
        if ptxt:
            p_pila, slon, slat = parse_pila_params(ptxt, model, lat0, lon0)
            if p_pila is not None:
                methods['PILA'] = (p_pila, slon, slat)
        else:
            print("  (no PILA params.txt found)")

        # --- Classical (SA + Bayesian) ---
        C = load_classical(args.classic, config_name, model)
        if C is not None:
            aeqd = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
            to_ll = pyproj.Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
            for label, key in [('SA', 'm_best_sa'), ('Bayesian', 'post_med')]:
                p_si = vec_to_si(model, C[key], C['param_names'])
                xo = p_si.get('xcen', p_si.get('xoff'))
                yo = p_si.get('ycen', p_si.get('yoff'))
                lon_s, lat_s = to_ll.transform(xo, yo)
                methods[label] = (p_si, lon_s, lat_s)
        else:
            print("  (no classical result .mat yet — run/await the PBS jobs)")

        # --- Evaluate every available method on the identical points ---
        for label, (p_si, slon, slat) in methods.items():
            pred = evaluate_los_mm(model, p_si, x_m, y_m, losE, losN, losU)
            r2, rmse = r2_rmse(pred, obs_mm)
            rows.append([config_name, volcano, model, track, epoch, N,
                         label, f"{rmse:.3f}", f"{r2:.4f}",
                         f"{slon:.5f}", f"{slat:.5f}"])
            print(f"    {label:9s}  RMSE={rmse:7.2f} mm   R2={r2:6.3f}")

        # --- Per-config source-location overlay map ---
        if methods:
            _plot_source_overlay(args.out, config_name, model, epoch,
                                 E, obs_mm, methods)

    # --- Write CSV table ---
    csv_path = os.path.join(args.out, 'comparison_table.csv')
    with open(csv_path, 'w') as fh:
        fh.write(','.join(header) + '\n')
        for r in rows:
            fh.write(','.join(str(c) for c in r) + '\n')
    print(f"\nWrote {csv_path}  ({len(rows)} rows)")

    # --- RMSE bar chart across configs ---
    _plot_rmse_bars(args.out, rows)
    print(f"Done. Comparison outputs in {args.out}/")


# ----------------------------------------------------------------------------
def _plot_source_overlay(out_dir, config_name, model, epoch, E, obs_mm, methods):
    """Observed LOS scatter (lon/lat) with each method's inverted source location."""
    lon = np.asarray(E['lon']).reshape(-1)
    lat = np.asarray(E['lat']).reshape(-1)
    vmax = np.nanmax(np.abs(obs_mm)) or 1.0
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(lon, lat, c=obs_mm, s=10, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Observed LOS (mm)')
    markers = {'PILA': ('*', 'k', 260), 'SA': ('^', 'lime', 130), 'Bayesian': ('o', 'magenta', 110)}
    for label, (_, slon, slat) in methods.items():
        mk, col, sz = markers.get(label, ('x', 'k', 100))
        ax.scatter([slon], [slat], marker=mk, s=sz, edgecolor='k',
                   facecolor=col, label=f'{label} source', zorder=5)
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(f'{config_name}  |  {model}  |  epoch {epoch}\nInverted source locations')
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    base = os.path.join(out_dir, f'{config_name}_source_overlay')
    fig.savefig(base + '.png', dpi=150)
    fig.savefig(base + '.pdf')
    plt.close(fig)


def _plot_rmse_bars(out_dir, rows):
    """Grouped bar chart of RMSE (mm) per config for PILA / SA / Bayesian."""
    if not rows:
        return
    configs = sorted({r[0] for r in rows})
    method_order = ['PILA', 'SA', 'Bayesian']
    rmse = {m: {c: np.nan for c in configs} for m in method_order}
    for r in rows:
        cfg, method, val = r[0], r[6], float(r[7])
        if method in rmse:
            rmse[method][cfg] = val

    x = np.arange(len(configs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(configs)), 6))
    colors = {'PILA': '#1f77b4', 'SA': '#2ca02c', 'Bayesian': '#d62728'}
    for j, m in enumerate(method_order):
        vals = [rmse[m][c] for c in configs]
        ax.bar(x + (j - 1) * w, vals, w, label=m, color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=90, fontsize=7)
    ax.set_ylabel('RMSE (mm)')
    ax.set_title('PILA vs classical (SA / Bayesian) — LOS misfit per config')
    ax.legend()
    plt.tight_layout()
    base = os.path.join(out_dir, 'rmse_comparison')
    fig.savefig(base + '.png', dpi=150)
    fig.savefig(base + '.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
