#!/usr/bin/env python
# Usage:       python aggregate_uq.py --runs-glob 'saved/etna_mogi_ta44_a_uqstr5_off*' \
#                  --name Etna_Mogi_TA44_uqstr5 [--epoch last|peak|<idx>|YYYY-MM-DD]
# Description: Aggregate a PILA strided-decimation UNCERTAINTY-QUANTIFICATION ensemble.
#              Each ensemble member is an independent PILA run trained on a disjoint
#              every-n-th-pixel subset of the same InSAR scene (see generate_uq_jobs.py).
#              This script collects each member's inverted source parameters at a common
#              reference epoch, then reports the SPREAD across members (median / IQR / std /
#              p5-p95) as the uncertainty estimate, writes a CSV, and renders per-parameter
#              violin+strip plots plus a source-location (lon/lat) scatter.
# Date:        2026-06-23
#
# Scientific notes (flagged):
#   * PILA is per-epoch / amortized: each member yields a parameter TIME SERIES (one set
#     per acquisition date). We compare members at ONE common reference epoch (default the
#     last cumulative date) so the spread reflects spatial-sampling uncertainty, not time.
#   * The spread is SYSTEMATIC (fixed K = stride n, disjoint subsets), not a random
#     bootstrap; read it as sensitivity to which pixels are sampled.
#   * Lengths are reported in metres, dV in m^3, angles in degrees (the decoder's native
#     rescaled units). Source location is also given in lon/lat via the scene's ENU origin.

import os
import sys
import csv
import json
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import pyproj
except ImportError as exc:  # pragma: no cover
    raise ImportError("pyproj is required for aggregate_uq (lon/lat back-projection): " + str(exc))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import data_loader.data_loaders as module_data
from model import PHYS_VAE_SMPL

# Parameter names per physics type (MUST match each decoder's rescale() enumeration order).
PHYSICS_ATTRS = {
    'Mogi_LOS':  ['xcen', 'ycen', 'd', 'dV'],
    'Sun69_LOS': ['xcen', 'ycen', 'depth', 'radius', 'dV'],
    'Okada_LOS': ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening'],
}
# Which (x, y) parameter pair locates the source horizontally (metres in the ENU frame).
LOC_PARAMS = {
    'Mogi_LOS':  ('xcen', 'ycen'),
    'Sun69_LOS': ('xcen', 'ycen'),
    'Okada_LOS': ('xoff', 'yoff'),
}


def load_model(config, checkpoint_path):
    """Build PHYS_VAE_SMPL from a config dict and load model_best.pth weights (CPU, eval)."""
    model = PHYS_VAE_SMPL(config)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if ckpt.get('tau_r_values') is not None:
        model.dec.set_tau_r_from_checkpoint(ckpt['tau_r_values'])
    model.eval()
    return model


def run_inference(config, checkpoint_path):
    """
    Encode every InSAR date deterministically for one member.

    Returns
    -------
    attrs       : list[str]                  physical parameter names
    params_phys : ndarray [n_dates, n_par]   inverted params, physical units
    dates       : list[str]                  acquisition dates (epoch order, no shuffle)
    n_points    : int                        number of LOS cells this member inverted
    """
    physics = config['arch']['args']['physics']
    if physics not in PHYSICS_ATTRS:
        raise ValueError(f"Unsupported physics '{physics}' (known: {list(PHYSICS_ATTRS)})")
    attrs = PHYSICS_ATTRS[physics]

    dl = getattr(module_data, config['data_loader']['type_test'])(
        insar=config['data_loader']['data_dir_test'],
        batch_size=512, shuffle=False, validation_split=0.0, num_workers=0,
        with_const=config['data_loader']['args'].get('with_const', False))

    model = load_model(config, checkpoint_path)
    # PHYS_VAE_SMPL(config) auto-sets input_dim to this member's coherent-cell count
    # (N/stride) during construction, so read it back here as the member's point count.
    n_points = int(config['arch']['args'].get('input_dim', -1))
    data_key = config['trainer']['input_key']

    params_list, dates = [], []
    with torch.no_grad():
        for batch in dl:
            data = batch[data_key]
            tfeat = batch.get('time_feats', None)
            if data.dim() == 3:
                data = data.view(-1, data.size(-1))
            # Deterministic latents (KL disabled in these configs -> hard_z=True).
            latent_phy, _, _, _ = model(
                data, t=tfeat, inference=True, hard_z_phy=True, hard_z_aux=True)
            resc = model.physics_model.rescale(latent_phy)        # dict name -> [batch]
            params = torch.stack([resc[k] for k in attrs], dim=1)
            params_list.append(params.cpu().numpy())
            if 'date' in batch:
                dates += list(batch['date'])

    return attrs, np.concatenate(params_list), dates, n_points


def resolve_epoch_index(epoch_arg, dates, params_phys):
    """
    Map the --epoch argument to a row index into params_phys / dates.

    'last' (default) -> last cumulative epoch; 'peak' -> largest |param-displacement| proxy
    via the largest absolute dV / opening if present, else last; an int -> that index;
    otherwise treated as a 'YYYY-MM-DD' date string to match exactly.
    """
    n = params_phys.shape[0]
    if epoch_arg == 'last':
        return n - 1
    if epoch_arg == 'peak':
        # Proxy for peak deformation: the epoch with max |last column| (dV or opening).
        return int(np.argmax(np.abs(params_phys[:, -1])))
    try:
        idx = int(epoch_arg)
        return idx if idx >= 0 else n + idx
    except ValueError:
        pass
    if epoch_arg in dates:
        return dates.index(epoch_arg)
    raise ValueError(f"--epoch '{epoch_arg}' not found in dates and is not last/peak/int")


def enu_m_to_lonlat(east_m, north_m, lat0, lon0):
    """Inverse azimuthal-equidistant: local ENU metres about (lat0,lon0) -> lon/lat (deg)."""
    aeqd = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    transformer = pyproj.Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(east_m, north_m)
    return float(lon), float(lat)


def load_truth_params(truth_path, ref_date):
    """
    Load synthetic ground-truth source parameters for overlay on the UQ violins.

    The truth JSON (written by the synthetic-cube builder) holds:
      param_names       : list[str]           parameter order
      epochs            : list[str]           acquisition dates (YYYY-MM-DD)
      params_per_epoch  : list[list[float]]   [n_epochs, n_par] true values per date
      peak_epoch_index  : int                 index of peak deformation

    We pick the truth row matching the ensemble's reference epoch (ref_date) so the
    overlaid line is the TRUE value at exactly the epoch the violins summarise. If the
    reference date is absent from the truth's epoch list we fall back to the peak epoch
    and warn (the violins and the line would then refer to different epochs).

    Returns
    -------
    truth_vals : dict[str, float]   parameter name -> true value at the chosen epoch
    truth_date : str                the epoch (date string) the values were taken from
    """
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"--truth file not found: {truth_path}")
    with open(truth_path) as fh:
        t = json.load(fh)
    epochs = t.get('epochs', [])
    if ref_date in epochs:
        idx = epochs.index(ref_date)
    else:
        idx = int(t['peak_epoch_index'])
        peak_date = epochs[idx] if idx < len(epochs) else f"idx{idx}"
        print(f"  WARNING: reference epoch {ref_date} not in truth epochs; "
              f"overlaying truth at PEAK epoch {peak_date} instead (epoch mismatch).")
    truth_vals = dict(zip(t['param_names'], t['params_per_epoch'][idx]))
    truth_date = epochs[idx] if idx < len(epochs) else f"idx{idx}"
    return truth_vals, truth_date


def find_members(runs_glob):
    """
    Resolve a runs glob to a sorted list of (offset, run_dir, checkpoint, config) tuples.

    A member 'run_dir' contains models/model_best.pth and models/config.json (the standard
    PILA save layout: save_dir/<name>/<timestamp>/models/). If several timestamps exist for
    one save_dir, the most recently modified checkpoint is taken.
    """
    pattern = runs_glob if os.path.isabs(runs_glob) else os.path.join(CURRENT_DIR, runs_glob)
    members = {}
    for ckpt in glob.glob(os.path.join(pattern, '**', 'model_best.pth'), recursive=True):
        cfg_path = os.path.join(os.path.dirname(ckpt), 'config.json')
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        ins = cfg.get('arch', {}).get('args', {}).get('insar', {})
        offset = int(ins.get('offset', 0))
        # Keep the newest checkpoint per offset (in case a member was retrained).
        if offset not in members or os.path.getmtime(ckpt) > os.path.getmtime(members[offset][1]):
            members[offset] = (offset, ckpt, cfg)
    return [members[o] for o in sorted(members)]


def main():
    ap = argparse.ArgumentParser(description="Aggregate a PILA strided-decimation UQ ensemble")
    ap.add_argument('--runs-glob', required=True,
                    help="glob matching the ensemble member save_dirs, e.g. "
                         "'saved/etna_mogi_ta44_a_uqstr5_off*'")
    ap.add_argument('--name', required=True, help="label for outputs (comparison_out/uq_<name>/)")
    ap.add_argument('--epoch', default='last',
                    help="reference epoch: last (default) | peak | <int index> | YYYY-MM-DD")
    ap.add_argument('--out-root', default=os.path.join(CURRENT_DIR, 'comparison_out'),
                    help="root output directory")
    ap.add_argument('--truth', default=None,
                    help="optional synthetic ground-truth JSON (e.g. "
                         "synthetic/cubes/<cube>/<cube>_truth.json); when given, the TRUE "
                         "value at the reference epoch is overlaid on each violin")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, f"uq_{args.name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Finding ensemble members matching: {args.runs_glob}")
    members = find_members(args.runs_glob)
    if len(members) < 2:
        raise SystemExit(f"Found {len(members)} member(s); need >= 2 to quantify spread. "
                         f"Check --runs-glob and that the runs have finished.")
    print(f"  Found {len(members)} members (offsets {[m[0] for m in members]}).")

    print(f"[2/4] Running inference per member (reference epoch = '{args.epoch}') ...")
    attrs = None
    ref_date = None
    rows = []           # (offset, n_points, {param: value})
    lat0 = lon0 = None
    for offset, ckpt, cfg in members:
        a, params_phys, dates, n_points = run_inference(cfg, ckpt)
        if attrs is None:
            attrs = a
            physics = cfg['arch']['args']['physics']
            ins = cfg['arch']['args']['insar']
            lat0, lon0 = ins['lat0'], ins['lon0']
        idx = resolve_epoch_index(args.epoch, dates, params_phys)
        ref_date = dates[idx] if dates else f"idx{idx}"
        vals = {name: float(params_phys[idx, i]) for i, name in enumerate(attrs)}
        rows.append((offset, n_points, vals))
        print(f"  offset {offset}: N={n_points} cells, epoch[{idx}]={ref_date}")

    # ---------------- Assemble the per-member parameter matrix ----------------
    param_matrix = np.array([[r[2][name] for name in attrs] for r in rows])  # [n_members, n_par]

    # ---------------- Georeferenced source location (lon/lat) ----------------
    # xcen/ycen (or xoff/yoff) are ENU metres relative to the config peg (lat0,lon0), which
    # is frame-dependent and not interpretable on a map. We ALSO report absolute lon/lat per
    # member (no extra inference -- derived from the already-inverted ENU coords). The metre
    # values are kept too, since their SPREAD in metres is the intuitive precision measure.
    physics = members[0][2]['arch']['args']['physics']
    xname, yname = LOC_PARAMS[physics]
    east_m = param_matrix[:, attrs.index(xname)]
    north_m = param_matrix[:, attrs.index(yname)]
    member_lon = np.array([enu_m_to_lonlat(float(e), float(n), lat0, lon0)[0]
                           for e, n in zip(east_m, north_m)])
    member_lat = np.array([enu_m_to_lonlat(float(e), float(n), lat0, lon0)[1]
                           for e, n in zip(east_m, north_m)])

    # Augmented matrix/labels = physics params + derived lon/lat, for one combined summary.
    aug_attrs = attrs + ['source_lon', 'source_lat']
    aug_matrix = np.column_stack([param_matrix, member_lon, member_lat])

    # ---------------- Per-parameter spread summary ----------------
    print("[3/4] Computing spread across members and writing CSV ...")
    summary = {}
    for i, name in enumerate(aug_attrs):
        col = aug_matrix[:, i]
        summary[name] = dict(
            median=float(np.median(col)), mean=float(np.mean(col)), std=float(np.std(col, ddof=1)),
            p5=float(np.percentile(col, 5)), p25=float(np.percentile(col, 25)),
            p75=float(np.percentile(col, 75)), p95=float(np.percentile(col, 95)),
            min=float(col.min()), max=float(col.max()))

    # Per-member rows + a summary block, written to one tidy CSV.
    csv_path = os.path.join(out_dir, 'uq_params.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['# reference_epoch', ref_date])
        w.writerow(['offset', 'n_points'] + aug_attrs)
        for ri, (offset, n_points, vals) in enumerate(rows):
            loc = [f"{member_lon[ri]:.6f}", f"{member_lat[ri]:.6f}"]
            w.writerow([offset, n_points] + [f"{vals[name]:.6g}" for name in attrs] + loc)
        w.writerow([])
        w.writerow(['statistic'] + aug_attrs)
        for stat in ('median', 'mean', 'std', 'p5', 'p25', 'p75', 'p95', 'min', 'max'):
            w.writerow([stat] + [f"{summary[name][stat]:.6g}" for name in aug_attrs])
    print(f"  Wrote {csv_path}")

    # Console summary.
    print("\n" + "=" * 70)
    print(f"PILA UQ ensemble '{args.name}'  ({len(members)} members, epoch {ref_date})")
    print("=" * 70)
    print(f"{'param':<12}{'median':>14}{'std':>14}{'p5':>14}{'p95':>14}")
    print("-" * 68)
    for name in aug_attrs:
        s = summary[name]
        fmt = '>14.6f' if name in ('source_lon', 'source_lat') else '>14.4g'
        print(f"{name:<12}{s['median']:{fmt}}{s['std']:{fmt}}{s['p5']:{fmt}}{s['p95']:{fmt}}")

    # ---------------- Figures ----------------
    print("[4/4] Rendering figures ...")
    # Optional synthetic ground truth to overlay on the violins (matched to ref_date).
    truth_vals, truth_date = (None, None)
    if args.truth is not None:
        truth_path = args.truth if os.path.isabs(args.truth) else os.path.join(CURRENT_DIR, args.truth)
        truth_vals, truth_date = load_truth_params(truth_path, ref_date)
        print(f"  Overlaying synthetic truth from epoch {truth_date}: "
              + ", ".join(f"{k}={truth_vals[k]:.4g}" for k in attrs if k in truth_vals))

    # (a) Per-parameter violin + member strip.
    n_par = len(attrs)
    fig, axes = plt.subplots(1, n_par, figsize=(3.2 * n_par, 4.2), constrained_layout=True)
    if n_par == 1:
        axes = [axes]
    for i, (name, ax) in enumerate(zip(attrs, axes)):
        col = param_matrix[:, i]
        ax.violinplot(col, showmedians=True)
        ax.scatter(np.ones_like(col) + (np.random.rand(col.size) - 0.5) * 0.12, col,
                   s=18, color='k', alpha=0.6, zorder=3)
        # Overlay the synthetic TRUE value (red dashed) when a truth file was supplied.
        if truth_vals is not None and name in truth_vals:
            ax.axhline(truth_vals[name], color='red', lw=2.0, ls='--', zorder=4,
                       label='synthetic truth' if i == 0 else None)
        ax.set_title(name)
        ax.set_ylabel(f"{name} (m)" if name in ('xcen', 'ycen', 'xoff', 'yoff', 'd', 'depth',
                                                 'length', 'width')
                      else (f"{name} (m^3)" if name == 'dV'
                            else (f"{name} (deg)" if name in ('strike', 'dip')
                                  else (f"{name} (m)" if name == 'opening' else name))))
        ax.set_xticks([])
    if truth_vals is not None:
        # Single shared legend entry for the truth line (label set on the first axis only).
        axes[0].legend(loc='best', fontsize=9)
    truth_note = f"  |  red dashed = synthetic truth (epoch {truth_date})" if truth_vals is not None else ""
    fig.suptitle(f"PILA UQ ensemble: source-parameter spread\n"
                 f"{args.name} — {len(members)} members, epoch {ref_date}{truth_note}", fontsize=12)
    p_png = os.path.join(out_dir, 'uq_param_distributions.png')
    fig.savefig(p_png, dpi=150)
    fig.savefig(os.path.join(out_dir, 'uq_param_distributions.pdf'))
    plt.close(fig)
    print(f"  Wrote {p_png}")

    # (b) Source-location scatter (lon/lat) across members (reuse the values from above).
    lons, lats = member_lon, member_lat

    fig2, ax2 = plt.subplots(figsize=(6, 6), constrained_layout=True)
    offsets = [r[0] for r in rows]
    sc = ax2.scatter(lons, lats, c=offsets, cmap='viridis', s=60, edgecolor='k', zorder=3)
    ax2.scatter([np.median(lons)], [np.median(lats)], marker='*', s=320, color='red',
                edgecolor='k', label='median source', zorder=4)
    ax2.scatter([lon0], [lat0], marker='^', s=120, color='white', edgecolor='k',
                label='ENU origin (config)', zorder=4)
    ax2.set_xlabel('Longitude (deg)')
    ax2.set_ylabel('Latitude (deg)')
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_title(f"PILA UQ: inverted source location across {len(members)} members\n"
                  f"{args.name} (epoch {ref_date})")
    ax2.legend(loc='best', fontsize=9)
    fig2.colorbar(sc, ax=ax2, label='ensemble member (offset)')
    l_png = os.path.join(out_dir, 'uq_source_location.png')
    fig2.savefig(l_png, dpi=150)
    fig2.savefig(os.path.join(out_dir, 'uq_source_location.pdf'))
    plt.close(fig2)
    print(f"  Wrote {l_png}")

    # Report 1-sigma location scatter in km for a quick physical sense of the spread.
    enu_e_km = east_m / 1000.0
    enu_n_km = north_m / 1000.0
    print(f"\nSource-location scatter: std(E)={np.std(enu_e_km, ddof=1):.3f} km, "
          f"std(N)={np.std(enu_n_km, ddof=1):.3f} km "
          f"(median lon/lat = {np.median(lons):.5f}, {np.median(lats):.5f})")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == '__main__':
    main()
