#!/usr/bin/env python
# Usage:       conda activate pila
#              python compare_uq_vs_fullscene.py \
#                  --uq-root comparison_out --saved-root saved/full_scene \
#                  --out comparison_out/uq_vs_fullscene \
#                  [--epoch last] [--configs SierraNegra_Sun69_InSAR_A Etna_Okada_TA44_A ...]
# Description: Compare each PILA strided-decimation UNCERTAINTY ensemble (already
#              aggregated by aggregate_uq.py) against the single FULL-SCENE PILA run for
#              the same volcano/track/source. The full-scene run inverts a 20x20
#              MULTILOOKED point set (~1000 cells); each UQ member inverts a disjoint
#              every-50th-pixel FULL-RESOLUTION subset (striding REPLACES multilooking).
#              The question answered here is: does the full-scene point estimate of every
#              source parameter fall inside the spread of the strided ensemble (i.e. is
#              the inversion robust to the spatial-sampling scheme)?
# Date:        2026-06-23
#
# Scientific notes (flagged):
#   * The full-scene estimate and the UQ members are inversions of the SAME scene but with
#     DIFFERENT spatial sampling (multilook=20 vs stride=50 full-res). Agreement therefore
#     tests robustness to sampling, NOT statistical bias — there is no "ground truth" here.
#   * Both go through the IDENTICAL inference path (aggregate_uq.run_inference): deterministic
#     latents (hard_z), decoder.rescale() -> native physical units (metres / m^3 / degrees).
#   * Compared at ONE common reference epoch (default last cumulative date), matching
#     aggregate_uq, so the comparison is over spatial sampling, not time.
#   * z-score = (full_scene - ensemble_median) / ensemble_std is a convenience flag, NOT a
#     formal significance test: the ensemble spread is SYSTEMATIC (fixed K disjoint subsets),
#     not a random bootstrap. Read |z|<~2 and "inside p5-p95" as "consistent with the ensemble".

import os
import sys
import csv
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Reuse the EXACT inference + geo helpers the UQ aggregator used, so the full-scene
# point estimate is computed identically to every ensemble member.
from aggregate_uq import (run_inference, resolve_epoch_index, enu_m_to_lonlat,
                          PHYSICS_ATTRS, LOC_PARAMS)


def find_fullscene_checkpoint(saved_root, config_name):
    """
    Newest full-scene model_best.pth + its config.json for a config, or (None, None).

    Layout: <saved_root>/<run_dir>/<ConfigName>/<timestamp>/models/model_best.pth.
    The config_name is matched case-insensitively against the <ConfigName> level so a
    save dir like 'etna_okada_ta44_A' still resolves the 'Etna_Okada_TA44_A' run inside.
    """
    pattern = os.path.join(saved_root, '**', 'model_best.pth')
    best = None
    for ckpt in glob.glob(pattern, recursive=True):
        # The directory two levels up from models/ is the <ConfigName>.
        cfg_name_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt))))
        if cfg_name_dir.lower() != config_name.lower():
            continue
        if best is None or os.path.getmtime(ckpt) > os.path.getmtime(best):
            best = ckpt
    if best is None:
        return None, None
    cfg_path = os.path.join(os.path.dirname(best), 'config.json')
    if not os.path.exists(cfg_path):
        return None, None
    return best, cfg_path


def read_uq_csv(uq_csv):
    """
    Parse a comparison_out/uq_<name>/uq_params.csv written by aggregate_uq.py.

    Returns (ref_epoch, attrs, member_matrix [n_members, n_par], summary{param: {stat: val}}).
    attrs are the physics params + ['source_lon', 'source_lat'] (the augmented set).
    """
    with open(uq_csv) as fh:
        rows = list(csv.reader(fh))
    ref_epoch = rows[0][1] if rows and rows[0] and rows[0][0].lstrip('# ').startswith('reference_epoch') else None
    # Find the member header (starts with 'offset') and the summary header (starts with 'statistic').
    hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == 'offset')
    sum_i = next(i for i, r in enumerate(rows) if r and r[0] == 'statistic')
    aug_attrs = rows[hdr_i][2:]                       # skip 'offset','n_points'
    member_vals = []
    for r in rows[hdr_i + 1:sum_i]:
        if not r or not r[0].strip():
            continue
        member_vals.append([float(x) for x in r[2:]])
    member_matrix = np.array(member_vals)
    summary = {}
    for r in rows[sum_i + 1:]:
        if not r or not r[0].strip():
            continue
        stat = r[0]
        for j, name in enumerate(aug_attrs):
            summary.setdefault(name, {})[stat] = float(r[1 + j])
    return ref_epoch, aug_attrs, member_matrix, summary


def fullscene_params(config_name, saved_root, epoch_arg):
    """
    Run identical inference on the full-scene checkpoint; return the augmented param dict
    (physics params + source_lon/source_lat in deg), the reference date, and n_points.
    """
    ckpt, cfg_path = find_fullscene_checkpoint(saved_root, config_name)
    if ckpt is None:
        return None
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    attrs, params_phys, dates, n_points = run_inference(cfg, ckpt)
    idx = resolve_epoch_index(epoch_arg, dates, params_phys)
    ref_date = dates[idx] if dates else f"idx{idx}"
    vals = {name: float(params_phys[idx, i]) for i, name in enumerate(attrs)}

    physics = cfg['arch']['args']['physics']
    xname, yname = LOC_PARAMS[physics]
    ins = cfg['arch']['args']['insar']
    lon, lat = enu_m_to_lonlat(vals[xname], vals[yname], ins['lat0'], ins['lon0'])
    aug = dict(vals)
    aug['source_lon'], aug['source_lat'] = lon, lat
    return dict(attrs=attrs, aug=aug, ref_date=ref_date, n_points=n_points,
                physics=physics, lat0=ins['lat0'], lon0=ins['lon0'], ckpt=ckpt)


UNIT = {'xcen': 'm', 'ycen': 'm', 'xoff': 'm', 'yoff': 'm', 'd': 'm', 'depth': 'm',
        'radius': 'm', 'length': 'm', 'width': 'm', 'opening': 'm', 'dV': 'm^3',
        'strike': 'deg', 'dip': 'deg', 'source_lon': 'deg', 'source_lat': 'deg'}


def main():
    ap = argparse.ArgumentParser(description="Compare PILA UQ ensembles vs full-scene runs")
    ap.add_argument('--uq-root', default=os.path.join(CURRENT_DIR, 'comparison_out'))
    ap.add_argument('--saved-root', default=os.path.join(CURRENT_DIR, 'saved', 'full_scene'))
    ap.add_argument('--out', default=os.path.join(CURRENT_DIR, 'comparison_out', 'uq_vs_fullscene'))
    ap.add_argument('--epoch', default='last')
    ap.add_argument('--configs', nargs='+', required=True,
                    help="config names, e.g. SierraNegra_Sun69_InSAR_A Etna_Okada_TA44_A")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    overview_rows = []   # one row per (config, param) for the combined CSV
    for ci, config_name in enumerate(args.configs, 1):
        print(f"\n[{ci}/{len(args.configs)}] === {config_name} ===")
        uq_csv = os.path.join(args.uq_root, f"uq_{config_name}", 'uq_params.csv')
        if not os.path.exists(uq_csv):
            print(f"  !! missing UQ csv {uq_csv} -- did aggregate_uq.py finish? skipping.")
            continue
        ref_epoch_uq, aug_attrs, member_matrix, summary = read_uq_csv(uq_csv)
        n_members = member_matrix.shape[0]

        fs = fullscene_params(config_name, args.saved_root, args.epoch)
        if fs is None:
            print(f"  !! no full-scene checkpoint for {config_name} under {args.saved_root}; skipping.")
            continue
        print(f"  full-scene: N={fs['n_points']} multilooked cells, epoch={fs['ref_date']}")
        print(f"  UQ ensemble: {n_members} members, epoch={ref_epoch_uq}")
        if ref_epoch_uq and fs['ref_date'] != ref_epoch_uq:
            print(f"  WARNING: reference epochs differ (full={fs['ref_date']} vs UQ={ref_epoch_uq})")

        # ---------------- Per-parameter comparison table ----------------
        out_cfg = os.path.join(args.out, config_name)
        os.makedirs(out_cfg, exist_ok=True)
        table = []
        for name in aug_attrs:
            full_v = fs['aug'][name]
            med = summary[name]['median']
            std = summary[name]['std']
            p5, p95 = summary[name]['p5'], summary[name]['p95']
            z = (full_v - med) / std if std > 0 else float('nan')
            inside = bool(p5 <= full_v <= p95)
            table.append((name, full_v, med, std, p5, p95, z, inside))
            overview_rows.append([config_name, name, UNIT.get(name, ''),
                                  f"{full_v:.6g}", f"{med:.6g}", f"{std:.6g}",
                                  f"{p5:.6g}", f"{p95:.6g}", f"{z:.3f}", inside])

        # Per-config CSV.
        with open(os.path.join(out_cfg, 'uq_vs_fullscene.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['# config', config_name, 'epoch', fs['ref_date'],
                        'n_members', n_members, 'full_scene_N', fs['n_points']])
            w.writerow(['param', 'unit', 'full_scene', 'uq_median', 'uq_std',
                        'uq_p5', 'uq_p95', 'z_score', 'inside_p5_p95'])
            for (name, full_v, med, std, p5, p95, z, inside) in table:
                w.writerow([name, UNIT.get(name, ''), f"{full_v:.6g}", f"{med:.6g}",
                            f"{std:.6g}", f"{p5:.6g}", f"{p95:.6g}", f"{z:.3f}", inside])

        # Console table.
        print(f"  {'param':<12}{'full_scene':>14}{'uq_median':>14}{'uq_std':>12}"
              f"{'z':>8}{'in90?':>7}")
        print("  " + "-" * 67)
        for (name, full_v, med, std, p5, p95, z, inside) in table:
            print(f"  {name:<12}{full_v:>14.5g}{med:>14.5g}{std:>12.4g}{z:>8.2f}"
                  f"{('yes' if inside else 'NO'):>7}")

        # ---------------- Figure: per-param violin (UQ) + full-scene marker ----------------
        phys_attrs = fs['attrs']                       # physics params only (exclude lon/lat)
        n_par = len(phys_attrs)
        fig, axes = plt.subplots(1, n_par, figsize=(3.0 * n_par, 4.2), constrained_layout=True)
        if n_par == 1:
            axes = [axes]
        for i, (name, ax) in enumerate(zip(phys_attrs, axes)):
            col = member_matrix[:, aug_attrs.index(name)]
            ax.violinplot(col, showmedians=True)
            jitter = np.linspace(-0.12, 0.12, col.size)
            ax.scatter(np.ones_like(col) + jitter, col, s=12, color='k', alpha=0.45, zorder=3)
            full_v = fs['aug'][name]
            ax.axhline(full_v, color='red', lw=2.0, zorder=4,
                       label='full-scene' if i == 0 else None)
            ax.set_title(name)
            ax.set_ylabel(f"{name} ({UNIT.get(name, '')})")
            ax.set_xticks([])
        fig.legend(loc='upper right', fontsize=9)
        fig.suptitle(f"PILA UQ ensemble vs full-scene: {config_name}\n"
                     f"{n_members} strided members (violin) vs full-scene (red), "
                     f"epoch {fs['ref_date']}", fontsize=11)
        fig.savefig(os.path.join(out_cfg, 'uq_vs_fullscene_params.png'), dpi=150)
        fig.savefig(os.path.join(out_cfg, 'uq_vs_fullscene_params.pdf'))
        plt.close(fig)

        # ---------------- Figure: source location (lon/lat) ----------------
        lons = member_matrix[:, aug_attrs.index('source_lon')]
        lats = member_matrix[:, aug_attrs.index('source_lat')]
        fig2, ax2 = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax2.scatter(lons, lats, s=42, color='steelblue', edgecolor='k', alpha=0.7,
                    zorder=3, label=f'UQ members (n={n_members})')
        ax2.scatter([np.median(lons)], [np.median(lats)], marker='*', s=320, color='gold',
                    edgecolor='k', zorder=4, label='UQ median')
        ax2.scatter([fs['aug']['source_lon']], [fs['aug']['source_lat']], marker='X', s=180,
                    color='red', edgecolor='k', zorder=5, label='full-scene')
        ax2.scatter([fs['lon0']], [fs['lat0']], marker='^', s=110, color='white',
                    edgecolor='k', zorder=4, label='ENU origin (config)')
        ax2.set_xlabel('Longitude (deg)')
        ax2.set_ylabel('Latitude (deg)')
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.set_title(f"Inverted source location: UQ ensemble vs full-scene\n{config_name}")
        ax2.legend(loc='best', fontsize=9)
        fig2.savefig(os.path.join(out_cfg, 'uq_vs_fullscene_location.png'), dpi=150)
        fig2.savefig(os.path.join(out_cfg, 'uq_vs_fullscene_location.pdf'))
        plt.close(fig2)
        print(f"  wrote {out_cfg}/")

    # ---------------- Combined overview CSV ----------------
    ov_path = os.path.join(args.out, 'uq_vs_fullscene_overview.csv')
    with open(ov_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['config', 'param', 'unit', 'full_scene', 'uq_median', 'uq_std',
                    'uq_p5', 'uq_p95', 'z_score', 'inside_p5_p95'])
        w.writerows(overview_rows)
    print(f"\nDone. Combined overview -> {ov_path}")
    # Quick agreement headline.
    inside = sum(1 for r in overview_rows if r[-1] is True)
    print(f"Agreement: {inside}/{len(overview_rows)} parameters have the full-scene "
          f"estimate inside the ensemble p5-p95.")


if __name__ == '__main__':
    main()
