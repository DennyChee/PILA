#!/usr/bin/env python
# Usage:       conda activate pila
#              python compare_mcmc_vs_uq.py \
#                  [--mcmc-csv comparison_out/inversion_params_pila_vs_mcmc.csv] \
#                  [--uq-root comparison_out] [--out comparison_out/uq_vs_mcmc]
# Description: Compare the classical MCMC (Bayesian posterior) inverted source parameters
#              against the PILA strided-decimation UNCERTAINTY ensemble (n=50 members) for
#              the Okada configs that have BOTH. The MCMC point estimate is checked against
#              the ensemble spread: is it inside the ensemble p5-p95, and what is its
#              z-score = (MCMC - ensemble_median)/ensemble_std?
# Date:        2026-06-24
#
# Scientific notes (flagged):
#   * MCMC params come from inversion_params_pila_vs_mcmc.csv (classical SA+Bayesian pipeline,
#     the 'MCMC' rows = posterior median). The PILA UQ ensemble is 50 disjoint stride-50
#     full-res subsets (see aggregate_uq.py). Both invert the SAME scene with DIFFERENT methods
#     AND different spatial sampling, so agreement tests cross-method robustness, not bias.
#   * Unit harmonization: MCMC csv stores depth/length/width in KM and lon/lat in deg; the UQ
#     ensemble stores them in METRES + source_lon/source_lat in deg. We convert MCMC km->m so
#     both are in the UQ's native units before comparing.
#   * strike is CIRCULAR (0-360, with a ~180 deg conjugate-plane ambiguity for a dislocation):
#     a large strike z-score often just means the two methods picked conjugate planes, not a
#     real disagreement. Flagged in the output.
#   * Okada length/width frequently RAIL at the prior bounds (poorly constrained from a single
#     LOS track) -- read those spreads as "unconstrained", not as a tight uncertainty.

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from compare_uq_vs_fullscene import read_uq_csv   # reuse the exact UQ-CSV parser

# Map each UQ ensemble parameter -> (MCMC csv column, scale to UQ units). Location is compared
# via source_lon/source_lat (the UQ also reports xoff/yoff in ENU m, but MCMC only has lon/lat).
OKADA_MAP = [
    ('source_lon', 'lon',        1.0,    'deg'),
    ('source_lat', 'lat',        1.0,    'deg'),
    ('depth',      'depth_km',   1000.0, 'm'),
    ('strike',     'strike_deg', 1.0,    'deg'),
    ('dip',        'dip_deg',    1.0,    'deg'),
    ('length',     'length_km',  1000.0, 'm'),
    ('width',      'width_km',   1000.0, 'm'),
    ('opening',    'opening_m',  1.0,    'm'),
]

# Configs that have BOTH an MCMC row and an n=50 UQ ensemble (Okada only -- Sun69/SierraNegra
# was not run through the classical MCMC pipeline, so it has no reference here).
CONFIGS = [
    'Etna_Okada_TA44_A',
    'Etna_Okada_TD124_A',
    'Nyiragongo_Okada_TA174_A',
    'Nyiragongo_Okada_TD21_A',
]


def read_method(mcmc_csv, method):
    """Return {config: {column: float}} for the rows of a given method ('MCMC' or 'PILA')."""
    out = {}
    with open(mcmc_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get('method', '').strip().upper() != method.upper():
                continue
            vals = {}
            for k, v in row.items():
                if k in ('config', 'model', 'method'):
                    continue
                v = (v or '').strip()
                if v != '':
                    vals[k] = float(v)
            out[row['config']] = vals
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare classical MCMC params vs PILA n=50 UQ ensemble")
    ap.add_argument('--mcmc-csv', default=os.path.join(CURRENT_DIR, 'comparison_out',
                                                       'inversion_params_pila_vs_mcmc.csv'))
    ap.add_argument('--uq-root', default=os.path.join(CURRENT_DIR, 'comparison_out'))
    ap.add_argument('--out', default=os.path.join(CURRENT_DIR, 'comparison_out', 'uq_vs_mcmc'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mcmc = read_method(args.mcmc_csv, 'MCMC')
    pila = read_method(args.mcmc_csv, 'PILA')   # original full_scene (ML=20) PILA point estimate
    overview = []   # (config, param, unit, pila, mcmc, uq_median, uq_std, p5, p95, z_mcmc, inside_mcmc, circular)

    for ci, cfg in enumerate(CONFIGS, 1):
        print(f"\n[{ci}/{len(CONFIGS)}] === {cfg} ===")
        uq_csv = os.path.join(args.uq_root, f"uq_{cfg}", 'uq_params.csv')
        if not os.path.exists(uq_csv):
            print(f"  !! missing UQ csv {uq_csv}; skipping."); continue
        if cfg not in mcmc:
            print(f"  !! no MCMC row for {cfg}; skipping."); continue
        ref_epoch, aug_attrs, member_matrix, summary = read_uq_csv(uq_csv)
        m = mcmc[cfg]
        p = pila.get(cfg, {})   # original full_scene (ML=20) PILA point estimate

        out_cfg = os.path.join(args.out, cfg)
        os.makedirs(out_cfg, exist_ok=True)
        table = []
        for uq_name, mcol, scale, unit in OKADA_MAP:
            if mcol not in m or uq_name not in summary:
                continue
            mcmc_v = m[mcol] * scale
            pila_v = p[mcol] * scale if mcol in p else float('nan')
            med = summary[uq_name]['median']
            std = summary[uq_name]['std']
            p5, p95 = summary[uq_name]['p5'], summary[uq_name]['p95']
            circular = (uq_name == 'strike')
            z = (mcmc_v - med) / std if std > 0 else float('nan')
            inside = bool(p5 <= mcmc_v <= p95)
            zp = (pila_v - med) / std if std > 0 else float('nan')
            inside_p = bool(p5 <= pila_v <= p95) if pila_v == pila_v else False  # NaN-safe
            table.append((uq_name, unit, pila_v, mcmc_v, med, std, p5, p95, z, inside, zp, inside_p, circular))
            overview.append([cfg, uq_name, unit, f"{pila_v:.6g}", f"{mcmc_v:.6g}", f"{med:.6g}",
                             f"{std:.6g}", f"{p5:.6g}", f"{p95:.6g}", f"{z:.3f}", inside, circular])

        # Per-config CSV.
        with open(os.path.join(out_cfg, 'mcmc_vs_uq.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['# config', cfg, 'uq_epoch', ref_epoch, 'n_members', member_matrix.shape[0]])
            w.writerow(['param', 'unit', 'pila', 'mcmc', 'uq_median', 'uq_std', 'uq_p5', 'uq_p95',
                        'z_mcmc', 'mcmc_inside_p5_p95', 'z_pila', 'pila_inside_p5_p95', 'circular'])
            for (name, unit, pv, mc, med, std, p5, p95, z, inside, zp, inside_p, circ) in table:
                w.writerow([name, unit, f"{pv:.6g}", f"{mc:.6g}", f"{med:.6g}", f"{std:.6g}",
                            f"{p5:.6g}", f"{p95:.6g}", f"{z:.3f}", inside, f"{zp:.3f}", inside_p, circ])

        # Console table.
        print(f"  {'param':<11}{'PILA':>13}{'MCMC':>13}{'uq_median':>13}{'uq_std':>11}"
              f"{'z_pila':>8}{'z_mcmc':>8}")
        print("  " + "-" * 75)
        for (name, unit, pv, mc, med, std, p5, p95, z, inside, zp, inside_p, circ) in table:
            tag = ' (circ)' if circ else ''
            print(f"  {name:<11}{pv:>13.5g}{mc:>13.5g}{med:>13.5g}{std:>11.4g}"
                  f"{zp:>8.2f}{z:>8.2f}{tag}")

        # Figure: per-param violin (UQ n=50) + original PILA (blue) + MCMC (red) reference lines.
        names = [t[0] for t in table]
        n_par = len(names)
        fig, axes = plt.subplots(1, n_par, figsize=(2.9 * n_par, 4.2), constrained_layout=True)
        if n_par == 1:
            axes = [axes]
        for i, (name, ax) in enumerate(zip(names, axes)):
            col = member_matrix[:, aug_attrs.index(name)]
            ax.violinplot(col, showmedians=True)
            jitter = np.linspace(-0.12, 0.12, col.size)
            ax.scatter(np.ones_like(col) + jitter, col, s=11, color='k', alpha=0.4, zorder=3)
            row = [t for t in table if t[0] == name][0]
            pv, mc = row[2], row[3]
            if pv == pv:   # not NaN
                ax.axhline(pv, color='tab:blue', lw=2.0, ls='--', zorder=4,
                           label='original PILA' if i == 0 else None)
            ax.axhline(mc, color='red', lw=2.0, zorder=5, label='MCMC' if i == 0 else None)
            ax.set_title(name + (' (circ)' if name == 'strike' else ''))
            ax.set_ylabel(name)
            ax.set_xticks([])
        fig.legend(loc='upper right', fontsize=9)
        fig.suptitle(f"PILA n=50 UQ ensemble (violin) vs original PILA (blue) & classical MCMC "
                     f"(red): {cfg}\nuq epoch {ref_epoch}", fontsize=11)
        fig.savefig(os.path.join(out_cfg, 'mcmc_vs_uq_params.png'), dpi=150)
        fig.savefig(os.path.join(out_cfg, 'mcmc_vs_uq_params.pdf'))
        plt.close(fig)
        print(f"  wrote {out_cfg}/")

    # Combined overview.
    ov = os.path.join(args.out, 'mcmc_vs_uq_overview.csv')
    with open(ov, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['config', 'param', 'unit', 'pila', 'mcmc', 'uq_median', 'uq_std', 'uq_p5',
                    'uq_p95', 'z_mcmc', 'mcmc_inside_p5_p95', 'circular'])
        w.writerows(overview)
    inside = sum(1 for r in overview if r[-2] is True)
    print(f"\nDone. Overview -> {ov}")
    print(f"Agreement: {inside}/{len(overview)} params have the MCMC estimate inside the "
          f"ensemble p5-p95 (strike is circular -- see note).")


if __name__ == '__main__':
    main()
