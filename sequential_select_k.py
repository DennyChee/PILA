#!/usr/bin/env python
# Usage:       python sequential_select_k.py stats --runs "saved/multisource_seq/main/*" --out <stats.csv>
#              python sequential_select_k.py calibrate --stats <stats.csv> --pattern "_cal" --out <thresholds.json>
#              python sequential_select_k.py select --stats <stats.csv> --thresholds <thresholds.json> --out <selection.csv>
# Description: Step 2 of the sequential multi-source plan: the "keep this source?" decision.
#   stats      For every album run (folder with K1..Kk sub-runs), re-render the physics-only prediction
#              of each K from the recovered per-acquisition parameters (fit_K<k>.npz) on the FULL album
#              grid (mogi_los_mm_torch; independent of loader subsetting, e.g. bootstrap members), and
#              compute the whitened misfit chi2_k = sum_t r_t^T C^-1 r_t of ramp-removed, multilooked
#              residuals (synthetic/multisource/noise_covariance.py). dchi2_k = chi2_{k-1} - chi2_k is
#              the improvement from adding source k (k = 1 compares against "no source").
#              DEFAULT score (user decision 2026-10-06): gain_pct_k = % of the no-source misfit removed by
#              source k, in PILA's OWN error (squared error of the physics fit; offset + ramp removed only
#              if the run's loss did). The whitened dchi2 failed on Step 3 (calibration nulls mostly
#              negative -> bars too low -> Marapi nulls over-counted) and is kept for reference only.
#   calibrate  Thresholds tau_k = 95th percentile of the chosen metric over noise-only calibration albums.
#   select     K_hat = largest k with dchi2_1..dchi2_k all > tau (sequential: stop at the first fail).
# Scientific notes:
#   * Physics-only residuals (PILA's low-rank residual is NOT a source). Offset + ramp are nuisance
#     terms removed at every K, so they never count as evidence for a source.
#   * Acquisitions share the reference-day APS, so chi2 is not chi2-distributed; that is why the bar is
#     set empirically on noise-only albums (5 % false-alarm by construction on those albums).
#   * Units: residuals in mm before whitening; chi2 dimensionless.
# Date:        2026-10-06

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

from synthetic.multisource.noise_covariance import remove_ramp_maps

from synthetic.multisource.noise_covariance import NoiseCovariance
from synthetic.multisource.torch_generator import mogi_los_mm_torch

DEFAULT_COV = 'synthetic/multisource/pila_stacks_v2/noise_covariance_final.npz'


def album_geometry(album_npz):
    """Data [T, H, W] mm, flattened coordinates (m) and LOS unit vectors of an album."""
    with np.load(album_npz) as f:
        data_mm = f['los_asc_m'].astype(np.float64) * 1e3
        x_m, y_m = f['x_m'], f['y_m']
        inc = np.radians(float(f['incidence_deg']))
        az = np.radians(90.0 - float(f['heading_asc_deg']))        # loader convention
    xE, yN = np.meshgrid(x_m, y_m)
    los = np.array([-np.sin(inc) * np.sin(az), np.sin(inc) * np.cos(az), np.cos(inc)])
    return data_mm, xE, yN, los


def render_mm(params_phys, xE, yN, los):
    """Sum of K Mogi fields per acquisition: params [T, K, 4] (km, km, km, m^3) -> [T, H, W] mm."""
    n_acq, n_src, _ = params_phys.shape
    if n_src == 0:
        return np.zeros((n_acq,) + xE.shape)
    p = torch.tensor(params_phys, dtype=torch.float64)
    fields = mogi_los_mm_torch(torch.tensor(xE.ravel()), torch.tensor(yN.ravel()),
                               torch.tensor(np.repeat(los[:, None], xE.size, axis=1)),
                               p[..., 0] * 1e3, p[..., 1] * 1e3, p[..., 2] * 1e3, p[..., 3])
    return fields.sum(dim=1).numpy().reshape((n_acq,) + xE.shape)


def run_stats(args):
    cov = NoiseCovariance.load(args.cov)
    rows = []
    t0 = time.time()
    album_roots = sorted(r for pattern in args.runs for r in glob.glob(pattern) if os.path.isdir(r))
    print(f"[1/2] {len(album_roots)} album runs, covariance {args.cov}")
    for root in album_roots:
        seq_path = os.path.join(root, 'sequence.json')
        if not os.path.exists(seq_path):
            print(f"  skip {root}: no sequence.json (incomplete)")
            continue
        seq = json.load(open(seq_path))
        album_npz = seq['album']
        data_mm, xE, yN, los = album_geometry(album_npz)
        # PILA's own error: squared error of the physics fit, with the offset + ramp removed only
        # if THIS run's training loss removed it (trainer.phys_vae.nuisance_ramp).
        k1_cfg = glob.glob(os.path.join(root, 'K1', 'config_K1.json'))
        ramp_on = bool(json.load(open(k1_cfg[0]))['trainer']['phys_vae'].get('nuisance_ramp', False)) if k1_cfg else False
        paras_file = json.load(open(k1_cfg[0]))['arch']['args']['mogi_paras'] if k1_cfg else 'configs/mogi_paras_symdV.json'
        paras = json.load(open(paras_file))
        bounds = {name: (float(paras[name]['min']), float(paras[name]['max'])) for name in ('xcen', 'ycen', 'd')}
        def pila_sse(maps):
            maps = remove_ramp_maps(maps, xE / 1e3, yN / 1e3) if ramp_on else maps
            return np.sum(maps.reshape(maps.shape[0], -1) ** 2, axis=1)  # per acquisition, mm^2
        chi2 = {0: cov.chi2(data_mm)}                                   # per-acquisition [T]
        sse = {0: pila_sse(data_mm)}
        params_by_k = {}
        for k_str, run in sorted(seq['runs'].items(), key=lambda kv: int(kv[0])):
            k = int(k_str)
            with np.load(run['fit']) as f:
                params_by_k[k] = f['params_phys']
            residual = data_mm - render_mm(params_by_k[k], xE, yN, los)
            chi2[k] = cov.chi2(residual)
            sse[k] = pila_sse(residual)
        k_max = max(chi2)
        row = dict(run=root, album=os.path.basename(album_npz)[:-4], k_max=k_max, nuisance_ramp=ramp_on)
        for k in range(k_max + 1):
            row[f'sse_{k}'] = float(sse[k].sum())
        for k in range(1, k_max + 1):
            # % of the no-source misfit removed by adding source k (PILA's own error)
            row[f'gain_pct_{k}'] = float(100.0 * (sse[k - 1].sum() - sse[k].sum()) / sse[0].sum())
        for k in range(k_max + 1):
            row[f'chi2_{k}'] = float(chi2[k].sum())
        for k in range(1, k_max + 1):
            delta_t = chi2[k - 1] - chi2[k]
            row[f'dchi2_{k}'] = float(delta_t.sum())
            row[f'frac_acq_favour_{k}'] = float(np.mean(delta_t > 0))
            # The newest source of the K=k fit (median over acquisitions), for railing diagnostics.
            med = np.median(params_by_k[k][:, -1], axis=0)
            row[f'src{k}_x_km'], row[f'src{k}_y_km'], row[f'src{k}_depth_km'], row[f'src{k}_dV_m3'] = map(float, med)
            # REPORT-ONLY flag (user decision 2026-10-06): newest source within 1 km of the prior
            # box wall (x/y edge or depth limits) -> likely fitting noise; not used in select_k.
            row[f'railed_{k}'] = bool(abs(med[0]) > bounds['xcen'][1] - 1.0 or abs(med[1]) > bounds['ycen'][1] - 1.0
                                      or med[2] > bounds['d'][1] - 1.0 or med[2] < bounds['d'][0] + 0.25)
        rows.append(row)
        print(f"  {row['album']:24s} " + ' '.join(f"gain_{k}={row[f'gain_pct_{k}']:6.2f}%" for k in range(1, k_max + 1)))
    if not rows:
        raise RuntimeError("no complete album runs found")
    fields = sorted({key for row in rows for key in row}, key=lambda c: (c not in ('run', 'album', 'k_max'), c))
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"[2/2] Done in {time.time() - t0:.1f}s. Output saved to {args.out}")


def read_stats(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run_calibrate(args):
    rows = [r for r in read_stats(args.stats) if args.pattern in r['album']]
    print(f"[1/1] Calibrating on {len(rows)} noise-only albums (pattern '{args.pattern}'), "
          f"false-alarm target {args.alpha:.0%}")
    if len(rows) < 10:
        raise RuntimeError("too few calibration albums for a percentile threshold")
    k_max = min(int(r['k_max']) for r in rows)
    thresholds = {}
    for k in range(1, k_max + 1):
        values = np.array([float(r[f'{args.metric}_{k}']) for r in rows])
        thresholds[k] = float(np.percentile(values, 100 * (1 - args.alpha)))
        print(f"  step {k}: {args.metric} median {np.median(values):.2f}, p95 = tau_{k} = {thresholds[k]:.2f} "
              f"(min {values.min():.1f}, max {values.max():.1f})")
    out = dict(alpha=args.alpha, metric=args.metric, n_albums=len(rows), pattern=args.pattern, stats=args.stats,
               thresholds={str(k): v for k, v in thresholds.items()})
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"Done. Output saved to {args.out}")


def select_k(row, thresholds, metric='gain_pct'):
    """Sequential stop: largest k such that every step 1..k beats its threshold (metric column prefix)."""
    k_hat = 0
    for k in range(1, int(row['k_max']) + 1):
        if str(k) not in thresholds or float(row[f'{metric}_{k}']) <= thresholds[str(k)]:
            break
        k_hat = k
    return k_hat


def run_select(args):
    cal = json.load(open(args.thresholds))
    rows = read_stats(args.stats)
    out_rows = []
    print(f"[1/1] Selecting K for {len(rows)} albums with thresholds {cal['thresholds']}")
    for r in rows:
        truth_path = os.path.join(args.truth_dir, f"{r['album']}_truth.json")
        true_k = json.load(open(truth_path))['n_sources'] if os.path.exists(truth_path) else None
        k_hat = select_k(r, cal['thresholds'], cal.get('metric', 'dchi2'))
        out_rows.append(dict(album=r['album'], true_k=true_k, k_hat=k_hat, run=r['run'],
                             **{k: r[k] for k in r if k.startswith(('dchi2_', 'gain_pct_'))}))
        print(f"  {r['album']:24s} true K={true_k}  selected K={k_hat}")
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader(); writer.writerows(out_rows)
    print(f"Done. Output saved to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('stats'); p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--cov', default=DEFAULT_COV); p.add_argument('--out', required=True)
    p = sub.add_parser('calibrate'); p.add_argument('--stats', required=True)
    p.add_argument('--pattern', default='_cal'); p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--metric', default='gain_pct', choices=['gain_pct', 'dchi2'],
                   help="gain_pct = % of misfit removed in PILA's own error (default); dchi2 = whitened")
    p.add_argument('--out', required=True)
    p = sub.add_parser('select'); p.add_argument('--stats', required=True); p.add_argument('--thresholds', required=True)
    p.add_argument('--truth-dir', default='synthetic/multisource/pila_stacks_v2/albums'); p.add_argument('--out', required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} exists; not overwriting")
    {'stats': run_stats, 'calibrate': run_calibrate, 'select': run_select}[args.cmd](args)


if __name__ == '__main__':
    main()
