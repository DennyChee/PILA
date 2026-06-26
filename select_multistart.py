#!/usr/bin/env python
# Usage:       conda activate pila
#              python select_multistart.py [--save-root saved/multistart] [--out comparison_out/multistart]
# Description: After the multi-start jobs finish (all 21 configs), score every (config, seed)
#              run by its physics-only LOS R^2 (recomputed from the checkpoint via
#              plot_insar_results.py) and report the BEST seed per config. For Okada this picks
#              the seed that escaped the null-source local minimum; the null flag (opening ~ 0)
#              never matches Mogi/Sun69, so those fall back to plain best-R^2. Writes a summary CSV.
# Date:        2026-06-24

import os
import re
import csv
import glob
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [
    'LaPalma_Mogi_TA60_A', 'LaPalma_Mogi_TD169_A',
    'LaPalma_Sun69_TA60_A', 'LaPalma_Sun69_TD169_A',
    'LaPalma_Okada_TA60_A', 'LaPalma_Okada_TD169_A',
    'Etna_Mogi_TA44_A', 'Etna_Mogi_TD124_A',
    'Etna_Sun69_TA44_A', 'Etna_Sun69_TD124_A',
    'Etna_Okada_TA44_A', 'Etna_Okada_TD124_A',
    'Nyiragongo_Mogi_TA174_A', 'Nyiragongo_Mogi_TD21_A',
    'Nyiragongo_Sun69_TA174_A', 'Nyiragongo_Sun69_TD21_A',
    'Nyiragongo_Okada_TA174_A', 'Nyiragongo_Okada_TD21_A',
    'SierraNegra_Mogi_InSAR_A', 'SierraNegra_Sun69_InSAR_A', 'SierraNegra_Okada_InSAR_A',
]

_R2 = re.compile(r'physics-only:\s*R²?=(-?\d+\.\d+)\s+RMSE=(-?\d+\.\d+)')
_OPEN = re.compile(r'opening\s*=\s*(-?\d+\.?\d*)\s*m')


def score_checkpoint(ckpt, scratch):
    """Run plot_insar_results.py on a checkpoint; return (r2_phys, rmse_mm, opening_m) or None."""
    env = dict(os.environ); env['MKL_THREADING_LAYER'] = 'GNU'
    out = subprocess.run(
        ['python', 'plot_insar_results.py', '--resume', ckpt, '--output_dir', scratch],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    txt = out.stdout + out.stderr
    m = _R2.search(txt); o = _OPEN.search(txt)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), (float(o.group(1)) if o else float('nan'))


def main():
    ap = argparse.ArgumentParser(description="Select best multi-start seed per Okada config")
    ap.add_argument('--save-root', default='saved/multistart')
    ap.add_argument('--out', default='comparison_out/multistart')
    ap.add_argument('--scratch', default=os.environ.get('TMPDIR', '/tmp') + '/ms_score')
    args = ap.parse_args()
    os.makedirs(os.path.join(REPO_ROOT, args.out), exist_ok=True)
    os.makedirs(args.scratch, exist_ok=True)

    rows = []
    for cfg in CONFIGS:
        cands = sorted(glob.glob(os.path.join(
            REPO_ROOT, args.save_root, f'{cfg.lower()}_s*', '*', '*', 'models', 'model_best.pth')))
        print(f"\n=== {cfg}: {len(cands)} seed run(s) ===")
        best = None
        for ck in cands:
            seed = re.search(rf'{cfg.lower()}_s(\d+)', ck)
            seed = seed.group(1) if seed else '?'
            sc = score_checkpoint(ck, os.path.join(args.scratch, f'{cfg}_s{seed}'))
            if sc is None:
                print(f"  seed {seed}: <no R2 parsed>"); continue
            r2, rmse, opening = sc
            null = abs(opening) < 1e-3
            flag = '  NULL(opening~0)' if null else ''
            print(f"  seed {seed:>3}: R2_phys={r2:+.3f}  RMSE={rmse:.1f} mm  opening={opening:.2f} m{flag}")
            rows.append({'config': cfg, 'seed': seed, 'r2_phys': r2, 'rmse_mm': rmse,
                         'opening_m': opening, 'null_source': null, 'checkpoint': ck})
            if (best is None or r2 > best['r2_phys']) and not null:
                best = rows[-1]
        if best:
            print(f"  -> BEST seed {best['seed']}  R2={best['r2_phys']:+.3f}  RMSE={best['rmse_mm']:.1f} mm")
        else:
            print(f"  -> NO non-null fit found across seeds (all collapsed)")

    csv_path = os.path.join(REPO_ROOT, args.out, 'multistart_scores.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['config', 'seed', 'r2_phys', 'rmse_mm',
                                           'opening_m', 'null_source', 'checkpoint'])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {csv_path}  ({len(rows)} runs scored)")


if __name__ == '__main__':
    main()
