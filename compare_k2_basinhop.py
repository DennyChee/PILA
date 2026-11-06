#!/usr/bin/env python
# Usage:       python compare_k2_basinhop.py [--out-dir figures/multisource_k2_basinhop] [--allow-missing]
#                  [--noise-tag n025|n050]   (n050 = SNR 6 album; arms with no runs at all are skipped)
# Description: Scores the basin-hopping test (generate_k2_basinhop_jobs.py) on E1_S3_K2_separated_<noise-tag> against
#              the 2026-10-22 keep-best std arm (same album, same first 100 epochs, warm restarts T_0=100 to epoch 400).
#              Each run is scored at up to three checkpoints: final epoch (models/fit_K2.npz), lowest val_rec_loss
#              (models/lowest_loss/fit_K2.npz) and, for basin-hop arms, lowest deterministic misfit
#              (models/basin_best/fit_K2.npz). Success rule = compare_k2_keepbest.score_fit (both sources < 1.5 km,
#              worst depth / dV error < 50 %). Truth is used ONLY for scoring.
#              Best-of-16 is selected per (arm, checkpoint type) by the DETERMINISTIC album misfit recomputed from each
#              fit_K2.npz the same way for every arm: mean over acquisitions of sum over coherent points of
#              ((x_PB - data) / x_scale)^2 (= the training rec_loss units; x_scale from each run's own config).
#              Checkpoint types are never mixed: final / lowest_val exist for every arm (fair comparison); basin_best is
#              the basin-hop arms' own selection (argmin of the per-epoch album misfit, which the keep-best arm never
#              logged, so it has no equivalent there). Selection uses the training album itself (as "best-of-16 by own
#              lowest loss"; no held-out data). phys_misfit (physics only, no low-rank residual) is reported as a check
#              that a low misfit is not the residual absorbing a missing source.
#              Also summarises basin_hops.csv (windows with > 0.5 % gain, slot-configuration changes).
# Output:      basinhop_runs.csv, basinhop_summary.csv, basinhop_bestof16.csv, basinhop_hops_summary.csv
# Date:        2026-10-23

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from compare_k2_keepbest import score_fit
from datasets.preprocessing.insar_mintpy import load_insar_mintpy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = 'saved/multisource_k2_tune'
DEFAULT_OUT_DIR = {'n025': 'figures/multisource_k2_basinhop', 'n050': 'figures/multisource_k2_basinhop_snr6'}
TRUTH = 'synthetic/multisource/pila_stacks_v2/albums_snr/E1_S3_K2_separated_{noise}_truth.json'
ARMS = {  # arm label: tag template
    'keepbest_std (warm restarts)': 'k2tune_snrE1{noise}kb_std_wr100_seed{seed}',
    'bh_slot': 'k2tune_snrE1{noise}bh_slot_seed{seed}',
    'bh_slot_resid': 'k2tune_snrE1{noise}bh_slot_resid_seed{seed}',
    'bh_wnoise': 'k2tune_snrE1{noise}bh_wnoise_seed{seed}',
    'bh_none': 'k2tune_snrE1{noise}bh_none_seed{seed}',
    'bh_none_full': 'k2tune_snrE1{noise}bh_none_full_seed{seed}',
    'observe_std (warm restarts, rerun)': 'k2tune_snrE1{noise}bh_observe_seed{seed}',
    # data-scaled dV box (60 mm anchor)
    'keepbest_box60 (warm restarts)': 'k2tune_snrE1{noise}kb_box60_wr100_seed{seed}',
    'observe_box60 (warm restarts, rerun)': 'k2tune_snrE1{noise}bh_observe_box60_seed{seed}',
    'bh_slot_resid_box60': 'k2tune_snrE1{noise}bh_slot_resid_box60_seed{seed}',
    # curvature temporal smoothing alone
    'observe_curv15000 (warm restarts + smoothing)': 'k2tune_snrE1{noise}bh_observe_curv15000_seed{seed}',
    'bh_slot_resid_curv15000': 'k2tune_snrE1{noise}bh_slot_resid_curv15000_seed{seed}',
    'bh_slot_resid_curv5000': 'k2tune_snrE1{noise}bh_slot_resid_curv5000_seed{seed}',
}
CHECKPOINTS = {'final': 'fit_K2.npz', 'lowest_val': os.path.join('lowest_loss', 'fit_K2.npz'),
               'basin_best': os.path.join('basin_best', 'fit_K2.npz')}


_SCALE_CACHE = {}


def run_x_scale(models_dir):
    """Global LOS scaler (mm) of a run, loaded with the run's own insar config (same call as dump_multisource_fit.py)."""
    with open(os.path.join(models_dir, 'config.json')) as handle:
        ins = json.load(handle)['arch']['args']['insar']
    key = json.dumps(ins, sort_keys=True)
    if key not in _SCALE_CACHE:
        d = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'), ins['lat0'], ins['lon0'],
                              multilook=ins.get('multilook', 1), coh_valid_frac=ins.get('coh_valid_frac', 0.5),
                              bbox=ins.get('bbox'), verbose=False, standardization=ins.get('std_mode', 'global'),
                              far_field=ins.get('far_field'), stride=ins.get('stride', 1), offset=ins.get('offset', 0),
                              bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
                              bootstrap_seed=ins.get('bootstrap_seed', 0), ref_date=ins.get('ref_date'))
        if np.ndim(d.x_scale_global) != 0:
            raise ValueError("expected a scalar global x_scale")
        _SCALE_CACHE[key] = float(d.x_scale_global)
    return _SCALE_CACHE[key]


def deterministic_misfit(fit_path, x_scale):
    """
    Album misfits of a fit_K2.npz in training rec_loss units (standardized, per-sample sum over points).
    Output: (misfit of x_PB = physics + low-rank residual, phys_misfit of physics only).
    """
    with np.load(fit_path) as fit:
        mask = fit['mask']
        data_mm = fit['data_mm'][:, mask].astype(np.float64)
        full_std = (fit['xPB_mm'][:, mask] - data_mm) / x_scale
        phys_std = (fit['xP_src_mm'].sum(axis=1)[:, mask] - data_mm) / x_scale
    return float((full_std ** 2).sum(axis=1).mean()), float((phys_std ** 2).sum(axis=1).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=None,
                        help='default figures/multisource_k2_basinhop (n025) or figures/multisource_k2_basinhop_snr6 (n050)')
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--allow-missing', action='store_true', help='skip runs that are not finished yet')
    parser.add_argument('--noise-tag', default='n025', choices=['n025', 'n050'])
    args = parser.parse_args()
    if args.out_dir is None:                # separate folders per album so SNR 6 never overwrites SNR 12
        args.out_dir = DEFAULT_OUT_DIR[args.noise_tag]
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[1/4] Loading truth and the album scaler...")
    with open(os.path.join(PROJECT_ROOT, TRUTH.format(noise=args.noise_tag))) as handle:
        truth = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km'], s['dV_m3']] for s in json.load(handle)['sources']])
    print(f"  truth: shape={truth.shape}, dtype={truth.dtype}")

    print(f"[2/4] Scoring {len(ARMS)} arms x {args.seeds} seeds...")
    rows, hop_rows = [], []
    for arm, template in ARMS.items():
        template = template.replace('{noise}', args.noise_tag)
        if not glob.glob(os.path.join(PROJECT_ROOT, SAVE_ROOT, template.format(seed='*'))):
            print(f"  skip arm {arm}: no runs for {args.noise_tag}")
            continue
        for seed in range(1, args.seeds + 1):
            models = glob.glob(os.path.join(PROJECT_ROOT, SAVE_ROOT, template.format(seed=seed), '*', 'K2', '*', '*', 'models'))
            finished = len(models) == 1 and os.path.exists(os.path.join(models[0], 'fit_K2.npz'))
            if not finished:
                if args.allow_missing:
                    continue
                raise FileNotFoundError(f'{arm} seed {seed}: {len(models)} run dirs / no final fit')
            x_scale = run_x_scale(models[0])
            for choice, relative in CHECKPOINTS.items():
                fit_path = os.path.join(models[0], relative)
                if not os.path.exists(fit_path):
                    continue                                    # basin_best only exists for basin-hop arms
                misfit, phys_misfit = deterministic_misfit(fit_path, x_scale)
                row = {'arm': arm, 'noise_tag': args.noise_tag, 'seed': seed, 'checkpoint': choice, 'misfit': misfit, 'phys_misfit': phys_misfit,
                       'x_scale_mm': x_scale, **score_fit(fit_path, truth)}
                if choice == 'basin_best':                      # dump must reproduce the training-time misfit
                    with open(os.path.join(models[0], 'basin_best', 'basin_best.json')) as handle:
                        logged = json.load(handle)['misfit']
                    row['rel_diff_vs_training_misfit'] = abs(misfit - logged) / logged
                    if row['rel_diff_vs_training_misfit'] > 1e-3:
                        raise ValueError(f"{arm} seed {seed}: dumped basin_best misfit {misfit:.3f} != logged {logged:.3f}")
                rows.append(row)
            hops_csv = os.path.join(models[0], 'basin_hops.csv')
            if os.path.exists(hops_csv):
                hops = pd.read_csv(hops_csv)
                hops = hops[hops['kind'] == 'hop']
                hop_rows.append({'arm': arm, 'seed': seed, 'n_hops': len(hops), 'n_gain_over_0p5pct': int(hops['accepted'].sum()),
                                 'n_gain_and_slots_changed': int((hops['accepted'] & hops['slots_changed']).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, 'basinhop_runs.csv'), index=False, float_format='%.6g')

    print("[3/4] Per-seed success counts and best-of-N by deterministic misfit:")
    summary = (table.groupby(['arm', 'checkpoint'])
               .agg(n_runs=('seed', 'count'), success=('success', 'sum'), loc_only=('loc_only_success', 'sum'),
                    median_loc_km=('loc_max_km', 'median'), median_misfit=('misfit', 'median'),
                    median_phys_misfit=('phys_misfit', 'median')).reset_index())
    summary.to_csv(os.path.join(out_dir, 'basinhop_summary.csv'), index=False, float_format='%.4g')
    print(summary.to_string(index=False))
    # Best-of-N per (arm, checkpoint type): one checkpoint type at a time, never mixed, so every arm gets the same
    # number of candidates per seed; the seed with the lowest deterministic misfit is the arm's honest answer.
    best_rows = []
    for (arm, choice), group in table.groupby(['arm', 'checkpoint']):
        if group['misfit'].isna().any():
            raise ValueError(f"{arm} {choice}: NaN misfit")
        ranked = group.sort_values('misfit')
        top = ranked.iloc[0]
        best_rows.append({'arm': arm, 'checkpoint': choice, 'n_seeds': len(group), 'best_seed': int(top['seed']),
                          'best_misfit': top['misfit'], 'best_phys_misfit': top['phys_misfit'],
                          'best_success': bool(top['success']), 'best_loc_max_km': top['loc_max_km'],
                          'best_depth_pct': top['depth_pct_max'], 'best_dV_pct': top['dV_pct_max'],
                          'top3_successes': int(ranked.head(3)['success'].sum())})
    best = pd.DataFrame(best_rows)
    best.to_csv(os.path.join(out_dir, 'basinhop_bestof16.csv'), index=False, float_format='%.4g')
    print(best.round(3).to_string(index=False))

    print("[4/4] Basin-hop acceptance:")
    if hop_rows:
        hops_table = pd.DataFrame(hop_rows)
        hops_summary = hops_table.groupby('arm')[['n_hops', 'n_gain_over_0p5pct', 'n_gain_and_slots_changed']].sum().reset_index()
        hops_summary.to_csv(os.path.join(out_dir, 'basinhop_hops_summary.csv'), index=False)
        print(hops_summary.to_string(index=False))
    print(f"Done. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
