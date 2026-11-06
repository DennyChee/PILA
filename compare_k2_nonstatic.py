#!/usr/bin/env python
# Usage:       python compare_k2_nonstatic.py [--out-dir figures/multisource_k2_nonstatic] [--allow-missing]
#                  [--jobs-dir jobs/multisource_k2_nonstatic] [--pila-root saved/multisource_k2_nonstatic]
# Description: Scores the non-static K=2 campaign (generate_k2_nonstatic_jobs.py) with the shared per-date rule
#              (synthetic/multisource/nonstatic_scoring.score_run) for PILA and the classical per-date fit:
#                * PILA: every (album, arm, seed) on the basin_best checkpoint (lowest deterministic album misfit),
#                  plus the honest best-of-N per (album, arm) chosen by that same misfit (truth used only to score);
#                * classical: per-date cold-start OLS (classical_k2_perdate.py).
#              Figures (PNG): 1 success vs album SNR per G1 scenario; 2 pooled per source-date success vs source-date
#              SNR with logistic SNR50; 3 per-date x / depth / dV trajectories (truth vs best-of-N PILA vs classical) on
#              G1 at SNR ~12 and ~6; 4 false-detection rate on N1/N3; 5 random layouts: success vs Pascal ratio / depth.
# Output:      runs.csv, bestofN.csv, classical.csv, source_dates.csv, snr50.csv + fig*.png in --out-dir
# Date:        2026-10-26

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, linear_sum_assignment

from compare_k2_basinhop import deterministic_misfit, run_x_scale
from synthetic.multisource.nonstatic_scoring import active_threshold_mm, match_slots, score_run, truth_arrays

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ALBUMS_DIR = 'synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2'
CLASSICAL_ROOT = 'saved/multisource_k2_nonstatic_classical_v2'
ARMS = ['slot_resid', 'slot_resid_dvl1u300', 'slot_resid_curv5000', 'slot_resid_curv15000_dvl1u300']
SCENARIOS = ['S3_static', 'N1_staggered', 'N2_opposite', 'N3_pulse', 'N4_moving']
COLORS = {'slot_resid': 'tab:blue', 'slot_resid_dvl1u300': 'tab:orange', 'slot_resid_curv5000': 'tab:green',
          'slot_resid_curv15000_dvl1u300': 'tab:red', 'classical': 'k'}


def find_models_dir(run_root):
    """The single models/ dir of a run (run_k2_tune layout: <tag>/<album>/K2/<album>_K2/<stamp>/models)."""
    models = glob.glob(os.path.join(run_root, '*', 'K2', '*', '*', 'models'))
    return models[0] if len(models) == 1 else None


def check_dates(fit_dates, truth):
    """PILA dumps ISO dates, truth stores YYYYMMDD: they must be the same dates in the same order."""
    iso = [str(d).replace('-', '') for d in fit_dates]
    if iso != truth['dates']:
        raise ValueError(f"{truth['album']}: fit dates do not match truth dates")


def logistic(log_snr, mid, width):
    """Success probability vs log10 SNR."""
    return 1.0 / (1.0 + np.exp(-(log_snr - mid) / width))


def fit_snr50(table):
    """Logistic SNR50 (linear SNR) from per source-date success; NaN if it cannot be fitted."""
    finite = table[np.isfinite(table['snr']) & (table['snr'] > 0)]
    if len(finite) < 20 or finite['success'].nunique() < 2:
        return float('nan')
    try:
        params, _ = curve_fit(logistic, np.log10(finite['snr']), finite['success'].astype(float), p0=[0.5, 0.2], maxfev=20000)
    except RuntimeError:
        return float('nan')
    return float(10 ** params[0])


def matched_fit(params, truth):
    """Per date, fitted slot params re-ordered to match the true sources with the scorer's own rule (match_slots).

    Dates with a non-finite fit stay NaN (the scorer counts them as failures; figures show a gap).
    """
    truth_params, peaks_mm = truth_arrays(truth)
    params = np.asarray(params, dtype=np.float64)
    if params.shape != truth_params.shape:
        raise ValueError(f"{truth['album']}: fit shape {params.shape} != truth shape {truth_params.shape}")
    active = peaks_mm >= active_threshold_mm(truth)
    out = np.full_like(truth_params, np.nan)
    for t in range(truth_params.shape[0]):
        if np.isfinite(params[t]).all():
            rows, cols = match_slots(truth_params[t], params[t], active[t])
            out[t, rows] = params[t, cols]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='figures/multisource_k2_nonstatic')
    parser.add_argument('--jobs-dir', default='jobs/multisource_k2_nonstatic')
    parser.add_argument('--pila-root', default='saved/multisource_k2_nonstatic')
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--allow-missing', action='store_true', help='skip runs that are not finished yet')
    args = parser.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("[1/5] Loading truth...")
    truths = {}
    for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, ALBUMS_DIR, '*_truth.json'))):
        if '_nonoise' not in path:
            truth = json.load(open(path))
            truths[truth['album']] = truth
    reused_path = os.path.join(PROJECT_ROOT, args.jobs_dir, 'reused_runs.json')
    reused = json.load(open(reused_path)) if os.path.exists(reused_path) else {}
    print(f"  {len(truths)} albums, {sum(len(v) for v in reused.values())} reused runs")

    print("[2/5] Scoring PILA runs (basin_best)...")
    run_rows, date_rows, fits = [], [], {}
    for album, truth in truths.items():
        for arm in ARMS:
            for seed in range(1, args.seeds + 1):
                if arm == 'slot_resid' and album in reused:
                    run_root = os.path.join(PROJECT_ROOT, reused[album][str(seed)])
                else:
                    run_root = os.path.join(PROJECT_ROOT, args.pila_root, f'k2ns_{album}_{arm}_seed{seed}')
                models = find_models_dir(run_root)
                fit_path = None if models is None else os.path.join(models, 'basin_best', 'fit_K2.npz')
                if fit_path is None or not os.path.exists(fit_path):
                    if args.allow_missing:
                        continue
                    raise FileNotFoundError(f'{album} {arm} seed {seed}: no basin_best fit under {run_root}')
                with np.load(fit_path) as fit:
                    params, fit_dates = fit['params_phys'].astype(np.float64), fit['dates']
                check_dates(fit_dates, truth)
                misfit, phys_misfit = deterministic_misfit(fit_path, run_x_scale(models))
                summary, table = score_run(params, truth)
                run_rows.append(dict(summary, arm=arm, seed=seed, misfit=misfit, phys_misfit=phys_misfit))
                fits[(album, arm, seed)] = params
                date_rows.append(table.assign(album=album, arm=arm, seed=seed, scenario=truth['scenario'],
                                              layout_set=truth['layout_set'], noise_scale=truth['noise_scale']))
    runs = pd.DataFrame(run_rows)
    runs.to_csv(os.path.join(out_dir, 'runs.csv'), index=False, float_format='%.6g')
    print(f"  {len(runs)} PILA runs scored")

    # Best-of-N per (album, arm): lowest deterministic album misfit (truth-free selection)
    best = (runs.sort_values('misfit').groupby(['album', 'arm'], as_index=False).first()
            .rename(columns={'seed': 'best_seed'}))
    per_seed = runs.groupby(['album', 'arm']).agg(n_runs=('seed', 'count'), n_run_success=('run_success', 'sum'),
                                                  median_success_frac=('success_frac', 'median')).reset_index()
    best = best.merge(per_seed, on=['album', 'arm'])
    best.to_csv(os.path.join(out_dir, 'bestofN.csv'), index=False, float_format='%.6g')

    print("[3/5] Loading classical per-date results...")
    classical_rows, classical_fits = [], {}
    for album, truth in truths.items():
        folder = os.path.join(PROJECT_ROOT, CLASSICAL_ROOT, album)
        if not os.path.exists(os.path.join(folder, 'score_summary.json')):
            if args.allow_missing:
                continue
            raise FileNotFoundError(f'classical missing for {album}')
        params = np.load(os.path.join(folder, 'per_date_params.npy'))
        summary, table = score_run(params, truth)          # re-score here so both methods use this exact code
        classical_rows.append(dict(summary, arm='classical'))
        classical_fits[album] = params
        date_rows.append(table.assign(album=album, arm='classical', seed=0, scenario=truth['scenario'],
                                      layout_set=truth['layout_set'], noise_scale=truth['noise_scale']))
    classical = pd.DataFrame(classical_rows)
    classical.to_csv(os.path.join(out_dir, 'classical.csv'), index=False, float_format='%.6g')
    dates_table = pd.concat(date_rows, ignore_index=True) if date_rows else pd.DataFrame()
    dates_table.to_csv(os.path.join(out_dir, 'source_dates.csv'), index=False, float_format='%.6g')
    print(f"  {len(classical)} classical albums scored")

    print("[4/5] SNR50 per method (pooled active source-dates; PILA = best-of-N picks and all runs)...")
    snr_rows = []
    active = dates_table[dates_table['active']].copy() if len(dates_table) else dates_table
    if len(active):
        best_keys = set(zip(best['album'], best['arm'], best['best_seed']))
        active['is_best'] = [(a, m, s) in best_keys for a, m, s in zip(active['album'], active['arm'], active['seed'])]
        for layout_set in ('G1', 'R'):
            for scenario in SCENARIOS + ['all']:
                subset = active[active['layout_set'] == layout_set]
                if scenario != 'all':
                    subset = subset[subset['scenario'] == scenario]
                for method in ARMS + ['classical']:
                    rows = subset[subset['arm'] == method]
                    entry = dict(layout_set=layout_set, scenario=scenario, method=method, snr50_all_runs=fit_snr50(rows))
                    if method != 'classical':
                        entry['snr50_bestofN'] = fit_snr50(rows[rows['is_best']])
                    snr_rows.append(entry)
    snr50 = pd.DataFrame(snr_rows)
    snr50.to_csv(os.path.join(out_dir, 'snr50.csv'), index=False, float_format='%.4g')
    if len(snr50):
        print(snr50[snr50['scenario'] == 'all'].round(2).to_string(index=False))

    print("[5/5] Figures...")
    snr_of = lambda album: truths[album]['snr_peak_over_rms'] or 60.0          # clean albums plotted at SNR 60
    # Figure 1: G1 success fraction vs album SNR, one panel per scenario
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        for arm in ARMS:
            rows = best[(best['arm'] == arm) & best['album'].str.startswith(f'G1_{scenario}')]
            rows = rows.assign(snr=[snr_of(a) for a in rows['album']]).sort_values('snr')
            ax.plot(rows['snr'], rows['success_frac'], 'o-', color=COLORS[arm], label=f'{arm} best-of-N')
            ax.plot(rows['snr'], rows['median_success_frac'], 's--', mfc='none', color=COLORS[arm], label=f'{arm} median run')
        rows = classical[classical['album'].str.startswith(f'G1_{scenario}')] if len(classical) else classical
        if len(rows):
            rows = rows.assign(snr=[snr_of(a) for a in rows['album']]).sort_values('snr')
            ax.plot(rows['snr'], rows['success_frac'], 'k^-', label='classical per-date')
        ax.axhline(0.8, color='gray', ls=':', lw=0.8)
        ax.set_xscale('log'); ax.set_xlim(2, 80); ax.set_ylim(-0.05, 1.05); ax.set_title(f'G1 {scenario}'); ax.set_xlabel('Album SNR (peak / noise RMS; clean at 60)')
    axes[0].set_ylabel('Fraction of active source-dates correct'); axes[0].legend(fontsize=7)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig1_G1_success_vs_snr.png'), dpi=150); plt.close(fig)

    # Figure 2: pooled per source-date success vs source-date SNR (binned) for each method
    if len(active):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        bins = np.logspace(np.log10(0.5), np.log10(60), 14)
        for ax, layout_set in zip(axes, ('G1', 'R')):
            subset = active[(active['layout_set'] == layout_set) & np.isfinite(active['snr'])]
            for method in ARMS + ['classical']:
                rows = subset[(subset['arm'] == method) & ((method == 'classical') | subset['is_best'])]
                if not len(rows):
                    continue
                index = np.digitize(rows['snr'], bins)
                centres = [np.sqrt(bins[i - 1] * bins[i]) for i in range(1, len(bins)) if (index == i).sum() >= 5]
                fracs = [rows['success'][index == i].astype(float).mean() for i in range(1, len(bins)) if (index == i).sum() >= 5]
                s50 = snr50[(snr50['layout_set'] == layout_set) & (snr50['scenario'] == 'all') & (snr50['method'] == method)]
                value = s50['snr50_all_runs' if method == 'classical' else 'snr50_bestofN'].iloc[0] if len(s50) else np.nan
                ax.plot(centres, fracs, 'o-', color=COLORS[method], label=f'{method} (SNR50 {value:.1f})')
            ax.set_xscale('log'); ax.set_xlim(0.5, 60); ax.set_ylim(-0.05, 1.05); ax.set_xlabel('Source-date SNR (own peak / noise RMS)')
            ax.set_title(f'{layout_set}: pooled active source-dates (PILA = best-of-N picks)'); ax.legend(fontsize=8)
        axes[0].set_ylabel('Fraction correct')
        plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig2_pooled_success_vs_sourcedate_snr.png'), dpi=150); plt.close(fig)

    # Figure 3: per-date trajectories on G1 at noise x0.25 (SNR ~12) and x0.5 (SNR ~6)
    for tag in ('n0250', 'n0500'):
        fig, axes = plt.subplots(5, 3, figsize=(16, 18), sharex=True)
        for row, scenario in enumerate(SCENARIOS):
            album = f'G1_{scenario}_{tag}'
            truth = truths[album]
            truth_params, _ = truth_arrays(truth)
            dates = np.array([f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in truth['dates']], dtype='datetime64[D]')
            series = []
            for arm in ARMS:
                pick = best[(best['album'] == album) & (best['arm'] == arm)]
                if len(pick):
                    series.append((f"{arm} (seed {int(pick['best_seed'].iloc[0])})", COLORS[arm],
                                   matched_fit(fits[(album, arm, int(pick["best_seed"].iloc[0]))], truth)))
            if album in classical_fits:
                series.append(('classical per-date', 'k', matched_fit(classical_fits[album], truth)))
            for col, (index, label) in enumerate([(0, 'East (km)'), (2, 'Depth (km)'), (3, 'dV (m$^3$)')]):
                ax = axes[row, col]
                for src, style in ((0, '-'), (1, '--')):
                    ax.plot(dates, truth_params[:, src, index], style, color='green', lw=2.5, alpha=0.6,
                            label=f'truth source {src}' if col == 0 else None)
                    for name, color, matched in series:
                        ax.plot(dates, matched[:, src, index], style, marker='.', color=color, lw=1,
                                label=f'{name} source {src}' if col == 0 else None)
                ax.set_ylabel(label); ax.set_title(f'{album}', fontsize=9)
                if index == 2:
                    ax.invert_yaxis()
            axes[row, 0].legend(fontsize=6)
        for ax in axes[-1]:
            ax.set_xlabel('Date'); ax.tick_params(axis='x', rotation=45)
        plt.tight_layout(); fig.savefig(os.path.join(out_dir, f'fig3_G1_trajectories_{tag}.png'), dpi=120); plt.close(fig)

    # Figure 4: false detections (slot claims a visible source on dates where the true source is not detectable)
    if len(dates_table):
        inactive = dates_table[(~dates_table['active']) & dates_table['scenario'].isin(['N1_staggered', 'N3_pulse'])]
        if len(inactive):
            rate = (inactive.groupby(['layout_set', 'scenario', 'arm', 'noise_scale'])['false_detection']
                    .apply(lambda s: s.astype(bool).mean()).reset_index())
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
            for ax, scenario in zip(axes, ['N1_staggered', 'N3_pulse']):
                for method in ARMS + ['classical']:
                    rows = rate[(rate['scenario'] == scenario) & (rate['arm'] == method) & (rate['layout_set'] == 'G1')]
                    ax.plot(rows['noise_scale'], rows['false_detection'], 'o-', color=COLORS[method], label=method)
                ax.set_xlabel('Noise scale (x E1 noise)'); ax.set_title(f'G1 {scenario}: inactive source-dates')
            axes[0].set_ylabel('False-detection rate (all PILA runs; classical)'); axes[0].legend(fontsize=8)
            plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig4_false_detections.png'), dpi=150); plt.close(fig)

    # Figure 5: random layouts, best-of-N success vs Pascal ratio and vs deeper source depth, coloured by SNR
    r_best = best[best['layout_set'] == 'R']
    if len(r_best):
        deeper = {a: max(truth_arrays(t)[0][:, :, 2].max(axis=0)) for a, t in truths.items()}
        fig, axes = plt.subplots(len(ARMS), 3, figsize=(17, 4.5 * len(ARMS)), sharey=True)   # one row per PILA arm
        for row_axes, method in zip(axes, ARMS):
            rows = r_best[r_best['arm'] == method]
            for ax, (key, label) in zip(row_axes, [('min_pascal', 'Min separation / shallower depth'),
                                                  ('deeper', 'Deeper source depth (km)'), ('album_snr', 'Album SNR')]):
                x = rows['album'].map(deeper) if key == 'deeper' else rows[key].fillna(60.0)
                image = ax.scatter(x, rows['success_frac'], c=np.log10(rows['album_snr'].fillna(60.0)), cmap='viridis',
                                   vmin=0.3, vmax=1.8)
                ax.set_xlabel(label); ax.set_title(f'R, {method} best-of-N')
                if key == 'album_snr':
                    ax.set_xscale('log'); ax.set_xlim(2, 80)
            row_axes[0].set_ylabel('Fraction of active source-dates correct')
        fig.colorbar(image, ax=axes, label='log10 album SNR (clean = 60)')
        fig.savefig(os.path.join(out_dir, 'fig5_random_layouts.png'), dpi=150); plt.close(fig)

    # Per-album parameter-recovery figures (vbICA-figure layout), always produced with the comparison
    if not args.allow_missing:
        print("[+] Per-album parameter-recovery figures...")
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'plot_k2_nonstatic_param_recovery.py'),
                        '--out-dir', os.path.join(args.out_dir, 'param_recovery'), '--compare-dir', args.out_dir,
                        '--jobs-dir', args.jobs_dir, '--pila-root', args.pila_root], check=True, cwd=PROJECT_ROOT)
        print("[+] Per-album input / model / residual maps...")
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'plot_k2_nonstatic_recon_maps.py'),
                        '--out-dir', os.path.join(args.out_dir, 'param_recovery'), '--compare-dir', args.out_dir,
                        '--jobs-dir', args.jobs_dir, '--pila-root', args.pila_root], check=True, cwd=PROJECT_ROOT)
    print(f"Done. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
