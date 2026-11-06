#!/usr/bin/env python
# Usage:       python plot_k2_smoothing_param_recovery.py [--album E1|E2] [--fig-root <default per album>]
# Description: 1-for-1 effect of curvature temporal smoothing (weights 5000 and 15000) on K=2 parameter recovery at
#              SNR 12 (static synthetic scene E1_S3_K2_separated_n025), laid out like plot_vbica_vs_pila_fixed.py: one
#              panel per parameter with all runs overlaid (truth black, without smoothing orange, curv5000 green,
#              curv15000 blue). For each of the 4 report scenarios in --fig-root (01-04, made without smoothing) the
#              SAME arm and seed rerun with smoothing is overlaid (baseline scenario 03: curv15000 only, no curv5000 arm). Rows = west / east source; columns = east, north, depth, dV, location
#              error. basin_best checkpoint (same checkpoint the 01-04 report folders used). Footer = each run's
#              stack-median scores (basinhop_runs.csv) and per-source RMS 2nd difference over epochs (what the
#              curvature penalty targets). Writes <scenario folder>/E1_S3_K2_separated_n025_K2_param_recovery_
#              smoothing_comparison.png. PNG only.
#              --album E2 (added 2026-10-26): second SNR-12 album E2_S3_K2_separated_n035 (same sources, independent noise
#              days); scenarios = each arm's lowest-misfit pick (seeds 6, 4) + seed 2 (fails in both), without smoothing vs
#              curv5000 only (the E2 runs); scores from figures/multisource_k2_basinhop/E2_basinhop_runs.csv; figures in
#              figures/multisource_k2_basinhop_E2/<scenario>/.
# Date:        2026-10-26

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = 'saved/multisource_k2_tune'
TRUTH = 'synthetic/multisource/pila_stacks_v2/albums_snr/E1_S3_K2_separated_n025_truth.json'
OUT_NAME = 'E1_S3_K2_separated_n025_K2_param_recovery_smoothing_comparison.png'
COLOR_TRUTH, COLOR_PLAIN, COLOR_CURV5000, COLOR_CURV15000 = 'k', 'tab:orange', 'tab:green', 'tab:blue'
# Runs overlaid per scenario: (legend label, colour, compare arm, run tag template)
SLOT_RESID_RUNS = [
    ('without smoothing', COLOR_PLAIN, 'bh_slot_resid', 'k2tune_snrE1n025bh_slot_resid_seed{seed}'),
    ('curv5000', COLOR_CURV5000, 'bh_slot_resid_curv5000', 'k2tune_snrE1n025bh_slot_resid_curv5000_seed{seed}'),
    ('curv15000', COLOR_CURV15000, 'bh_slot_resid_curv15000', 'k2tune_snrE1n025bh_slot_resid_curv15000_seed{seed}'),
]
# The warm-restart baseline was only rerun with smoothing at weight 15000 (no curv5000 arm exists)
OBSERVE_RUNS = [
    ('without smoothing', COLOR_PLAIN, 'observe_std (warm restarts, rerun)', 'k2tune_snrE1n025bh_observe_seed{seed}'),
    ('curv15000', COLOR_CURV15000, 'observe_curv15000 (warm restarts + smoothing)',
     'k2tune_snrE1n025bh_observe_curv15000_seed{seed}'),
]
# scenario folder -> (seed, runs to overlay)
SCENARIOS = {
    '01_slot_resid_seed7_PICK': (7, SLOT_RESID_RUNS),
    '02_slot_resid_seed1_rescued': (1, SLOT_RESID_RUNS),
    '03_baseline_seed1_depth_wrong': (1, OBSERVE_RUNS),
    '04_slot_resid_seed2_fails': (2, SLOT_RESID_RUNS),
}
# Second album (E2 noise days); its runs CSV names arms without the 'bh_' prefix
E2_RUNS = [
    ('without smoothing', COLOR_PLAIN, 'slot_resid', 'k2tune_snrE2n035bh_slot_resid_seed{seed}'),
    ('curv5000', COLOR_CURV5000, 'slot_resid_curv5000', 'k2tune_snrE2n035bh_slot_resid_curv5000_seed{seed}'),
]
E2_SCENARIOS = {
    '01_E2_seed6_PICK_no_smoothing': (6, E2_RUNS),
    '02_E2_seed4_PICK_curv5000': (4, E2_RUNS),
    '03_E2_seed2_fails': (2, E2_RUNS),
}
ALBUMS = {  # album -> (truth json, default fig root, runs CSV relative to project, scenarios, output file name)
    'E1': (TRUTH, 'figures/multisource_k2_basinhop', 'figures/multisource_k2_basinhop/basinhop_runs.csv', SCENARIOS,
           OUT_NAME),
    'E2': ('synthetic/multisource/pila_stacks_v2/albums_snr/E2_S3_K2_separated_n035_truth.json',
           'figures/multisource_k2_basinhop_E2', 'figures/multisource_k2_basinhop/E2_basinhop_runs.csv', E2_SCENARIOS,
           'E2_S3_K2_separated_n035_K2_param_recovery_smoothing_comparison.png'),
}
# (param index in params_phys, panel title, axis label, scale to display units)
PARAMS = [(0, 'xcen', 'East (km)', 1.0), (1, 'ycen', 'North (km)', 1.0), (2, 'depth', 'Depth (km)', 1.0),
          (3, 'dV', r'$\Delta V$ ($10^6$ m$^3$)', 1e-6)]
SOURCE_NAMES = ['west source', 'east source']


def load_matched_fit(tag, seed, truth):
    """Load one run's basin_best fit_K2.npz and reorder its slots to match truth (Hungarian on median location).

    Inputs: tag template, seed, truth [2 x (x km, y km, depth km, dV m3)].
    Output: (dates as datetime64 [30], params [30 epochs, 2 sources (truth order), 4] in km / m3).
    """
    paths = glob.glob(os.path.join(PROJECT_ROOT, SAVE_ROOT, tag.format(seed=seed), '*', 'K2', '*', '*', 'models',
                                   'basin_best', 'fit_K2.npz'))
    if len(paths) != 1:
        raise FileNotFoundError(f'{tag.format(seed=seed)}: expected 1 basin_best fit, found {len(paths)}')
    with np.load(paths[0]) as fit:
        params = fit['params_phys']
        dates = fit['dates'].astype('datetime64[D]')                    # stored as ISO 'YYYY-MM-DD' strings
    if not np.all(np.diff(dates) > np.timedelta64(0, 'D')):
        raise ValueError(f'{paths[0]}: dates not chronological, 2nd differences would be meaningless')
    # Slot order is arbitrary in PILA, so match slots to truth sources by the run's median location
    # (same rule as compare_k2_keepbest.score_fit, so the footer scores describe the same pairing).
    median_params = np.median(params, axis=0)
    truth_idx, slot_idx = linear_sum_assignment(
        np.linalg.norm(truth[:, None, :2] - median_params[None, :, :2], axis=-1))
    matched = np.empty_like(params)
    matched[:, truth_idx, :] = params[:, slot_idx, :]
    return dates, matched


def footer_line(label, runs, arm, seed, params):
    """One footer line: run scores from compare_k2_basinhop.py + per-source RMS 2nd difference of depth and dV."""
    row = runs[(runs.arm == arm) & (runs.seed == seed)]
    if len(row) != 1:
        raise ValueError(f'{arm} seed {seed}: expected 1 basin_best row in basinhop_runs.csv, found {len(row)}')
    row = row.iloc[0]
    verdict = 'SUCCESS' if row.success else 'FAIL'
    # RMS 2nd difference over epoch index: the quantity the curvature penalty pushes down (static truth -> 0)
    curvature = [f'{name.split()[0]} depth {np.sqrt(np.mean(np.diff(params[:, src, 2], 2) ** 2)):.2f} km / '
                 f'dV {np.sqrt(np.mean(np.diff(params[:, src, 3], 2) ** 2)) * 1e-6:.2f}e6 m3'
                 for src, name in enumerate(SOURCE_NAMES)]
    return (f'{label}: {verdict} (worst loc {row.loc_max_km:.2f} km, depth {row.depth_pct_max:.0f} %, '
            f'dV {row.dV_pct_max:.0f} %, misfit {row.misfit:.0f}, physics R2 {row.physics_r2:.2f}) | '
            f'RMS 2nd diff: {"; ".join(curvature)}')


def plot_scenario(scenario, seed, runs_to_plot, truth, runs, out_png):
    """One figure, all runs overlaid per panel: rows = sources, columns = 4 parameters + location error."""
    fits = {}
    for label, _, arm, tag in runs_to_plot:
        fits[label] = load_matched_fit(tag, seed, truth)
        print(f"  {tag.format(seed=seed)}: params shape={fits[label][1].shape}, dtype={fits[label][1].dtype}")

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'legend.fontsize': 8, 'lines.linewidth': 1.7})
    fig, axes = plt.subplots(2, len(PARAMS) + 1, figsize=(5.0 * (len(PARAMS) + 1), 3.6 * 2), squeeze=False)
    for src, src_name in enumerate(SOURCE_NAMES):
        for col, (p_idx, title, ylabel, scale) in enumerate(PARAMS):
            axis = axes[src][col]
            dates = fits[runs_to_plot[0][0]][0]
            axis.plot(dates, np.full(len(dates), truth[src, p_idx] * scale), '-', color=COLOR_TRUTH, label='truth')
            for label, colour, _, _ in runs_to_plot:
                run_dates, params = fits[label]
                axis.plot(run_dates, params[:, src, p_idx] * scale, '-', color=colour, label=label)
            axis.set_title(f'{src_name}: {title}')
            axis.set_ylabel(ylabel)
            axis.set_xlabel('Date')
            axis.tick_params(axis='x', rotation=30)
            if src == 0 and col == 0:
                axis.legend(loc='best', framealpha=0.9)
        # Last column: horizontal location error per epoch vs the (static) true centre
        axis = axes[src][-1]
        for label, colour, _, _ in runs_to_plot:
            run_dates, params = fits[label]
            location_error_km = np.hypot(params[:, src, 0] - truth[src, 0], params[:, src, 1] - truth[src, 1])
            axis.plot(run_dates, location_error_km, '-', color=colour, label=label)
        axis.axhline(1.5, color='0.45', ls='-.', label='success threshold 1.5 km')
        axis.set_title(f'{src_name}: location error')
        axis.set_ylabel('Location error (km)')
        axis.set_xlabel('Date')
        axis.tick_params(axis='x', rotation=30)
        if src == 0:
            axis.legend(loc='best', framealpha=0.9)

    footer = '\n'.join(footer_line(label, runs, arm, seed, fits[label][1]) for label, _, arm, _ in runs_to_plot)
    footer += ('\nSame arm and seed in every run; basin_best checkpoint; truth is static, so any epoch-to-epoch '
               'change is error. Scores are stack medians (compare_k2_basinhop.py).')
    if runs_to_plot is OBSERVE_RUNS:
        footer += ' No curv5000 run exists for the warm-restart baseline.'
    elif runs_to_plot is E2_RUNS:
        footer += ' E2 album was run without smoothing and at curv5000 only.'
    weights = ' and '.join(label.replace('curv', '') for label, _, _, _ in runs_to_plot[1:])
    fig.suptitle(f'{scenario}   [K=2 Mogi, SNR 12, {os.path.basename(out_png).split("_S3")[0]} album, seed {seed}]\n'
                 f'without vs with curvature temporal smoothing (weight {weights})', fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.text(0.5, 0.012, footer, ha='center', va='bottom', fontsize=9, color='0.15')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', default='E1', choices=list(ALBUMS))
    parser.add_argument('--fig-root', default=None)
    args = parser.parse_args()
    truth_json, default_root, runs_rel, scenarios, out_name = ALBUMS[args.album]
    fig_root = os.path.join(PROJECT_ROOT, args.fig_root or default_root)

    print("[1/2] Loading truth and run scores...")
    with open(os.path.join(PROJECT_ROOT, truth_json)) as handle:
        truth = np.array([[s['xcen_km'], s['ycen_km'], s['depth_km'], s['dV_m3']] for s in json.load(handle)['sources']])
    print(f"  truth: shape={truth.shape}, dtype={truth.dtype}")
    runs_csv = os.path.join(PROJECT_ROOT, runs_rel)
    if not os.path.exists(runs_csv):
        raise FileNotFoundError(f'{runs_csv} (run compare_k2_basinhop.py first)')
    runs = pd.read_csv(runs_csv)
    if 'checkpoint' in runs:                       # E1 CSV holds several checkpoints / albums; E2 CSV is basin_best only
        runs = runs[(runs.checkpoint == 'basin_best') & (runs.noise_tag == 'n025')]

    print(f"[2/2] Plotting {len(scenarios)} scenario comparisons ({args.album})...")
    for scenario, (seed, runs_to_plot) in scenarios.items():
        scenario_dir = os.path.join(fig_root, scenario)
        if args.album == 'E1' and not os.path.isdir(scenario_dir):
            raise FileNotFoundError(scenario_dir)          # E1 figures go next to the existing report folders
        os.makedirs(scenario_dir, exist_ok=True)
        out_png = os.path.join(scenario_dir, out_name)
        plot_scenario(scenario, seed, runs_to_plot, truth, runs, out_png)
        print(f"  saved {out_png}")
    print(f"Done. Output saved to {fig_root}/<scenario>/")


if __name__ == '__main__':
    main()
