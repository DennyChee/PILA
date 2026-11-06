#!/usr/bin/env python
# Usage:       python autopsy_k2_nonstatic.py [--field-arms slot_resid] [--out-dir figures/multisource_k2_nonstatic/autopsy]
# Description: Failure-mode autopsy of the K=2 non-static PILA campaign (read-only on existing basin_best fits).
#              Every ACTIVE source-date that fails the campaign scorer is put into ONE primary failure class, using
#              the same per-date slot matching as compare_k2_nonstatic.py (nonstatic_scoring.match_slots):
#                nonfinite  : fit parameters not finite on that date
#                dipole     : the two slots sit within 3 km of each other with opposite-sign dV, both >1.5x the
#                             largest true |dV| that date (two huge sources cancelling = the SNR-6 trap)
#                double_up  : both slots within 3 km of the SAME true source (the other source has no slot)
#                dead       : the matched slot is invisible (own peak |LOS| < activity threshold) -> source missed
#                railed     : matched slot at a prior bound (|x|,|y|>=14.5 km, depth>=11.5 or <=0.6 km, |dV|>=0.95 max)
#                misplaced  : visible, not railed, but >=1.5 km from the source
#                sign_flip  : location ok, dV has the wrong sign
#                dV_only / depth_only / depth_dV : location ok, amplitude and/or depth outside 50 %
#              Spare-slot behaviour on dates with exactly one active source, slot-identity swaps per run, and
#              (for --field-arms) where the missed source's LOS signal went: physics / low-rank residual / misfit.
#              Truth is used ONLY for scoring/diagnosis here, never fed back into any inversion.
# Assumptions: Mogi box from configs/mogi_paras_symdV.json (x,y +-15 km, depth 0.5-12 km, dV +-2e8 m^3);
#              LOS mm, + toward satellite; truth fields are unreferenced, so field projections are de-meaned over
#              the mask (removes the reference-pixel constant, <=5 % of each source peak).
# Date:        2026-11-02

import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic.forward_numpy import mogi
from synthetic.multisource.nonstatic_scoring import (DV_TOL_FRAC, DEPTH_TOL_FRAC, LOC_TOL_KM, _grid,
                                                     active_threshold_mm, match_slots, own_peak_mm, truth_arrays)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ALBUMS_DIR = os.path.join(PROJECT_ROOT, 'synthetic/multisource/pila_stacks_v2/albums_nonstatic_v2')
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'saved/multisource_k2_nonstatic')
SOURCE_DATES_CSV = os.path.join(PROJECT_ROOT, 'figures/multisource_k2_nonstatic/source_dates.csv')
REUSED_JSON = os.path.join(PROJECT_ROOT, 'jobs/multisource_k2_nonstatic/reused_runs.json')
ARMS = ['slot_resid', 'slot_resid_dvl1u300', 'slot_resid_curv5000', 'slot_resid_curv15000_dvl1u300']
SEEDS = range(1, 9)

# --- Prior box (configs/mogi_paras_symdV.json) and railing / dipole tolerances ---
BOX_XY_KM, BOX_DEPTH_KM, BOX_DV_M3 = 15.0, (0.5, 12.0), 2.0e8
RAIL_XY_KM, RAIL_DEPTH_LO_KM, RAIL_DEPTH_HI_KM, RAIL_DV_FRAC = 14.5, 0.6, 11.5, 0.95
PAIR_CLOSE_KM = 3.0          # "two slots on top of each other" / "on the same source"
DIPOLE_DV_FACTOR = 1.5       # each slot |dV| > 1.5 x largest true |dV| that date

CLASS_ORDER = ['nonfinite', 'dipole', 'double_up', 'dead', 'railed', 'misplaced',
               'sign_flip', 'dV_only', 'depth_only', 'depth_dV']
CLASS_COLOR = dict(nonfinite='#444444', dipole='#d62728', double_up='#ff7f0e', dead='#7f7f7f', railed='#8c564b',
                   misplaced='#9467bd', sign_flip='#e377c2', dV_only='#1f77b4', depth_only='#17becf',
                   depth_dV='#2ca02c')


def is_railed(slot):
    """True if a fitted slot [x km, y km, depth km, dV m^3] sits at any prior bound."""
    return bool(abs(slot[0]) >= RAIL_XY_KM or abs(slot[1]) >= RAIL_XY_KM or slot[2] >= RAIL_DEPTH_HI_KM
                or slot[2] <= RAIL_DEPTH_LO_KM or abs(slot[3]) >= RAIL_DV_FRAC * BOX_DV_M3)


def find_fit(album, arm, seed, reused):
    """Path of the basin_best fit_K2.npz for one run (handles the reused G1_S3_static runs), or None."""
    key = f'{album}|{arm}|{seed}'
    if key in reused:
        root = os.path.join(PROJECT_ROOT, reused[key])
    else:
        root = os.path.join(SAVE_ROOT, f'k2ns_{album}_{arm}_seed{seed}')
    hits = sorted(glob.glob(os.path.join(root, '**', 'models', 'basin_best', 'fit_K2.npz'), recursive=True))
    return hits[-1] if hits else None


def load_reused():
    """reused_runs.json maps album/arm/seed -> old K=2-study run dir; format checked defensively."""
    if not os.path.exists(REUSED_JSON):
        return {}
    # format: {album: {seed: run_root}}; reused runs are the slot_resid arm only (as in compare_k2_nonstatic.py)
    raw = json.load(open(REUSED_JSON))
    return {f'{album}|slot_resid|{int(seed)}': run_root
            for album, seeds in raw.items() for seed, run_root in seeds.items()}


def classify_date(true_t, fit_t, active_t, threshold_mm, finite):
    """
    Classify one date of one run.

    Inputs : true_t [2,4], fit_t [2,4] (x km, y km, depth km, dV m^3), active_t [2] bool, threshold_mm (float),
             finite (bool: fit finite on this date).
    Output : list of dicts, one per true source (matched slot, success, primary failure class, flags).
    """
    if not finite:
        return [dict(source=s, slot=-1, active=bool(active_t[s]), success=False if active_t[s] else None,
                     cls='nonfinite' if active_t[s] else None, slot_peak_mm=np.nan, slot_railed=False,
                     spare_state='nonfinite' if not active_t[s] else None) for s in range(2)]
    true_idx, slot_idx = match_slots(true_t, fit_t, active_t)
    slot_peaks = [own_peak_mm(*fit_t[k]) for k in range(2)]
    slot_sep_km = float(np.hypot(*(fit_t[0, :2] - fit_t[1, :2])))
    dv_true_max = float(np.max(np.abs(true_t[:, 3])))
    dipole = (slot_sep_km < PAIR_CLOSE_KM and np.sign(fit_t[0, 3]) != np.sign(fit_t[1, 3])
              and np.all(np.abs(fit_t[:, 3]) > DIPOLE_DV_FACTOR * dv_true_max))
    # distance of every slot to every true source [source, slot]
    dist_km = np.linalg.norm(true_t[:, None, :2] - fit_t[None, :, :2], axis=-1)
    doubled_source = [s for s in range(2) if np.all(dist_km[s] < PAIR_CLOSE_KM)]
    rows = []
    for src, slot in zip(true_idx, slot_idx):
        truth_row, fit_row = true_t[src], fit_t[slot]
        loc_km = float(np.hypot(*(fit_row[:2] - truth_row[:2])))
        depth_err = (fit_row[2] - truth_row[2]) / truth_row[2]
        dv_err = (fit_row[3] - truth_row[3]) / abs(truth_row[3]) if truth_row[3] != 0 else np.inf
        loc_ok, depth_ok, dv_ok = loc_km < LOC_TOL_KM, abs(depth_err) < DEPTH_TOL_FRAC, abs(dv_err) < DV_TOL_FRAC
        railed = is_railed(fit_row)
        row = dict(source=int(src), slot=int(slot), active=bool(active_t[src]), loc_err_km=loc_km,
                   depth_err_pct=100 * depth_err, dV_err_pct=100 * dv_err, slot_peak_mm=slot_peaks[slot],
                   slot_railed=railed, slot_sep_km=slot_sep_km, dipole=bool(dipole),
                   double_up=bool(doubled_source), success=None, cls=None, spare_state=None)
        if active_t[src]:
            success = loc_ok and depth_ok and dv_ok
            row['success'] = bool(success)
            if not success:
                if dipole:
                    cls = 'dipole'
                elif doubled_source and src not in doubled_source:
                    cls = 'double_up'       # both slots sit on the OTHER source; this one has none
                elif slot_peaks[slot] < threshold_mm:
                    cls = 'dead'
                elif railed:
                    cls = 'railed'
                elif not loc_ok:
                    cls = 'misplaced'
                elif np.sign(fit_row[3]) != np.sign(truth_row[3]):
                    cls = 'sign_flip'
                elif not depth_ok and not dv_ok:
                    cls = 'depth_dV'
                elif not dv_ok:
                    cls = 'dV_only'
                else:
                    cls = 'depth_only'
                row['cls'] = cls
        else:
            # what does the slot matched to an INACTIVE source do?
            other_src = 1 - src
            if slot_peaks[slot] < threshold_mm:
                row['spare_state'] = 'invisible'
            elif dist_km[other_src, slot] < PAIR_CLOSE_KM:
                row['spare_state'] = 'on_active_source'
            elif railed:
                row['spare_state'] = 'visible_railed'
            else:
                row['spare_state'] = 'visible_elsewhere'
        rows.append(row)
    return rows


def truth_field_mm(source_params):
    """LOS field (mm) of one true Mogi [x km, y km, depth km, dV m^3] on the album grid, shape of the grid."""
    grid = _grid()
    uE, uN, uU = mogi(grid['xE_m'], grid['yN_m'], source_params[0] * 1e3, source_params[1] * 1e3,
                      source_params[2] * 1e3, source_params[3])
    # scorer grid is flat (16384,), row-major 128x128 -- verified against the dump's xE_km/yN_km (<1e-6 km)
    return (grid['losE'] * uE + grid['losN'] * uN + grid['losU'] * uU).reshape(128, 128)


def field_shares(fit_npz, true_params, active, mask_flat_cache={}):
    """
    Where each active true source's signal went, per date: share in physics, low-rank residual, misfit.

    Inputs : open npz (xP_src_mm [T,K,H,W], xPB_mm [T,H,W], data_mm [T,H,W], mask [H,W]); true_params [T,2,4];
             active [T,2] bool.
    Output : dict (t, src) -> (a_phys, a_lowrank, a_misfit): de-meaned projections <f, X>/<f, f> over the mask.
             data ~ phys + lowrank + misfit, so the three shares of a source's own field sum to ~ its share in data
             (~1 for clean albums; cross-talk with the other source / noise adds scatter).
    """
    mask = fit_npz['mask']
    xP_src = fit_npz['xP_src_mm']
    phys = xP_src.sum(axis=1)
    lowrank = fit_npz['xPB_mm'] - phys
    misfit = fit_npz['data_mm'] - fit_npz['xPB_mm']
    out = {}
    for t in range(true_params.shape[0]):
        for src in range(2):
            if not active[t, src]:
                continue
            field = truth_field_mm(true_params[t, src])
            if field.shape != mask.shape:
                raise ValueError(f'truth field shape {field.shape} != fit grid {mask.shape}')
            f = field[mask] - field[mask].mean()
            norm = float(f @ f)
            shares = []
            for comp in (phys[t], lowrank[t], misfit[t]):
                c = comp[mask] - comp[mask].mean()
                shares.append(float(f @ c) / norm)
            out[(t, src)] = tuple(shares)
    return out


def main():
    parser = argparse.ArgumentParser(description='Failure-mode autopsy of the K=2 non-static PILA campaign.')
    parser.add_argument('--field-arms', nargs='*', default=['slot_resid'],
                        help='arms for the (slower) signal-share analysis on the dumped fields')
    parser.add_argument('--out-dir', default=os.path.join(PROJECT_ROOT, 'figures/multisource_k2_nonstatic/autopsy'))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    # --- 1. Truth ---
    print('[1/6] Loading truth...')
    truths = {}
    for path in sorted(glob.glob(os.path.join(ALBUMS_DIR, '*_truth.json'))):
        if '_nonoise' in path:
            continue
        truth = json.load(open(path))
        truths[truth['album']] = truth
    reused = load_reused()
    print(f'  {len(truths)} albums, {len(reused)} reused run entries')

    # --- 2. Classify every run / date / source ---
    print('[2/6] Classifying every basin_best fit (all arms, seeds 1-8)...')
    t0 = time.time()
    rows, field_rows, missing = [], [], []
    for album_index, (album, truth) in enumerate(truths.items()):
        true_params, true_peaks = truth_arrays(truth)
        threshold_mm = active_threshold_mm(truth)
        active = true_peaks >= threshold_mm
        for arm in ARMS:
            for seed in SEEDS:
                path = find_fit(album, arm, seed, reused)
                if path is None:
                    missing.append((album, arm, seed))
                    continue
                with np.load(path, allow_pickle=True) as fit_npz:
                    params = np.asarray(fit_npz['params_phys'], dtype=np.float64)
                    if params.shape != true_params.shape:
                        raise ValueError(f'{path}: params {params.shape} != truth {true_params.shape}')
                    finite = np.isfinite(params).all(axis=(1, 2))
                    shares = field_shares(fit_npz, true_params, active) if arm in args.field_arms else {}
                for t in range(true_params.shape[0]):
                    for row in classify_date(true_params[t], params[t], active[t], threshold_mm, finite[t]):
                        row.update(album=album, scenario=truth['scenario'], layout_set=truth['layout_set'],
                                   noise_scale=truth['noise_scale'], arm=arm, seed=seed, date_index=t,
                                   tau=truth['tau'][t], true_peak_mm=float(true_peaks[t, row['source']]),
                                   threshold_mm=threshold_mm,
                                   snr=(np.inf if truth['noise_rms_mm'] == 0
                                        else float(true_peaks[t, row['source']] / truth['noise_rms_mm'])))
                        key = (t, row['source'])
                        if key in shares:
                            row['share_phys'], row['share_lowrank'], row['share_misfit'] = shares[key]
                        rows.append(row)
        if (album_index + 1) % 10 == 0:
            print(f'  {album_index + 1}/{len(truths)} albums ({time.time() - t0:.0f}s)')
    table = pd.DataFrame(rows)
    print(f'  {len(table)} source-date rows; missing runs: {len(missing)}')
    if missing:
        print(f'  first missing: {missing[:5]}')

    # --- 3. Sanity: our success flag must reproduce the campaign scorer exactly ---
    print('[3/6] Cross-checking success flags against compare_k2_nonstatic source_dates.csv...')
    ref = pd.read_csv(SOURCE_DATES_CSV)
    ref_pila = ref[ref['arm'] != 'classical'][['album', 'arm', 'seed', 'date_index', 'source', 'active', 'success']]
    merged = table.merge(ref_pila, on=['album', 'arm', 'seed', 'date_index', 'source'], suffixes=('', '_ref'))
    act = merged[merged['active']]
    mismatch = int((act['success'].astype(bool) != act['success_ref'].astype(bool)).sum())
    print(f'  matched {len(merged)}/{len(table)} rows; active success mismatches: {mismatch}')
    if len(merged) != len(table) or mismatch:
        raise SystemExit('FAIL: autopsy success flags do not reproduce the campaign scorer')
    print('  PASS')
    classical = ref[ref['arm'] == 'classical'][['album', 'date_index', 'source', 'success']].rename(
        columns={'success': 'classical_success'})
    table = table.merge(classical, on=['album', 'date_index', 'source'], how='left')
    table.to_csv(os.path.join(args.out_dir, 'autopsy_source_dates.csv'), index=False)

    # --- 4. Summary tables ---
    print('[4/6] Summaries...')
    active_rows = table[table['active']].copy()
    active_rows['success'] = active_rows['success'].astype(bool)
    active_rows['classical_success'] = active_rows['classical_success'].fillna(False).astype(bool)
    failed = active_rows[~active_rows['success']]
    # the interesting failures: classical succeeded on the very same source-date
    pila_only_fail = failed[failed['classical_success']]
    class_by_scen = (pila_only_fail.groupby(['arm', 'scenario'])['cls'].value_counts(normalize=True)
                     .unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0))
    class_by_scen.to_csv(os.path.join(args.out_dir, 'failclass_by_arm_scenario.csv'))
    clean_fail = failed[failed['noise_scale'] == 0]
    class_clean = (clean_fail.groupby(['arm', 'scenario'])['cls'].value_counts(normalize=True)
                   .unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0))
    class_clean.to_csv(os.path.join(args.out_dir, 'failclass_clean_albums.csv'))
    with pd.option_context('display.width', 200, 'display.precision', 2):
        print('  Failure classes where classical SUCCEEDED (fraction of PILA failures):')
        print(class_by_scen.loc['slot_resid'])
        print('  Failure classes on CLEAN albums (slot_resid):')
        print(class_clean.loc['slot_resid'] if 'slot_resid' in class_clean.index.get_level_values(0) else 'none')

    spare = table[(~table['active']) & table['spare_state'].notna()]
    spare_tab = spare.groupby(['arm', 'scenario'])['spare_state'].value_counts(normalize=True).unstack(fill_value=0)
    spare_tab.to_csv(os.path.join(args.out_dir, 'spare_slot_state.csv'))
    with pd.option_context('display.width', 200, 'display.precision', 2):
        print('  Spare slot (matched to an inactive source) state, slot_resid:')
        print(spare_tab.loc['slot_resid'] if 'slot_resid' in spare_tab.index.get_level_values(0) else 'none')

    # identity swaps: per run and source, how often the matched slot index changes between successive
    # SUCCESSFUL active dates (PILA slots carry no identity across dates; a swap is not penalised by the scorer)
    swap_rows = []
    for (album, arm, seed, src), grp in active_rows.groupby(['album', 'arm', 'seed', 'source']):
        good = grp[grp['success']].sort_values('date_index')['slot'].to_numpy()
        swap_rows.append(dict(album=album, arm=arm, seed=seed, source=src, n_good=len(good),
                              n_swaps=int(np.sum(good[1:] != good[:-1])) if len(good) > 1 else 0))
    swaps = pd.DataFrame(swap_rows)
    swaps.to_csv(os.path.join(args.out_dir, 'slot_swaps.csv'), index=False)
    print(f"  slot swaps per source (runs with >=5 good dates): median "
          f"{swaps[swaps['n_good'] >= 5]['n_swaps'].median():.1f}, "
          f"frac with >=1 swap {(swaps[swaps['n_good'] >= 5]['n_swaps'] > 0).mean():.2f}")

    # --- 5. Figures ---
    print('[5/6] Figures...')
    # (a) failure classes per scenario/arm, PILA-only failures
    scenarios = sorted(active_rows['scenario'].unique())
    fig, axes = plt.subplots(1, len(ARMS), figsize=(4.2 * len(ARMS), 4.5), sharey=True)
    for ax, arm in zip(axes, ARMS):
        sub = class_by_scen.loc[arm] if arm in class_by_scen.index.get_level_values(0) else None
        bottom = np.zeros(len(scenarios))
        for cls in CLASS_ORDER:
            vals = np.array([sub.loc[s, cls] if sub is not None and s in sub.index else 0 for s in scenarios])
            ax.bar(range(len(scenarios)), vals, bottom=bottom, color=CLASS_COLOR[cls], label=cls)
            bottom += vals
        counts = pila_only_fail[pila_only_fail['arm'] == arm]['scenario'].value_counts()
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([f'{s}\nn={counts.get(s, 0)}' for s in scenarios], rotation=45, ha='right', fontsize=8)
        ax.set_title(arm, fontsize=9)
    axes[0].set_ylabel('Fraction of PILA failures (classical succeeded)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(os.path.join(args.out_dir, 'fig_a_failclass_by_scenario.png'), dpi=150)
    plt.close(fig)

    # (b) per-date success (pooled over seeds) vs date, G1, slot_resid vs classical, per noise level
    noise_levels = sorted(active_rows[active_rows['layout_set'] == 'G1']['noise_scale'].unique())
    g1_scen = sorted(table[table['layout_set'] == 'G1']['scenario'].unique())
    fig, axes = plt.subplots(len(g1_scen), 2, figsize=(11, 2.4 * len(g1_scen)), sharex=True, squeeze=False)
    cmap = plt.get_cmap('viridis')
    for row, scen in enumerate(g1_scen):
        for src in range(2):
            ax = axes[row, src]
            for k, noise in enumerate(noise_levels):
                sub = table[(table['layout_set'] == 'G1') & (table['scenario'] == scen) & (table['source'] == src)
                            & (table['noise_scale'] == noise) & (table['arm'] == 'slot_resid')]
                if sub.empty:
                    continue
                rate = sub.groupby('date_index').apply(
                    lambda g: g['success'].astype(float).mean() if g['active'].all() else np.nan)
                ax.plot(rate.index, rate.values, '-o', ms=2, color=cmap(k / max(1, len(noise_levels) - 1)),
                        label=f'noise x{noise:g}')
            peaks = table[(table['layout_set'] == 'G1') & (table['scenario'] == scen) & (table['source'] == src)
                          & (table['noise_scale'] == 0) & (table['arm'] == 'slot_resid') & (table['seed'] == 1)]
            ax2 = ax.twinx()
            ax2.plot(peaks['date_index'], peaks['true_peak_mm'], 'k--', lw=1)
            ax2.set_ylabel('True own peak (mm)', fontsize=7)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'{scen} - source {src}', fontsize=9)
            ax.set_ylabel('PILA success rate (8 seeds)', fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('Date index')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out_dir, 'fig_b_G1_success_vs_date_slot_resid.png'), dpi=150)
    plt.close(fig)

    # (c) where the missed signal went (field arms only)
    if 'share_lowrank' in table.columns:
        fs = active_rows.dropna(subset=['share_lowrank'])
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, col, name in zip(axes, ['share_phys', 'share_lowrank', 'share_misfit'],
                                 ['physics', 'low-rank residual', 'misfit (data - recon)']):
            bins = np.linspace(-0.5, 1.5, 61)
            ax.hist(fs[fs['success']][col].clip(-0.5, 1.5), bins=bins, alpha=0.6, density=True, label='success')
            ax.hist(fs[~fs['success']][col].clip(-0.5, 1.5), bins=bins, alpha=0.6, density=True, label='failure')
            ax.set_xlabel(f'Share of true source field in {name} (-)')
            ax.set_ylabel('Density (-)')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, 'fig_c_signal_shares.png'), dpi=150)
        plt.close(fig)
        share_tab = fs.groupby(['scenario', 'success'])[['share_phys', 'share_lowrank', 'share_misfit']].median()
        share_tab.to_csv(os.path.join(args.out_dir, 'signal_shares_median.csv'))
        with pd.option_context('display.width', 200, 'display.precision', 2):
            print('  Median signal shares (field arms):')
            print(share_tab)

    # --- 6. Done ---
    print(f'[6/6] Total {time.time() - t_start:.0f}s')
    print(f'Done. Output saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
