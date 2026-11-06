#!/usr/bin/env python
# Usage:       python compare_dvbox_campaign.py [--out-dir figures/dvbox_rerun/campaign]
# Description: Three-way comparison of the PILA-vs-vbICA campaign cell (curv15000 + dvl1u300) on the 12 synthetic cubes:
#                old   = original 2026-09-07 runs (11/12 had the decoder standardised with rising_combined's mean/std)
#                a     = decoder-path fix, old amplitude box            (cell curv15000_dvl1u300_pathfix)
#                b     = decoder-path fix + data-scaled amplitude box   (cell curv15000_dvl1u300_pathfix_dvbox)
#              Scores use score_pila_variants.score_variant (identical metric definitions to the published table),
#              plus the median amplitude ratio (recovered / true dV or opening) over strong epochs. vbICA location error
#              and the static floor are read (read-only) from pila_bench/results/summary.csv for context.
#              Writes ONLY into --out-dir; the published pila_bench/results/*.csv are not touched.
# Output:      campaign_3arm.csv, campaign_3arm.png, recovery/<cube>_old_a_b.png (param-recovery panels side by side)
# Date:        2026-10-22

import argparse
import importlib.util
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = '/eos-rs/INSAR_processing/denny/matlab/vbICA/Code_denny/pila_bench'
VBICA_SUMMARY = os.path.join(BENCH_DIR, 'results', 'summary.csv')
CUBES = ['marapi_mogi_move_loc', 'marapi_mogi_move_loc_clean', 'marapi_mogi_move_loc_hisnr', 'marapi_mogi_move_locdepth',
         'marapi_mogi_rising_clean', 'marapi_mogi_rising_combined', 'marapi_mogi_rising_lineardV',
         'marapi_mogi_rising_lineardV_hisnr', 'marapi_mogi_staticloc_lineardV', 'marapi_mogi_staticloc_sigmoiddV',
         'marapi_okada_dike_combined', 'marapi_sun69_sill_combined']
ARMS = {'old': ('curv15000_dvl1u300', 'old (wrong decoder std)', '0.6'),
        'a': ('curv15000_dvl1u300_pathfix', 'a: path fix, old box', 'C0'),
        'b': ('curv15000_dvl1u300_pathfix_dvbox', 'b: path fix + data-scaled box', 'C3')}
AMP_PARAM = {'mogi': 'dV', 'sun69': 'dV', 'okada': 'opening'}
DEPTH_PARAM = {'mogi': 'd', 'sun69': 'depth', 'okada': 'depth'}


def load_scorer():
    """Import pila_bench/score_pila_variants.py by path (its main() writes results; we only call its functions)."""
    spec = importlib.util.spec_from_file_location('score_pila_variants', os.path.join(BENCH_DIR, 'score_pila_variants.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def amplitude_ratio(csv_path, truth, model):
    """Median over strong epochs of recovered / true amplitude (dV m^3 or opening m); 1 = unbiased."""
    series = pd.read_csv(csv_path)
    param = AMP_PARAM[model]
    true_values = np.asarray(truth['params_per_epoch'], dtype=float)[:, truth['param_names'].index(param)]
    rms_mm = np.asarray(truth['signal_rms_mm_per_epoch'], dtype=float)
    strong = rms_mm >= 0.5 * rms_mm.max()
    return float(np.median(series[param].to_numpy(float)[strong] / true_values[strong]))


def side_by_side(cube, out_path):
    """Param-recovery PNGs of the three arms, left to right."""
    images, titles = [], []
    for arm, (cell, label, _) in ARMS.items():
        png = os.path.join(PROJECT_ROOT, 'saved', f'combined_curvdvl1u_{cube}', cell,
                           f'param_recovery_combined_{cell}_{cube}.png')
        if not os.path.exists(png):
            print(f"  WARNING: missing {png}")
            return
        images.append(plt.imread(png))
        titles.append(label)
    fig, axes = plt.subplots(1, 3, figsize=(30, 30 * images[0].shape[0] / images[0].shape[1] / 3 + 1))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=16)
        axis.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='figures/dvbox_rerun/campaign')
    args = parser.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(os.path.join(out_dir, 'recovery'), exist_ok=True)

    # --- [1/3] Score every cube x arm ---
    print("[1/3] Scoring 12 cubes x 3 arms...")
    scorer = load_scorer()
    rows = []
    for cube in CUBES:
        truth = scorer.load_truth(cube)
        model = str(truth['forward_model']).lower()
        for arm, (cell, _, _) in ARMS.items():
            csv_path = os.path.join(PROJECT_ROOT, 'saved', f'combined_curvdvl1u_{cube}', cell, 'uspace_series.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(csv_path)
            score = scorer.score_variant(cube, f'combined_curvdvl1u/{cell}', csv_path, truth)
            rows.append({'cube': cube, 'model': model, 'arm': arm,
                         'loc_err_km': score['source_loc_err_km_strong_mean'],
                         'static_floor_km': score['source_loc_err_km_static_floor'],
                         'depth_nmae_pct': 100 * score[f"nmae_{DEPTH_PARAM[model]}"],
                         'amp_ratio_median': amplitude_ratio(csv_path, truth, model),
                         'nmae_mean': score['selection_score_nmae_mean']})
    table = pd.DataFrame(rows)
    vbica = pd.read_csv(VBICA_SUMMARY).set_index('tag')['source_loc_err_km_strong_mean']
    table['vbica_loc_err_km'] = table['cube'].map(vbica)
    table.to_csv(os.path.join(out_dir, 'campaign_3arm.csv'), index=False, float_format='%.4g')
    wide = table.pivot(index='cube', columns='arm', values=['amp_ratio_median', 'loc_err_km', 'depth_nmae_pct', 'nmae_mean'])
    print(wide.round(3).reindex(CUBES).to_string())

    # --- [2/3] Summary figure ---
    print("[2/3] Summary figure...")
    short = [cube.replace('marapi_', '').replace('_combined', '') for cube in CUBES]
    positions = np.arange(len(CUBES))
    fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)
    panels = (('amp_ratio_median', 'Amplitude recovered / true\n(median, strong epochs)', 'log'),
              ('loc_err_km', 'Location error (km)\n(mean, strong epochs)', 'log'),
              ('depth_nmae_pct', 'Depth error (% of true)\n(mean, strong epochs)', 'log'),
              ('nmae_mean', 'Mean normalised error\n(all parameters)', 'log'))
    for axis, (key, ylabel, scale) in zip(axes, panels):
        for index, (arm, (_, label, colour)) in enumerate(ARMS.items()):
            values = table[table['arm'] == arm].set_index('cube').loc[CUBES, key].to_numpy()
            axis.bar(positions + (index - 1) * 0.27, np.abs(values), width=0.27, color=colour, label=label)
        axis.set_yscale(scale)
        axis.set_ylabel(ylabel)
    axes[0].axhline(1.0, color='k', lw=1)
    floor = table[table['arm'] == 'old'].set_index('cube').loc[CUBES]
    axes[1].scatter(positions, floor['vbica_loc_err_km'], marker='D', color='k', zorder=3, label='vbICA (N=5)')
    axes[1].scatter(positions, floor['static_floor_km'].clip(lower=1e-3), marker='_', s=400, color='k', zorder=3,
                    label='static floor')
    axes[0].legend(fontsize=9, ncol=3)
    axes[1].legend(fontsize=9, ncol=2)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(short, rotation=35, ha='right')
    fig.suptitle('PILA-vs-vbICA campaign (curv15000 + dvl1u300): old vs decoder-path fix vs + data-scaled box')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'campaign_3arm.png'), dpi=150)
    plt.close(fig)

    # --- [3/3] Per-cube recovery panels side by side ---
    print("[3/3] Side-by-side recovery panels...")
    for cube in CUBES:
        side_by_side(cube, os.path.join(out_dir, 'recovery', f'{cube}_old_a_b.png'))
    print(f"Done. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
