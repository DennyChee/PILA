#!/usr/bin/env python3
# Usage:       python plot_iterprior_vs_truth.py \
#                  --series saved/iterprior_marapi_mogi_rising/base/iterprior_series.csv \
#                  --truth  synthetic/cubes/marapi_mogi_rising_combined/marapi_mogi_rising_combined_truth.json \
#                  [--out saved/iterprior_marapi_mogi_rising/base/iterprior_vs_truth]
# Description: Overlay the PILA iterative-prior recovered per-epoch source parameters
#              (from run_iterative_prior_refit.py) against the synthetic GROUND TRUTH
#              trajectory, styled to MATCH synthetic/plots.py::plot_param_recovery_compare
#              (the *_param_recovery_compare.png figures): one subplot per parameter vs
#              epoch index, black solid truth, dashed recovered track, legend in the
#              spare grid slot, matching suptitle.
# Date:        2026-07-02
#
# Scientific notes:
#   * Title/noise-state/method are derived from the truth sidecar + --label, so the figure
#     annotation matches the actual run (cube name, noise-free vs realized SNR, method).
#   * For a *rising* (inflation) source the signal RMS grows over epochs, so early low-SNR
#     epochs are noise-dominated and the recovered track is unreliable there; for a
#     constant-amplitude moving source every epoch shares the same SNR.
#   * Units: xcen/ycen/d in metres, dV in m^3. LOS convention positive = toward sat.

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless HPC
import matplotlib.pyplot as plt
import pandas as pd

# Match the presentation defaults used by synthetic/plots.py so fonts/line widths agree.
plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'legend.framealpha': 0.9, 'figure.titlesize': 18,
    'axes.linewidth': 1.1, 'lines.linewidth': 2.2,
})


def main():
    parser = argparse.ArgumentParser(description='Overlay iterprior recovered params vs synthetic truth')
    parser.add_argument('--series', required=True, type=str,
                        help='iterprior_series.csv from run_iterative_prior_refit.py')
    parser.add_argument('--truth', required=True, type=str,
                        help='*_truth.json sidecar with params_per_epoch')
    parser.add_argument('--out', default=None, type=str,
                        help='output figure stem (default: next to --series as iterprior_vs_truth)')
    parser.add_argument('--label', default='PILA recovered', type=str,
                        help='legend text for the recovered track and the method shown in the '
                             'title (e.g. "iterprior tight", "no prior (independent E0)"). '
                             'Set this per run so the figure annotation matches what was run.')
    parser.add_argument('--dv-eps-frac', default=0.05, type=float,
                        help='Epochs whose RECOVERED |dV| < dv_eps_frac * peak|dV| are treated '
                             'as having no resolvable source: the recovered geometry (location/'
                             'depth) is masked there, since a Mogi field of ~0 amplitude leaves '
                             'depth/location unidentifiable. dV itself is never masked. '
                             'Set 0 to disable masking.')
    args = parser.parse_args()

    for path in (args.series, args.truth):
        if not os.path.exists(path):
            raise FileNotFoundError(f"input not found: {path}")

    # --- Load recovered series and truth trajectory ---
    print(f"[1/3] Loading recovered series: {args.series}")
    df = pd.read_csv(args.series)
    print(f"  Loaded: shape={df.shape}")

    with open(args.truth) as fh:
        truth_meta = json.load(fh)
    param_names = truth_meta['param_names']                       # ['xcen','ycen','d','dV']
    param_units = truth_meta.get('param_units', [''] * len(param_names))
    truth = np.asarray(truth_meta['params_per_epoch'], dtype=float)   # (n_epoch, n_param)
    rec = df[param_names].values                                  # (n_epoch, n_param)
    n_epoch, n_param = truth.shape
    if len(df) != n_epoch:
        n_epoch = min(len(df), n_epoch)
        print(f"  WARNING: series rows ({len(df)}) != truth epochs ({truth.shape[0]}); "
              f"plotting first {n_epoch}.")
        truth, rec = truth[:n_epoch], rec[:n_epoch]
    print(f"  Truth: {n_epoch} epochs, params={param_names}, model={truth_meta['forward_model']}")

    # --- Depth/location gating on near-zero recovered dV -------------------------------
    # Physics: the Mogi/Sun69 surface field is linear in dV, so at dV~0 the output is ~0
    # for ANY depth/location -> geometry is unidentifiable (no source). We therefore mask
    # the RECOVERED geometry tracks at epochs whose recovered |dV| < dv_eps_frac*peak|dV|,
    # so a null-epoch depth (an arbitrary floating value) is not read as signal. Truth and
    # the dV track itself are left untouched. `dv_gated` records the masked epochs for the
    # light shading + caption note below.
    AMP_NAMES = {'dV', 'opening'}                 # source strength; never gated
    rec_plot = rec.copy()
    dv_gated = np.zeros(n_epoch, dtype=bool)
    if args.dv_eps_frac > 0 and 'dV' in param_names and np.isfinite(rec[:, param_names.index('dV')]).any():
        dv_col = param_names.index('dV')
        peak_abs_dV = float(np.nanmax(np.abs(rec[:, dv_col])))
        dv_eps = args.dv_eps_frac * peak_abs_dV
        dv_gated = np.abs(rec[:, dv_col]) < dv_eps
        if dv_gated.any():
            for j, pname in enumerate(param_names):
                if pname not in AMP_NAMES:
                    rec_plot[dv_gated, j] = np.nan   # break the recovered geometry track
            print(f"  Gating: {int(dv_gated.sum())}/{n_epoch} epochs have recovered "
                  f"|dV| < {dv_eps:.3g} m^3 (={100*args.dv_eps_frac:.0f}% of peak "
                  f"{peak_abs_dV:.3g}); masking geometry there.")

    # --- Grid of subplots matching plot_param_recovery_compare (3 cols, spare-slot legend) ---
    print("[2/3] Plotting (matched to *_param_recovery_compare style)...")
    ncol = min(3, n_param)
    nrow = int(np.ceil(n_param / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow), squeeze=False)
    ep = np.arange(n_epoch)  # x-axis is the epoch index, exactly like the reference figure

    for j, (pname, punit) in enumerate(zip(param_names, param_units)):
        ax = axes[j // ncol][j % ncol]
        # Truth = solid black; recovered iterprior track = dashed blue squares (mirrors the
        # 'PILA blended' styling in plot_param_recovery_compare, since iterprior IS the
        # temporally-coupled analogue of that blended series).
        ax.plot(ep, truth[:, j], 'k-', lw=2, label='truth')
        ax.plot(ep, rec_plot[:, j], 's--', color='C0', ms=3, alpha=0.8,
                label=args.label)
        # Shade epochs where the source is unresolved (dV~0) on the gated (geometry)
        # subplots, so the break in the recovered track reads as "no estimate here".
        if dv_gated.any() and pname not in AMP_NAMES:
            for e in np.where(dv_gated)[0]:
                ax.axvspan(e - 0.5, e + 0.5, color='0.85', alpha=0.5, lw=0,
                           label='_nolegend_')
        ax.set_xlabel('Epoch index')
        ax.set_ylabel(pname if not punit else f'{pname} ({punit})')
        ax.set_title(pname)
        ax.grid(alpha=0.3)

        # --- Fluctuation annotation: RMSE of the recovered track about the TRUE trajectory.
        # Captures scatter around the (constant OR moving) truth; ~= jitter since ~unbiased.
        # Shown as % of |truth| where the truth stays away from zero (d, dV -> % is meaningful);
        # shown as absolute (in param units) for params whose truth crosses zero (xcen/ycen),
        # where a "%" would blow up. The parenthetical is the burn-in-excluded value (epochs 2+),
        # since the one-off round-1 retraining transient otherwise dominates the constant params.
        # nan-aware: gated (masked) epochs are NaN in rec_plot and are excluded from the
        # fluctuation metric, so the RMSE reflects only epochs with a resolvable source.
        resid = rec_plot[:, j] - truth[:, j]
        rmse_all = float(np.sqrt(np.nanmean(resid ** 2))) if np.isfinite(resid).any() else float('nan')
        rmse_be = (float(np.sqrt(np.nanmean(resid[2:] ** 2)))
                   if n_epoch > 2 and np.isfinite(resid[2:]).any() else rmse_all)
        tmean = float(np.mean(np.abs(truth[:, j])))
        crosses_zero = (truth[:, j].min() < 0.0 < truth[:, j].max())
        if (not crosses_zero) and tmean > 0:
            txt = (f"fluctuation (RMSE vs truth)\n{100*rmse_all/tmean:.1f}%  |  "
                   f"excl. burn-in {100*rmse_be/tmean:.1f}%")
        else:
            u = (' ' + punit) if punit else ''
            txt = (f"fluctuation (RMSE vs truth)\n{rmse_all:.0f}{u}  |  "
                   f"excl. burn-in {rmse_be:.0f}{u}")
        # Auto-place the box in the emptiest corner (least overlap with truth+recovered points).
        # nan-aware: masked recovered epochs (NaN) are ignored in the extent and, via the
        # nan<threshold comparisons below, contribute no occupancy count.
        yall = np.concatenate([truth[:, j], rec_plot[:, j]])
        ymin, ymax = float(np.nanmin(yall)), float(np.nanmax(yall))
        yr = (ymax - ymin) or 1.0
        xr = (float(ep.max()) - float(ep.min())) or 1.0
        px = np.concatenate([(ep - ep.min()) / xr, (ep - ep.min()) / xr])
        py = np.concatenate([(truth[:, j] - ymin) / yr, (rec_plot[:, j] - ymin) / yr])
        corners = {'tl': (0.02, 0.98, 'left', 'top'), 'tr': (0.98, 0.98, 'right', 'top'),
                   'bl': (0.02, 0.02, 'left', 'bottom'), 'br': (0.98, 0.02, 'right', 'bottom')}
        counts = {c: int(np.sum((np.abs(px - cx) < 0.45) & (np.abs(py - cy) < 0.45)))
                  for c, (cx, cy) in {'tl': (0, 1), 'tr': (1, 1), 'bl': (0, 0), 'br': (1, 0)}.items()}
        tx, ty, ha, va = corners[min(counts, key=counts.get)]
        ax.text(tx, ty, txt, transform=ax.transAxes, va=va, ha=ha,
                fontsize=8.5, bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))

    # Legend goes in the first empty grid slot (same as the reference figure).
    handles, labels = axes[0][0].get_legend_handles_labels()
    empty = list(range(n_param, nrow * ncol))
    for j in empty:
        axes[j // ncol][j % ncol].axis('off')
    if empty:
        slot = axes[empty[0] // ncol][empty[0] % ncol]
        slot.legend(handles, labels, loc='center', frameon=True)
    else:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02))

    # Suptitle derived from the truth sidecar so it reflects the ACTUAL cube + noise state
    # + method (no hardcoded "rising"/"iterative prior"/SNR that lie on other runs).
    pretty_name = truth_meta.get('name', 'synthetic').replace('_', ' ').title()
    noise_meta = truth_meta.get('noise', {})
    if noise_meta.get('type', 'none') == 'none' or 'realized_snr_db' not in noise_meta:
        noise_desc = 'noise-free'
    else:
        noise_desc = f"peak SNR {noise_meta['realized_snr_db']:.0f} dB"
    suptitle = (f'{pretty_name}: parameter recovery vs epoch '
                f'— {args.label}  [{noise_desc}]')
    fig.suptitle(suptitle, y=1.0)

    # Note the gating so a reader understands the broken/shaded recovered geometry tracks.
    if dv_gated.any():
        fig.text(0.5, -0.03,
                 f'Shaded epochs: recovered |dV| < {100*args.dv_eps_frac:.0f}% of peak '
                 f'— source unresolved, recovered location/depth undefined (masked).',
                 ha='center', va='top', fontsize=9, color='0.35')

    # --- Save png(dpi=150) + pdf, per house style ---
    out_stem = args.out or os.path.join(os.path.dirname(args.series), 'iterprior_vs_truth')
    os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
    print("[3/3] Saving figures...")
    fig.tight_layout()
    fig.savefig(out_stem + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_stem + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Done. Saved: {out_stem}.png and {out_stem}.pdf")


if __name__ == '__main__':
    main()
