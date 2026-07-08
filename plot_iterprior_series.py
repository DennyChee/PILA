#!/usr/bin/env python3
# Usage:       python plot_iterprior_series.py --series saved/.../iterprior_series_<tag>.csv \
#                  [--raw test_out/<ckpt>_testset_analyzer.csv] [--out figures/iterprior]
# Description: Plot the per-epoch physical source-parameter time series produced by
#              run_iterative_prior_refit.py (previous-epoch-as-prior retraining), one
#              subplot per parameter vs date. Optionally overlays a RAW per-epoch
#              encoder series (no retraining) so the smoothing/coupling introduced by
#              the sequential prior can be read off directly.
# Date:        2026-07-01

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # headless HPC
import matplotlib.pyplot as plt
import pandas as pd

# Known physical units per Mogi parameter for axis labels (see Physics_Mogi.rescale:
# xcen/ycen/d are metres after the km->m *1000; dV is m^3). Unknown params fall back
# to a bare name so the script stays model-agnostic for Okada/Sun69.
UNITS = {'xcen': 'm', 'ycen': 'm', 'd': 'm', 'dV': 'm$^3$',
         'xoff': 'km', 'yoff': 'km', 'depth': 'km', 'strike': 'deg', 'dip': 'deg',
         'length': 'km', 'width': 'km', 'opening': 'm', 'radius': 'km'}


def _phys_param_cols(df):
    """Physical-parameter columns = not bookkeeping and not the u-space ('u_') copies."""
    skip = {'round', 'date', 'source'}
    return [c for c in df.columns if c not in skip and not c.startswith('u_')]


def main():
    parser = argparse.ArgumentParser(description='Plot iterprior per-epoch parameter series')
    parser.add_argument('--series', required=True, type=str,
                        help='iterprior_series_<tag>.csv written by run_iterative_prior_refit.py')
    parser.add_argument('--raw', default=None, type=str,
                        help='optional RAW per-epoch series CSV to overlay (e.g. latent_* from test_pila_mogi.py)')
    parser.add_argument('--out', default=None, type=str,
                        help='output figure path stem (default: next to --series)')
    parser.add_argument('--dv-eps-frac', default=0.05, type=float,
                        help='Epochs whose |dV| < dv_eps_frac * peak|dV| are treated as '
                             'having no resolvable source: geometry (location/depth) is masked '
                             'there because a ~0-amplitude Mogi/Sun69 field leaves depth/location '
                             'unidentifiable. dV itself is never masked. Set 0 to disable.')
    args = parser.parse_args()

    if not os.path.exists(args.series):
        raise FileNotFoundError(f"series CSV not found: {args.series}")

    print(f"[1/3] Loading retrained series: {args.series}")
    df = pd.read_csv(args.series)
    # Both date sources emit dashed ISO dates (YYYY-MM-DD): insar_mintpy
    # date_bytes_to_iso() and displacementGPS strftime('%Y-%m-%d'). Parsing with
    # a dotted format silently coerces every date to NaT (blank x-axis), so match
    # the actual format and fail loudly if anything fails to parse.
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    if df['date'].isna().any():
        n_bad = int(df['date'].isna().sum())
        raise ValueError(
            f"{n_bad}/{len(df)} dates in {args.series} failed to parse as "
            f"YYYY-MM-DD; refusing to plot with a garbage time axis.")
    params = _phys_param_cols(df)
    print(f"  {len(df)} epochs, params = {params}")

    raw_df = None
    if args.raw is not None and os.path.exists(args.raw):
        raw_df = pd.read_csv(args.raw)
        # test_pila_* stores physical params under 'latent_<name>'; map to bare names.
        rename = {f'latent_{p}': p for p in params if f'latent_{p}' in raw_df.columns}
        raw_df = raw_df.rename(columns=rename)
        if 'date' in raw_df.columns:
            # Same dashed-ISO format as the main series (see note above).
            raw_df['date'] = pd.to_datetime(raw_df['date'], format='%Y-%m-%d', errors='coerce')
            if raw_df['date'].isna().any():
                n_bad = int(raw_df['date'].isna().sum())
                raise ValueError(
                    f"{n_bad}/{len(raw_df)} dates in {args.raw} failed to parse "
                    f"as YYYY-MM-DD; refusing to plot raw overlay with a bad axis.")
            raw_df = raw_df.sort_values('date')
        print(f"  raw overlay: {len(raw_df)} epochs from {args.raw}")

    # --- Depth/location gating on near-zero dV ----------------------------------------
    # A Mogi/Sun69 field is linear in dV, so at dV~0 there is no source and depth/location
    # are unidentifiable. Mask the geometry tracks at epochs whose |dV| < dv_eps_frac*peak,
    # so a null-epoch depth (arbitrary floating value) is not read as signal. dV itself is
    # never masked. The retrained and raw series are each gated by their OWN dV column.
    AMP_NAMES = {'dV', 'opening'}

    def gated_dates(frame):
        """Epoch dates in `frame` whose |dV| < dv_eps_frac * peak|dV| (empty if disabled)."""
        if args.dv_eps_frac <= 0 or 'dV' not in frame.columns or not len(frame):
            return frame['date'].iloc[0:0]
        peak = float(frame['dV'].abs().max())
        return frame['date'][frame['dV'].abs() < args.dv_eps_frac * peak]

    def masked(frame, p):
        """Series `p` with geometry values NaN'd at that frame's gated (dV~0) epochs."""
        s = frame[p].copy()
        if p not in AMP_NAMES and args.dv_eps_frac > 0 and 'dV' in frame.columns and len(frame):
            peak = float(frame['dV'].abs().max())
            s = s.mask(frame['dV'].abs() < args.dv_eps_frac * peak)
            return s, True
        return s, False

    df_gated_dates = gated_dates(df)

    print("[2/3] Plotting one subplot per parameter...")
    n = len(params)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, params):
        # Shade the retrained series' unresolved epochs on geometry subplots.
        if p not in AMP_NAMES and len(df_gated_dates):
            for d in df_gated_dates:
                ax.axvline(d, color='0.88', lw=3, alpha=0.7, zorder=0, label='_nolegend_')
        # Retrained (previous-epoch-as-prior) series; mark the bootstrap epoch 0.
        rec_series, _ = masked(df, p)
        ax.plot(df['date'], rec_series, '-o', color='C0', ms=4, label='retrained (prior=prev epoch)')
        boot = df[df['source'] == 'bootstrap']
        if len(boot):
            boot_series, _ = masked(boot, p)
            ax.plot(boot['date'], boot_series, 's', color='k', ms=7, label='bootstrap (raw E0)')
        if raw_df is not None and p in raw_df.columns:
            raw_series, _ = masked(raw_df, p)
            ax.plot(raw_df['date'], raw_series, '--x', color='C3', ms=4, alpha=0.7,
                    label='raw encoder (no retrain)')
        unit = UNITS.get(p, '')
        ax.set_ylabel(f'{p}' + (f' ({unit})' if unit else ''))
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8)
    axes[-1].set_xlabel('Date')
    fig.suptitle('PILA sequential previous-epoch-as-prior: per-epoch source parameters')
    if len(df_gated_dates):
        print(f"  Gating: {len(df_gated_dates)}/{len(df)} epochs have |dV| < "
              f"{100*args.dv_eps_frac:.0f}% of peak; geometry masked (shaded) there.")
        fig.text(0.5, 0.005,
                 f'Shaded epochs: |dV| < {100*args.dv_eps_frac:.0f}% of peak '
                 f'— source unresolved, location/depth undefined (masked).',
                 ha='center', va='bottom', fontsize=8, color='0.35')
    fig.tight_layout()

    out_stem = args.out or os.path.splitext(args.series)[0]
    os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
    print("[3/3] Saving figures...")
    fig.savefig(out_stem + '.png', dpi=150)
    fig.savefig(out_stem + '.pdf')
    plt.close(fig)
    print(f"Done. Saved: {out_stem}.png and {out_stem}.pdf")


if __name__ == '__main__':
    main()
