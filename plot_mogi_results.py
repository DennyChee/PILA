#!/usr/bin/env python
# Usage:       python plot_mogi_results.py --csv path/to/model_best_testset_analyzer.csv
# Description: Visualise PILA Mogi test-set results: predicted vs target displacement,
#              per-station RMSE, inferred Mogi source parameters, and time-series comparison.
# Date:        2026-06-12

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# --- Station list and component definitions ---
STATIONS = ['AKGG', 'AKLV', 'AKMO', 'AKRB', 'AV06', 'AV07', 'AV08',
            'AV10', 'AV12', 'AV13', 'AV14', 'AV15']
COMPONENTS = ['ux', 'uy', 'uz']
COMPONENT_LABELS = {'ux': 'East (ux)', 'uy': 'North (uy)', 'uz': 'Up (uz)'}

# Mogi latent parameter metadata
MOGI_PARAMS = {
    'latent_xcen': ('Source Easting',  'm'),
    'latent_ycen': ('Source Northing', 'm'),
    'latent_d':    ('Source Depth',    'm'),
    'latent_dV':   ('Volume Change',   'm³'),
}


def compute_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root mean squared error between predicted and true arrays."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def load_results(csv_path: str) -> pd.DataFrame:
    """Load testset analyzer CSV and parse date column."""
    print(f"[1/5] Loading results from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Parse date column (format: YYYY.MM.DD)
    df['date_parsed'] = pd.to_datetime(df['date'].astype(str), format='%Y.%m.%d')
    print(f"  Loaded: {len(df)} samples, {len(df.columns)} columns")
    print(f"  Date range: {df['date_parsed'].min().date()} to {df['date_parsed'].max().date()}")
    return df


def plot_scatter(df: pd.DataFrame, out_dir: str):
    """
    Figure 1 — Predicted vs target scatter for each displacement component.
    All stations pooled together; 1:1 line; R² and RMSE annotated.
    """
    print("[2/5] Plotting predicted vs target scatter...")
    t0 = time.time()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, comp in zip(axes, COMPONENTS):
        pred_cols   = [f'output_{comp}_{s}' for s in STATIONS]
        target_cols = [f'target_{comp}_{s}' for s in STATIONS]

        y_pred = df[pred_cols].values.ravel()
        y_true = df[target_cols].values.ravel()

        rmse = compute_rmse(y_pred, y_true)
        r2   = compute_r2(y_pred, y_true)

        ax.scatter(y_true, y_pred, alpha=0.15, s=4, color='steelblue', rasterized=True)

        # 1:1 reference line
        lim_min = min(y_true.min(), y_pred.min())
        lim_max = max(y_true.max(), y_pred.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=1.2, label='1:1')

        ax.set_xlabel('Target (normalized)', fontsize=11)
        ax.set_ylabel('Predicted (normalized)', fontsize=11)
        ax.set_title(COMPONENT_LABELS[comp], fontsize=12)
        ax.annotate(f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
                    xy=(0.05, 0.92), xycoords='axes fraction',
                    fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.legend(fontsize=8)

    fig.suptitle('PILA Mogi — Predicted vs Target Displacement (all stations)', fontsize=13)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fpath = os.path.join(out_dir, f'scatter_pred_vs_target.{ext}')
        fig.savefig(fpath, dpi=150)
        print(f"  Saved: {fpath}")
    plt.close(fig)
    print(f"  Done in {time.time() - t0:.1f}s")


def plot_per_station_rmse(df: pd.DataFrame, out_dir: str):
    """
    Figure 2 — Per-station RMSE bar chart for each displacement component.
    Helps identify which stations are harder to reconstruct.
    """
    print("[3/5] Plotting per-station RMSE...")
    t0 = time.time()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    bar_width = 0.6

    for ax, comp in zip(axes, COMPONENTS):
        rmse_per_station = []
        for station in STATIONS:
            y_pred = df[f'output_{comp}_{station}'].values
            y_true = df[f'target_{comp}_{station}'].values
            rmse_per_station.append(compute_rmse(y_pred, y_true))

        bars = ax.bar(STATIONS, rmse_per_station, width=bar_width, color='steelblue', edgecolor='white')
        ax.set_xlabel('Station', fontsize=11)
        ax.set_ylabel('RMSE (normalized)', fontsize=11)
        ax.set_title(f'Per-station RMSE — {COMPONENT_LABELS[comp]}', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # Annotate bar values
        for bar, val in zip(bars, rmse_per_station):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('PILA Mogi — Per-station Reconstruction RMSE', fontsize=13)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fpath = os.path.join(out_dir, f'per_station_rmse.{ext}')
        fig.savefig(fpath, dpi=150)
        print(f"  Saved: {fpath}")
    plt.close(fig)
    print(f"  Done in {time.time() - t0:.1f}s")


def plot_timeseries(df: pd.DataFrame, out_dir: str, n_stations: int = 4):
    """
    Figure 3 — Time-series of predicted vs target vertical displacement (uz)
    for a selection of stations.
    """
    print("[4/5] Plotting displacement time series...")
    t0 = time.time()

    # Pick n_stations spread across the list
    indices = np.linspace(0, len(STATIONS) - 1, n_stations, dtype=int)
    selected_stations = [STATIONS[i] for i in indices]

    dates = df['date_parsed'].values

    fig, axes = plt.subplots(n_stations, 1, figsize=(14, 3 * n_stations), sharex=True)

    for ax, station in zip(axes, selected_stations):
        y_pred = df[f'output_uz_{station}'].values
        y_true = df[f'target_uz_{station}'].values
        rmse   = compute_rmse(y_pred, y_true)

        ax.plot(dates, y_true, color='black', lw=0.8, alpha=0.8, label='Target')
        ax.plot(dates, y_pred, color='tomato', lw=0.8, alpha=0.9, label='Predicted', linestyle='--')
        ax.set_ylabel('Up uz\n(normalized)', fontsize=9)
        ax.set_title(f'Station {station}   (RMSE = {rmse:.3f})', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Date', fontsize=11)
    fig.suptitle('PILA Mogi — Vertical Displacement Time Series (uz)', fontsize=13)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fpath = os.path.join(out_dir, f'timeseries_uz.{ext}')
        fig.savefig(fpath, dpi=150)
        print(f"  Saved: {fpath}")
    plt.close(fig)
    print(f"  Done in {time.time() - t0:.1f}s")


def plot_mogi_parameters(df: pd.DataFrame, out_dir: str):
    """
    Figure 4 — Inferred Mogi source parameters over time.
    Shows xcen, ycen (m), depth (m), and volume change (m³).
    """
    print("[5/5] Plotting inferred Mogi source parameters...")
    t0 = time.time()

    dates = df['date_parsed'].values
    param_keys = list(MOGI_PARAMS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes_flat = axes.ravel()

    for ax, param_key in zip(axes_flat, param_keys):
        label, unit = MOGI_PARAMS[param_key]
        values = df[param_key].values

        ax.plot(dates, values, lw=0.7, color='steelblue', alpha=0.8)
        ax.set_ylabel(f'{label} ({unit})', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_title(label, fontsize=11)

        # Annotate median
        median_val = np.median(values)
        ax.axhline(median_val, color='tomato', lw=1.2, linestyle='--',
                   label=f'Median: {median_val:.1f} {unit}')
        ax.legend(fontsize=8)

    fig.suptitle('PILA Mogi — Inferred Source Parameters Over Time', fontsize=13)
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fpath = os.path.join(out_dir, f'mogi_parameters.{ext}')
        fig.savefig(fpath, dpi=150)
        print(f"  Saved: {fpath}")
    plt.close(fig)
    print(f"  Done in {time.time() - t0:.1f}s")


def main(args: argparse.Namespace):
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    df = load_results(args.csv)

    plot_scatter(df, out_dir)
    plot_per_station_rmse(df, out_dir)
    plot_timeseries(df, out_dir)
    plot_mogi_parameters(df, out_dir)

    print(f"\nDone. All figures saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot PILA Mogi test-set results.')
    parser.add_argument('--csv', required=True, type=str,
                        help='Path to model_best_testset_analyzer.csv')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save figures (default: same dir as CSV)')
    args = parser.parse_args()
    main(args)
