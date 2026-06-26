#!/usr/bin/env python
# Usage:       python -m synthetic.validate_generation \
#                  --spec synthetic/specs/marapi_sun69_buildup_combined.json
#                  [--epochs all | --epochs 0,10,20,75]
# Description: Manual, component-by-component validation of a synthetic cube's
#              GENERATION. Reconstructs every additive layer separately on the same
#              grid/LOS the generator used:
#                deformation | stratified (trop) | turbulent (atm) | orbital (linear)
#                | white | combined-noise | total (signal+noise)
#              and renders (a) a per-epoch decomposition figure (all components side
#              by side, each with its own colorbar so structure is visible), (b) a
#              per-component time-series montage, and (c) a per-component RMS-vs-epoch
#              summary. Also CROSS-CHECKS that trop+atm+orbit+white == 'combined'.
# Scientific notes:
#   * Components are referenced to epoch 0 and scaled EXACTLY as the generator does
#     (cumulative referencing; same global scale), so total == the saved cube.
#   * LOS sign: + toward satellite. Units shown in mm.
# Date:        2026-06-22

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic.generate_timeseries import (
    load_full_res_geometry, read_base_dates_attrs, build_clean_cube, _read_mask)
from synthetic import trajectories as traj
from synthetic import noise as noisemod

# Individual physical components and display labels (order = additive layers).
COMPONENT_LABELS = [
    ('trop', 'Stratified APS (trop)'),
    ('turbulent', 'Turbulent APS (atm)'),
    ('orbit', 'Orbital ramp (linear)'),
    ('white', 'White noise'),
]


def reconstruct(spec):
    """Rebuild every additive layer of a synthetic cube. Returns a dict of
    [n_epoch, L, W] cubes in METRES plus dates, mask, params, and the sum-check."""
    base = spec['base_scene']
    model_name = spec['model']

    xE_m, yN_m, losE, losN, losU, (L, W) = load_full_res_geometry(
        base['geometry'], base['lat0'], base['lon0'])
    raw_dates, _, _ = read_base_dates_attrs(base['timeseries'])
    dates = [d.decode() if isinstance(d, (bytes, bytearray)) else str(d) for d in raw_dates]

    tr = spec['trajectory']
    n_epochs = tr.get('n_epochs', len(raw_dates))
    if tr['type'] == 'static_buildup':
        names, params = traj.static_buildup(
            model_name, n_epochs, tr['geometry'], tr['amplitude_key'],
            tr['amp_start'], tr['amp_end'], tr.get('profile', 'sigmoid'))
    else:
        names, params = traj.build_trajectory(
            model_name, n_epochs, tr['start'], tr['end'], tr.get('profile', 'linear'))

    # --- Deformation (clean signal) ---
    deformation = build_clean_cube(model_name, xE_m, yN_m, losE, losN, losU, names, params)

    mask = _read_mask(base.get('mask'), (L, W))
    layers = {'deformation': deformation}

    ns = spec.get('noise', {'type': 'none'})
    sumcheck = None
    if ns.get('type') == 'marapi':
        scale = float(ns.get('scale', 1.0))
        ref0 = ns.get('reference_epoch0', True)

        def _prep(cube):
            c = cube[:n_epochs].astype(np.float64)
            if ref0:
                c = c - c[0]
            return (c * scale).astype(np.float32)

        for key, _ in COMPONENT_LABELS:
            layers[key] = _prep(noisemod.load_marapi_noise([key]))
        combined = _prep(noisemod.load_marapi_noise(['combined']))
        layers['combined'] = combined
        comp_sum = sum(layers[k] for k, _ in COMPONENT_LABELS)
        sumcheck = float(np.max(np.abs(comp_sum - combined))) * 1000.0  # mm
        layers['total'] = (deformation + combined).astype(np.float32)
    else:
        layers['total'] = deformation

    return layers, dates, mask, names, params, sumcheck, n_epochs


def _panel(ax, field_mm, mask, title):
    if mask is not None:
        field_mm = np.where(mask, field_mm, np.nan)
    vmax = np.nanmax(np.abs(field_mm)) + 1e-9
    im = ax.imshow(field_mm, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
    ax.set_title(f'{title}\n(±{vmax:.1f} mm)', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)


def per_epoch_figures(out_dir, name, layers, dates, mask, epochs):
    """One figure per epoch: deformation + each noise component + total."""
    ed = os.path.join(out_dir, 'per_epoch'); os.makedirs(ed, exist_ok=True)
    has_noise = 'combined' in layers
    panels = [('deformation', 'Deformation (signal)')]
    if has_noise:
        panels += COMPONENT_LABELS + [('combined', 'Combined noise')]
    panels += [('total', 'Total (signal+noise)')]

    ncol = len(panels)
    for ep in epochs:
        fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.4))
        if ncol == 1:
            axes = [axes]
        for ax, (key, lbl) in zip(axes, panels):
            _panel(ax, layers[key][ep] * 1000.0, mask, lbl)
        fig.suptitle(f'{name}: epoch {ep} ({dates[ep]})  —  LOS mm, + toward satellite',
                     y=1.04)
        fig.tight_layout()
        fig.savefig(os.path.join(ed, f'epoch_{ep:03d}_{dates[ep]}.png'),
                    dpi=130, bbox_inches='tight')
        plt.close(fig)
    print(f"  wrote {len(epochs)} per-epoch figures to {ed}")


def component_montages(out_dir, name, layers, dates, mask, n_frames=None):
    """
    One contact-sheet montage per layer showing its temporal evolution. By default
    EVERY epoch is shown (n_frames=None); pass an int to subsample evenly. One shared
    symmetric color scale per layer so the buildup/decay is comparable across epochs.
    """
    md = os.path.join(out_dir, 'component_montages'); os.makedirs(md, exist_ok=True)
    n_epoch = next(iter(layers.values())).shape[0]
    if n_frames is None or n_frames >= n_epoch:
        idx = list(range(n_epoch))                       # every epoch
    else:
        idx = sorted(set(np.linspace(0, n_epoch - 1, n_frames).round().astype(int).tolist()))
    ncol = 8 if len(idx) > 24 else 4                     # wider sheet for many epochs
    nrow = int(np.ceil(len(idx) / ncol))
    for key, cube in layers.items():
        cube_mm = cube * 1000.0
        if mask is not None:
            cube_mm = np.where(mask[None], cube_mm, np.nan)
        vmax = np.nanmax(np.abs(cube_mm[idx])) + 1e-9
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow), squeeze=False)
        im = None
        for k, ep in enumerate(idx):
            ax = axes[k // ncol][k % ncol]
            im = ax.imshow(cube_mm[ep], cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
            ax.set_title(f'{ep}: {dates[ep]}', fontsize=6); ax.set_xticks([]); ax.set_yticks([])
        for k in range(len(idx), nrow * ncol):
            axes[k // ncol][k % ncol].axis('off')
        fig.suptitle(f'{name}: {key}  —  all {len(idx)} epochs (±{vmax:.1f} mm, + toward sat)',
                     y=1.0, fontsize=11)
        fig.subplots_adjust(right=0.91)
        cax = fig.add_axes([0.93, 0.15, 0.012, 0.7]); fig.colorbar(im, cax=cax, label='mm')
        fig.savefig(os.path.join(md, f'montage_{key}.png'), dpi=130, bbox_inches='tight')
        plt.close(fig)
    print(f"  wrote {len(layers)} component montages ({len(idx)} epochs each) to {md}")


def rms_summary(out_dir, name, layers, dates, mask):
    """RMS (mm) per epoch for each layer on one plot — quantitative buildup view."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, cube in layers.items():
        cube_mm = cube * 1000.0
        rms = np.array([noisemod._valid_rms(cube_mm[i], mask) for i in range(cube_mm.shape[0])])
        ax.plot(rms, label=key, lw=2 if key in ('deformation', 'total') else 1)
    ax.set_xlabel('Epoch index'); ax.set_ylabel('RMS over valid pixels (mm)')
    ax.set_title(f'{name}: per-component RMS vs epoch'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f'{name}_component_rms.png')
    fig.savefig(p, dpi=150); fig.savefig(p.replace('.png', '.pdf')); plt.close(fig)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(description="Component-by-component validation of synthetic generation.")
    ap.add_argument('--spec', required=True)
    ap.add_argument('--epochs', default='all',
                    help="'all' or comma-separated epoch indices for per-epoch figures.")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    name = spec['name']
    out_dir = args.out or os.path.join('synthetic', 'validation', name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Reconstructing additive layers for {name}")
    layers, dates, mask, names, params, sumcheck, n_epoch = reconstruct(spec)
    print(f"  layers: {list(layers.keys())}; {n_epoch} epochs")
    if sumcheck is not None:
        ok = 'OK' if sumcheck < 1e-3 else 'WARNING (components do not sum to combined!)'
        print(f"  sum-check max|(trop+atm+orbit+white) - combined| = {sumcheck:.3e} mm  [{ok}]")

    if args.epochs == 'all':
        epochs = list(range(n_epoch))
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    print(f"[2/4] Per-epoch decomposition figures")
    per_epoch_figures(out_dir, name, layers, dates, mask, epochs)
    print(f"[3/4] Component montages")
    component_montages(out_dir, name, layers, dates, mask)
    print(f"[4/4] RMS summary")
    rms_summary(out_dir, name, layers, dates, mask)
    print(f"Done. Validation plots in {out_dir}")


if __name__ == '__main__':
    main()
