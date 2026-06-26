#!/usr/bin/env python
# Usage:       python -m synthetic.validate_all --spec <spec.json> --multilook 1 \
#                  [--ckpt <model_best.pth>]
#              (or)  from synthetic.validate_all import validate_all
# Description: Run the FULL validation suite for one synthetic scenario, producing every
#              plot used in the manual-validation workflow:
#                generation : per-epoch component decomposition (all epochs),
#                             per-component montages (all epochs), RMS-vs-epoch summary
#                loader     : full-res vs multilook, multilook suppression by component,
#                             the standardized point set PILA ingests, loader metrics
#                inference  : observed / prediction / physics-only / residual montages
#                             (all epochs) — only if a checkpoint is given
#              Used standalone and per-run inside synthetic/snr_sweep.py so every SNR
#              point gets the same full set of validation plots.
# Date:        2026-06-22

import argparse
import json
import os

import numpy as np

from synthetic import validate_generation as vg
from synthetic import validate_loader as vl
from synthetic import validate_inference as vi
from datasets.preprocessing.insar_mintpy import load_insar_mintpy


def validate_all(spec, multilook, ckpt=None, coh_valid_frac=0.5, out_dir=None):
    """Produce the full generation+loader(+inference) validation plot set for a spec."""
    name = spec['name']
    base = spec['base_scene']
    out_dir = out_dir or os.path.join('synthetic', 'validation', name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Generation: component decomposition ----
    layers, dates, mask, pnames, params, sumcheck, n_epoch = vg.reconstruct(spec)
    vg.per_epoch_figures(out_dir, name, layers, dates, mask, list(range(n_epoch)))
    vg.component_montages(out_dir, name, layers, dates, mask)        # all epochs
    vg.rms_summary(out_dir, name, layers, dates, mask)

    # ---- Loader: multilook effect + PILA input ----
    loader_dir = os.path.join(out_dir, 'loader'); os.makedirs(loader_dir, exist_ok=True)
    peak = int(np.argmax([vl._rms(layers['deformation'][i], mask) for i in range(n_epoch)]))
    coarse, cmask = vl.multilook_layers(layers, mask, multilook, coh_valid_frac)
    rows, (snr_f, snr_c, sig_f, sig_c, noise_f, noise_c) = vl.snr_table(
        layers, mask, coarse, cmask, peak)
    vl.plot_fullres_vs_multilook(loader_dir, name, layers['total'], coarse['total'],
                                 mask, cmask, peak, dates[peak])
    vl.plot_suppression(loader_dir, name, rows)
    data = load_insar_mintpy(vl.spec_h5(spec), base['geometry'], base.get('mask'),
                             base['lat0'], base['lon0'], multilook=multilook,
                             coh_valid_frac=coh_valid_frac, verbose=False)
    vl.plot_pila_input(loader_dir, name, data, peak)
    with open(os.path.join(loader_dir, f'{name}_loader_metrics.json'), 'w') as f:
        json.dump({'name': name, 'multilook': multilook,
                   'n_coherent_cells': int(cmask.sum()),
                   'snr_db_fullres': snr_f, 'snr_db_multilook': snr_c,
                   'snr_gain_db': snr_c - snr_f,
                   'component_rms_mm': {k: {'fullres': fr, 'multilook': cr, 'ratio': ra}
                                        for k, fr, cr, ra in rows}}, f, indent=2)

    # ---- Inference montages (needs a trained checkpoint) ----
    if ckpt and os.path.exists(ckpt):
        infer_dir = os.path.join(out_dir, 'inference'); os.makedirs(infer_dir, exist_ok=True)
        with open(os.path.join(os.path.dirname(ckpt), 'config.json')) as f:
            config = json.load(f)
        obs, pb, pp, idata, idates = vi.run_inference_full(config, ckpt)
        obs_g = vi.points_to_grid(obs, idata.mask_d)
        pb_g = vi.points_to_grid(pb, idata.mask_d)
        pp_g = vi.points_to_grid(pp, idata.mask_d)
        res_g = obs_g - pb_g
        vmax = max(np.nanmax(np.abs(obs_g)), np.nanmax(np.abs(pb_g))) + 1e-9
        vi.montage(os.path.join(infer_dir, f'{name}_infer_observed.png'),
                   f'{name}: OBSERVED (PILA input)', obs_g, idates, vmax)
        vi.montage(os.path.join(infer_dir, f'{name}_infer_prediction.png'),
                   f'{name}: PILA prediction (x_PB)', pb_g, idates, vmax)
        vi.montage(os.path.join(infer_dir, f'{name}_infer_physics_only.png'),
                   f'{name}: PILA physics-only (x_P)', pp_g, idates, vmax)
        vi.montage(os.path.join(infer_dir, f'{name}_infer_residual.png'),
                   f'{name}: RESIDUAL (observed - prediction)', res_g, idates)

    print(f"  validate_all: wrote full plot set to {out_dir}"
          f"{' (incl inference)' if ckpt else ''}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Full validation plot set for a synthetic scenario.")
    ap.add_argument('--spec', required=True)
    ap.add_argument('--multilook', type=int, default=1)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--coh-valid-frac', type=float, default=0.5)
    args = ap.parse_args()
    with open(args.spec) as f:
        spec = json.load(f)
    validate_all(spec, args.multilook, ckpt=args.ckpt, coh_valid_frac=args.coh_valid_frac)


if __name__ == '__main__':
    main()
