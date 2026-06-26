#!/usr/bin/env python
# Usage:       python -m synthetic.validate_inference \
#                  --truth synthetic/cubes/<name>/<name>_truth.json \
#                  --ckpt  saved/synth/<run>/.../models/model_best.pth
# Description: Render the FULL inference output of a trained PILA model as per-epoch
#              contact-sheet montages, mapped back onto the (coarse) grid so they line
#              up directly with the input/component montages from validate_generation:
#                observed (input)  |  PILA physics+residual (x_PB)  |  physics-only (x_P)
#                |  residual (obs - x_PB)
#              All de-standardized to LOS mm. Lets you compare what PILA reconstructs
#              against what it was fed, epoch by epoch.
# Date:        2026-06-22

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data_loader.data_loaders as module_data
from model import PHYS_VAE_SMPL
from datasets.preprocessing.insar_mintpy import load_insar_mintpy


def run_inference_full(config, ckpt_path):
    """Per-epoch deterministic inference. Returns obs, x_PB, x_P (all [n_epoch, N],
    de-standardized to mm), the InSARData bundle, and dates."""
    model = PHYS_VAE_SMPL(config)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if ckpt.get('tau_r_values') is not None:
        model.dec.set_tau_r_from_checkpoint(ckpt['tau_r_values'])
    model.eval()

    dl = getattr(module_data, config['data_loader']['type_test'])(
        insar=config['data_loader']['data_dir_test'],
        batch_size=512, shuffle=False, validation_split=0.0, num_workers=0,
        with_const=config['data_loader']['args'].get('with_const', False))

    data_key = config['trainer']['input_key']; target_key = config['trainer']['output_key']
    scale = float(model.physics_model.x_scale.flatten()[0].cpu())
    mean = float(model.physics_model.x_mean.flatten()[0].cpu())

    obs, pb, pp, dates = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch[data_key]; tgt = batch[target_key]; tf = batch.get('time_feats', None)
            if x.dim() == 3: x = x.view(-1, x.size(-1))
            if tgt.dim() == 3: tgt = tgt.view(-1, tgt.size(-1))
            _, _, x_PB, x_P = model(x, t=tf, inference=True, hard_z_phy=True, hard_z_aux=True)
            obs.append(tgt.cpu().numpy()); pb.append(x_PB.cpu().numpy()); pp.append(x_P.cpu().numpy())
            if 'date' in batch: dates += list(batch['date'])
    # De-standardize to mm
    obs = np.concatenate(obs) * scale + mean
    pb = np.concatenate(pb) * scale + mean
    pp = np.concatenate(pp) * scale + mean

    ins = config['arch']['args']['insar']
    # Forward stride/offset so the reloaded coherence mask is thinned to the SAME decimated
    # subset this member was trained on (mask_d.sum() == N/stride). Without this, a strided
    # UQ member's point count (N/stride) would not match the full mask and points_to_grid
    # would raise a shape error. stride=1/offset=0 (the default) is the full-scene case.
    data = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'),
                             ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 20),
                             coh_valid_frac=ins.get('coh_valid_frac', 0.5), verbose=False,
                             stride=ins.get('stride', 1), offset=ins.get('offset', 0),
                             bootstrap_k=ins.get('bootstrap_k'),
                             bootstrap_block=ins.get('bootstrap_block', 3),
                             bootstrap_seed=ins.get('bootstrap_seed', 0))
    return obs, pb, pp, data, dates


def points_to_grid(values_2d, mask_d):
    """Map [n_epoch, N] point values (row-major over mask_d True cells) to [n_epoch, Hd, Wd]."""
    n_epoch = values_2d.shape[0]
    Hd, Wd = mask_d.shape
    cube = np.full((n_epoch, Hd, Wd), np.nan, dtype=np.float64)
    for i in range(n_epoch):
        g = np.full((Hd, Wd), np.nan); g[mask_d] = values_2d[i]; cube[i] = g
    return cube


def montage(out_path, title, cube_mm, dates, vmax=None):
    n_epoch = cube_mm.shape[0]
    idx = list(range(n_epoch))
    ncol = 8 if n_epoch > 24 else 4; nrow = int(np.ceil(n_epoch / ncol))
    if vmax is None:
        vmax = np.nanmax(np.abs(cube_mm)) + 1e-9
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow), squeeze=False)
    im = None
    for k, ep in enumerate(idx):
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(cube_mm[ep], cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
        ax.set_title(f'{ep}: {dates[ep]}', fontsize=6); ax.set_xticks([]); ax.set_yticks([])
    for k in range(n_epoch, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'{title}  —  all {n_epoch} epochs (±{vmax:.1f} mm, + toward sat)', y=1.0, fontsize=11)
    fig.subplots_adjust(right=0.91)
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.7]); fig.colorbar(im, cax=cax, label='mm')
    fig.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close(fig)
    return vmax


def main():
    ap = argparse.ArgumentParser(description="Full inference montages vs input, all epochs.")
    ap.add_argument('--truth', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.truth) as f: truth = json.load(f)
    with open(os.path.join(os.path.dirname(args.ckpt), 'config.json')) as f: config = json.load(f)
    name = truth['name']
    out_dir = args.out or os.path.join('synthetic', 'validation', name, 'inference')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/3] Running inference over all epochs")
    obs, pb, pp, data, dates = run_inference_full(config, args.ckpt)
    print(f"  {obs.shape[0]} epochs x {obs.shape[1]} cells; grid {data.mask_d.shape}")

    print(f"[2/3] Mapping points -> grid")
    obs_g = points_to_grid(obs, data.mask_d)
    pb_g = points_to_grid(pb, data.mask_d)
    pp_g = points_to_grid(pp, data.mask_d)
    res_g = obs_g - pb_g

    print(f"[3/3] Montages")
    # Shared scale for observed/prediction/physics so they compare directly.
    vmax = max(np.nanmax(np.abs(obs_g)), np.nanmax(np.abs(pb_g))) + 1e-9
    montage(os.path.join(out_dir, f'{name}_infer_observed.png'),
            f'{name}: OBSERVED (PILA input)', obs_g, dates, vmax)
    montage(os.path.join(out_dir, f'{name}_infer_prediction.png'),
            f'{name}: PILA prediction (physics+residual x_PB)', pb_g, dates, vmax)
    montage(os.path.join(out_dir, f'{name}_infer_physics_only.png'),
            f'{name}: PILA physics-only (x_P)', pp_g, dates, vmax)
    montage(os.path.join(out_dir, f'{name}_infer_residual.png'),
            f'{name}: RESIDUAL (observed - prediction)', res_g, dates)
    print(f"Done. Inference montages in {out_dir}")


if __name__ == '__main__':
    main()
