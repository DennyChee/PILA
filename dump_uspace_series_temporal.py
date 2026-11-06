#!/usr/bin/env python3
# Usage:       python dump_uspace_series_temporal.py \
#                  --config configs/synth/marapi_mogi_rising_lineardV_original.json \
#                  --resume saved/original_marapi_mogi_rising_lineardV/run/<RUN>/models/model_best.pth \
#                  --alpha_spatial 0.5 --alpha_dv 0.3 [--out ...]
# Description: INFERENCE-TIME temporal blend for the InSAR point/MLP path (the h5-native
#              analogue of test_pila_mogi_temporal.py, which is GPS-CSV only). Encodes
#              every epoch in chronological order and blends each epoch's u-space encoder
#              mean with the PREVIOUS (already-blended) epoch in u-space:
#                  u_blend[t] = (1-alpha)*u_enc[t] + alpha*u_blend[t-1]   (causal EMA)
#              per-dimension alpha (spatial for xcen/ycen/d, dv for dV). No retraining:
#              this is a post-hoc smoother on a fixed encoder. Writes a per-epoch CSV
#              (same columns as dump_uspace_series.py) so plot_iterprior_vs_truth.py and
#              compare_prior_variants.py consume it unchanged.
# Date:        2026-07-09
#
# Scientific notes:
#   * Blend is in u-space (unbounded); physical value = affine(sigmoid(u_blend)) via the
#     paras JSON, exactly as training/inference. alpha in [0,1]: 0 = raw encoder, 1 = frozen.
#   * Causal (only past informs present) -> matches real-time monitoring; a single railed
#     epoch propagates forward through the EMA (a known downside of causal blending).
#   * Units: xcen/ycen/d in metres, dV in m^3. LOS positive = toward satellite.

import argparse
import os

import numpy as np
import pandas as pd
import torch

from model import PHYS_VAE_SMPL
from utils import prepare_device, read_json
from run_iterative_prior_refit import load_epoch_series, load_checkpoint_into_model
from test_pila_mogi_temporal import build_alpha, temporal_blend

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description='Inference-time temporal blend (InSAR point path).')
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('-d', '--device', default=None)
    ap.add_argument('--alpha', type=float, default=0.3, help='default per-dim blend weight')
    ap.add_argument('--alpha_spatial', type=float, default=None, help='override for xcen,ycen,d')
    ap.add_argument('--alpha_dv', type=float, default=None, help='override for dV')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device, _ = prepare_device(1)

    print("[1/4] Loading config + checkpoint...")
    config_dict = read_json(os.path.join(CURRENT_DIR, args.config)
                            if not os.path.isabs(args.config) else args.config)
    ckpt = os.path.join(CURRENT_DIR, args.resume) if not os.path.isabs(args.resume) else args.resume
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    input_dim = config_dict['arch']['args']['input_dim']
    time_feat_dim = config_dict['arch']['args'].get('time_feat_dim', 4)

    model = PHYS_VAE_SMPL(config_dict).to(device)
    load_checkpoint_into_model(model, ckpt, device)
    model.eval()
    attrs = list(model.physics_model.z_phy_ranges.keys())
    dim_z_phy = len(attrs)
    alpha = build_alpha(dim_z_phy, args.alpha, args.alpha_spatial, args.alpha_dv, device)  # (dim_z_phy,)
    print(f"  params={attrs}, alpha per dim = {alpha.detach().cpu().numpy().round(3).tolist()}")

    print("[2/4] Loading epoch time series (chronological)...")
    epochs, dates = load_epoch_series(config_dict, device, input_dim, time_feat_dim)
    print(f"  {len(epochs)} epochs, {dates[0]} .. {dates[-1]}")

    print("[3/4] Sequential causal temporal blend in u-space...")
    records = []
    u_blend_prev = None
    with torch.no_grad():
        for i, ep in enumerate(epochs):
            z_phy_stat, _ = model.encode(ep['data'], ep['tfeat'])
            u_enc = z_phy_stat['mean']                              # (1, dim_z_phy) u-space
            if u_blend_prev is None:
                u_blend = u_enc.clone()                            # epoch 0: raw encoder
            else:
                u_blend = temporal_blend(u_enc, u_blend_prev, alpha)
            u_blend_prev = u_blend
            resc = model.physics_model.rescale(torch.sigmoid(u_blend))  # dict attr -> (1,)
            um = u_blend.squeeze(0).cpu().numpy()
            rec = {'epoch_idx': i, 'date': dates[i]}
            rec.update({f'u_{a}': float(um[j]) for j, a in enumerate(attrs)})
            rec.update({a: float(resc[a].squeeze().cpu()) for a in attrs})
            records.append(rec)

    print("[4/4] Writing CSV...")
    out_csv = args.out or os.path.join(os.path.dirname(ckpt), 'uspace_series_temporalblend.csv')
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"  Wrote {len(records)} epochs -> {out_csv}")
    print("Done.")


if __name__ == '__main__':
    main()
