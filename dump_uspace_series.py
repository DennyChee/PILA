#!/usr/bin/env python3
# Usage:       python dump_uspace_series.py \
#                  --config saved/temporalprior_marapi_mogi_move_loc/run/<RUN>/models/config.json \
#                  --resume saved/temporalprior_marapi_mogi_move_loc/run/<RUN>/models/model_best.pth \
#                  [--out uspace_series.csv] [-d 0]
# Description: Dump the trained encoder's per-epoch u-space posterior for the InSAR
#              point/MLP path: for every epoch it writes the u-space MEAN and
#              LOG-VARIANCE of each physical parameter (what the KL prior is defined
#              over), plus the rescaled physical value. This lets you inspect exactly
#              what the u-space parameters are and how the temporal prior shapes the
#              posterior spread (u_lnvar) across the time series.
# Date:        2026-07-09
#
# Scientific notes:
#   * u-space = unbounded latent. Physical value = affine( sigmoid(u_mean) ) via the
#     paras JSON; u_lnvar is the posterior log-variance IN u-space (not physical units).
#   * The physical-space distribution is the pushforward of N(u_mean, exp(u_lnvar))
#     through sigmoid->affine and is therefore NOT Gaussian/symmetric near the bounds.
#   * LOS convention: positive = motion toward satellite. MintPy metres -> mm at load.
#   * This mirrors run_iterative_prior_refit.infer_epoch_uspace (raw encoder posterior
#     mean, no temporal blending, no sampling) and adds the missing u_lnvar columns.

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from model import PHYS_VAE_SMPL
from utils import prepare_device, read_json
# Reuse the existing per-epoch loaders/inference so this stays consistent with the
# iterprior driver (single source of truth for u-space extraction).
from run_iterative_prior_refit import (
    load_epoch_series,
    infer_epoch_uspace,
    load_checkpoint_into_model,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description='Dump the trained encoder per-epoch u-space posterior (mean + lnvar).')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='config.json saved alongside the trained model')
    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='trained checkpoint (model_best.pth) to load')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='GPU index (default: auto)')
    parser.add_argument('--out', type=str, default=None,
                        help='output CSV path (default: uspace_series.csv next to --resume)')
    args = parser.parse_args()

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device, _ = prepare_device(1)

    # --- Load config + checkpoint ---
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
    if config_dict['arch']['args']['physics'] == 'MultiMogi_LOS':
        raise SystemExit("MultiMogi_LOS (K sources) is not supported here; "
                         "use dump_multisource_fit.py for per-source parameters and fields")
    attrs = list(model.physics_model.z_phy_ranges.keys())  # u-space / rescale column order
    print(f"  Physics = {config_dict['arch']['args']['physics']}, params = {attrs}")

    # --- Load the per-epoch series in chronological order ---
    print("[2/4] Loading epoch time series (chronological)...")
    t_load0 = time.time()
    epochs, dates = load_epoch_series(config_dict, device, input_dim, time_feat_dim)
    load_sec = time.time() - t_load0
    print(f"  Loaded: {len(epochs)} epochs, {dates[0]} .. {dates[-1]}  ({load_sec:.2f} s)")

    # --- Infer u-space posterior per epoch ---
    # INFERENCE COST: this loop is PILA's whole-timeseries "inversion" -- one encoder
    # forward pass per epoch, with the source parameters already learned at training
    # time. It is the direct cost analogue of running a classical per-epoch source
    # inversion over the same 76 epochs, so it is timed per epoch and written to
    # infer_usage.json alongside the trainer's compute_usage.json. Note the asymmetry
    # that must travel with any comparison: PILA amortises a one-off training cost
    # (train_sec) across all epochs, whereas the classical inversion pays its full
    # search cost on every epoch independently.
    print("[3/4] Inferring u-space posterior per epoch...")
    records = []
    epoch_sec = []
    t_infer0 = time.time()
    for i, epoch_item in enumerate(epochs):
        t_ep0 = time.time()
        u_mean, u_lnvar, phys = infer_epoch_uspace(model, epoch_item)
        epoch_sec.append(time.time() - t_ep0)
        # u_lnvar -> per-param u-space std for readability: std = exp(0.5*lnvar).
        u_std = np.exp(0.5 * u_lnvar)
        rec = {'epoch_idx': i, 'date': dates[i]}
        rec.update({f'u_{a}': float(u_mean[j]) for j, a in enumerate(attrs)})
        rec.update({f'u_lnvar_{a}': float(u_lnvar[j]) for j, a in enumerate(attrs)})
        rec.update({f'u_std_{a}': float(u_std[j]) for j, a in enumerate(attrs)})
        rec.update({a: float(phys[a]) for a in attrs})
        records.append(rec)

    infer_sec = time.time() - t_infer0

    # --- Write CSV ---
    print("[4/4] Writing CSV...")
    out_csv = args.out or os.path.join(os.path.dirname(ckpt), 'uspace_series.csv')
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"  Wrote {len(records)} epochs x {len(records[0])} columns -> {out_csv}")

    # --- Inference-cost sidecar (telemetry must never break the dump) ---
    try:
        ep = np.asarray(epoch_sec, dtype=float)
        usage = {
            'device': str(device),
            'physics': config_dict['arch']['args'].get('physics'),
            'n_obs_points': input_dim,
            'n_epochs': int(len(epochs)),
            'series_load_sec': round(float(load_sec), 3),
            'infer_sec_total': round(float(infer_sec), 3),
            'infer_sec_per_epoch_mean': round(float(ep.mean()), 5) if ep.size else None,
            'infer_sec_per_epoch_max': round(float(ep.max()), 5) if ep.size else None,
            'model_params': int(sum(p.numel() for p in model.parameters())),
            'note': ('infer_sec_total is the whole-timeseries inference cost, comparable '
                     'to a classical per-epoch inversion sweep. It EXCLUDES training '
                     '(see compute_usage.json train_sec), which PILA pays once and '
                     'amortises over all epochs.'),
        }
        out_json = os.path.join(os.path.dirname(out_csv) or '.', 'infer_usage.json')
        with open(out_json, 'w') as fh:
            json.dump(usage, fh, indent=2)
        print(f"  Inference cost: {infer_sec:.2f} s for {len(epochs)} epochs "
              f"({ep.mean()*1000:.1f} ms/epoch) -> {out_json}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"  WARNING: could not write inference-cost sidecar: {exc}")

    print("Done.")


if __name__ == '__main__':
    main()
