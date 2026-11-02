#!/usr/bin/env python
# Usage:       python dump_multisource_fit.py --resume saved/<...>/model_best.pth [--out <run_dir>/fit_K<k>.npz]
# Description: Per-acquisition inference of a (Multi)Mogi_LOS PILA run, for the sequential
#              multi-source selection (Step 2). Saves, for EVERY acquisition of the training stack:
#                params_phys [T, K, 4]  xcen_km, ycen_km, depth_km, dV_m3 of each source
#                u_mean      [T, 4K]    encoder u-space means (posterior means, pre-sigmoid)
#                xP_src_mm   [T, K, H, W] physics LOS of each source (mm, + toward satellite)
#                xPB_mm      [T, H, W]  physics + low-rank residual (the full PILA reconstruction)
#                data_mm     [T, H, W]  observed LOS (mm)
#              plus grid axes, dates and the coherence mask, and writes infer_usage.json with the
#              inference timing (sec per acquisition, physics decode ms per acquisition).
# Notes:       Deterministic inference (posterior means), as in plot_insar_results.py. Points are
#              scattered back to the [H, W] grid through the loader's coherence mask (NaN elsewhere).
#              Single-source Mogi_LOS runs are dumped with K = 1 so Step 2 can treat both alike.
# Date:        2026-10-06

import argparse
import json
import os
import time

import numpy as np
import torch

from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from plot_insar_results import build_model
from utils.timing import PhaseTimer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True, help='checkpoint (.pth) of a Mogi_LOS / MultiMogi_LOS run')
    parser.add_argument('--out', default=None, help='output npz (default: <ckpt dir>/fit_K<K>.npz)')
    args = parser.parse_args()
    if not os.path.exists(args.resume):
        raise FileNotFoundError(args.resume)
    device = torch.device('cpu')

    print(f"[1/4] Loading checkpoint {args.resume}...")
    t0 = time.perf_counter()
    config = torch.load(args.resume, map_location='cpu', weights_only=False)['config']
    config = config.config if hasattr(config, 'config') else config
    physics_name = config['arch']['args']['physics']
    if physics_name not in ('Mogi_LOS', 'MultiMogi_LOS') or config['arch']['args'].get('encoder_type', 'mlp') != 'mlp':
        raise ValueError(f"needs a point/MLP Mogi_LOS or MultiMogi_LOS run, got {physics_name}")
    model = build_model(config, args.resume, device)
    physics = model.physics_model
    n_sources = int(getattr(physics, 'n_sources', 1))
    load_sec = time.perf_counter() - t0
    print(f"  physics={physics_name}, K={n_sources}, loaded in {load_sec:.1f}s")

    print("[2/4] Loading the training stack through the PILA loader...")
    ins = config['arch']['args']['insar']
    d = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'), ins['lat0'], ins['lon0'],
                          multilook=ins.get('multilook', 1), coh_valid_frac=ins.get('coh_valid_frac', 0.5),
                          bbox=ins.get('bbox'), verbose=False, standardization=ins.get('std_mode', 'global'),
                          far_field=ins.get('far_field'), stride=ins.get('stride', 1), offset=ins.get('offset', 0),
                          bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
                          bootstrap_seed=ins.get('bootstrap_seed', 0), ref_date=ins.get('ref_date'))
    data_pts_mm = d.los_points_mm.astype(np.float64)                     # [T, N]
    series_std = (data_pts_mm - d.x_mean_global) / d.x_scale_global      # same global scaler as training
    n_acq, n_pts = data_pts_mm.shape
    print(f"  Loaded: shape={data_pts_mm.shape}, dtype={data_pts_mm.dtype}")

    print("[3/4] Inference (posterior means)...")
    timer = PhaseTimer(enabled=True)
    timer.attach_module(physics, 'physics_decode', within='decode', remainder_name='residual_decode')
    with torch.no_grad():
        x = torch.tensor(series_std, dtype=torch.float32)
        with timer.phase('encode'):
            z_phy_stat, z_aux_stat = model.encode(x, None)
            z_phy, z_aux = model.draw(z_phy_stat, z_aux_stat, hard_z_phy=True, hard_z_aux=True)
        with timer.phase('decode'):
            x_PB, x_P, _, _, _ = model.decode(z_phy, z_aux, full=True, use_inference_values=True)
        if n_sources > 1:
            xP_src = physics.per_source_los_mm(z_phy).numpy()                          # [T, K, N] mm
        else:
            xP_src = (x_P.numpy() * float(physics.x_scale) + float(physics.x_mean))[:, None]
        xPB_mm = x_PB.numpy() * float(physics.x_scale) + float(physics.x_mean)
        u_mean = z_phy_stat['mean'].numpy()
        params = physics.rescale(z_phy.reshape(n_acq * n_sources, 4))                 # m, m^3
    params_phys = np.stack([params['xcen'].numpy() / 1e3, params['ycen'].numpy() / 1e3,
                            params['d'].numpy() / 1e3, params['dV'].numpy()], axis=-1).reshape(n_acq, n_sources, 4)
    infer_sec = sum(timer.total_s.values())

    # Consistency: sum of per-source fields must equal the decoder's physics output.
    xP_mm_decoder = x_P.numpy() * float(physics.x_scale) + float(physics.x_mean)
    max_diff = float(np.abs(xP_src.sum(axis=1) - xP_mm_decoder).max())
    print(f"  per-source sum vs decoder x_P: max |diff| {max_diff:.2e} mm")
    if max_diff > 1e-3:
        raise RuntimeError("per-source fields do not add up to the decoder output")

    print("[4/4] Writing outputs...")
    rows, cols = np.nonzero(d.mask_d)
    def to_grid(points):                                   # [..., N] -> [..., H, W], NaN off-mask
        grid = np.full(points.shape[:-1] + d.mask_d.shape, np.nan, dtype=np.float32)
        grid[..., rows, cols] = points
        return grid
    out_path = args.out or os.path.join(os.path.dirname(args.resume), f'fit_K{n_sources}.npz')
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists; not overwriting")
    np.savez_compressed(out_path, params_phys=params_phys.astype(np.float64), u_mean=u_mean,
                        xP_src_mm=to_grid(xP_src), xPB_mm=to_grid(xPB_mm), data_mm=to_grid(data_pts_mm),
                        mask=d.mask_d, xE_km=d.xE_grid, yN_km=d.yN_grid, dates=np.array(d.dates),
                        n_sources=n_sources, checkpoint=args.resume)
    usage = dict(checkpoint=args.resume, n_sources=n_sources, n_acquisitions=n_acq, n_points=n_pts,
                 load_sec=round(load_sec, 3), infer_sec=round(infer_sec, 4),
                 sec_per_acquisition=round(infer_sec / n_acq, 6),
                 physics_decode_ms_per_acquisition=round(1e3 * timer.total_s.get('physics_decode', 0.0) / n_acq, 4),
                 timing=timer.summary(n_samples_seen=n_acq))
    usage_path = os.path.join(os.path.dirname(out_path), f'infer_usage_K{n_sources}.json')
    with open(usage_path, 'w') as f:
        json.dump(usage, f, indent=2)
    print(f"  inference {infer_sec:.3f}s ({1e3 * infer_sec / n_acq:.2f} ms per acquisition)")
    print(f"Done. Output saved to {out_path} and {usage_path}")


if __name__ == '__main__':
    main()
