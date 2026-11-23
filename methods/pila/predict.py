#!/usr/bin/env python
# Usage:       python methods/pila/predict.py --checkpoint <run>/models/model_best.pth --scenario-id <id> \
#                  --out runs/pila/<id>/<run_id> [--label best]
# Description: Turn a trained PILA run into the common method output (methods/contract.py): per-date source
#              parameters [T, K, P] in SI units, the reconstructed LOS field (physics + low-rank residual) and the
#              physics-only field, timing and metadata. Works for Mogi_LOS, MultiMogi_LOS, Sun69_LOS and
#              Okada_LOS point/MLP runs. Inference = posterior means (deterministic), as dump_fit.py.
# Notes:       * PILA has no explicit constant offset: offset_m is NaN; its low-rank residual absorbs the
#                reference-pixel constant (the user-agreed per-date offset rule: PILA keeps its residual).
#              * The reconstruction is written on the grid the loader produced (row 0 = NORTH, as MintPy);
#                x_m / y_m record that grid, the scorer aligns grids by their axes.
#              * Training time is read from the run's compute_usage.json (written by run_pila.py).
# Date:        2026-11-23
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(METHOD_DIR))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, METHOD_DIR)

import pila.parse_config as _pila_parse_config  # noqa: E402
sys.modules.setdefault('parse_config', _pila_parse_config)     # legacy checkpoints pickle 'parse_config'
from pila.datasets.preprocessing.insar_mintpy import load_insar_mintpy  # noqa: E402
from pila.utils.timing import PhaseTimer  # noqa: E402
from plot_insar_results import build_model  # noqa: E402
from methods.contract import grid_axes_m, pixel_fingerprint, write_method_output  # noqa: E402
from scenarios.schema import PARAM_NAMES  # noqa: E402

PHYSICS_MODEL = {'Mogi_LOS': 'mogi', 'MultiMogi_LOS': 'mogi', 'Sun69_LOS': 'sun69', 'Okada_LOS': 'okada'}


def load_stack(config):
    """Load the scene exactly as PILA trained on it; return the loader result."""
    ins = config['arch']['args']['insar']
    return load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'), ins['lat0'], ins['lon0'],
                             multilook=ins.get('multilook', 1), coh_valid_frac=ins.get('coh_valid_frac', 0.5),
                             bbox=ins.get('bbox'), verbose=False, standardization=ins.get('std_mode', 'global'),
                             far_field=ins.get('far_field'), stride=ins.get('stride', 1), offset=ins.get('offset', 0),
                             bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
                             bootstrap_seed=ins.get('bootstrap_seed', 0), ref_date=ins.get('ref_date'))


def params_from_z(physics, z_phy, model, n_sources):
    """
    Physical parameters [T, K, P] (SI, PARAM_NAMES order) from the decoded z in (0, 1).

    Inputs:  physics (PILA physics module, with rescale); z_phy [T, K*P] tensor; model name; n_sources K.
    """
    names = PARAM_NAMES[model]
    n_dates = z_phy.shape[0]
    rescaled = physics.rescale(z_phy.reshape(n_dates * n_sources, len(names)))
    missing = [name for name in names if name not in rescaled]
    if missing:
        raise KeyError(f"physics.rescale did not return {missing} (got {list(rescaled)})")
    stacked = np.stack([rescaled[name].detach().cpu().numpy().astype(np.float64) for name in names], axis=-1)
    return stacked.reshape(n_dates, n_sources, len(names))


def predict(checkpoint, scenario_id, out_dir, label='best'):
    """Run inference for one checkpoint and write the contract outputs; return out_dir."""
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    device = torch.device('cpu')
    print(f"[1/4] Loading checkpoint {checkpoint} ...")
    t0 = time.perf_counter()
    config = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
    config = config.config if hasattr(config, 'config') else config
    physics_name = config['arch']['args']['physics']
    if physics_name not in PHYSICS_MODEL or config['arch']['args'].get('encoder_type', 'mlp') != 'mlp':
        raise ValueError(f"predict supports point/MLP {sorted(PHYSICS_MODEL)}; got {physics_name}")
    model_name = PHYSICS_MODEL[physics_name]
    model = build_model(config, checkpoint, device)
    physics = model.physics_model
    n_sources = int(getattr(physics, 'n_sources', 1))
    load_s = time.perf_counter() - t0
    print(f"  physics={physics_name} -> {model_name}, K={n_sources}, loaded in {load_s:.1f}s")

    print("[2/4] Loading the stack through the PILA loader ...")
    stack = load_stack(config)
    data_pts_mm = stack.los_points_mm.astype(np.float64)                         # [T, N]
    series_std = (data_pts_mm - stack.x_mean_global) / stack.x_scale_global
    n_dates, n_points = data_pts_mm.shape
    print(f"  Loaded: shape={data_pts_mm.shape}, dtype={data_pts_mm.dtype}")

    print("[3/4] Inference (posterior means) ...")
    timer = PhaseTimer(enabled=True)
    with torch.no_grad():
        x = torch.tensor(series_std, dtype=torch.float32)
        t_inf = time.perf_counter()
        z_phy_stat, z_aux_stat = model.encode(x, None)
        z_phy, z_aux = model.draw(z_phy_stat, z_aux_stat, hard_z_phy=True, hard_z_aux=True)
        x_pb, x_p, _, _, _ = model.decode(z_phy, z_aux, full=True, use_inference_values=True)
        infer_s = time.perf_counter() - t_inf
    scale, mean = float(physics.x_scale), float(physics.x_mean)
    recon_mm = x_pb.numpy() * scale + mean                                       # physics + residual, mm
    phys_mm = x_p.numpy() * scale + mean                                          # physics only, mm
    params = params_from_z(physics, z_phy, model_name, n_sources)
    print(f"  inference {infer_s:.3f}s ({1e3 * infer_s / n_dates:.2f} ms per date); params {params.shape}")

    print("[4/4] Writing contract outputs ...")
    rows, cols = np.nonzero(stack.mask_d)

    def to_grid_m(points_mm):
        grid = np.full((n_dates,) + stack.mask_d.shape, np.nan, dtype=np.float32)
        grid[:, rows, cols] = points_mm / 1000.0                                  # mm -> m
        return grid

    track = str(config['arch']['args']['insar']['geometry']).lower()
    track = track if track in ('asc', 'desc') else 'asc' if 'asc' in track.lower() else 'desc'
    x_m, y_m = grid_axes_m(stack.xE_grid, stack.yN_grid, stack.mask_d)            # valid cells only (masked = 0)
    run_models_dir = os.path.dirname(os.path.abspath(checkpoint))
    if os.path.basename(run_models_dir) in ('lowest_loss', 'basin_best'):
        run_models_dir = os.path.dirname(run_models_dir)
    usage_path = os.path.join(run_models_dir, 'compute_usage.json')
    usage = json.load(open(usage_path)) if os.path.exists(usage_path) else {}
    train_s = float(usage.get('train_sec', float('nan')))
    timing = dict(train_s=train_s, data_load_s=float(usage.get('data_load_sec', float('nan'))),
                  infer_s=infer_s, per_date_s=infer_s / n_dates,
                  total_s=(train_s if np.isfinite(train_s) else 0.0) + infer_s, hardware=str(usage.get('device', 'cpu')))
    meta = dict(method='pila', scenario_id=scenario_id, run_id=os.path.basename(out_dir), label=label,
                checkpoint=os.path.abspath(checkpoint), seed=config.get('seed'), physics=physics_name,
                n_points=int(n_points), pixel_mask_sha1=pixel_fingerprint(stack.mask_d), config=config)
    write_method_output(out_dir, stack.dates, [model_name] * n_sources, params, {track: to_grid_m(recon_mm)},
                        x_m, y_m, timing, meta, phys_by_track={track: to_grid_m(phys_mm)})
    print(f"Done. Output saved to {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Write the common method output for a trained PILA run')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', default='best', help='which checkpoint this is (best / lowest_loss / basin_best)')
    args = parser.parse_args()
    predict(args.checkpoint, args.scenario_id, args.out, args.label)


if __name__ == '__main__':
    main()
