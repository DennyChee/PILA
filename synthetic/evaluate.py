#!/usr/bin/env python
# Usage:       python -m synthetic.evaluate \
#                  --truth synthetic/cubes/<name>/<name>_truth.json \
#                  --ckpt  saved/synth/<name>/.../models/model_best.pth
# Description: Quantify how well a trained PILA model recovered the KNOWN synthetic
#              source parameters. Runs deterministic per-epoch inference, denorms
#              z -> physical params via the decoder's own rescale(), and compares to
#              the ground-truth sidecar: per-parameter error (physical units), LOS
#              reconstruction RMSE (mm), source horizontal-location error (km), and
#              (for moving sources) the trajectory error. Writes a JSON metrics file,
#              a console table, and recovery / LOS / trajectory plots.
# Scientific notes:
#   * Recovery is meaningful only on strong-signal epochs: at near-zero deformation
#     (early cumulative epochs) the source parameters are unconstrained, so we
#     headline the PEAK epoch and a strong-signal subset, and also report all epochs.
#   * dV convention matches PILA rescale(): physical dV_si = z*1e5 - 1e7.
# Date:        2026-06-22

import argparse
import json
import os
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data_loader.data_loaders as module_data
from model import PHYS_VAE_SMPL
from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from synthetic import plots

# Physical-parameter names per physics type (must match each decoder's rescale()).
PHYSICS_ATTRS = {
    'Mogi_LOS':  ['xcen', 'ycen', 'd', 'dV'],
    'Sun69_LOS': ['xcen', 'ycen', 'depth', 'radius', 'dV'],
    'Okada_LOS': ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening'],
}
# Which two parameters give the horizontal source location (km after /1000).
LOC_KEYS = {'Mogi_LOS': ('xcen', 'ycen'), 'Sun69_LOS': ('xcen', 'ycen'),
            'Okada_LOS': ('xoff', 'yoff')}


def los_field_r2(recon_mm, truth_mm):
    """Squared Pearson correlation r^2 between a reconstructed and a truth LOS field.

    Both inputs are 1-D arrays of LOS displacement in mm sampled at the SAME observation
    points. Returns r^2 = corrcoef(recon, truth)^2, bounded in [0, 1]: 1.0 = the two
    fields share the same SPATIAL PATTERN, 0 = no linear correlation. Used as a continuous
    reconstruction-quality metric vs SNR.

    NOTE (scientific caveat): Pearson r^2 is amplitude- and offset-INSENSITIVE — a
    reconstruction whose deformation pattern matches truth but is, say, 10x too large
    (a wrong dV / opening) still scores ~1. It measures pattern match only, so it
    complements, and does NOT replace, the per-parameter error (which carries amplitude).
    This is the deliberate trade for a bounded [0,1] metric instead of the unbounded-below
    coefficient of determination (1 - SS_res/SS_tot).

    Returns NaN when the truth field is flat (std ~ 0, correlation undefined) — callers
    gate on a minimum signal RMS so this should not trigger for kept epochs. Returns 0.0
    when the truth varies but the reconstruction is flat (e.g. collapsed to a null source):
    a total pattern-recovery failure -> no correlation.
    """
    recon = np.asarray(recon_mm, float)
    truth = np.asarray(truth_mm, float)
    if truth.std() <= 0:
        return float('nan')        # truth has no spatial signal -> r undefined
    if recon.std() <= 0:
        return 0.0                 # null reconstruction over a varying truth -> no correlation
    r = float(np.corrcoef(recon, truth)[0, 1])
    return r ** 2


def run_inference(config, checkpoint_path):
    """Deterministic per-epoch inference. Returns attrs, params_phys[n,np],
    pred_std[n,N], target_std[n,N], x_scale(mm), dates."""
    physics = config['arch']['args']['physics']
    attrs = PHYSICS_ATTRS[physics]

    dl = getattr(module_data, config['data_loader']['type_test'])(
        insar=config['data_loader']['data_dir_test'],
        batch_size=512, shuffle=False, validation_split=0.0, num_workers=0,
        with_const=config['data_loader']['args'].get('with_const', False))

    model = PHYS_VAE_SMPL(config)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if ckpt.get('tau_r_values') is not None:
        model.dec.set_tau_r_from_checkpoint(ckpt['tau_r_values'])
    model.eval()

    data_key = config['trainer']['input_key']
    target_key = config['trainer']['output_key']
    params_list, pred_list, targ_list, dates = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            data = batch[data_key]; target = batch[target_key]
            tfeat = batch.get('time_feats', None)
            if data.dim() == 3:
                data = data.view(-1, data.size(-1))
            if target.dim() == 3:
                target = target.view(-1, target.size(-1))
            latent_phy, _, x_PB, _ = model(
                data, t=tfeat, inference=True, hard_z_phy=True, hard_z_aux=True)
            resc = model.physics_model.rescale(latent_phy)
            params = torch.stack([resc[k] for k in attrs], dim=1)
            params_list.append(params.cpu().numpy())
            pred_list.append(x_PB.cpu().numpy())
            targ_list.append(target.cpu().numpy())
            if 'date' in batch:
                dates += list(batch['date'])
    x_scale = float(model.physics_model.x_scale.flatten()[0].cpu())
    return (attrs, np.concatenate(params_list), np.concatenate(pred_list),
            np.concatenate(targ_list), x_scale, dates)


def evaluate(truth_json, checkpoint_path, out_dir=None):
    with open(truth_json, 'r') as f:
        truth = json.load(f)
    cfg_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    name = truth['name']
    physics = cfg['arch']['args']['physics']
    out_dir = out_dir or os.path.join('synthetic', 'eval', name)
    os.makedirs(out_dir, exist_ok=True)

    attrs, params_inf, pred_std, targ_std, x_scale, dates = run_inference(cfg, checkpoint_path)

    truth_names = truth['param_names']
    truth_params = np.asarray(truth['params_per_epoch'])      # [n_epoch, n_param] physical
    signal_rms = np.asarray(truth['signal_rms_mm_per_epoch'])
    n_epoch = truth_params.shape[0]

    # Inference order matches the loader (shuffle=False) -> same epoch order as truth.
    if params_inf.shape[0] != n_epoch:
        print(f"  WARNING: {params_inf.shape[0]} inferred epochs vs {n_epoch} truth epochs")
    n = min(params_inf.shape[0], n_epoch)

    # Reorder inferred columns to the truth's parameter order (they match for these
    # physics types, but be explicit).
    col = [attrs.index(p) for p in truth_names]
    inferred = params_inf[:n, col]
    tru = truth_params[:n]
    rms = signal_rms[:n]

    peak = int(truth.get('peak_epoch_index', int(np.argmax(rms))))
    strong = rms >= 0.5 * rms.max()                            # well-constrained epochs

    # --- LOS reconstruction RMSE (mm) ---
    resid_mm = (pred_std[:n] - targ_std[:n]) * x_scale
    los_rmse_all = float(np.sqrt(np.mean(resid_mm ** 2)))
    los_rmse_peak = float(np.sqrt(np.mean(resid_mm[peak] ** 2)))

    # --- Per-parameter recovery error (peak + strong-epoch mean abs error) ---
    per_param = {}
    for j, pname in enumerate(truth_names):
        err_peak = float(inferred[peak, j] - tru[peak, j])
        mae_strong = float(np.mean(np.abs(inferred[strong, j] - tru[strong, j]))) \
            if strong.any() else float('nan')
        # %err is only meaningful when the truth value is non-trivial; for a param
        # whose truth is ~0 (e.g. a source centred at xcen=ycen=0) report NaN% and
        # rely on the absolute error / the source-location-error metric instead.
        scale_ref = max(abs(tru[:, j]).max(), 1e-12)
        if abs(tru[peak, j]) < 1e-3 * scale_ref:
            pct = float('nan')
        else:
            pct = 100.0 * abs(err_peak) / abs(tru[peak, j])
        per_param[pname] = {
            'truth_peak': float(tru[peak, j]),
            'inferred_peak': float(inferred[peak, j]),
            'abs_err_peak': abs(err_peak),
            'pct_err_peak': pct,
            'mae_strong_epochs': mae_strong,
        }

    # --- Source horizontal-location error (km) ---
    kx, ky = LOC_KEYS[physics]
    ix, iy = truth_names.index(kx), truth_names.index(ky)
    loc_err_km = np.sqrt((inferred[:, ix] - tru[:, ix]) ** 2
                         + (inferred[:, iy] - tru[:, iy]) ** 2) / 1000.0
    loc_err_peak_km = float(loc_err_km[peak])
    loc_err_strong_km = float(np.mean(loc_err_km[strong])) if strong.any() else float('nan')

    metrics = {
        'name': name, 'physics': physics, 'checkpoint': checkpoint_path,
        'n_epochs': n, 'peak_epoch_index': peak,
        'n_strong_epochs': int(strong.sum()),
        'los_rmse_mm_all': los_rmse_all, 'los_rmse_mm_peak': los_rmse_peak,
        'source_loc_err_km_peak': loc_err_peak_km,
        'source_loc_err_km_strong_mean': loc_err_strong_km,
        'per_param': per_param,
        'x_scale_mm': x_scale,
    }

    # --- Plots ---
    plots.plot_param_recovery_vs_epoch(out_dir, name, truth_names,
                                       truth['param_units'], tru, inferred, rms)
    truth_xy_km = np.column_stack([tru[:, ix], tru[:, iy]]) / 1000.0
    inf_xy_km = np.column_stack([inferred[:, ix], inferred[:, iy]]) / 1000.0
    plots.plot_trajectory_map(out_dir, name, truth_xy_km, inf_xy_km, rms)

    # LOS maps at the peak epoch (need point coords from the loader)
    ins = cfg['arch']['args']['insar']
    data = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'),
                             ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 20),
                             coh_valid_frac=ins.get('coh_valid_frac', 0.5), verbose=False)
    obs_mm = targ_std[peak] * x_scale
    pred_mm = pred_std[peak] * x_scale
    plots.plot_los_maps(out_dir, name, data.xE_pts, data.yN_pts, obs_mm, pred_mm,
                        truth['epochs'][peak] if peak < len(truth['epochs']) else str(peak))

    with open(os.path.join(out_dir, f'{name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    _print_report(metrics, truth_names)
    print(f"\nPlots + metrics written to {out_dir}")
    return metrics


def _print_report(m, truth_names):
    print("\n" + "=" * 70)
    print(f"SYNTHETIC RECOVERY: {m['name']}  ({m['physics']})")
    print("=" * 70)
    print(f"epochs={m['n_epochs']}  peak={m['peak_epoch_index']}  "
          f"strong-signal epochs={m['n_strong_epochs']}")
    print(f"LOS RMSE: all={m['los_rmse_mm_all']:.3f} mm   peak={m['los_rmse_mm_peak']:.3f} mm")
    print(f"Source location error: peak={m['source_loc_err_km_peak']:.4f} km   "
          f"strong-mean={m['source_loc_err_km_strong_mean']:.4f} km")
    print(f"\n{'param':<10}{'truth(peak)':>16}{'PILA(peak)':>16}{'abs_err':>14}{'%err':>9}")
    for p in truth_names:
        d = m['per_param'][p]
        print(f"{p:<10}{d['truth_peak']:>16.4g}{d['inferred_peak']:>16.4g}"
              f"{d['abs_err_peak']:>14.4g}{d['pct_err_peak']:>8.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Evaluate PILA synthetic parameter recovery.")
    ap.add_argument('--truth', required=True, help='Ground-truth sidecar JSON.')
    ap.add_argument('--ckpt', required=True, help='Trained model_best.pth.')
    ap.add_argument('--out', default=None, help='Output dir (default synthetic/eval/<name>).')
    args = ap.parse_args()
    evaluate(args.truth, args.ckpt, args.out)


if __name__ == '__main__':
    main()
