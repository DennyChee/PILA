# Usage:       python compare_sun69_vs_mogi.py \
#                  --mogi  saved/sierranegra_mogi_insar_A/.../models/model_best.pth \
#                  --sun69 saved/sierranegra_sun69_insar_A/.../models/model_best.pth \
#                  --out   saved/sun69_vs_mogi_sierranegra
# Description: Compare the PILA Sierra Negra Sun (1969) penny-shaped crack inversion
#              against the Mogi point-source inversion on the SAME InSAR LOS scene.
#              Reports per-model LOS reconstruction error (mm) and the inverted
#              physical source parameters (median across dates), and renders a
#              side-by-side observed/predicted/residual LOS map for one date.
# Date:        2026-06-17

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import data_loader.data_loaders as module_data
from model import PHYS_VAE_SMPL
from datasets.preprocessing.insar_mintpy import load_insar_mintpy

# Inferred physics-parameter names per physics type (must match each decoder's
# rescale() enumeration order). Used to label the inverted parameter columns.
PHYSICS_ATTRS = {
    'Mogi_LOS':  ['xcen', 'ycen', 'd', 'dV'],
    'Sun69_LOS': ['xcen', 'ycen', 'depth', 'radius', 'dV'],
}


def load_model(config, checkpoint_path):
    """Build PHYS_VAE_SMPL from a config dict and load model_best.pth weights."""
    model = PHYS_VAE_SMPL(config)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if ckpt.get('tau_r_values') is not None:
        model.dec.set_tau_r_from_checkpoint(ckpt['tau_r_values'])
    model.eval()
    return model


def run_inference(config, checkpoint_path):
    """
    Encode every InSAR date deterministically and return:
      attrs       : list[str]  physical parameter names
      params_phys : ndarray [n_dates, n_params]  inverted params (physical units)
      x_PB        : ndarray [n_dates, N]  predicted LOS (standardized)
      target      : ndarray [n_dates, N]  observed LOS  (standardized)
      x_scale     : float  global standardization scale (mm per std unit)
      dates       : list   acquisition dates
    """
    physics = config['arch']['args']['physics']
    attrs = PHYSICS_ATTRS[physics]

    # Same loader the trainer used (whole time series, no shuffle for stable order).
    # The h5-native InSAR loaders take the `insar` dict (timeseries/geometry/...),
    # which lives in data_dir_test for these configs.
    dl = getattr(module_data, config['data_loader']['type_test'])(
        insar=config['data_loader']['data_dir_test'],
        batch_size=512, shuffle=False, validation_split=0.0, num_workers=0,
        with_const=config['data_loader']['args'].get('with_const', False))

    model = load_model(config, checkpoint_path)
    data_key = config['trainer']['input_key']
    target_key = config['trainer']['output_key']

    params_list, pred_list, targ_list, dates = [], [], [], []
    with torch.no_grad():
        for batch in dl:
            data = batch[data_key]
            target = batch[target_key]
            tfeat = batch.get('time_feats', None)
            if data.dim() == 3:
                data = data.view(-1, data.size(-1))
            if target.dim() == 3:
                target = target.view(-1, target.size(-1))
            # Deterministic latents (KL disabled in these configs -> hard_z=True).
            latent_phy, _, x_PB, _ = model(
                data, t=tfeat, inference=True, hard_z_phy=True, hard_z_aux=True)
            resc = model.physics_model.rescale(latent_phy)  # dict name -> [batch]
            params = torch.stack([resc[k] for k in attrs], dim=1)
            params_list.append(params.cpu().numpy())
            pred_list.append(x_PB.cpu().numpy())
            targ_list.append(target.cpu().numpy())
            if 'date' in batch:
                dates += list(batch['date'])

    x_scale = float(model.physics_model.x_scale.flatten()[0].cpu())
    return (attrs, np.concatenate(params_list), np.concatenate(pred_list),
            np.concatenate(targ_list), x_scale, dates)


def summarize(attrs, params_phys, x_scale):
    """Return per-model summary: LOS-mm not needed here; param medians in their
    physical/native units (m for lengths, m^3 for volume)."""
    med = np.median(params_phys, axis=0)
    p25 = np.percentile(params_phys, 25, axis=0)
    p75 = np.percentile(params_phys, 75, axis=0)
    return {name: (med[i], p25[i], p75[i]) for i, name in enumerate(attrs)}


def los_rmse_mm(pred_std, targ_std, x_scale):
    """RMSE between predicted and observed LOS in physical mm (undo z-score)."""
    resid_mm = (pred_std - targ_std) * x_scale
    return float(np.sqrt(np.mean(resid_mm**2)))


def main():
    ap = argparse.ArgumentParser(description="Compare Sun69 vs Mogi PILA inversions.")
    ap.add_argument('--mogi', required=True, help="Mogi model_best.pth")
    ap.add_argument('--sun69', required=True, help="Sun69 model_best.pth")
    ap.add_argument('--out', default=os.path.join(CURRENT_DIR, 'saved/sun69_vs_mogi_sierranegra'),
                    help="output directory for the comparison figure/CSV")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for tag, ckpt in [('Mogi', args.mogi), ('Sun69', args.sun69)]:
        cfg_path = os.path.join(os.path.dirname(ckpt), 'config.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"[{tag}] loading {ckpt}")
        attrs, params, pred, targ, x_scale, dates = run_inference(cfg, ckpt)
        rmse = los_rmse_mm(pred, targ, x_scale)
        summary = summarize(attrs, params, x_scale)
        results[tag] = dict(cfg=cfg, attrs=attrs, params=params, pred=pred,
                            targ=targ, x_scale=x_scale, dates=dates,
                            rmse_mm=rmse, summary=summary)
        print(f"[{tag}] LOS RMSE = {rmse:.3f} mm over {pred.shape[0]} dates, "
              f"{pred.shape[1]} points")

    # ---------------- Console report ----------------
    print("\n" + "=" * 64)
    print("PILA Sierra Negra inversion: Sun69 penny-crack vs Mogi point source")
    print("=" * 64)
    print(f"{'metric':<28}{'Mogi':>16}{'Sun69':>16}")
    print("-" * 60)
    print(f"{'LOS reconstruction RMSE (mm)':<28}"
          f"{results['Mogi']['rmse_mm']:>16.3f}{results['Sun69']['rmse_mm']:>16.3f}")
    print("\nInverted source parameters (median [IQR]); lengths in m, dV in m^3:")
    for tag in ('Mogi', 'Sun69'):
        print(f"\n  {tag}:")
        for name, (med, p25, p75) in results[tag]['summary'].items():
            print(f"    {name:<8} = {med:12.3f}  [{p25:12.3f}, {p75:12.3f}]")

    # ---------------- Comparison figure ----------------
    # Use the InSAR geometry to scatter LOS on the map for a representative date
    # (the date with the largest observed signal, i.e. peak deformation).
    ins = results['Mogi']['cfg']['arch']['args']['insar']
    d = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'),
                          ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 20),
                          coh_valid_frac=ins.get('coh_valid_frac', 0.5),
                          bbox=ins.get('bbox'), verbose=False)
    xE, yN = d.xE_pts, d.yN_pts   # km

    targ_mm = results['Mogi']['targ'] * results['Mogi']['x_scale']
    date_idx = int(np.argmax(np.max(np.abs(targ_mm), axis=1)))
    date_lbl = results['Mogi']['dates'][date_idx] if results['Mogi']['dates'] else f"idx{date_idx}"

    obs_mm = results['Mogi']['targ'][date_idx] * results['Mogi']['x_scale']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    vmax = np.percentile(np.abs(obs_mm), 99)
    for row, tag in enumerate(('Mogi', 'Sun69')):
        R = results[tag]
        pred_mm = R['pred'][date_idx] * R['x_scale']
        resid_mm = pred_mm - obs_mm
        panels = [('Observed LOS (mm)', obs_mm, 'RdBu_r', vmax),
                  (f'{tag} predicted LOS (mm)', pred_mm, 'RdBu_r', vmax),
                  (f'{tag} residual (mm)', resid_mm, 'RdBu_r', vmax)]
        for col, (title, vals, cmap, vm) in enumerate(panels):
            ax = axes[row, col]
            sc = ax.scatter(xE, yN, c=vals, cmap=cmap, vmin=-vm, vmax=vm, s=6)
            ax.set_title(f'{title}\n(RMSE {R["rmse_mm"]:.2f} mm all dates)' if col == 2 else title)
            ax.set_xlabel('East (km)')
            ax.set_ylabel('North (km)')
            ax.set_aspect('equal')
            fig.colorbar(sc, ax=ax, label='LOS (mm)')
    fig.suptitle(f'Sierra Negra InSAR — Sun69 penny-crack vs Mogi (date {date_lbl})',
                 fontsize=14)
    png = os.path.join(args.out, 'sun69_vs_mogi_los_maps.png')
    pdf = os.path.join(args.out, 'sun69_vs_mogi_los_maps.pdf')
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    print(f"\nSaved comparison figure: {png}")
    print(f"                         {pdf}")
    print("Done.")


if __name__ == '__main__':
    main()
