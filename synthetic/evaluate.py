#!/usr/bin/env python
# Usage:       # plain per-epoch recovery:
#              python -m synthetic.evaluate \
#                  --truth synthetic/cubes/<name>/<name>_truth.json \
#                  --ckpt  saved/synth/<name>/.../models/model_best.pth
#              # moving-source temporal-blending test (raw vs blended):
#              python -m synthetic.evaluate --truth ..._truth.json --ckpt ...model_best.pth \
#                  --alpha 0 [--alpha-loc 0] [--alpha-depth 0.3] [--alpha-amp 0.0]
# Description: Quantify how well a trained PILA model recovered the KNOWN synthetic
#              source parameters. Runs deterministic per-epoch inference, denorms
#              z -> physical params via the decoder's own rescale(), and compares to
#              the ground-truth sidecar: per-parameter error (physical units), LOS
#              reconstruction RMSE (mm), source horizontal-location error (km), and
#              (for moving sources) the trajectory error. Writes a JSON metrics file,
#              a console table, and recovery / LOS / trajectory plots.
#              With --alpha it ALSO runs inference-time temporal blending (sequential,
#              u-space, per-parameter weights) and reports RAW vs BLENDED side by side
#              -- the moving-source temporal test. Blending trades per-epoch variance
#              for temporal-lag bias; for a moving source a high location alpha lags
#              the true migration, so keep --alpha-loc LOW.
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

# Per-physics grouping of parameters for temporal-blending alpha control.
#   'loc'   = horizontal position + remaining geometry -> evolves SLOWLY / usually well
#             constrained. For a MOVING source a high alpha here LAGS the true migration,
#             so keep it LOW when testing a mover.
#   'depth' = source depth -- its OWN group because it is the ill-constrained half of the
#             depth-amplitude trade-off and usually wants harder smoothing than the
#             (well-converged) horizontal position. Mogi names this parameter 'd'.
#   'amp'   = source amplitude/strength (dV / opening) -> genuinely time-varying (episodic
#             inflation/deflation), so a low alpha lets the encoder track it per epoch.
# Note: for Okada, length/width are fault-plane GEOMETRY (evolve slowly) -> 'loc'; only
# 'opening' is the true amplitude -> 'amp'.
LOC_GROUPS = {
    'Mogi_LOS':  {'loc': ['xcen', 'ycen'],           'depth': ['d'],     'amp': ['dV']},
    'Sun69_LOS': {'loc': ['xcen', 'ycen', 'radius'], 'depth': ['depth'], 'amp': ['dV']},
    'Okada_LOS': {'loc': ['xoff', 'yoff', 'strike', 'dip', 'length', 'width'],
                  'depth': ['depth'], 'amp': ['opening']},
}


def build_alpha_vec(attrs, physics, alpha_default, alpha_loc, alpha_depth, alpha_amp):
    """Per-parameter temporal-blending weight vector, in the encoder's `attrs` order.

    Args:
        attrs         : list of physical-parameter names = PHYSICS_ATTRS[physics], whose
                        order matches the z_phy / latent columns the decoder rescales.
        physics       : physics key (e.g. 'Mogi_LOS') -> selects the loc/depth/amp grouping.
        alpha_default : blend weight applied to every parameter unless overridden.
                        0 = no temporal smoothing (raw per-epoch); 1 = freeze at prior epoch.
        alpha_loc     : override weight for the position/geometry group (or None).
        alpha_depth   : override weight for the depth group (or None).
        alpha_amp     : override weight for the amplitude group (or None).

    Returns:
        np.ndarray shape (len(attrs),) of blend weights, dtype float64.
    """
    alpha = np.full(len(attrs), float(alpha_default), dtype=np.float64)
    groups = LOC_GROUPS.get(physics, {})
    for grp_alpha, grp_key in [(alpha_loc, 'loc'), (alpha_depth, 'depth'),
                               (alpha_amp, 'amp')]:
        if grp_alpha is not None:
            for nm in groups.get(grp_key, []):
                if nm in attrs:
                    alpha[attrs.index(nm)] = float(grp_alpha)
    return alpha


def alpha_tag(alpha, alpha_loc, alpha_depth, alpha_amp):
    """Filesystem-safe tag encoding the blend weights, e.g. 'a0_loc0_dep0.3_amp0'.

    Used as the per-run output subfolder name so an alpha sweep keeps every run
    instead of overwriting. loc/depth/amp appear only when explicitly overridden.
    """
    tag = f"a{alpha:g}"
    if alpha_loc is not None:
        tag += f"_loc{alpha_loc:g}"
    if alpha_depth is not None:
        tag += f"_dep{alpha_depth:g}"
    if alpha_amp is not None:
        tag += f"_amp{alpha_amp:g}"
    return tag


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


def _load_model_and_loader(config, checkpoint_path):
    """Build the test dataloader and load the trained model (CPU, eval mode).

    shuffle=False is REQUIRED so the epoch order matches the truth sidecar (and so the
    temporal blend in run_inference_blended sweeps epochs chronologically). Shared by
    both the raw and the blended inference paths so tau/r loading stays consistent.
    """
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
    return model, dl


def run_inference(config, checkpoint_path):
    """Deterministic per-epoch inference. Returns attrs, params_phys[n,np],
    pred_std[n,N], target_std[n,N], x_scale(mm), dates."""
    physics = config['arch']['args']['physics']
    attrs = PHYSICS_ATTRS[physics]

    model, dl = _load_model_and_loader(config, checkpoint_path)

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


def _decode_epochs(model, z_phy_all, z_aux_all, tfeat_all):
    """Decode each epoch's z to LOS, one epoch at a time so model.time_feats (read by the
    residual in decode()) stays aligned with the epoch being decoded. Returns pred[n, N]."""
    n = z_phy_all.shape[0]
    preds = []
    with torch.no_grad():
        for t in range(n):
            model.time_feats = tfeat_all[t:t + 1] if tfeat_all is not None else None
            # epoch/epochs_pretrain are ignored when use_inference_values=True (the residual
            # scale comes from get_r_for_inference); set explicitly for clarity.
            x_PB, _x_P, _y, _d, _c = model.decode(
                z_phy_all[t:t + 1], z_aux_all[t:t + 1],
                epoch=0, epochs_pretrain=0,
                full=True, use_inference_values=True,
                detach_x_P_for_bias=model.detach_x_P_for_bias)
            preds.append(x_PB.cpu().numpy())
    return np.concatenate(preds)


def encode_pass(config, checkpoint_path):
    """Load the model + data ONCE and encode every epoch to its u-space mean.

    Everything returned here is INDEPENDENT of the blend weights, so an alpha sweep can
    compute it a single time and then call blend_decode() per combo. The RAW (no-blend)
    decode + rescale is alpha-independent too, so it is done here as well.

    Returns a context dict with: model, attrs, physics, mu_enc[n,np] (u-space means),
    z_aux_all[n,na], tfeat_all (or None), targ[n,N] (numpy), x_scale(mm), dates,
    z_raw[n,np], pred_raw[n,N] (numpy), params_raw[n,np] (numpy, physical).
    """
    physics = config['arch']['args']['physics']
    attrs = PHYSICS_ATTRS[physics]

    model, dl = _load_model_and_loader(config, checkpoint_path)
    data_key = config['trainer']['input_key']
    target_key = config['trainer']['output_key']

    mu_list, zaux_list, tfeat_list, targ_list, dates = [], [], [], [], []
    have_tfeat = True
    with torch.no_grad():
        for batch in dl:
            data = batch[data_key]; target = batch[target_key]
            tfeat = batch.get('time_feats', None)
            if data.dim() == 3:
                data = data.view(-1, data.size(-1))
            if target.dim() == 3:
                target = target.view(-1, target.size(-1))
            z_phy_stat, z_aux_stat = model.encode(data, tfeat)
            mu_list.append(z_phy_stat['mean'])          # u-space mean, (b, dim_z_phy)
            zaux_list.append(z_aux_stat['mean'])        # aux mean (hard draw uses the mean)
            targ_list.append(target)
            if tfeat is None:
                have_tfeat = False
            else:
                tfeat_list.append(tfeat)
            if 'date' in batch:
                dates += list(batch['date'])

    mu_enc = torch.cat(mu_list, dim=0)                  # (n, dim_z_phy)
    z_aux_all = torch.cat(zaux_list, dim=0)             # (n, dim_z_aux)
    targ_all = torch.cat(targ_list, dim=0)             # (n, N)
    tfeat_all = torch.cat(tfeat_list, dim=0) if have_tfeat else None
    n = mu_enc.shape[0]
    print(f"  Encoded {n} epochs: mu_enc shape={tuple(mu_enc.shape)} dtype={mu_enc.dtype}, "
          f"targets shape={tuple(targ_all.shape)}, time_feats={'yes' if tfeat_all is not None else 'none'}")

    # RAW (alpha-independent) decode + rescale.
    z_raw = torch.sigmoid(mu_enc)
    pred_raw = _decode_epochs(model, z_raw, z_aux_all, tfeat_all)
    with torch.no_grad():
        resc_raw = model.physics_model.rescale(z_raw)
    params_raw = torch.stack([resc_raw[k] for k in attrs], dim=1).cpu().numpy()
    x_scale = float(model.physics_model.x_scale.flatten()[0].cpu())

    return {'model': model, 'attrs': attrs, 'physics': physics, 'mu_enc': mu_enc,
            'z_aux_all': z_aux_all, 'tfeat_all': tfeat_all, 'targ': targ_all.cpu().numpy(),
            'x_scale': x_scale, 'dates': dates, 'z_raw': z_raw, 'pred_raw': pred_raw,
            'params_raw': params_raw}


def blend_decode(ctx, alpha_np):
    """Apply the u-space temporal-blend recurrence for ONE alpha vector, reusing the
    precomputed encoder means in ctx (NO re-encode), then decode the blended z.

    Recurrence (per-parameter weight a = alpha_np, in attrs order):
        u_blend[t] = (1 - a) * mu_enc[t] + a * logit(z_blend[t-1])     (z_blend[0] = raw)
    Carries the BLENDED result forward (mirrors the trainer's Stage-B smoother and
    test_pila_mogi_temporal.py). For a MOVING source a large location alpha lags the
    true migration -- that variance-vs-lag trade-off is what a sweep maps out.

    Returns (params_blend[n,np] physical numpy, pred_blend[n,N] numpy).
    """
    mu_enc = ctx['mu_enc']; model = ctx['model']; attrs = ctx['attrs']
    eps = 1e-6
    alpha = torch.tensor(alpha_np, dtype=mu_enc.dtype)  # (dim_z_phy,)
    n = mu_enc.shape[0]
    z_blend = torch.empty_like(mu_enc)
    u_prev = None
    for t in range(n):
        u_b = mu_enc[t] if u_prev is None else (1.0 - alpha) * mu_enc[t] + alpha * u_prev
        z_t = torch.sigmoid(u_b)
        z_blend[t] = z_t
        u_prev = torch.logit(z_t.clamp(eps, 1.0 - eps))

    pred_blend = _decode_epochs(model, z_blend, ctx['z_aux_all'], ctx['tfeat_all'])
    with torch.no_grad():
        resc = model.physics_model.rescale(z_blend)
    params_blend = torch.stack([resc[k] for k in attrs], dim=1).cpu().numpy()
    return params_blend, pred_blend


def run_inference_blended(config, checkpoint_path, alpha_np):
    """Sequential temporal-blended per-epoch inference for a (possibly moving) source.

    Thin wrapper: encode_pass() (load + encode + raw decode, all alpha-independent) then
    blend_decode() for the given alpha. Kept for the single-alpha evaluate() path; a sweep
    should call encode_pass() once and blend_decode() per combo instead.

    Returns:
        attrs, params_raw[n,np], params_blend[n,np], pred_raw[n,N], pred_blend[n,N],
        targ[n,N], x_scale(mm), dates
    """
    ctx = encode_pass(config, checkpoint_path)
    params_blend, pred_blend = blend_decode(ctx, alpha_np)
    return (ctx['attrs'], ctx['params_raw'], params_blend, ctx['pred_raw'], pred_blend,
            ctx['targ'], ctx['x_scale'], ctx['dates'])


def _recovery_metrics(inferred, pred_std, targ_std, x_scale, tru, rms, peak, strong,
                      truth_names, physics):
    """Recovery-metric dict for ONE inference run (raw OR temporally blended).

    inferred, tru : [n, n_param] physical units, already in the truth's parameter order.
    pred_std, targ_std : [n, N] standardized LOS (multiply by x_scale for mm).
    peak/strong/rms : peak epoch index, strong-signal boolean mask, per-epoch signal RMS.
    Returns the same metric schema used by the original raw evaluate() (minus the
    name/physics/checkpoint header, which the caller adds).
    """
    n = inferred.shape[0]

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

    return {
        'n_epochs': n, 'peak_epoch_index': peak,
        'n_strong_epochs': int(strong.sum()),
        'los_rmse_mm_all': los_rmse_all, 'los_rmse_mm_peak': los_rmse_peak,
        'source_loc_err_km_peak': loc_err_peak_km,
        'source_loc_err_km_strong_mean': loc_err_strong_km,
        'per_param': per_param,
        'x_scale_mm': x_scale,
    }


def evaluate(truth_json, checkpoint_path, out_dir=None,
             alpha=None, alpha_loc=None, alpha_depth=None, alpha_amp=None):
    """Quantify synthetic source recovery.

    alpha=None (default) : original behaviour -- plain per-epoch inference, one metrics
                           file + recovery/trajectory/LOS plots.
    alpha is not None     : ALSO run inference-time temporal blending (per-parameter
                           weights from build_alpha_vec, loc/depth/amp groups) and report
                           RAW vs BLENDED side by side -- the moving-source temporal test.
    """
    with open(truth_json, 'r') as f:
        truth = json.load(f)
    cfg_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    name = truth['name']
    physics = cfg['arch']['args']['physics']
    # Default under the repo ROOT so it is independent of the invoking directory.
    out_dir = out_dir or os.path.join(ROOT, 'synthetic', 'eval', name)
    os.makedirs(out_dir, exist_ok=True)

    truth_names = truth['param_names']
    truth_params = np.asarray(truth['params_per_epoch'])      # [n_epoch, n_param] physical
    signal_rms = np.asarray(truth['signal_rms_mm_per_epoch'])
    n_epoch = truth_params.shape[0]
    kx, ky = LOC_KEYS[physics]
    ix, iy = truth_names.index(kx), truth_names.index(ky)

    attrs = PHYSICS_ATTRS[physics]
    col = [attrs.index(p) for p in truth_names]     # reorder inferred -> truth order

    # ---- point coords for LOS maps (shared by both modes) ----
    ins = cfg['arch']['args']['insar']
    data = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'),
                             ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 20),
                             coh_valid_frac=ins.get('coh_valid_frac', 0.5), verbose=False)

    # ============================ RAW-ONLY MODE ============================
    if alpha is None:
        attrs_, params_inf, pred_std, targ_std, x_scale, dates = run_inference(
            cfg, checkpoint_path)
        if params_inf.shape[0] != n_epoch:
            print(f"  WARNING: {params_inf.shape[0]} inferred epochs vs {n_epoch} truth epochs")
        n = min(params_inf.shape[0], n_epoch)
        inferred = params_inf[:n, col]
        tru = truth_params[:n]; rms = signal_rms[:n]
        peak = int(truth.get('peak_epoch_index', int(np.argmax(rms))))
        strong = rms >= 0.5 * rms.max()

        metrics = {'name': name, 'physics': physics, 'checkpoint': checkpoint_path}
        metrics.update(_recovery_metrics(inferred, pred_std, targ_std, x_scale,
                                         tru, rms, peak, strong, truth_names, physics))

        plots.plot_param_recovery_vs_epoch(out_dir, name, truth_names,
                                           truth['param_units'], tru, inferred, rms)
        truth_xy_km = np.column_stack([tru[:, ix], tru[:, iy]]) / 1000.0
        inf_xy_km = np.column_stack([inferred[:, ix], inferred[:, iy]]) / 1000.0
        plots.plot_trajectory_map(out_dir, name, truth_xy_km, inf_xy_km, rms)
        plots.plot_los_maps(out_dir, name, data.xE_pts, data.yN_pts,
                            targ_std[peak] * x_scale, pred_std[peak] * x_scale,
                            truth['epochs'][peak] if peak < len(truth['epochs']) else str(peak))

        with open(os.path.join(out_dir, f'{name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        _print_report(metrics, truth_names)
        print(f"\nPlots + metrics written to {out_dir}")
        return metrics

    # ===================== TEMPORAL (RAW vs BLENDED) MODE =====================
    # Put every alpha run in its own alpha-tagged subfolder so a sweep does not
    # overwrite previous runs (the tag carries the blend weights, e.g. a0_loc0_dep0.3_amp0).
    out_dir = os.path.join(out_dir, alpha_tag(alpha, alpha_loc, alpha_depth, alpha_amp))
    os.makedirs(out_dir, exist_ok=True)

    alpha_vec = build_alpha_vec(attrs, physics, alpha, alpha_loc, alpha_depth, alpha_amp)
    (attrs_, params_raw, params_blend, pred_raw, pred_blend,
     targ_std, x_scale, dates) = run_inference_blended(cfg, checkpoint_path, alpha_vec)

    if params_raw.shape[0] != n_epoch:
        print(f"  WARNING: {params_raw.shape[0]} inferred epochs vs {n_epoch} truth epochs")
    n = min(params_raw.shape[0], n_epoch)
    inferred_raw = params_raw[:n, col]
    inferred_blend = params_blend[:n, col]
    tru = truth_params[:n]; rms = signal_rms[:n]
    peak = int(truth.get('peak_epoch_index', int(np.argmax(rms))))
    strong = rms >= 0.5 * rms.max()

    m_raw = _recovery_metrics(inferred_raw, pred_raw, targ_std, x_scale,
                              tru, rms, peak, strong, truth_names, physics)
    m_blend = _recovery_metrics(inferred_blend, pred_blend, targ_std, x_scale,
                                tru, rms, peak, strong, truth_names, physics)

    if not strong.any():
        print("  WARNING: no strong-signal epochs (rms >= 0.5*max) — "
              "strong-mean metrics will be NaN.")

    # Positive delta = blending REDUCED the error (improvement).
    improvement = {
        'source_loc_err_km_strong_mean': m_raw['source_loc_err_km_strong_mean']
        - m_blend['source_loc_err_km_strong_mean'],
        'source_loc_err_km_peak': m_raw['source_loc_err_km_peak']
        - m_blend['source_loc_err_km_peak'],
        'los_rmse_mm_all': m_raw['los_rmse_mm_all'] - m_blend['los_rmse_mm_all'],
        'los_rmse_mm_peak': m_raw['los_rmse_mm_peak'] - m_blend['los_rmse_mm_peak'],
    }

    alpha_map = {attrs[i]: float(alpha_vec[i]) for i in range(len(attrs))}
    alpha_label = (f"alpha={alpha}"
                   + (f", loc={alpha_loc}" if alpha_loc is not None else '')
                   + (f", depth={alpha_depth}" if alpha_depth is not None else '')
                   + (f", amp={alpha_amp}" if alpha_amp is not None else ''))

    metrics = {
        'name': name, 'physics': physics, 'checkpoint': checkpoint_path,
        'mode': 'temporal_blend',
        'alpha_default': alpha, 'alpha_loc': alpha_loc, 'alpha_depth': alpha_depth,
        'alpha_amp': alpha_amp,
        'alpha_per_param': alpha_map,
        'raw': m_raw, 'temporal': m_blend,
        'improvement_raw_minus_temporal': improvement,
    }

    # --- Comparison plots (truth vs raw vs blended) ---
    plots.plot_param_recovery_compare(out_dir, name, truth_names, truth['param_units'],
                                      tru, inferred_raw, inferred_blend, rms,
                                      alpha_label=alpha_label)
    truth_xy_km = np.column_stack([tru[:, ix], tru[:, iy]]) / 1000.0
    raw_xy_km = np.column_stack([inferred_raw[:, ix], inferred_raw[:, iy]]) / 1000.0
    blend_xy_km = np.column_stack([inferred_blend[:, ix], inferred_blend[:, iy]]) / 1000.0
    plots.plot_trajectory_compare(out_dir, name, truth_xy_km, raw_xy_km, blend_xy_km,
                                  rms, alpha_label=alpha_label)
    # LOS maps at peak for the BLENDED reconstruction (raw maps come from the raw run).
    plots.plot_los_maps(out_dir, name, data.xE_pts, data.yN_pts,
                        targ_std[peak] * x_scale, pred_blend[peak] * x_scale,
                        truth['epochs'][peak] if peak < len(truth['epochs']) else str(peak))

    with open(os.path.join(out_dir, f'{name}_metrics_temporal.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    _print_compare(metrics, truth_names)
    print(f"\nTemporal comparison plots + metrics written to {out_dir}")
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


def _print_compare(m, truth_names):
    """Print a raw-vs-temporal recovery comparison table for the moving-source test."""
    r, t = m['raw'], m['temporal']
    print("\n" + "=" * 78)
    print(f"TEMPORAL MOVING-SOURCE RECOVERY: {m['name']}  ({m['physics']})")
    print(f"alpha per param: {m['alpha_per_param']}")
    print("=" * 78)
    print(f"epochs={r['n_epochs']}  peak={r['peak_epoch_index']}  "
          f"strong-signal epochs={r['n_strong_epochs']}")
    print(f"\n{'metric':<34}{'raw':>13}{'temporal':>13}{'delta(raw-tmp)':>16}")
    rows = [
        ('source loc err (km, strong-mean)', 'source_loc_err_km_strong_mean'),
        ('source loc err (km, peak)',        'source_loc_err_km_peak'),
        ('LOS RMSE (mm, all)',               'los_rmse_mm_all'),
        ('LOS RMSE (mm, peak)',              'los_rmse_mm_peak'),
    ]
    for label, key in rows:
        rv, tv = r[key], t[key]
        print(f"{label:<34}{rv:>13.4f}{tv:>13.4f}{rv - tv:>16.4f}")
    print("\n(positive delta = temporal blending REDUCED the error)")
    # Per-parameter strong-epoch MAE: where smoothing helped vs where it lagged.
    print(f"\n{'param':<10}{'raw MAE(strong)':>18}{'tmp MAE(strong)':>18}{'delta':>12}")
    for p in truth_names:
        rv = r['per_param'][p]['mae_strong_epochs']
        tv = t['per_param'][p]['mae_strong_epochs']
        print(f"{p:<10}{rv:>18.4g}{tv:>18.4g}{rv - tv:>12.4g}")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate PILA synthetic parameter recovery. With --alpha, also runs "
                    "inference-time temporal blending and reports raw vs blended (the "
                    "moving-source temporal test).")
    ap.add_argument('--truth', required=True, help='Ground-truth sidecar JSON.')
    ap.add_argument('--ckpt', required=True, help='Trained model_best.pth.')
    ap.add_argument('--out', default=None, help='Output dir (default synthetic/eval/<name>).')
    # --- Temporal-blending controls (omit --alpha for the original raw-only eval) ---
    ap.add_argument('--alpha', type=float, default=None,
                    help='Enable temporal blending with this default per-parameter weight. '
                         '0 = raw per-epoch; 1 = freeze at previous epoch. Typical: 0.2-0.4.')
    ap.add_argument('--alpha-loc', type=float, default=None,
                    help='Override blend weight for the position/geometry group. '
                         'For a MOVING source keep this LOW (e.g. 0.1) -- a high value lags '
                         'the true migration. (default: uses --alpha)')
    ap.add_argument('--alpha-depth', type=float, default=None,
                    help='Override blend weight for the depth group (Mogi d / Okada-Sun69 '
                         'depth). Depth is the ill-constrained half of the depth-amplitude '
                         'trade-off, so a HIGHER alpha here pools it across epochs. '
                         '(default: uses --alpha)')
    ap.add_argument('--alpha-amp', type=float, default=None,
                    help='Override blend weight for the source amplitude group (dV / opening). '
                         'Low lets the encoder track fast (episodic) changes. '
                         '(default: uses --alpha)')
    args = ap.parse_args()
    evaluate(args.truth, args.ckpt, args.out,
             alpha=args.alpha, alpha_loc=args.alpha_loc,
             alpha_depth=args.alpha_depth, alpha_amp=args.alpha_amp)


if __name__ == '__main__':
    main()
