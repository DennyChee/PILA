#!/usr/bin/env python
# Usage:       python train_pila_basinhop.py -c <config.json>     (same arguments as train_pila.py)
# Description: train_pila_keepbest.py (per_epoch_sources.csv + lowest-val_rec_loss checkpoint) PLUS basin hopping in
#              place of blind warm restarts: after an initial phase, the run repeatedly (1) goes back to the BEST state
#              seen so far, (2) perturbs it, (3) lets it settle for a window of epochs, keeping any new best. A bad try
#              can never lose a good state. Settings: config trainer.phys_vae.basin_hop (see BASIN_HOP_DEFAULTS).
#              "Best" = DETERMINISTIC album misfit: one eval pass over every acquisition once (posterior means), same
#              per-sample MSE as the training rec_loss. val_rec_loss is NOT used for selection: its loader draws random
#              windows, so it scatters 5-8 % at fixed weights (ai-expert review 2026-10-23).
#              Perturbations (honest: random or PILA's own fit; the truth is never read):
#                slot         move the REDUNDANT slot (the one whose removal raises the misfit least) to a random
#                             x, y, depth in the honest start box, dV sign from PILA's own residual there,
#                             |dV| = the other slot's |dV|
#                slot_resid   as slot, but x, y drawn with probability ~ smoothed |stack-mean residual|^2 of the fit
#                             without the redundant slot
#                weight_noise Gaussian noise on both layers of the z_phy-mean head, scaled so RMS du ~ 1
#                none         control: revert + same LR re-warm + same Adam reset, no perturbation
#                off          observe only: NO windows, no LR override (the config scheduler runs unchanged, e.g. warm
#                             restarts); only logs the album misfit and keeps basin_best -> the warm-restart baseline
#                             selected by the same deterministic rule
#              A settle window is extended (by extend_epochs, at most max_extensions times) only while the GLOBAL best
#              epoch lies in its last extend_if_best_in_last epochs, i.e. the window is still setting new bests.
#              Outputs (run models dir): basin_hops.csv (one row per window), basin_epochs.csv (one row per epoch),
#              basin_hop.json (settings), basin_best/{model_basin_best.pth, fit_K<k>.npz, basin_best.json};
#              plus everything train_pila_keepbest.py writes. model_best.pth (val_rec_loss) is left as base_trainer
#              writes it, for comparison.
# Assumptions: MultiMogi_LOS point/MLP path, KL off (deterministic z_phy), no nuisance ramp, no input constants, time
#              features unused by the encoder (use_time_in_residual false) -- all checked at the first epoch.
#              The hop happens at the START of an epoch, so no checkpoint is ever written for a just-perturbed state.
# Date:        2026-10-23

import copy
import csv
import json
import math
import os
import runpy
import subprocess
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import train_pila_keepbest  # noqa: E402  (importing applies the keep-lowest + delayed-penalty patches)
from trainer import PhysVAETrainerSMPL  # noqa: E402
from model.loss import mse_loss  # noqa: E402
from utils.warm_start import _z_from_physical, _logit, RESIDUAL_SMOOTH_KM, Z_CLIP  # noqa: E402

BASIN_HOP_DEFAULTS = dict(initial_epochs=100, settle_epochs=40, extend_if_best_in_last=5, extend_epochs=10, max_extensions=3,
                          polish_epochs=60, total_epochs=400, perturbation='slot', settle_lr_peak_frac=0.5,
                          settle_lr_floor_frac=0.1, min_rel_improvement=0.005, seed=0)
PERTURBATIONS = ('slot', 'slot_resid', 'weight_noise', 'none', 'off')     # off = observe only (see header)
PARAMETERS_PER_SOURCE = 4             # MultiMogi block order: xcen, ycen, d, dV
GEOMETRY_ROW_STD_MAX_U = 0.2          # moved slot: album std of its x/y/depth u after the row shrink
MAX_ABS_U = 6.0                       # sigmoid'(6) ~ 2.5e-3: shrink the dV row too if the move pushes |u| past this
WEIGHT_NOISE_TARGET_RMS_DU = 1.0      # weight-noise arm: album RMS change of u over all z_phy dims
WEIGHT_NOISE_PROBE_SCALE = 0.1        # first noise scale (x tensor RMS) used to calibrate the target
SLOT_MOVED_KM = 2.0                   # "slot configuration changed": a matched slot moved more than this
DV_FLOOR_M3 = 1e7                     # moved slot |dV| floor (same floor as the honest random start box)
# Honest start box, identical to run_k2_tune.draw_random_init_sources (never the truth)
BOX_XY_KM, BOX_DEPTH_KM = 14.0, (0.5, 12.0)
_base_train_epoch = PhysVAETrainerSMPL._train_epoch     # keep-lowest wrapper (-> delayed-penalty -> original)
_run_state = {'trainer': None}


# ----------------------------------------------------------------------------------------------------------------
# Album-wide deterministic evaluation
# ----------------------------------------------------------------------------------------------------------------
class _FrozenRng:
    """Save/restore the global torch + numpy RNG so the hop code never shifts the training data order."""
    def __enter__(self):
        self.torch_state, self.numpy_state = torch.get_rng_state(), np.random.get_state()
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self.torch_state)
        np.random.set_state(self.numpy_state)
        return False


def _head_layers(model):
    """(penultimate Sequential, last Linear) of the z_phy-mean head: u = last(penultimate(feature))."""
    net = model.enc.func_z_phy_mean.net
    return net[:-1], net[-1]


def evaluate_album(trainer, epoch):
    """
    One deterministic pass over every acquisition of the album (posterior means, eval mode, no grad).

    Inputs:  trainer (PhysVAETrainerSMPL); epoch (int) for the decoder's tau / r schedule (same values as training).
    Output:  dict -- misfit (per-sample MSE of the full reconstruction, standardized units, = training rec_loss),
             phys_misfit (physics only, no low-rank residual), u [T, 4K], h [T, 128] (penultimate head features),
             z [T, 4K], x_PB / x_P / data [T, N] standardized, src_mm [T, K, N] per-source LOS (mm),
             params [K, 4] album-median (x km, y km, depth km, dV m^3).
    """
    model = trainer.model
    physics = model.physics_model
    data = trainer._album_std
    was_training = model.training
    model.eval()
    with _FrozenRng(), torch.no_grad():
        z_phy_stat, z_aux_stat = model.encode(data, None)
        z_phy, z_aux = model.draw(z_phy_stat, z_aux_stat, hard_z_phy=True, hard_z_aux=True)
        x_PB, x_P, _, _, _ = model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=trainer.epochs_pretrain, full=True)
        penultimate, _last = _head_layers(model)
        h = penultimate(model.enc.func_feat(data, None))
        src_mm = physics.per_source_los_mm(z_phy)
        rescaled = physics.rescale(z_phy.reshape(-1, PARAMETERS_PER_SOURCE))
    if was_training:
        model.train()
    n_acq = data.shape[0]
    params = torch.stack([rescaled['xcen'] / 1e3, rescaled['ycen'] / 1e3, rescaled['d'] / 1e3, rescaled['dV']], dim=-1)
    params = params.reshape(n_acq, -1, PARAMETERS_PER_SOURCE).median(dim=0).values.cpu().numpy().astype(np.float64)
    return dict(misfit=float(mse_loss(x_PB, data)), phys_misfit=float(mse_loss(x_P, data)), u=z_phy_stat['mean'],
                h=h, z=z_phy, x_PB=x_PB, x_P=x_P, data=data, src_mm=src_mm, params=params)


def slot_removal_misfits(trainer, evaluation):
    """Misfit of the full reconstruction with each slot's physics field removed (standardized units), [K]."""
    x_scale = trainer.model.physics_model.x_scale.to(evaluation['x_PB'].device)
    return [float(mse_loss(evaluation['x_PB'] - evaluation['src_mm'][:, k] / x_scale, evaluation['data']))
            for k in range(evaluation['src_mm'].shape[1])]


def slots_changed(params_before, params_after):
    """True if, after Hungarian matching on (x, y), any slot moved more than SLOT_MOVED_KM."""
    cost = np.linalg.norm(params_before[:, None, :2] - params_after[None, :, :2], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    return bool(cost[rows, cols].max() > SLOT_MOVED_KM)


# ----------------------------------------------------------------------------------------------------------------
# Perturbations
# ----------------------------------------------------------------------------------------------------------------
def _residual_grid(trainer, evaluation, slot):
    """Smoothed stack-mean residual (mm) of the physics fit WITHOUT `slot`, on the grid (as utils/warm_start.py)."""
    grid = trainer._album_grid
    other_mm = evaluation['src_mm'].sum(dim=1) - evaluation['src_mm'][:, slot]
    residual_mean_mm = (trainer._album_mm - other_mm).mean(dim=0).cpu().numpy()          # [N]
    rows, cols = grid['rows'], grid['cols']
    grid_res = np.zeros(grid['mask'].shape)
    grid_weight = np.zeros(grid['mask'].shape)
    grid_res[rows, cols] = residual_mean_mm
    grid_weight[rows, cols] = 1.0
    sigma_px = RESIDUAL_SMOOTH_KM / grid['pixel_km']
    smooth = gaussian_filter(grid_res, sigma_px) / np.maximum(gaussian_filter(grid_weight, sigma_px), 1e-6)
    smooth[~grid['mask']] = 0.0
    return smooth


def _reachable_depth_km(ranges):
    """Honest depth box intersected with the range _logit can reach (z clipped to [Z_CLIP, 1 - Z_CLIP]), km."""
    lo, hi = float(ranges['d']['min']), float(ranges['d']['max'])
    return max(BOX_DEPTH_KM[0], lo + Z_CLIP * (hi - lo)), min(BOX_DEPTH_KM[1], hi - Z_CLIP * (hi - lo))


def draw_slot_target(trainer, evaluation, slot, residual_guided, rng):
    """
    Physical target for the moved slot (honest: random box or PILA's own residual, never the truth).

    Output: dict x_km, y_km, depth_km, dV_m3, residual_at_target_mm, unit_los_mm_per_m3.
    """
    grid = trainer._album_grid
    smooth = _residual_grid(trainer, evaluation, slot)
    if residual_guided:
        weight = smooth[grid['rows'], grid['cols']] ** 2
        if not np.isfinite(weight).all() or weight.sum() <= 0.0:
            raise ValueError("residual-guided placement: residual map is empty or non-finite")
        point = int(rng.choice(len(weight), p=weight / weight.sum()))
        x_km, y_km = float(grid['x_pts_km'][point]), float(grid['y_pts_km'][point])
    else:
        x_km, y_km = (float(value) for value in rng.uniform(-BOX_XY_KM, BOX_XY_KM, size=2))
        point = int(np.argmin(np.hypot(grid['x_pts_km'] - x_km, grid['y_pts_km'] - y_km)))
    depth_km = float(rng.uniform(*_reachable_depth_km(trainer.model.physics_model.z_phy_ranges)))
    residual_mm = float(smooth[grid['rows'][point], grid['cols'][point]])
    # Signed unit-dV LOS of a source at the target, at the target's nearest point (LOS is linear in dV). The sign of
    # dV comes from residual / unit field, not from the residual alone (steep geometry can flip an inflation's sign).
    physics = trainer.model.physics_model
    ranges = physics.z_phy_ranges
    probe_dv_m3 = 1e6
    z_probe = torch.full((1, evaluation['z'].shape[1]), 0.5, device=evaluation['z'].device)
    z_probe[0, slot * PARAMETERS_PER_SOURCE:(slot + 1) * PARAMETERS_PER_SOURCE] = torch.tensor(
        [_z_from_physical(ranges, 'xcen', x_km), _z_from_physical(ranges, 'ycen', y_km),
         _z_from_physical(ranges, 'd', depth_km), _z_from_physical(ranges, 'dV', probe_dv_m3)])
    with torch.no_grad():
        unit_mm_per_m3 = float(physics.per_source_los_mm(z_probe)[0, slot, point]) / probe_dv_m3
    sign = float(np.sign(residual_mm * unit_mm_per_m3))
    sign_fallback = sign == 0.0                     # no residual at the point (e.g. off the coherent grid): coin flip
    if sign_fallback:
        sign = float(rng.choice([-1.0, 1.0]))
    other = [k for k in range(evaluation['params'].shape[0]) if k != slot]
    magnitude_m3 = max(DV_FLOOR_M3, float(np.max(np.abs(evaluation['params'][other, 3]))))
    return dict(x_km=x_km, y_km=y_km, depth_km=depth_km, dV_m3=float(sign * magnitude_m3),
                residual_at_target_mm=residual_mm, residual_over_map_max=abs(residual_mm) / max(float(np.abs(smooth).max()), 1e-12),
                unit_los_mm_per_m3=unit_mm_per_m3, sign_fallback=sign_fallback)


def move_slot(trainer, evaluation, slot, target):
    """
    Move `slot` so its album-mean u equals the target: shrink its x/y/depth weight rows of the last head layer to
    album std(u) <= GEOMETRY_ROW_STD_MAX_U (the dV row keeps its weights unless the move pushes |u| past MAX_ABS_U),
    then set each bias so mean(W h) + b = u_target. The other slot's rows are untouched.
    Output: dict u_target, max_abs_u_before / _after (moved slot), dv_row_shrunk.
    """
    ranges = trainer.model.physics_model.z_phy_ranges
    u_target = [_logit(_z_from_physical(ranges, 'xcen', target['x_km'])),
                _logit(_z_from_physical(ranges, 'ycen', target['y_km'])),
                _logit(_z_from_physical(ranges, 'd', target['depth_km'])),
                _logit(_z_from_physical(ranges, 'dV', target['dV_m3']))]
    dims = list(range(slot * PARAMETERS_PER_SOURCE, (slot + 1) * PARAMETERS_PER_SOURCE))
    _penultimate, last = _head_layers(trainer.model)
    h = evaluation['h']
    max_abs_u_before = float(evaluation['u'][:, dims].abs().max())
    dv_row_shrunk = False
    with torch.no_grad():
        for offset, dim in enumerate(dims):
            row_u = h @ last.weight[dim]                                    # [T], bias excluded
            if offset < 3:                                                  # x, y, depth: static rows
                factor = min(1.0, GEOMETRY_ROW_STD_MAX_U / max(float(row_u.std()), 1e-12))
                last.weight[dim] *= factor
                row_u = row_u * factor
            last.bias[dim] = u_target[offset] - row_u.mean()
            if offset == 3:                                                 # dV row: only shrink if it saturates
                u_row = row_u + last.bias[dim]
                spread = float((u_row - u_row.mean()).abs().max())
                if float(u_row.abs().max()) > MAX_ABS_U and spread > 1e-12:
                    factor = max(0.0, (MAX_ABS_U - abs(u_target[3])) / spread)
                    last.weight[dim] *= factor
                    last.bias[dim] = u_target[3] - (row_u * factor).mean()
                    dv_row_shrunk = True
        u_after = h @ last.weight[dims].T + last.bias[dims]
    return dict(u_target=u_target, max_abs_u_before=max_abs_u_before, max_abs_u_after=float(u_after.abs().max()),
                dv_row_shrunk=dv_row_shrunk)


def add_weight_noise(trainer, generator):
    """
    Gaussian noise on both layers of the z_phy-mean head (weights: x tensor RMS; biases: x paired weight's RMS),
    calibrated so the album RMS change of u ~ WEIGHT_NOISE_TARGET_RMS_DU (one probe at WEIGHT_NOISE_PROBE_SCALE,
    then rescaled; the head is linear in its last-layer noise and near-linear in the first).
    Output: dict noise_scale, rms_du.
    """
    head = trainer.model.enc.func_z_phy_mean.net
    linears = [module for module in head if isinstance(module, torch.nn.Linear)]
    clean = {id(p): p.detach().clone() for layer in linears for p in (layer.weight, layer.bias)}
    unit_noise = {}
    for layer in linears:
        weight_rms = float(layer.weight.pow(2).mean().sqrt())
        for param in (layer.weight, layer.bias):
            noise = torch.randn(param.shape, generator=generator, dtype=param.dtype).to(param.device)
            unit_noise[id(param)] = noise * weight_rms
    u_clean = trainer._last_eval['u']

    def apply(scale):
        with torch.no_grad():
            for layer in linears:
                for param in (layer.weight, layer.bias):
                    param.copy_(clean[id(param)] + scale * unit_noise[id(param)])
        return float((evaluate_album(trainer, trainer._current_epoch)['u'] - u_clean).pow(2).mean().sqrt())

    rms_probe = apply(WEIGHT_NOISE_PROBE_SCALE)
    scale = WEIGHT_NOISE_PROBE_SCALE * WEIGHT_NOISE_TARGET_RMS_DU / max(rms_probe, 1e-12)
    rms_du = apply(scale)
    return dict(noise_scale=scale, rms_du=rms_du)


def perturbed_parameters(trainer, perturbation):
    """Parameters whose Adam state is reset after the hop (the control arm uses the slot arm's scope)."""
    head = trainer.model.enc.func_z_phy_mean.net
    if perturbation == 'weight_noise':
        return [p for module in head if isinstance(module, torch.nn.Linear) for p in (module.weight, module.bias)]
    last = head[-1]
    return [last.weight, last.bias]


# ----------------------------------------------------------------------------------------------------------------
# Schedule, snapshots and logging
# ----------------------------------------------------------------------------------------------------------------
def _setup(trainer):
    """First-epoch checks, album tensors, grid, settings, RNGs."""
    cfg = trainer.config
    settings = dict(BASIN_HOP_DEFAULTS, **cfg['trainer']['phys_vae'].get('basin_hop', {}))
    if settings['perturbation'] not in PERTURBATIONS:
        raise ValueError(f"basin_hop.perturbation must be one of {PERTURBATIONS}, got {settings['perturbation']!r}")
    if int(settings['total_epochs']) != int(trainer.epochs):
        raise ValueError(f"basin_hop.total_epochs={settings['total_epochs']} != trainer.epochs={trainer.epochs}")
    if '-r' in sys.argv or '--resume' in sys.argv:
        raise ValueError("basin hopping does not support resume (-r): the hop state is not checkpointed")
    if getattr(trainer, 'use_iterative', False):
        raise ValueError("basin hopping evaluates with a single encode pass; iterative refinement is not supported")
    head = trainer.model.enc.func_z_phy_mean.net
    if not all(isinstance(m, (torch.nn.Linear, torch.nn.ELU, torch.nn.ReLU, torch.nn.Tanh, torch.nn.SiLU,
                              torch.nn.LeakyReLU, torch.nn.Softplus)) for m in head) or not isinstance(head[-1], torch.nn.Linear):
        raise ValueError(f"z_phy-mean head must be Linear/activation layers ending in Linear (no dropout/batchnorm): {head}")
    if settings['perturbation'] != 'off' and (settings['initial_epochs'] + settings['settle_epochs']
                                              + settings['polish_epochs'] > settings['total_epochs']):
        raise ValueError("initial + one settle window + polish must fit in total_epochs")
    if cfg['arch']['args']['physics'] != 'MultiMogi_LOS' or cfg['arch']['args'].get('encoder_type', 'mlp') != 'mlp':
        raise ValueError("basin hopping supports the MultiMogi_LOS point/MLP path only")
    # Effective KL: the delayed-penalty wrapper (runs inside this one) keeps it off while its factor is 0
    schedule = cfg['trainer']['phys_vae'].get('delayed_penalty') or {}
    kl_scheduled_off = bool(schedule) and (int(schedule['off_epochs']) >= int(trainer.epochs)
                                           or bool(schedule.get('kl_stays_off', False)))
    kl_on = bool(cfg['trainer']['phys_vae'].get('use_kl_term_z_phy', False)) and not kl_scheduled_off
    if kl_on or cfg['trainer']['phys_vae'].get('nuisance_ramp', False) or trainer.input_const_keys:
        raise ValueError("basin hopping assumes KL off (deterministic z_phy), no nuisance ramp, no input constants")
    if getattr(trainer.model.dec, 'use_time_in_residual', False):
        raise ValueError("basin hopping evaluates without time features; use_time_in_residual must be false")
    trainer._bh = settings
    physics = trainer.model.physics_model
    series = np.asarray(trainer.data_loader.dataset.series, dtype=np.float32)            # [T, N] standardized
    trainer._album_std = torch.tensor(series, device=trainer.device)
    trainer._album_mm = (trainer._album_std * physics.x_scale.to(trainer.device)
                         + physics.x_mean.to(trainer.device))                             # same scaler as the decoder
    # Grid + point coordinates for the residual map (same loader call as utils/warm_start.py)
    from datasets.preprocessing.insar_mintpy import load_insar_mintpy
    ins = cfg['arch']['args']['insar']
    loaded = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'), ins['lat0'], ins['lon0'],
                               multilook=ins.get('multilook', 1), coh_valid_frac=ins.get('coh_valid_frac', 0.5),
                               bbox=ins.get('bbox'), verbose=False, standardization=ins.get('std_mode', 'global'),
                               far_field=ins.get('far_field'), stride=ins.get('stride', 1), offset=ins.get('offset', 0),
                               bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
                               bootstrap_seed=ins.get('bootstrap_seed', 0), ref_date=ins.get('ref_date'))
    rows, cols = np.nonzero(loaded.mask_d)                          # row-major order == point order
    if len(rows) != series.shape[1]:
        raise ValueError(f"grid mask has {len(rows)} points, album has {series.shape[1]}")
    mm_check = float(np.abs(loaded.los_points_mm - trainer._album_mm.cpu().numpy()).max())
    if mm_check > 1e-2:
        raise ValueError(f"album mm reconstruction differs from the loader by {mm_check:.3e} mm")
    trainer._album_grid = dict(mask=loaded.mask_d.astype(bool), rows=rows, cols=cols,
                               x_pts_km=loaded.xE_grid[rows, cols], y_pts_km=loaded.yN_grid[rows, cols],
                               pixel_km=float(np.median(np.abs(np.diff(loaded.xE_grid, axis=1)))))
    trainer._bh_rng = np.random.RandomState(int(settings['seed']) + 1000)
    trainer._bh_generator = torch.Generator().manual_seed(int(settings['seed']) + 1000)
    trainer._bh_base_lr = float(cfg['optimizer']['args']['lr'])
    trainer._bh_best = None                 # dict(misfit, epoch, model, optimizer, params)
    trainer._bh_window = None               # active window bookkeeping (None during the initial phase)
    trainer._bh_hop_index = 0
    trainer._bh_dir = str(cfg.save_dir)
    os.makedirs(os.path.join(trainer._bh_dir, 'basin_best'), exist_ok=True)
    with open(os.path.join(trainer._bh_dir, 'basin_hop.json'), 'w') as handle:
        json.dump(dict(settings=settings, base_lr=trainer._bh_base_lr, n_acquisitions=int(series.shape[0]),
                       n_points=int(series.shape[1]), selection='deterministic album misfit (per-sample MSE)'),
                  handle, indent=2)
    print(f"[basin_hop] {settings}")


def _window_lr(trainer, window, epoch):
    """Cosine from peak to floor over the nominal window, then hold the floor during extensions."""
    peak = trainer._bh_base_lr * float(window['peak_frac'])
    floor = peak * float(trainer._bh['settle_lr_floor_frac'])
    t = epoch - window['start']
    length = max(1, window['nominal_length'] - 1)
    if t >= length:
        return floor
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * t / length))


def _snapshot(trainer, epoch, evaluation):
    """Keep the new best (model + optimizer, deep copies) in memory and on disk (basin_best/)."""
    trainer._bh_best = dict(misfit=evaluation['misfit'], phys_misfit=evaluation['phys_misfit'], epoch=epoch,
                            params=evaluation['params'].copy(),
                            model={k: v.detach().clone() for k, v in trainer.model.state_dict().items()},
                            optimizer=copy.deepcopy(trainer.optimizer.state_dict()))
    tau_r = trainer.model.dec.get_current_tau_r(epoch, trainer.epochs_pretrain) if trainer.model.dim_z_aux > 0 else None
    best_dir = os.path.join(trainer._bh_dir, 'basin_best')
    torch.save({'arch': type(trainer.model).__name__, 'epoch': epoch, 'state_dict': trainer._bh_best['model'],
                'monitor_best': evaluation['misfit'], 'config': trainer.config, 'tau_r_values': tau_r},
               os.path.join(best_dir, 'model_basin_best.pth'))
    with open(os.path.join(best_dir, 'basin_best.json'), 'w') as handle:
        json.dump(dict(epoch=epoch, misfit=evaluation['misfit'], phys_misfit=evaluation['phys_misfit'],
                       params_km_m3=evaluation['params'].tolist()), handle, indent=2)


def _append_csv(path, row):
    """Append one row; an existing file keeps its header (missing keys -> empty cells, unknown keys -> error)."""
    fieldnames = list(row)
    if os.path.exists(path):
        with open(path, newline='') as handle:
            fieldnames = next(csv.reader(handle))
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval='')
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _params_columns(params, prefix):
    return {f'{prefix}s{k + 1}_{name}': float(params[k, j]) for k in range(params.shape[0])
            for j, name in enumerate(('x_km', 'y_km', 'depth_km', 'dV_m3'))}


def _close_window(trainer):
    """Write the finished window's row to basin_hops.csv."""
    window = trainer._bh_window
    if window is None:
        return
    gain = (window['best_before'] - window['window_min']) / window['best_before']
    row = dict(hop=window['hop'], kind=window['kind'], start_epoch=window['start'], end_epoch=window['last_epoch'],
               extended_epochs=window['extended'], perturbation=window['perturbation'], slot=window.get('slot', ''),
               removal_misfits=window.get('removal_misfits', ''), best_before=window['best_before'],
               window_min=window['window_min'], epoch_of_min=window['epoch_of_min'], rel_gain=gain,
               accepted=bool(gain > float(trainer._bh['min_rel_improvement'])),
               slots_changed=slots_changed(window['params_before'], window['params_at_min']),
               phys_misfit_at_min=window['phys_misfit_at_min'], **window.get('details', {}),
               **_params_columns(window['params_before'], 'before_'), **_params_columns(window['params_at_min'], 'min_'))
    _append_csv(os.path.join(trainer._bh_dir, 'basin_hops.csv'), row)
    trainer.logger.info(f"[basin_hop] window {window['hop']} ({window['kind']}) epochs {window['start']}-"
                        f"{window['last_epoch']}: min {window['window_min']:.2f} vs best-before "
                        f"{window['best_before']:.2f} (gain {100 * gain:.2f} %), slots changed {row['slots_changed']}")
    trainer._bh_window = None


def _start_window(trainer, epoch, kind):
    """Revert to the best state, perturb (hop) or not (polish), reset Adam on the perturbed tensors."""
    _close_window(trainer)
    settings, best = trainer._bh, trainer._bh_best
    trainer.model.load_state_dict(best['model'])
    trainer.optimizer.load_state_dict(copy.deepcopy(best['optimizer']))
    perturbation = settings['perturbation'] if kind == 'hop' else 'none'
    window = dict(kind=kind, start=epoch, perturbation=perturbation, extended=0, n_extensions=0, best_before=best['misfit'],
                  params_before=best['params'].copy(), window_min=float('inf'), epoch_of_min=None,
                  params_at_min=best['params'].copy(), phys_misfit_at_min=float('nan'), last_epoch=epoch,
                  peak_frac=float(settings['settle_lr_peak_frac']))
    if kind == 'hop':
        trainer._bh_hop_index += 1
        window['hop'] = trainer._bh_hop_index
        window['nominal_length'] = int(settings['settle_epochs'])
        # sanity: evaluated at the best state's own epoch (decoder tau / r warm-ups) it must equal best_before
        misfit_after_revert = evaluate_album(trainer, best['epoch'])['misfit']
        if abs(misfit_after_revert - best['misfit']) > 1e-6 * best['misfit']:
            raise RuntimeError(f"revert failed: misfit {misfit_after_revert} != best {best['misfit']}")
        evaluation = evaluate_album(trainer, epoch)
        trainer._last_eval = evaluation
        if perturbation in ('slot', 'slot_resid'):
            removal = slot_removal_misfits(trainer, evaluation)
            slot = int(np.argmin(removal))                                  # removing it hurts the fit least
            target = draw_slot_target(trainer, evaluation, slot, perturbation == 'slot_resid', trainer._bh_rng)
            moved = move_slot(trainer, evaluation, slot, target)
            after = evaluate_album(trainer, epoch)
            window.update(slot=slot + 1, removal_misfits=';'.join(f'{value:.2f}' for value in removal))
            window['details'] = dict(**{f'target_{k}': v for k, v in target.items()}, **moved,
                                     moved_slot_params_after=';'.join(f'{v:.4g}' for v in after['params'][slot]),
                                     other_slot_max_abs_change=float(np.abs(np.delete(after['params'], slot, 0)
                                                                            - np.delete(evaluation['params'], slot, 0)).max()),
                                     moved_geometry_u_std_after=float(after['u'][:, slot * PARAMETERS_PER_SOURCE:
                                                                                 slot * PARAMETERS_PER_SOURCE + 3].std(dim=0).max()),
                                     misfit_after_perturb=after['misfit'])
            window['details']['u_target'] = ';'.join(f'{v:.3f}' for v in moved['u_target'])
        elif perturbation == 'weight_noise':
            noise = add_weight_noise(trainer, trainer._bh_generator)
            window['details'] = dict(**noise, misfit_after_perturb=evaluate_album(trainer, epoch)['misfit'])
        else:
            window['details'] = dict(misfit_after_perturb=evaluation['misfit'])
        window['details']['misfit_after_revert'] = misfit_after_revert
        for param in perturbed_parameters(trainer, perturbation):           # hops only; polish keeps Adam state
            trainer.optimizer.state[param] = {}                             # fresh Adam / AMSGrad state
    else:
        window['hop'] = 'polish'
        window['nominal_length'] = int(settings['polish_epochs'])
        window['details'] = dict(misfit_after_revert=evaluate_album(trainer, best['epoch'])['misfit'])
    trainer._bh_window = window
    trainer.logger.info(f"[basin_hop] epoch {epoch}: start {kind} {window['hop']} from best epoch {best['epoch']} "
                        f"(misfit {best['misfit']:.2f}), perturbation {perturbation}, details {window['details']}")


def basin_hop_train_epoch(self, epoch):
    """Hop/polish decisions at the start of the epoch, LR override, inner epoch, deterministic misfit, snapshot."""
    if not hasattr(self, '_bh'):
        _setup(self)
    self._current_epoch = epoch
    settings = self._bh
    polish_start = int(settings['total_epochs']) - int(settings['polish_epochs']) + 1
    window = self._bh_window
    if settings['perturbation'] != 'off' and epoch > int(settings['initial_epochs']):
        if epoch == polish_start:
            _start_window(self, epoch, 'polish')
        elif epoch < polish_start:
            end = None if window is None else window['start'] + window['nominal_length'] + window['extended']
            if window is None or epoch >= end:
                # extend only while the window is setting new GLOBAL bests near its end (comparable across arms:
                # a perturbed state that is merely still descending from a worse misfit does not extend)
                recent = (window is not None and window['n_extensions'] < int(settings['max_extensions'])
                          and self._bh_best['epoch'] >= window['start']
                          and self._bh_best['epoch'] > end - 1 - int(settings['extend_if_best_in_last']))
                fits = epoch + int(settings['settle_epochs']) - 1 < polish_start
                if window is not None and (recent or not fits):
                    if recent:
                        window['n_extensions'] += 1
                    window['extended'] += min(int(settings['extend_epochs']), polish_start - epoch) if recent \
                        else polish_start - epoch
                else:
                    _start_window(self, epoch, 'hop')
        if self._bh_window is not None:
            lr = _window_lr(self, self._bh_window, epoch)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
    lr_used = float(self.optimizer.param_groups[0]['lr'])

    result = _base_train_epoch(self, epoch)

    evaluation = evaluate_album(self, epoch)
    self._last_eval = evaluation
    if self._bh_best is None or evaluation['misfit'] < self._bh_best['misfit']:
        _snapshot(self, epoch, evaluation)
    window = self._bh_window
    if window is not None:
        window['last_epoch'] = epoch
        if evaluation['misfit'] < window['window_min']:
            window.update(window_min=evaluation['misfit'], epoch_of_min=epoch, params_at_min=evaluation['params'].copy(),
                          phys_misfit_at_min=evaluation['phys_misfit'])
    _append_csv(os.path.join(self._bh_dir, 'basin_epochs.csv'), dict(
        epoch=epoch, phase='initial' if window is None else window['kind'], hop=self._bh_hop_index if window else 0,
        lr_used=lr_used, misfit=evaluation['misfit'], phys_misfit=evaluation['phys_misfit'],
        best_misfit=self._bh_best['misfit'], best_epoch=self._bh_best['epoch'],
        val_rec_loss=float(result.get('val_rec_loss', float('nan'))), rec_loss=float(result.get('rec_loss', float('nan'))),
        **_params_columns(evaluation['params'], '')))
    result['basin_misfit'] = evaluation['misfit']
    _run_state['trainer'] = self
    return result


PhysVAETrainerSMPL._train_epoch = basin_hop_train_epoch       # in memory only

if __name__ == '__main__':
    runpy.run_path(os.path.join(PROJECT_ROOT, 'train_pila.py'), run_name='__main__')
    trainer = _run_state['trainer']
    if trainer is None:
        raise RuntimeError("basin hop: no epoch ran")
    _close_window(trainer)
    print(f"[basin_hop] best misfit {trainer._bh_best['misfit']:.3f} at epoch {trainer._bh_best['epoch']}")
    # --- Dump the lowest-val_rec_loss model (as train_pila_keepbest.py) and the basin-best model ---
    models_dir = str(trainer.config.save_dir)
    for checkpoint, out_dir in ((os.path.join(models_dir, 'lowest_loss', 'model_lowest_loss.pth'), 'lowest_loss'),
                                (os.path.join(models_dir, 'basin_best', 'model_basin_best.pth'), 'basin_best')):
        if not os.path.exists(checkpoint):
            raise RuntimeError(f"basin hop: {checkpoint} was not written")
        n_sources = int(trainer.config['arch']['args'].get('n_sources', 1))
        out_npz = os.path.join(models_dir, out_dir, f'fit_K{n_sources}.npz')
        print(f"[basin_hop] Dumping {checkpoint} -> {out_npz}")
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'dump_multisource_fit.py'), '--resume', checkpoint,
                        '--out', out_npz], check=True, cwd=PROJECT_ROOT)
    print(f"Done. Output saved to {models_dir}/")
