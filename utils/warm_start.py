#!/usr/bin/env python
# Usage:       (called by train_pila.py when config arch.phys_vae.init_from is set)
#              from utils.warm_start import warm_start_multimogi
#              info = warm_start_multimogi(model, config, data_loader, logger)
# Description: Warm-start a K-source MultiMogi PILA model from a trained (K-1)-source checkpoint,
#              the "add one source at a time" step of the sequential multi-source plan:
#                * every weight with a matching shape is copied (feature extractor, z_aux heads,
#                  low-rank residual B, s, coeff);
#                * the z_phy mean/lnvar output layers get the (K-1) old sources' rows copied,
#                  and the NEW source's 4 rows are initialised so that it starts at the peak of
#                  the stack-mean residual (data - physics fit of the K-1 sources), CLEAN-style.
# Scientific notes:
#   * The new source must NOT start at dV = 0: Mogi LOS is proportional to dV, so at dV = 0 the
#     field does not depend on x, y or depth and their gradients vanish. It starts with
#     peak |LOS| = 0.5 x the smoothed residual peak, with the residual's sign (inflation/deflation),
#     at mid-range depth.
#   * Residual smoothing: Gaussian, sigma = 2 km, normalised convolution over coherent cells, so the
#     peak reflects a coherent source-like blob rather than a single noisy pixel.
#   * Only the point/MLP path (encoder_type 'mlp') is supported; the CNN grid path raises.
#   * Units: LOS mm (+ toward satellite), coordinates km, dV m^3.
# Date:        2026-10-06

import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

RESIDUAL_SMOOTH_KM = 2.0          # Gaussian sigma for residual-peak placement
NEW_SOURCE_PEAK_FRACTION = 0.5    # new source starts at this fraction of the residual peak
NEW_ROW_WEIGHT_SCALE = 0.01       # new rows' weights = this x std of existing rows' weights
Z_CLIP = 0.02                     # keep the start away from the sigmoid rails (logit finite)
INIT_SOURCES_LNVAR = -6.0         # explicit start: sampling std = exp(-6/2) ~ 0.05 in u-space (default head gives std ~ 1)


def _logit(z):
    z = float(np.clip(z, Z_CLIP, 1.0 - Z_CLIP))
    return float(np.log(z / (1.0 - z)))


def _z_from_physical(ranges, name, value):
    """
    Invert Physics_Mogi.rescale for one parameter: physical value -> z in (0, 1).
    value units: km for xcen/ycen/d (the paras-file units, before rescale's x1000), m^3 for dV.
    """
    lo, hi = float(ranges[name]['min']), float(ranges[name]['max'])
    if name == 'dV':
        scale = float(ranges['dV'].get('scale', 1e5))
        shift = float(ranges['dV'].get('shift', -1e7))
        value = (value - shift) / scale
    return (value - lo) / (hi - lo)


def warm_start_multimogi(model, config, data_loader, logger):
    """
    Initialise `model` (K sources) from the (K-1)-source checkpoint at arch.phys_vae.init_from.

    Inputs : model        PHYS_VAE_SMPL with physics MultiMogi_LOS (not yet on device is fine)
             config       ConfigParser / dict of THIS run
             data_loader  training loader (its dataset provides the standardized series)
             logger       run logger
    Output : dict with copied / partially copied / new tensor names, new-source start
             (x_km, y_km, depth_km, dV_m3), residual peak (mm) and warm_start_sec.
    """
    t_start = time.perf_counter()
    init_path = config['arch']['phys_vae']['init_from']
    physics = model.physics_model
    n_sources = int(getattr(physics, 'n_sources', 1))
    if config['arch']['args'].get('encoder_type', 'mlp') != 'mlp':
        raise NotImplementedError("warm start supports the point/MLP path only")
    if n_sources < 2:
        raise ValueError("init_from warm start needs n_sources >= 2")
    n_per = len(physics.z_phy_ranges)
    n_old_dims = n_per * (n_sources - 1)

    print(f"[warm start 1/4] Loading (K-1) checkpoint {init_path}")
    ckpt = torch.load(init_path, map_location='cpu', weights_only=False)
    old_state = ckpt['state_dict']
    new_state = model.state_dict()
    head_keys = {}
    for head in ('func_z_phy_mean', 'func_z_phy_lnvar'):
        last = max(int(k.split('.')[3]) for k in new_state if k.startswith(f'enc.{head}.net.'))
        head_keys[head] = (f'enc.{head}.net.{last}.weight', f'enc.{head}.net.{last}.bias')

    print("[warm start 2/4] Copying weights...")
    copied, partial, fresh = [], [], []
    head_key_set = {k for pair in head_keys.values() for k in pair}
    with torch.no_grad():
        for key, tensor in new_state.items():
            if key in head_key_set:
                old = old_state[key]
                if old.shape[0] != n_old_dims:
                    raise ValueError(f"{key}: checkpoint has {old.shape[0]} output rows, expected "
                                     f"{n_old_dims} for {n_sources - 1} sources")
                tensor[:n_old_dims] = old
                partial.append(key)
            elif key in old_state and old_state[key].shape == tensor.shape:
                tensor.copy_(old_state[key])
                copied.append(key)
            else:
                fresh.append(key)
        # New rows: tiny weights (break symmetry, ~source-independent start), biases set below.
        for head, (w_key, b_key) in head_keys.items():
            w = new_state[w_key]
            w[n_old_dims:] = torch.randn_like(w[n_old_dims:]) * NEW_ROW_WEIGHT_SCALE * w[:n_old_dims].std()
            if head == 'func_z_phy_lnvar':
                new_state[b_key][n_old_dims:] = new_state[b_key][:n_old_dims].mean()
            else:
                new_state[b_key][n_old_dims:] = 0.0                    # placeholder until step 4
    model.load_state_dict(new_state)

    print("[warm start 3/4] Residual of the (K-1)-source physics fit over the whole stack...")
    from datasets.preprocessing.insar_mintpy import load_insar_mintpy
    ins = config['arch']['args']['insar']
    d = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'), ins['lat0'], ins['lon0'],
                          multilook=ins.get('multilook', 1), coh_valid_frac=ins.get('coh_valid_frac', 0.5),
                          bbox=ins.get('bbox'), verbose=False, standardization=ins.get('std_mode', 'global'),
                          far_field=ins.get('far_field'), stride=ins.get('stride', 1), offset=ins.get('offset', 0),
                          bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
                          bootstrap_seed=ins.get('bootstrap_seed', 0), ref_date=ins.get('ref_date'))
    series_std = torch.tensor(np.asarray(data_loader.dataset.series), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        z_phy_stat, _ = model.encode(series_std, None)
        z_phy = torch.sigmoid(z_phy_stat['mean'])
        old_fit_mm = physics.per_source_los_mm(z_phy)[:, :n_sources - 1].sum(dim=1).cpu().numpy()   # [T, N]
    model.train()
    residual_mm = d.los_points_mm - old_fit_mm                                                       # [T, N]
    if config['trainer']['phys_vae'].get('nuisance_ramp', False):
        # Same nuisance model as the loss: remove each acquisition's LS offset + planar ramp.
        G = np.stack([np.ones(d.n_points), d.xE_pts, d.yN_pts], axis=1)
        residual_mm = residual_mm - (residual_mm @ np.linalg.pinv(G).T) @ G.T
    residual_mean_mm = np.nanmean(residual_mm, axis=0)                                                # [N]

    # Scatter to the grid, smooth with a normalised Gaussian over coherent cells.
    grid_res = np.zeros(d.mask_d.shape); grid_w = np.zeros(d.mask_d.shape)
    rows, cols = np.nonzero(d.mask_d)                     # row-major order == point order
    grid_res[rows, cols] = residual_mean_mm; grid_w[rows, cols] = 1.0
    pixel_km = float(np.median(np.abs(np.diff(d.xE_grid, axis=1))))
    sigma_px = RESIDUAL_SMOOTH_KM / pixel_km
    smooth = gaussian_filter(grid_res, sigma_px) / np.maximum(gaussian_filter(grid_w, sigma_px), 1e-6)
    smooth[~d.mask_d] = 0.0
    r_peak, c_peak = np.unravel_index(np.argmax(np.abs(smooth)), smooth.shape)
    peak_mm = float(smooth[r_peak, c_peak])
    x_km, y_km = float(d.xE_grid[r_peak, c_peak]), float(d.yN_grid[r_peak, c_peak])

    print("[warm start 4/4] Placing the new source at the residual peak...")
    ranges = physics.z_phy_ranges
    z_x = _z_from_physical(ranges, 'xcen', x_km)
    z_y = _z_from_physical(ranges, 'ycen', y_km)
    seed_init = config['arch']['phys_vae'].get('seed_init') or {}
    if seed_init.get('depth_km') is not None:             # opt-in test (2026-10-06): shallow-depth start
        z_d = _z_from_physical(ranges, 'd', float(seed_init['depth_km']))
    else:
        z_d = 0.5                                         # mid-range depth
    depth_km = float(ranges['d']['min']) + z_d * (float(ranges['d']['max']) - float(ranges['d']['min']))
    # Unit-dV field of the new source (LOS is linear in dV) -> dV for the target peak.
    probe_dv_m3 = 1e6
    z_probe = torch.full((1, n_sources * n_per), 0.5)
    z_probe[0, -n_per:] = torch.tensor([z_x, z_y, z_d, _z_from_physical(ranges, 'dV', probe_dv_m3)])
    # SIGNED unit-dV LOS at the residual-peak cell (point order = row-major over the mask): the
    # LOS of an inflation source can be negative off-centre for steep / horizontal geometry, so the
    # dV sign must come from residual / unit-field, not from the residual sign alone.
    peak_point = int(np.flatnonzero(d.mask_d.ravel()).tolist().index(r_peak * d.mask_d.shape[1] + c_peak))
    with torch.no_grad():
        unit_at_peak_mm = float(physics.per_source_los_mm(z_probe)[0, -1, peak_point]) / probe_dv_m3
    dv_m3 = NEW_SOURCE_PEAK_FRACTION * peak_mm / unit_at_peak_mm if abs(unit_at_peak_mm) > 1e-30 else 0.0
    z_dv = _z_from_physical(ranges, 'dV', dv_m3)
    if not (Z_CLIP <= z_dv <= 1.0 - Z_CLIP):
        logger.warning(f"warm start: target dV {dv_m3:.3e} m^3 lies outside the prior range; "
                       f"start clipped to z={np.clip(z_dv, Z_CLIP, 1 - Z_CLIP):.3f}")
    u_new = [_logit(z_x), _logit(z_y), _logit(z_d), _logit(z_dv)]
    with torch.no_grad():
        model.state_dict()[head_keys['func_z_phy_mean'][1]][n_old_dims:] = torch.tensor(u_new)
    warm_start_sec = time.perf_counter() - t_start

    info = dict(init_from=init_path, n_sources=n_sources, copied=copied, partially_copied=partial,
                fresh=fresh, residual_peak_mm=peak_mm, new_source_start=dict(
                    xcen_km=x_km, ycen_km=y_km, depth_km=depth_km, dV_m3=float(dv_m3)),
                residual_smooth_km=RESIDUAL_SMOOTH_KM, warm_start_sec=warm_start_sec)
    logger.info(f"Warm start from {init_path}: copied {len(copied)} tensors, partially copied {partial}, "
                f"fresh {fresh}")
    logger.info(f"  new source {n_sources} starts at x={x_km:.2f} km, y={y_km:.2f} km, depth={depth_km:.2f} km, "
                f"dV={dv_m3:.3e} m^3 (residual peak {peak_mm:.2f} mm); took {warm_start_sec:.1f}s")
    return info


def seed_first_source(model, config, data_loader, logger):
    """
    Opt-in start for the FIRST source (K = 1), enabled by config arch.phys_vae.seed_init
    = {"depth_km": D, "at_peak": bool}. Only the bias of the z_phy-mean output layer is set
    (weights untouched): depth is initialised at D km; if at_peak, x, y and dV are placed at the
    peak of the smoothed stack-mean data (K = 0 residual, 2 km Gaussian, as in warm_start_multimogi)
    with 0.5 x peak LOS and the sign from the unit-dV field. Otherwise x, y, dV keep the default.
    Units: km, m^3, mm. Returns dict with the start used.
    """
    seed_init = config['arch']['phys_vae']['seed_init']
    physics = model.physics_model
    ranges = physics.z_phy_ranges
    if int(getattr(physics, 'n_sources', 1)) != 1:
        raise ValueError("seed_first_source is for K = 1")
    n_per = len(ranges)
    w_key = [k for k in model.state_dict() if k.startswith('enc.func_z_phy_mean.net.') and k.endswith('.bias')]
    b_key = sorted(w_key, key=lambda k: int(k.split('.')[3]))[-1]
    u_start = {'d': _logit(_z_from_physical(ranges, 'd', float(seed_init['depth_km'])))}
    start = dict(depth_km=float(seed_init['depth_km']), at_peak=bool(seed_init.get('at_peak', False)))
    if seed_init.get('at_peak', False):
        from datasets.preprocessing.insar_mintpy import load_insar_mintpy
        ins = config['arch']['args']['insar']
        d = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'), ins['lat0'], ins['lon0'],
                              multilook=ins.get('multilook', 1), coh_valid_frac=ins.get('coh_valid_frac', 0.5),
                              bbox=ins.get('bbox'), verbose=False, standardization=ins.get('std_mode', 'global'),
                              far_field=ins.get('far_field'), stride=ins.get('stride', 1), offset=ins.get('offset', 0),
                              bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
                              bootstrap_seed=ins.get('bootstrap_seed', 0), ref_date=ins.get('ref_date'))
        residual_mean_mm = np.nanmean(d.los_points_mm, axis=0)                     # [N]
        grid_res = np.zeros(d.mask_d.shape); grid_w = np.zeros(d.mask_d.shape)
        rows, cols = np.nonzero(d.mask_d)
        grid_res[rows, cols] = residual_mean_mm; grid_w[rows, cols] = 1.0
        pixel_km = float(np.median(np.abs(np.diff(d.xE_grid, axis=1))))
        sigma_px = RESIDUAL_SMOOTH_KM / pixel_km
        smooth = gaussian_filter(grid_res, sigma_px) / np.maximum(gaussian_filter(grid_w, sigma_px), 1e-6)
        smooth[~d.mask_d] = 0.0
        r_peak, c_peak = np.unravel_index(np.argmax(np.abs(smooth)), smooth.shape)
        peak_mm = float(smooth[r_peak, c_peak])
        x_km, y_km = float(d.xE_grid[r_peak, c_peak]), float(d.yN_grid[r_peak, c_peak])
        z_x, z_y = _z_from_physical(ranges, 'xcen', x_km), _z_from_physical(ranges, 'ycen', y_km)
        z_d = _z_from_physical(ranges, 'd', float(seed_init['depth_km']))
        probe_dv_m3 = 1e6
        z_probe = torch.full((1, n_per), 0.5)
        z_probe[0] = torch.tensor([z_x, z_y, z_d, _z_from_physical(ranges, 'dV', probe_dv_m3)])
        peak_point = int(np.flatnonzero(d.mask_d.ravel()).tolist().index(r_peak * d.mask_d.shape[1] + c_peak))
        with torch.no_grad():
            unit_at_peak_mm = float(physics.per_source_los_mm(z_probe)[0, -1, peak_point]) / probe_dv_m3
        dv_m3 = NEW_SOURCE_PEAK_FRACTION * peak_mm / unit_at_peak_mm if abs(unit_at_peak_mm) > 1e-30 else 0.0
        u_start.update(xcen=_logit(z_x), ycen=_logit(z_y), dV=_logit(_z_from_physical(ranges, 'dV', dv_m3)))
        start.update(xcen_km=x_km, ycen_km=y_km, dV_m3=float(dv_m3), peak_mm=peak_mm)
    names = list(ranges.keys())                                                    # xcen, ycen, d, dV
    with torch.no_grad():
        bias = model.state_dict()[b_key]
        for name, u_value in u_start.items():
            bias[names.index(name)] = u_value
    logger.info(f"seed_init (K=1): {start}")
    print(f"[seed_init] K=1 start: {start}")
    return start


def warm_start_explicit_sources(model, config, logger):
    """
    Start a FRESH K-source MultiMogi network at explicit sources (K>1 initialisation test), enabled by
    config arch.phys_vae.init_sources = [[xcen_km, ycen_km, depth_km, dV_m3], ...] (K rows, source-major
    like the latent: [x1,y1,d1,dV1, x2,...]). No checkpoint is used: backbone and aux/low-rank params keep
    their seeded random init; the z_phy-mean output layer's weights are set to zero
    (so every output starts EXACTLY constant) and its biases are set to logit(z) of the requested sources, so
    sigmoid(u) -> the requested physical parameters for any input at epoch 0.
    Units: km, m^3. dV must be non-zero (gradients of x, y, depth vanish at dV = 0 - see header).
    The lnvar head is also started tight (zero weights, bias INIT_SOURCES_LNVAR) so sampled latent noise does
    not wash out the start. Returns dict(requested, z, clipped, lnvar_init, warm_start_sec, init_from,
    new_source_start) for the logs / compute_usage.json / sanity checks.
    """
    t_start = time.perf_counter()
    sources = config['arch']['phys_vae']['init_sources']
    physics = model.physics_model
    ranges = physics.z_phy_ranges
    n_sources = int(getattr(physics, 'n_sources', 1))
    n_per = len(ranges)
    if len(sources) != n_sources or any(len(row) != n_per for row in sources):
        raise ValueError(f"init_sources must be {n_sources} rows of {n_per} values (xcen, ycen, d, dV)")
    if any(float(row[3]) == 0.0 for row in sources):
        raise ValueError("init_sources: dV must be non-zero (no gradient for x, y, depth at dV = 0)")
    if config['arch']['args'].get('encoder_type', 'mlp') != 'mlp':
        raise NotImplementedError("explicit init supports the point/MLP path only")
    head_w = [k for k in model.state_dict() if k.startswith('enc.func_z_phy_mean.net.') and k.endswith('.weight')]
    last = max(int(k.split('.')[3]) for k in head_w)
    w_key, b_key = f'enc.func_z_phy_mean.net.{last}.weight', f'enc.func_z_phy_mean.net.{last}.bias'
    lnvar_w_key = f'enc.func_z_phy_lnvar.net.{last}.weight'
    lnvar_b_key = f'enc.func_z_phy_lnvar.net.{last}.bias'
    names = ['xcen', 'ycen', 'd', 'dV']
    z_requested = np.array([[_z_from_physical(ranges, name, float(value)) for name, value in zip(names, row)]
                            for row in sources])                                   # [K, 4]
    clipped = bool(np.any((z_requested < Z_CLIP) | (z_requested > 1.0 - Z_CLIP)))
    if clipped:
        logger.warning(f"init_sources: z outside [{Z_CLIP}, {1 - Z_CLIP}] was clipped: {z_requested.round(3).tolist()}")
    u_start = np.array([[_logit(z) for z in row] for row in z_requested]).reshape(-1)   # source-major [4K]
    with torch.no_grad():
        state = model.state_dict()
        state[w_key].zero_()          # exact start: a x0.01 scaling left a ~0.3% input-dependent offset (up to ~2 mm); the
                                      # zeroed weights still receive gradients (dL/dW = dL/du * hidden activation)
        state[b_key].copy_(torch.tensor(u_start, dtype=state[b_key].dtype))
        # Training samples u = mean + eps*std when the KL term is on; the default lnvar head (std ~ 1 in u-space) would
        # blur a truth start by +-2.6 km in depth and +-1e8 m^3 in dV. Start tight: lnvar = INIT_SOURCES_LNVAR.
        state[lnvar_w_key].zero_()
        state[lnvar_b_key].fill_(INIT_SOURCES_LNVAR)
    z_back = 1.0 / (1.0 + np.exp(-u_start))
    info = dict(requested=[list(map(float, row)) for row in sources], z=z_back.reshape(n_sources, n_per).round(4).tolist(),
                clipped=clipped, lnvar_init=INIT_SOURCES_LNVAR, init_from='explicit_sources',
                new_source_start=[list(map(float, row)) for row in sources],
                warm_start_sec=time.perf_counter() - t_start)       # keys train_pila.log_compute_usage reads
    logger.info(f"init_sources (K={n_sources}): {info}")
    print(f"[init_sources] K={n_sources} start: {info['requested']} -> z {info['z']}")
    return info
