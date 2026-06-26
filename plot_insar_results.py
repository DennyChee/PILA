# Usage:       python plot_insar_results.py \
#                  --resume saved/sierranegra_mogi_insar_A/<ts>/models/model_best.pth \
#                  [--config <config.json>] [--epoch -1] [--output_dir DIR]
# Description: h5-native inference + verification plots for a PILA InSAR (Mogi_LOS /
#              Okada_LOS, MLP) checkpoint. Runs the trained physics-VAE on the
#              multilooked MintPy LOS field, de-standardizes to mm, and plots
#              reconstructed-vs-actual LOS as (a) a 3-panel map over a hillshade
#              background derived from geo_geometryRadar.h5 and (b) a 1:1 scatter.
# Date:        2026-06-13
#
# Scientific conventions (READ — these affect interpretation):
#   * LOS sign: positive = motion TOWARD the satellite (range decrease), MintPy
#     convention. Displacement maps use a diverging colormap centred at 0.
#   * The dataset z-scores LOS with a single GLOBAL (scene-wide) scaler. The model output
#     (x_PB) and target are in standardized units, so both are inverted with
#     mm = std*x_scale_global + x_mean_global before plotting/RMSE in mm.
#   * Hillshade uses the geo_geometryRadar.h5 `height` DEM as spatial context only;
#     displacement cells are georeferenced from the MintPy geocoding attributes
#     (X_FIRST/Y_FIRST/X_STEP/Y_STEP), multilooked by the same factor as the data.
#   * lat0/lon0 in the config is the local-ENU peg, NOT the inverted source location.

import argparse
import os
import time

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless (HPC / no display)
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from model import PHYS_VAE_SMPL
from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from datasets.displacementGPS import time_feats
from utils import read_json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(config, checkpoint_path, device):
    """
    Build PHYS_VAE_SMPL, load the checkpoint state, and restore tau/r for inference.

    Inputs:  config (dict), checkpoint_path (str, .pth), device (torch.device)
    Output:  model in eval() mode on `device`.
    """
    model = PHYS_VAE_SMPL(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # Restore the annealed temperature / residual-scale used at the end of training,
    # so inference matches the trained decoder behaviour (mirrors test_pila_insar.py).
    no_phy = config['arch']['phys_vae']['no_phy']
    tau_r_values = checkpoint.get('tau_r_values', None)
    if tau_r_values is not None and not no_phy:
        model.dec.set_tau_r_from_checkpoint(tau_r_values)
        print(f"  Restored tau={tau_r_values['tau']:.3f}, r={tau_r_values['r']:.3f} from checkpoint")
    else:
        print("  Using default tau/r values for inference")

    model = model.to(device)
    model.eval()
    return model


def hard_z_flags(config):
    """Deterministic latents when the matching KL term was disabled during training."""
    pv = config['trainer']['phys_vae']
    if 'use_kl_term' in pv:                       # legacy single-flag configs
        use_kl_phy = use_kl_aux = pv['use_kl_term']
    else:
        use_kl_phy = pv.get('use_kl_term_z_phy', False)
        use_kl_aux = pv.get('use_kl_term_z_aux', False)
    return (not use_kl_phy), (not use_kl_aux)


def coarse_extent_lonlat(geometry_h5, hd, wd, multilook):
    """
    Geographic extent (in degrees) of the multilooked coarse grid, for imshow.

    Reconstructs cell edges from the MintPy geocoding attributes. The coarse grid
    spans the fine grid cropped to a whole multiple of `multilook` (matches the
    cropping in insar_mintpy.mask_aware_multilook). Returns
    [lon_min, lon_max, lat_min, lat_max] suitable for imshow(origin='upper').
    """
    with h5py.File(geometry_h5, 'r') as f:
        attrs = dict(f.attrs)
    x_first = float(attrs['X_FIRST'])             # lon of left edge of col 0
    y_first = float(attrs['Y_FIRST'])             # lat of top edge of row 0
    x_step = float(attrs['X_STEP'])               # deg / fine pixel (lon, +)
    y_step = float(attrs['Y_STEP'])               # deg / fine pixel (lat, -)
    lon_min = x_first
    lon_max = x_first + x_step * (wd * multilook)
    lat_top = y_first
    lat_bot = y_first + y_step * (hd * multilook)
    # imshow extent order is [left, right, bottom, top]; y_step<0 so bottom<top.
    return [lon_min, lon_max, lat_bot, lat_top]


def load_hillshade(geometry_h5, azimuth_deg=315.0, altitude_deg=45.0):
    """
    Grayscale hillshade of the DEM in geo_geometryRadar.h5, plus its lon/lat extent.

    Inputs:  geometry_h5 (str), illumination azimuth/altitude (deg).
    Output:  (hillshade [L, W] in 0..1, extent [lon_min, lon_max, lat_min, lat_max]).
    """
    with h5py.File(geometry_h5, 'r') as f:
        height_m = np.array(f['height'], dtype=np.float64)   # DEM, metres
        attrs = dict(f.attrs)
    # NaN/voids -> sea level so the shader does not propagate NaNs.
    height_m = np.nan_to_num(height_m, nan=0.0)
    light = LightSource(azdeg=azimuth_deg, altdeg=altitude_deg)
    hs = light.hillshade(height_m, vert_exag=1.0)            # 0..1

    x_first = float(attrs['X_FIRST'])
    y_first = float(attrs['Y_FIRST'])
    x_step = float(attrs['X_STEP'])
    y_step = float(attrs['Y_STEP'])
    n_rows, n_cols = height_m.shape
    extent = [x_first, x_first + x_step * n_cols,
              y_first + y_step * n_rows, y_first]            # [l, r, b, t]
    return hs, extent


def points_to_grid(values_at_points, mask_d):
    """Scatter per-point values (row-major over coherent cells) back to a [Hd,Wd] grid."""
    grid = np.full(mask_d.shape, np.nan, dtype=np.float64)
    rows, cols = np.where(mask_d)                            # same order as los_points_mm
    grid[rows, cols] = values_at_points
    return grid


# Parameters whose rescaled value is in metres but is naturally read in km.
_KM_PARAMS = {'xcen', 'ycen', 'd', 'xoff', 'yoff', 'depth', 'length', 'width', 'radius'}


def _param_scalar(v):
    """Pull a python float out of a rescale() entry (torch tensor or array)."""
    arr = v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)
    return float(arr.reshape(-1)[0])


# Horizontal-position params are shown via the map itself, not the annotation box.
_SKIP_IN_BOX = {'xcen', 'ycen', 'xoff', 'yoff'}


def format_param_text(params):
    """Human-readable multi-line string of inferred source parameters (with units).

    Horizontal-position params (xcen/ycen, xoff/yoff) are omitted — they're the location
    on the map, not informative as text.
    """
    lines = []
    for k, v in params.items():
        if k in _SKIP_IN_BOX:
            continue
        val = _param_scalar(v)
        if k == 'dV':
            lines.append(f"dV = {val/1e6:+.1f}e6 m³")        # volume change
        elif k in _KM_PARAMS:
            lines.append(f"{k} = {val/1000.0:.2f} km")           # m -> km
        elif k in ('strike', 'dip'):
            lines.append(f"{k} = {val:.1f}°")
        elif k == 'opening':
            lines.append(f"opening = {val:.2f} m")
        else:
            lines.append(f"{k} = {val:.3g}")
    return "\n".join(lines)


# A parameter is flagged RAILED when its inferred value sits within this fraction of the
# full prior range from either bound -- i.e. the inversion pinned it to the edge of the
# prior box rather than finding an interior optimum (a hallmark of the Okada degeneracy).
_RAILED_FRAC = 0.01


def prior_bounds_report(physics_model, params):
    """Compare each inferred parameter against its prior bounds and flag railing.

    The physics decoder stores the per-parameter prior box in `physics_model.z_phy_ranges`
    (a dict {name: {min, max}}). Bounds for the length-like parameters listed in
    `physics_model.km_params` are expressed in km, while the rescaled values in `params`
    are in metres -- so those bounds are scaled by 1000 before comparison. Angles
    (strike/dip, degrees) and opening (m) are compared as-is.

    Inputs:
        physics_model : the trained decoder (must expose z_phy_ranges + km_params).
        params        : dict of rescaled inferred parameters (metres / degrees), as
                        returned by physics_model.rescale().
    Output:
        (txt_block, footnote) -- a multi-line string for the saved *_params.txt file and a
        short one/two-line footnote for the figure annotation. Returns ("", "") if the
        model does not expose prior ranges (e.g. a no-physics run).
    """
    ranges = getattr(physics_model, 'z_phy_ranges', None)
    if not ranges:
        return "", ""
    km_params = set(getattr(physics_model, 'km_params', []))

    bound_lines = []          # one '  name = val  [min, max] unit  RAILED?' line per param
    railed_names = []         # names pinned to a bound, for the figure footnote
    for para_name, rng in ranges.items():
        if para_name not in params:
            continue
        # Inferred value in physical units (m for lengths/opening, deg for angles).
        val = _param_scalar(params[para_name])
        # Prior bounds in the SAME physical units as `val` (km -> m for length-like params).
        min_phys = rng['min'] * 1000.0 if para_name in km_params else float(rng['min'])
        max_phys = rng['max'] * 1000.0 if para_name in km_params else float(rng['max'])
        span = max_phys - min_phys
        # Normalised position in the prior box, 0 at min and 1 at max.
        z_norm = (val - min_phys) / span if span > 0 else 0.0
        is_railed = (z_norm <= _RAILED_FRAC) or (z_norm >= 1.0 - _RAILED_FRAC)
        if is_railed:
            railed_names.append(para_name)
        # Report km-scaled params in km (and opening in m) to match format_param_text.
        if para_name in km_params:
            disp_val, disp_min, disp_max, unit = val / 1000.0, rng['min'], rng['max'], 'km'
        elif para_name in ('strike', 'dip'):
            disp_val, disp_min, disp_max, unit = val, rng['min'], rng['max'], 'deg'
        else:
            disp_val, disp_min, disp_max, unit = val, rng['min'], rng['max'], 'm'
        flag = '  <-- RAILED' if is_railed else ''
        bound_lines.append(
            f"  {para_name:8s} = {disp_val:9.3f}  [{disp_min:g}, {disp_max:g}] {unit}{flag}")

    txt_block = "# prior bounds (inferred value vs [min, max]):\n" + "\n".join(bound_lines)
    if railed_names:
        footnote = "RAILED (pinned to prior bound): " + ", ".join(railed_names)
    else:
        footnote = "no parameters railed to prior bounds"
    return txt_block, footnote


def source_lonlat(params, lat0, lon0):
    """
    Inferred source horizontal position -> (lon, lat) in degrees.

    Uses the inverse of the same azimuthal-equidistant projection (about lat0/lon0)
    that mapped pixels into the local ENU frame. Mogi uses xcen/ycen; Okada xoff/yoff.
    Returns None if no horizontal-position parameter is present.
    """
    if 'xcen' in params and 'ycen' in params:
        x_m, y_m = _param_scalar(params['xcen']), _param_scalar(params['ycen'])
    elif 'xoff' in params and 'yoff' in params:
        x_m, y_m = _param_scalar(params['xoff']), _param_scalar(params['yoff'])
    else:
        return None
    import pyproj
    aeqd = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    transformer = pyproj.Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x_m, y_m)               # inverse of the forward proj
    return float(lon), float(lat)


def plot_maps(actual_mm, phys_mm, full_mm, disp_extent, hs, hs_extent, out_base,
              param_text=None, fit_text=None, disp_alpha=0.8, suptitle=None):
    """
    4-panel map over a hillshade:
      Actual LOS | Physics-only (x_P) | Physics+residual (x_PB) | Physics residual.

    x_P is the PURE physics (Mogi/Okada) reconstruction — it can only contain a smooth
    source field, so it is the denoised view. x_PB additionally includes the learned
    low-rank residual augmentation. The physics residual (actual - x_P) is what the source
    alone cannot explain (atmosphere/noise + model mismatch).

    param_text : inferred source parameters, annotated on the physics panel.
    fit_text   : R^2/RMSE lines (physics and full) for the annotation box.
    disp_alpha : opacity of the displacement overlay.
    """
    phys_resid_mm = phys_mm - actual_mm
    vmax = np.nanmax(np.abs(np.concatenate([actual_mm.ravel(), phys_mm.ravel(), full_mm.ravel()])))
    rmax = np.nanmax(np.abs(phys_resid_mm))
    panels = [("Actual LOS", actual_mm, vmax),
              ("Physics only (x_P)", phys_mm, vmax),
              ("Physics + residual (x_PB)", full_mm, vmax),
              ("Physics residual (actual − x_P)", phys_resid_mm, rmax)]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    for ax, (title, grid, scale) in zip(axes, panels):
        # Hillshade background (grayscale), then displacement on top (NaN = transparent).
        ax.imshow(hs, extent=hs_extent, cmap='gray', origin='upper',
                  vmin=0.0, vmax=1.0, zorder=0)
        im = ax.imshow(np.ma.masked_invalid(grid), extent=disp_extent, origin='upper',
                       cmap='RdBu_r', vmin=-scale, vmax=scale, alpha=disp_alpha, zorder=1)
        ax.set_title(title)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_xlim(disp_extent[0], disp_extent[1])
        ax.set_ylim(disp_extent[2], disp_extent[3])
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('LOS displacement (mm)')

    # Fit metrics + inferred parameters, placed BELOW the panels as a figure footer so
    # the annotation never overlaps the displacement maps. Laid out side-by-side.
    fig.suptitle(suptitle or 'PILA InSAR: physics vs. actual LOS', fontsize=14)
    box_props = dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9, ec='0.4')
    if fit_text:
        fig.text(0.01, -0.02, fit_text, ha='left', va='top', fontsize=9.0,
                 family='monospace', bbox=box_props)
    if param_text:
        fig.text(0.30, -0.02, param_text, ha='left', va='top', fontsize=9.0,
                 family='monospace', bbox=box_props)

    # bbox_inches='tight' expands the saved canvas to include the footer text.
    fig.savefig(out_base + '_maps.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_base + '_maps.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_base}_maps.png / .pdf")


def _r2_rmse(pred_mm, actual_mm):
    """R^2 and RMSE (mm) of a reconstruction against the observed field."""
    rmse = float(np.sqrt(np.mean((pred_mm - actual_mm) ** 2)))
    ss_res = float(np.sum((actual_mm - pred_mm) ** 2))
    ss_tot = float(np.sum((actual_mm - actual_mm.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return r2, rmse


def plot_scatter(actual_pts_mm, recon_pts_mm, out_base):
    """1:1 scatter of reconstructed vs. actual LOS (mm) with R^2 and RMSE."""
    rmse = float(np.sqrt(np.mean((recon_pts_mm - actual_pts_mm) ** 2)))
    ss_res = float(np.sum((actual_pts_mm - recon_pts_mm) ** 2))
    ss_tot = float(np.sum((actual_pts_mm - actual_pts_mm.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    lim = np.nanmax(np.abs(np.concatenate([actual_pts_mm, recon_pts_mm])))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual_pts_mm, recon_pts_mm, s=6, alpha=0.4, edgecolors='none')
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='1:1')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Actual LOS displacement (mm)')
    ax.set_ylabel('Reconstructed LOS displacement (mm)')
    ax.set_title(f'Reconstructed vs. actual LOS\nR$^2$={r2:.3f}, RMSE={rmse:.2f} mm, N={actual_pts_mm.size}')
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_base + '_scatter.png', dpi=150)
    fig.savefig(out_base + '_scatter.pdf')
    plt.close(fig)
    print(f"  Saved {out_base}_scatter.png / .pdf  (R2={r2:.3f}, RMSE={rmse:.2f} mm)")
    return r2, rmse


def main():
    parser = argparse.ArgumentParser(description='PILA InSAR reconstruction-vs-actual plots')
    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='path to model checkpoint (.pth)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config.json (default: <checkpoint_dir>/config.json)')
    parser.add_argument('--epoch', default=-1, type=int,
                        help='time-series epoch index to invert (default: -1 = final cumulative)')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='figure output directory (default: <experiment_dir>/figures)')
    parser.add_argument('--hs_azimuth', default=315.0, type=float, help='hillshade azimuth (deg)')
    parser.add_argument('--hs_altitude', default=45.0, type=float, help='hillshade altitude (deg)')
    args = parser.parse_args()

    t0 = time.time()
    resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(CURRENT_DIR, args.resume)
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    # --- [1/6] Load config (from the checkpoint's models/ dir by default) ---
    cfg_path = args.config or os.path.join(os.path.dirname(resume_path), 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    print(f"[1/6] Loading config from {cfg_path} ...")
    config = read_json(cfg_path)

    insar_args = config['arch']['args']['insar']
    geometry_h5 = insar_args['geometry']
    multilook = int(insar_args.get('multilook', 1))   # default 1 = no multilook
    # UQ strided-decimation OR block-bootstrap: plot the SAME subset of cells the model inverted.
    stride = int(insar_args.get('stride', 1))
    offset = int(insar_args.get('offset', 0))
    bootstrap_k = insar_args.get('bootstrap_k')
    bootstrap_block = int(insar_args.get('bootstrap_block', 3))
    bootstrap_seed = int(insar_args.get('bootstrap_seed', 0))

    # --- [2/6] Build model + restore weights/tau/r ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[2/6] Building model and loading checkpoint on {device} ...")
    model = build_model(config, resume_path, device)

    # --- [3/6] Load + multilook the MintPy LOS field (same memoized points the model used) ---
    print("[3/6] Loading multilooked InSAR field ...")
    d = load_insar_mintpy(
        insar_args['timeseries'], insar_args['geometry'], insar_args.get('mask'),
        insar_args['lat0'], insar_args['lon0'],
        multilook=multilook, coh_valid_frac=insar_args.get('coh_valid_frac', 0.5),
        bbox=insar_args.get('bbox'), verbose=False, stride=stride, offset=offset,
        bootstrap_k=bootstrap_k, bootstrap_block=bootstrap_block, bootstrap_seed=bootstrap_seed)
    print(f"  N points={d.n_points}, coarse grid={d.mask_d.shape}, epochs={len(d.dates)}")

    epoch = args.epoch
    date_str = d.dates[epoch]
    # Global standardized LOS at the chosen epoch (the model's input/target space).
    # Must match the GLOBAL scaler the dataset and the physics decoder use.
    std_series = (d.los_points_mm - d.x_mean_global) / d.x_scale_global
    target_std = std_series[epoch].astype(np.float32)                  # [N]
    print(f"  Inverting epoch index {epoch} (date {date_str})")

    # --- [4/6] Inference: reconstructed (x_PB) vs. target ---
    print("[4/6] Running inference ...")
    tfeat_dim = config['arch']['args'].get('time_feat_dim', 4)
    tfeat = torch.tensor(time_feats(date_str, time_feat_dim=tfeat_dim),
                         dtype=torch.float32).unsqueeze(0).to(device)   # [1, T]
    x_in = torch.tensor(target_std, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N]
    hz_phy, hz_aux = hard_z_flags(config)
    with torch.no_grad():
        latent_phy, latent_aux, x_PB, x_P = model(
            x_in, t=tfeat, inference=True, hard_z_phy=hz_phy, hard_z_aux=hz_aux, const=None)
    # x_P = pure physics (denoised); x_PB = physics + low-rank residual augmentation.
    full_std = x_PB[0].cpu().numpy()                                    # [N], standardized
    phys_std = x_P[0].cpu().numpy()                                     # [N], standardized

    # De-standardize to mm (global scaler). actual_mm reproduces the observed field.
    actual_mm = target_std * d.x_scale_global + d.x_mean_global
    full_mm = full_std * d.x_scale_global + d.x_mean_global             # physics + residual
    phys_mm = phys_std * d.x_scale_global + d.x_mean_global             # pure physics (denoised)

    # Inferred physics parameters (z in (0,1) -> physical units).
    param_text, src_ll, bounds_block, prior_footnote = None, None, "", ""
    if not config['arch']['phys_vae']['no_phy']:
        params = model.physics_model.rescale(latent_phy)
        param_text = format_param_text(params)
        src_ll = source_lonlat(params, insar_args['lat0'], insar_args['lon0'])
        # Report the prior box used and flag any parameter pinned to a bound (railing) --
        # the diagnostic for the Okada non-identifiability / degenerate-corner problem.
        bounds_block, prior_footnote = prior_bounds_report(model.physics_model, params)
        print("  Inferred source parameters:")
        for line in param_text.split("\n"):
            print(f"    {line}")
        if src_ll is not None:
            print(f"    source (lon, lat) = ({src_ll[0]:.4f}, {src_ll[1]:.4f})")
        if prior_footnote:
            print(f"    prior check: {prior_footnote}")

    # Fit quality: physics-only (the source) vs physics+residual (the augmented fit).
    r2_phys, rmse_phys = _r2_rmse(phys_mm, actual_mm)
    r2_full, rmse_full = _r2_rmse(full_mm, actual_mm)
    fit_text = (f"physics : R²={r2_phys:.3f}  RMSE={rmse_phys:.1f} mm\n"
                f"+resid  : R²={r2_full:.3f}  RMSE={rmse_full:.1f} mm")
    print(f"  Fit  physics-only: R²={r2_phys:.3f} RMSE={rmse_phys:.1f} mm | "
          f"physics+residual: R²={r2_full:.3f} RMSE={rmse_full:.1f} mm")

    # --- [5/6] Build grids + hillshade ---
    print("[5/6] Building grids and hillshade ...")
    actual_grid = points_to_grid(actual_mm, d.mask_d)
    phys_grid = points_to_grid(phys_mm, d.mask_d)
    full_grid = points_to_grid(full_mm, d.mask_d)
    disp_extent = list(d.extent_lonlat)   # [lon_min, lon_max, lat_min, lat_max] (honours bbox crop)
    hs, hs_extent = load_hillshade(geometry_h5, args.hs_azimuth, args.hs_altitude)

    # --- [6/6] Plot ---
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(resume_path)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"{config['name']}_{date_str}")
    print(f"[6/6] Writing figures to {out_dir} ...")
    # Surface the prior-railing check on the figure itself (appended to the fit box).
    fit_text_fig = fit_text + ("\n" + prior_footnote if prior_footnote else "")
    plot_maps(actual_grid, phys_grid, full_grid, disp_extent, hs, hs_extent, out_base,
              param_text=param_text, fit_text=fit_text_fig,
              suptitle=f"{config['name']} ({date_str}): physics vs. actual LOS")
    plot_scatter(actual_mm, phys_mm, out_base)   # scatter on the pure-physics source fit

    # Persist the inferred parameters next to the figures (so they're stored, not just shown).
    if param_text is not None:
        params_path = out_base + '_params.txt'
        # Name the prior-bounds file (okada_paras / mogi_paras / ...) for provenance.
        paras_ref = next((config['arch']['args'][k] for k in
                          ('okada_paras', 'mogi_paras', 'sun69_paras', 'rtm_paras')
                          if k in config['arch']['args']), None)
        with open(params_path, 'w') as f:
            f.write(f"# PILA inferred source parameters\n# config: {config['name']}\n")
            f.write(f"# epoch: {date_str}\n")
            if paras_ref is not None:
                f.write(f"# prior paras file: {paras_ref}\n")
            f.write(param_text + "\n")
            if src_ll is not None:
                f.write(f"source_lon = {src_ll[0]:.5f}\nsource_lat = {src_ll[1]:.5f}\n")
            # Prior box + railing diagnostic (empty for no-physics runs).
            if bounds_block:
                f.write(bounds_block + "\n")
                f.write(f"# prior check: {prior_footnote}\n")
        print(f"  Saved {params_path}")

    print(f"Done in {time.time() - t0:.1f}s. Output saved to {out_dir}")


if __name__ == '__main__':
    main()
