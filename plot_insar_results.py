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
import re
import time

import h5py
import numpy as np
import pandas as pd
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
        elif para_name == 'dV':
            disp_val, disp_min, disp_max, unit = val, rng['min'], rng['max'], 'm^3'
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


def load_chamber_section(design_csv, scene_npz, n_samples=721):
    """
    True chamber outline for a single-chamber COMSOL npz scene, for the cross-section panel.

    The COMSOL wk6 design gives each axisymmetric chamber as
        r(s) = ar * sin(s) * Q(cos s),  z(s) = -d + az * cos(s),  Q(x) = sum_k b_k x^k,
    s in [0, pi] (same formula as the catalogue's scripts/wk6_chambers.py).

    Inputs : design_csv str  metadata/wk6_design.csv (one row per cid; '#' header line)
             scene_npz  str  single-chamber npz (must hold one 'cid'); a '_cone.npz' /
                             '_flat.npz' suffix selects the surface profile to draw.
    Output : dict(cid, r_m [n], z_m [n] (chamber wall, z up, m), z_centroid_m,
                  surface_r_m, surface_z_m (None if no profile)); None if the scene has no cid.
    """
    with np.load(scene_npz) as scene:
        if 'cid' not in scene.files or scene['cid'].size != 1:
            return None
        cid = int(scene['cid'][0])
    if not os.path.exists(design_csv):
        raise FileNotFoundError(design_csv)
    design = pd.read_csv(design_csv, comment='#')
    row = design.loc[design['cid'] == cid]
    if row.empty:
        raise ValueError(f"cid {cid} not found in {design_csv}")
    row = row.iloc[0]
    # Polynomial degree follows the design file (b0..bK columns), not a hardcoded 8.
    n_coeffs = sum(1 for column in design.columns if re.fullmatch(r'b\d+', column))
    coeffs = [row[f'b{k}'] for k in range(n_coeffs)]
    s_rad = np.linspace(0.0, np.pi, n_samples)
    r_m = row['ar_m'] * np.sin(s_rad) * np.polyval(coeffs[::-1], np.cos(s_rad))
    z_m = -row['d_m'] + row['az_m'] * np.cos(s_rad)
    # Surface elevation from the raw COMSOL profile (cone runs: up to 1 km high).
    surface_r_m = surface_z_m = None
    topo_match = re.search(r'_(flat|cone)\.npz$', os.path.basename(scene_npz))
    if topo_match:
        profile_csv = os.path.join(os.path.dirname(os.path.dirname(design_csv)), 'profiles',
                                   f'wk6_cid{cid}_{topo_match.group(1)}.csv')
        if os.path.exists(profile_csv):
            profile = pd.read_csv(profile_csv)
            surface_r_m, surface_z_m = profile['r_m'].values, profile['z_m'].values
    # True cavity volume change is per surface (flat/cone COMSOL runs differ), so it lives in
    # the sibling wk6_truth.csv, not the design file. None if unavailable (table shows '-').
    dV_true_m3 = None
    truth_csv = os.path.join(os.path.dirname(design_csv), 'wk6_truth.csv')
    if topo_match and os.path.exists(truth_csv):
        truth = pd.read_csv(truth_csv)
        truth_row = truth.loc[(truth['cid'] == cid) & (truth['surface'] == topo_match.group(1))]
        if not truth_row.empty:
            dV_true_m3 = float(truth_row.iloc[0]['dV_cavity_m3'])
    return dict(cid=cid, r_m=r_m, z_m=z_m, z_centroid_m=float(row['z_centroid_m']),
                z_top_m=float(row['z_top_m']), dV_true_m3=dV_true_m3,
                elongation_A=row['A'], kind=str(row['kind']),
                surface=topo_match.group(1) if topo_match else None,
                surface_r_m=surface_r_m, surface_z_m=surface_z_m)


def format_truth_table(outline, xcen_m, ycen_m, depth_m, dV_m3):
    """
    Recovered-vs-true table for a COMSOL chamber scene (monospace text for the figure footer).

    Truth conventions (same as collect_comsol_results.py):
      xcen/ycen truth = chamber axis at (0, 0) km in the npz local frame;
      depth truth     = centroid depth -z_centroid (flat z = 0 datum), NOT the 2 km top;
      dV truth        = COMSOL cavity volume change dV_cavity (not identical to Mogi strength
                        for a finite chamber even with a perfect fit).
    Error sign: + = PILA east/north / too deep / too large.
    """
    depth_true_m = -outline['z_centroid_m']
    offset_m = float(np.hypot(xcen_m, ycen_m))
    lines = ["PILA recovered vs COMSOL truth",
             f"{'parameter':<14}{'true':>9}{'PILA':>9}{'error':>16}",
             f"{'xcen (km)':<14}{0.0:>+9.2f}{xcen_m / 1000.0:>+9.2f}{xcen_m:>+14.0f} m",
             f"{'ycen (km)':<14}{0.0:>+9.2f}{ycen_m / 1000.0:>+9.2f}{ycen_m:>+14.0f} m",
             f"{'depth (km)':<14}{depth_true_m / 1000.0:>9.2f}{depth_m / 1000.0:>9.2f}"
             f"{100.0 * (depth_m / depth_true_m - 1.0):>+14.1f} %"]
    if dV_m3 is not None and outline['dV_true_m3'] is not None and outline['dV_true_m3'] > 0:
        lines.append(f"{'dV (1e6 m³)':<14}{outline['dV_true_m3'] / 1e6:>9.2f}{dV_m3 / 1e6:>9.2f}"
                     f"{100.0 * (dV_m3 / outline['dV_true_m3'] - 1.0):>+14.1f} %")
    elif dV_m3 is not None:
        lines.append(f"{'dV (1e6 m³)':<14}{'-':>9}{dV_m3 / 1e6:>9.2f}{'-':>16}")
    lines.append(f"horizontal offset = {offset_m:.0f} m;  chamber top = {-outline['z_top_m'] / 1000.0:.2f} km")
    lines.append("depth truth = centroid; dV truth = cavity dV")
    return "\n".join(lines)


def plot_chamber_section(ax, section, xcen_m, ycen_m, depth_m, half_width_km=6.0):
    """
    Vertical E-W cross-section through the chamber axis: true outline (revolved profile,
    so both walls +/- r), true centroid, surface, and the recovered point source (xcen, -d).
    ycen is quoted in the legend (it lies off the section plane).
    Datum: chamber, centroid and Mogi depth are all referenced to the flat z = 0 datum. The
    cone surface is drawn for context only -- the COMSOL catalogue placed cone displacements
    by radius alone (surface elevation ignored), so the flat-datum comparison is consistent.
    """
    # Surface: mirrored profile elevation, or the flat free surface z = 0.
    if section['surface_r_m'] is not None:
        r_km = section['surface_r_m'] / 1000.0
        z_km = section['surface_z_m'] / 1000.0
        ax.plot(np.concatenate([-r_km[::-1], r_km]), np.concatenate([z_km[::-1], z_km]),
                color='#52514e', lw=1.5, label='Surface')
    else:
        ax.axhline(0.0, color='#52514e', lw=1.5, label='Surface')
    wall_x_km = np.concatenate([section['r_m'], -section['r_m'][::-1]]) / 1000.0
    wall_z_km = np.concatenate([section['z_m'], section['z_m'][::-1]]) / 1000.0
    ax.fill(wall_x_km, wall_z_km, color='#2a78d6', alpha=0.25, lw=0)
    ax.plot(wall_x_km, wall_z_km, color='#2a78d6', lw=2.0, label='True chamber')
    ax.plot(0.0, section['z_centroid_m'] / 1000.0, marker='+', color='#2a78d6', ms=12, mew=2.0,
            ls='none', label=f"True centroid ({-section['z_centroid_m'] / 1000.0:.2f} km)")
    ax.plot(xcen_m / 1000.0, -depth_m / 1000.0, marker='o', color='#eb6834', ms=9,
            mec='white', mew=1.2, ls='none',
            label=f"PILA Mogi ({depth_m / 1000.0:.2f} km; y={ycen_m / 1000.0:+.2f} km)")
    ax.set_xlim(-half_width_km, half_width_km)
    ax.set_ylim(min(-10.0, wall_z_km.min() - 0.5), 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('East (km)')
    ax.set_ylabel('Elevation (km)')
    ax.set_title(f"Chamber cid {section['cid']}: E-W section")
    ax.grid(color='#e5e5e2', lw=0.8)
    # Legend below the axes so it never hides the chamber (tall chambers reach ~-8.6 km).
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)


def plot_maps(actual_mm, phys_mm, full_mm, disp_extent, hs, hs_extent, out_base,
              param_text=None, fit_text=None, disp_alpha=0.8, suptitle=None, section=None):
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
    section    : optional dict(outline=load_chamber_section(...), xcen_m, ycen_m, depth_m);
                 adds a 5th panel with the true chamber cross-section + recovered source.
    """
    phys_resid_mm = phys_mm - actual_mm
    vmax = np.nanmax(np.abs(np.concatenate([actual_mm.ravel(), phys_mm.ravel(), full_mm.ravel()])))
    rmax = np.nanmax(np.abs(phys_resid_mm))
    panels = [("Actual LOS", actual_mm, vmax),
              ("Physics only (x_P)", phys_mm, vmax),
              ("Physics + residual (x_PB)", full_mm, vmax),
              ("Physics residual (actual − x_P)", phys_resid_mm, rmax)]

    n_panels = 5 if section is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 6), constrained_layout=True)
    for ax, (title, grid, scale) in zip(axes, panels):
        # Hillshade background (grayscale), then displacement on top (NaN = transparent).
        # hs is None for local-frame npz scenes (no DEM): plot the displacement alone, in km.
        if hs is not None:
            ax.imshow(hs, extent=hs_extent, cmap='gray', origin='upper',
                      vmin=0.0, vmax=1.0, zorder=0)
        im = ax.imshow(np.ma.masked_invalid(grid), extent=disp_extent, origin='upper',
                       cmap='RdBu_r', vmin=-scale, vmax=scale,
                       alpha=disp_alpha if hs is not None else 1.0, zorder=1)
        ax.set_title(title)
        ax.set_xlabel('Longitude (deg)' if hs is not None else 'East (km)')
        ax.set_ylabel('Latitude (deg)' if hs is not None else 'North (km)')
        ax.set_xlim(disp_extent[0], disp_extent[1])
        ax.set_ylim(disp_extent[2], disp_extent[3])
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('LOS displacement (mm)')
    if section is not None:
        plot_chamber_section(axes[4], section['outline'], section['xcen_m'],
                             section['ycen_m'], section['depth_m'])

    # Fit metrics + inferred parameters, placed BELOW the panels as a figure footer so
    # the annotation never overlaps the displacement maps. Laid out side-by-side.
    fig.suptitle(suptitle or 'PILA InSAR: physics vs. actual LOS', fontsize=14)
    box_props = dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9, ec='0.4')
    if fit_text:
        fig.text(0.01, -0.02, fit_text, ha='left', va='top', fontsize=9.0,
                 family='monospace', bbox=box_props)
    if param_text:
        fig.text(0.30, -0.02, param_text, ha='left', va='top', fontsize=11.0,
                 family='monospace', bbox=box_props)

    # bbox_inches='tight' expands the saved canvas to include the footer text.
    # PNG only (user preference: no PDF export).
    fig.savefig(out_base + '_maps.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_base}_maps.png")


# Presentation colours: one identity per quantity across every panel (reference categorical
# palette slots 1-2, a validated pair). Text stays in neutral ink, never the series colour.
_TRUE_COLOR = '#2a78d6'
_PILA_COLOR = '#eb6834'
_INK_PRIMARY = '#1f1f1d'
_INK_MUTED = '#6b6a66'
_GRID_COLOR = '#e5e5e2'


def _style_axes(ax):
    """Recessive axes for slides: no top/right spines, light grid behind the data."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_INK_MUTED)
    ax.tick_params(colors=_INK_PRIMARY)
    ax.grid(color=_GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)


def _paired_bars(ax, true_value, pila_value, title, ylabel, value_fmt):
    """
    Two bars (COMSOL truth, PILA) for one parameter, value labels on top and the signed
    percent error (+ = PILA too large) in the title. One quantity per axis -- no dual axes.
    """
    x_positions = [0, 1]
    bars = ax.bar(x_positions, [true_value, pila_value], width=0.6,
                  color=[_TRUE_COLOR, _PILA_COLOR], edgecolor='white', linewidth=2)
    for bar, value in zip(bars, [true_value, pila_value]):
        # Label beyond the bar end: above for positive, below for negative (e.g. deflation).
        ax.text(bar.get_x() + bar.get_width() / 2, value, value_fmt.format(value),
                ha='center', va='bottom' if value >= 0 else 'top', fontsize=14,
                color=_INK_PRIMARY)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['True', 'PILA'])
    ax.set_xlim(-0.6, 1.6)
    # Limits always include zero (bars start there) plus headroom for the value labels.
    low, high = min(0.0, true_value, pila_value), max(0.0, true_value, pila_value)
    span = (high - low) or 1.0
    ax.set_ylim(low - (0.15 * span if low < 0 else 0.0), high + 0.2 * span)
    if low < 0:
        ax.axhline(0.0, color=_INK_MUTED, lw=1)
    if true_value != 0:
        error_pct = 100.0 * (pila_value / true_value - 1.0)
        ax.set_title(f"{title}\nerror {error_pct:+.1f} %")
    else:
        ax.set_title(f"{title}\nerror n/a (zero truth)")
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    ax.grid(axis='x', visible=False)


def plot_presentation_maps(actual_mm, phys_mm, disp_extent, section, dV_pila_m3, out_base,
                           prior_footnote=None):
    """
    Slide-ready (16:9) summary of one COMSOL chamber inversion: PILA Mogi vs the known truth.

    Top row   : Actual LOS | PILA Mogi LOS (physics only, x_P) | Residual (actual - x_P),
                true axis (+) and PILA source (o) marked on the displacement maps.
    Bottom row: E-W cross-section (true chamber + centroid vs PILA source) | plan-view zoom
                of the horizontal offset | depth bars | dV bars.
    The low-rank-augmented field x_PB is left out on purpose: it fits ~everything and does
    not help the truth comparison (it is still in the standard *_maps figure for non-COMSOL
    scenes).

    Inputs : actual_mm, phys_mm [ny, nx] LOS grids (mm); disp_extent (km, local frame);
             section dict(outline=load_chamber_section(...), xcen_m, ycen_m, depth_m) in m;
             dV_pila_m3 recovered volume change (m^3) or None; prior_footnote optional str.
    Truth  : axis (0, 0); depth = centroid (flat z = 0 datum); dV = COMSOL cavity dV.
    Output : <out_base>_maps.png (PNG only, user preference).
    """
    outline = section['outline']
    xcen_km, ycen_km = section['xcen_m'] / 1000.0, section['ycen_m'] / 1000.0
    depth_pila_km = section['depth_m'] / 1000.0
    depth_true_km = -outline['z_centroid_m'] / 1000.0
    offset_m = float(np.hypot(section['xcen_m'], section['ycen_m']))
    # Metrics on valid pixels only: gridded fields are NaN outside the mask.
    valid = np.isfinite(actual_mm) & np.isfinite(phys_mm)
    r2, rmse_mm = _r2_rmse(phys_mm[valid], actual_mm[valid])

    # imshow treats extent as pixel EDGES, but disp_extent spans pixel CENTRES: pad by half a
    # pixel so the true/PILA markers sit on the right pixels (50 m on the 100 m COMSOL grid).
    n_rows, n_cols = actual_mm.shape
    half_dx_km = 0.5 * (disp_extent[1] - disp_extent[0]) / max(n_cols - 1, 1)
    half_dy_km = 0.5 * (disp_extent[3] - disp_extent[2]) / max(n_rows - 1, 1)
    image_extent = (disp_extent[0] - half_dx_km, disp_extent[1] + half_dx_km,
                    disp_extent[2] - half_dy_km, disp_extent[3] + half_dy_km)

    # Slide-sized fonts: readable when the 16x9 in figure is shrunk onto a projected slide.
    with plt.rc_context({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
                         'axes.titlecolor': _INK_PRIMARY, 'axes.labelcolor': _INK_PRIMARY}):
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 0.75, 0.75])
        top = grid[0, :].subgridspec(1, 3)
        map_axes = [fig.add_subplot(top[0, k]) for k in range(3)]

        # --- Top row: displacement maps (shared symmetric scale for actual and PILA) ---
        vmax_mm = max(float(np.nanmax(np.abs(np.concatenate([actual_mm.ravel(), phys_mm.ravel()])))), 1e-9)
        resid_mm = actual_mm - phys_mm                       # what the point source cannot explain
        rmax_mm = max(float(np.nanmax(np.abs(resid_mm))), 1e-9)
        map_panels = [("Actual LOS (COMSOL)", actual_mm, vmax_mm),
                      ("PILA Mogi LOS", phys_mm, vmax_mm),
                      (f"Residual (actual − PILA)\nR² = {r2:.3f}, RMSE = {rmse_mm:.1f} mm", resid_mm, rmax_mm)]
        for ax, (title, field_mm, scale_mm) in zip(map_axes, map_panels):
            image = ax.imshow(np.ma.masked_invalid(field_mm), extent=image_extent, origin='upper',
                              cmap='RdBu_r', vmin=-scale_mm, vmax=scale_mm)
            ax.set_title(title)
            ax.set_xlabel('East (km)')
            ax.set_aspect('equal')
            colorbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
            colorbar.set_label('LOS (mm)')
        map_axes[0].set_ylabel('North (km)')
        for ax in map_axes[:2]:
            ax.plot(0.0, 0.0, marker='+', color=_TRUE_COLOR, ms=16, mew=3, ls='none',
                    label='True axis')
            ax.plot(xcen_km, ycen_km, marker='o', color=_PILA_COLOR, ms=10, mec='white', mew=1.5,
                    ls='none', label='PILA source')
        for ax in map_axes:
            # Pin to the image: a source outside the scene must not rescale the map.
            ax.set_xlim(image_extent[0], image_extent[1])
            ax.set_ylim(image_extent[2], image_extent[3])
        map_axes[0].legend(loc='lower left', framealpha=0.9)

        # --- Bottom row (a): E-W cross-section, true chamber vs PILA point source ---
        ax_section = fig.add_subplot(grid[1, 0])
        if outline['surface_r_m'] is not None:
            surface_r_km = outline['surface_r_m'] / 1000.0
            surface_z_km = outline['surface_z_m'] / 1000.0
            ax_section.plot(np.concatenate([-surface_r_km[::-1], surface_r_km]),
                            np.concatenate([surface_z_km[::-1], surface_z_km]),
                            color=_INK_MUTED, lw=2, label='Surface')
        else:
            ax_section.axhline(0.0, color=_INK_MUTED, lw=2, label='Surface')
        wall_x_km = np.concatenate([outline['r_m'], -outline['r_m'][::-1]]) / 1000.0
        wall_z_km = np.concatenate([outline['z_m'], outline['z_m'][::-1]]) / 1000.0
        ax_section.fill(wall_x_km, wall_z_km, color=_TRUE_COLOR, alpha=0.2, lw=0)
        ax_section.plot(wall_x_km, wall_z_km, color=_TRUE_COLOR, lw=2, label='True chamber')
        ax_section.plot(0.0, -depth_true_km, marker='+', color=_TRUE_COLOR, ms=16, mew=3,
                        ls='none', label=f'True centroid ({depth_true_km:.2f} km)')
        ax_section.plot(xcen_km, -depth_pila_km, marker='o', color=_PILA_COLOR, ms=11,
                        mec='white', mew=1.5, ls='none', label=f'PILA source ({depth_pila_km:.2f} km)')
        ax_section.set_xlim(-6.0, 6.0)
        ax_section.set_ylim(min(-10.0, wall_z_km.min() - 0.5), 4.0)
        ax_section.set_aspect('equal')
        ax_section.set_xlabel('East (km)')
        ax_section.set_ylabel('Elevation (km)')
        ax_section.set_title('E-W cross-section')
        _style_axes(ax_section)
        # Legend in the empty band above the surface (ylim top raised for it): tall chambers
        # reach ~-8.6 km, so any placement inside the crust can hide the chamber or the source.
        section_handles, section_labels = ax_section.get_legend_handles_labels()
        keep = [k for k, label in enumerate(section_labels) if label != 'Surface']
        ax_section.legend([section_handles[k] for k in keep], [section_labels[k] for k in keep],
                          loc='upper right', framealpha=0.9, fontsize=10, handlelength=1.2)

        # --- Bottom row (b): plan-view zoom of the horizontal offset ---
        ax_plan = fig.add_subplot(grid[1, 1])
        half_width_km = max(1.0, 1.4 * offset_m / 1000.0)    # always frame both markers
        ax_plan.plot(0.0, 0.0, marker='+', color=_TRUE_COLOR, ms=18, mew=3, ls='none')
        ax_plan.plot([0.0, xcen_km], [0.0, ycen_km], color=_INK_MUTED, lw=1.5, ls='--')
        ax_plan.plot(xcen_km, ycen_km, marker='o', color=_PILA_COLOR, ms=12, mec='white', mew=1.5,
                     ls='none')
        ax_plan.set_xlim(-half_width_km, half_width_km)
        ax_plan.set_ylim(-half_width_km, half_width_km)
        ax_plan.set_aspect('equal')
        ax_plan.set_xlabel('East (km)')
        ax_plan.set_ylabel('North (km)')
        ax_plan.set_title(f'Horizontal position\noffset {offset_m:.0f} m')
        _style_axes(ax_plan)

        # --- Bottom row (c, d): depth and dV, truth vs PILA ---
        _paired_bars(fig.add_subplot(grid[1, 2]), depth_true_km, depth_pila_km,
                     'Depth (centroid)', 'Depth below datum (km)', '{:.2f}')
        ax_dV = fig.add_subplot(grid[1, 3])
        if dV_pila_m3 is not None and outline['dV_true_m3'] is not None and outline['dV_true_m3'] > 0:
            _paired_bars(ax_dV, outline['dV_true_m3'] / 1e6, dV_pila_m3 / 1e6,
                         'Volume change', 'dV (10⁶ m³)', '{:.2f}')
        else:
            ax_dV.set_axis_off()
            ax_dV.text(0.5, 0.5, 'dV truth unavailable', ha='center', va='center',
                       transform=ax_dV.transAxes, color=_INK_MUTED)

        # Title names the case; the footnote carries conventions + the prior-railing check.
        surface_label = {'flat': 'flat surface', 'cone': '1 km cone'}.get(outline['surface'],
                                                                         'surface n/a')
        fig.suptitle(f"Chamber cid {outline['cid']} (A = {outline['elongation_A']}, "
                     f"{surface_label}): PILA Mogi vs COMSOL truth",
                     fontsize=20, color=_INK_PRIMARY)
        footnote = ("Truth: chamber axis at (0, 0); depth = volume centroid from the flat z = 0 "
                    f"datum (chamber top {-outline['z_top_m'] / 1000.0:.1f} km); dV = COMSOL cavity "
                    "volume change. Descending LOS.")
        if prior_footnote:
            footnote += f"  Prior check: {prior_footnote}."
        fig.text(0.5, -0.01, footnote, ha='center', va='top', fontsize=11, color=_INK_MUTED)

        fig.savefig(out_base + '_maps.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f"  Saved {out_base}_maps.png (presentation layout)")


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
    plt.close(fig)
    print(f"  Saved {out_base}_scatter.png  (R2={r2:.3f}, RMSE={rmse:.2f} mm)")
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
    parser.add_argument('--chamber-design', default=None, type=str,
                        help='COMSOL design CSV (metadata/wk6_design.csv); for single-chamber npz '
                             'scenes adds a cross-section panel with the true chamber outline')
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
    geometry_h5 = insar_args['geometry']   # for .npz scenes this is the track token ('asc'/'desc')
    is_npz_scene = str(insar_args['timeseries']).lower().endswith('.npz')
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
        bootstrap_k=bootstrap_k, bootstrap_block=bootstrap_block, bootstrap_seed=bootstrap_seed,
        ref_date=insar_args.get('ref_date'))
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
        # Local-frame npz scenes have no geographic peg: xcen/ycen (km) already locate the source.
        src_ll = (None if is_npz_scene else
                  source_lonlat(params, insar_args['lat0'], insar_args['lon0']))
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
    if is_npz_scene:
        # npz scenes: extent is local km and there is no DEM to hillshade.
        hs, hs_extent = None, None
    else:
        hs, hs_extent = load_hillshade(geometry_h5, args.hs_azimuth, args.hs_altitude)

    # --- [6/6] Plot ---
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(resume_path)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"{config['name']}_{date_str}")
    print(f"[6/6] Writing figures to {out_dir} ...")
    # Surface the prior-railing check on the figure itself (appended to the fit box).
    fit_text_fig = fit_text + ("\n" + prior_footnote if prior_footnote else "")
    # Optional true-chamber cross-section (COMSOL single-chamber npz scenes only).
    section = None
    if (args.chamber_design and is_npz_scene and not config['arch']['phys_vae']['no_phy']
            and not all(key in params for key in ('xcen', 'ycen', 'd'))):
        print("  NOTE: --chamber-design needs a Mogi-type source (xcen/ycen/d); skipping section panel")
    elif args.chamber_design and is_npz_scene and not config['arch']['phys_vae']['no_phy']:
        outline = load_chamber_section(args.chamber_design, insar_args['timeseries'])
        if outline is None:
            print("  NOTE: scene npz has no single 'cid'; skipping the chamber section panel")
        else:
            section = dict(outline=outline, xcen_m=_param_scalar(params['xcen']),
                           ycen_m=_param_scalar(params['ycen']), depth_m=_param_scalar(params['d']))
    # With a known chamber, the recovered-vs-true numbers also go into *_params.txt.
    truth_text = None
    if section is not None:
        truth_text = format_truth_table(
            section['outline'], section['xcen_m'], section['ycen_m'], section['depth_m'],
            _param_scalar(params['dV']) if 'dV' in params else None)
        print("  " + truth_text.replace("\n", "\n  "))
    if section is not None:
        # Known COMSOL chamber: slide-ready truth-vs-PILA layout replaces the 4-map figure.
        plot_presentation_maps(actual_grid, phys_grid, disp_extent, section,
                               _param_scalar(params['dV']) if 'dV' in params else None,
                               out_base, prior_footnote=prior_footnote)
    else:
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
            if truth_text:
                f.write("# " + truth_text.replace("\n", "\n# ") + "\n")
        print(f"  Saved {params_path}")

    print(f"Done in {time.time() - t0:.1f}s. Output saved to {out_dir}")


if __name__ == '__main__':
    main()
