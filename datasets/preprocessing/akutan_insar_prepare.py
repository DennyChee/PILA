#!/usr/bin/env python
# Usage:
#   python akutan_insar_prepare.py \
#       --timeseries /path/geo_timeseries_ERA5_demErr.h5 \
#       --geometry   /path/geo_geometryRadar.h5 \
#       --mask       /path/geo_maskTempCoh.h5 \
#       --lat0 54.134 --lon0 -165.986 \
#       --n_points 300 \
#       --out_dir ../../data/processed/insar_akutan \
#       --points_json ../../configs/akutan_points_insar.json \
#       [--ref_date 20170101] [--ref_lalo 54.20 -166.10] \
#       [--episode_start_date 20180101 --episode_end_date 20180601]
#
# Description: Build PILA InSAR inputs from a MintPy LOS time series for the
#              Akutan volcano inverse problem. Downsamples the LOS field to N
#              coherent observation points, converts pixel lon/lat to a local
#              ENU grid (km) centred on the source origin, derives per-point LOS
#              unit vectors (consistent with MintPy's sign convention), and
#              writes the points config, per-epoch CSVs (train/valid/test),
#              a high-SNR episode.csv (Stage A), and per-point standardization.
# Date:        2026-06-11
#
# Scientific conventions (READ — these affect interpretation):
#   * LOS sign: positive LOS displacement = motion TOWARD the satellite (range
#     decrease), matching the MintPy convention. The per-point LOS unit vector is
#     derived by projecting unit E/N/U through MintPy's enu2los (or the documented
#     fallback), so model-side projection (e_E*uE + e_N*uN + e_U*uU) is consistent.
#   * Local ENU origin: (lat0, lon0) MUST be the SAME origin used to define the
#     Mogi/Okada source-parameter ranges (xoff/yoff, xcen/ycen). The origin maps
#     to (xE, yN) = (0, 0) km.
#   * Units: model physics output is in mm, so LOS displacement is written in mm.
#   * Reference ambiguity: InSAR LOS is relative. The MintPy time series is
#     already referenced to a pixel/date; --ref_date re-references the whole cube
#     to a chosen (ideally pre-event) epoch, and --ref_lalo subtracts a far-field
#     reference pixel per epoch. Document whichever you use.

import argparse
import json
import os
import time
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

try:
    import pyproj
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("pyproj is required (conda activate isce2): " + str(exc))


# --- LOS unit-vector helper -------------------------------------------------
def enu_to_los_unit_vectors(incidence_angle_deg, azimuth_angle_deg):
    """
    Per-point ground->satellite LOS unit-vector components (e_E, e_N, e_U).

    Derived by projecting the E, N, U unit vectors through MintPy's enu2los so
    that the resulting projection matches the sign convention of the LOS data
    MintPy produced (positive = motion toward the satellite). Falls back to the
    documented MintPy formula if MintPy is not importable.

    Parameters
    ----------
    incidence_angle_deg : np.ndarray [N]   LOS incidence angle from vertical, deg.
    azimuth_angle_deg   : np.ndarray [N]   MintPy azimuthAngle of the LOS vector, deg.

    Returns
    -------
    los_E, los_N, los_U : np.ndarray [N]   LOS unit-vector components.
    """
    try:
        from mintpy.utils import utils0 as ut0  # MintPy's own convention
        los_E = ut0.enu2los(1.0, 0.0, 0.0, inc_angle=incidence_angle_deg, az_angle=azimuth_angle_deg)
        los_N = ut0.enu2los(0.0, 1.0, 0.0, inc_angle=incidence_angle_deg, az_angle=azimuth_angle_deg)
        los_U = ut0.enu2los(0.0, 0.0, 1.0, inc_angle=incidence_angle_deg, az_angle=azimuth_angle_deg)
        print("  LOS unit vectors derived via mintpy.utils.utils0.enu2los")
    except ImportError:
        # Documented MintPy fallback (VERIFY against your MintPy version):
        #   e_U = cos(inc); e_E = -sin(inc)*cos(az); e_N = -sin(inc)*sin(az)
        warn = ("  WARNING: MintPy not importable; using documented enu2los "
                "fallback. VERIFY the azimuth convention for your data.")
        print(warn)
        inc = np.deg2rad(incidence_angle_deg)
        az = np.deg2rad(azimuth_angle_deg)
        los_U = np.cos(inc)
        los_E = -np.sin(inc) * np.cos(az)
        los_N = -np.sin(inc) * np.sin(az)

    # Normalize to unit length (guards tiny numerical drift).
    norm = np.sqrt(los_E ** 2 + los_N ** 2 + los_U ** 2)
    norm = np.where(norm == 0, 1.0, norm)
    return los_E / norm, los_N / norm, los_U / norm


# --- Geometry / coordinate helpers ------------------------------------------
def read_lonlat_grid(geom_file, n_rows, n_cols):
    """
    Per-pixel longitude/latitude arrays [n_rows, n_cols] (degrees).

    Uses the 'longitude'/'latitude' datasets if present (radar-coord geometry),
    otherwise reconstructs the grid from geocoding attributes (X_FIRST, Y_FIRST,
    X_STEP, Y_STEP) on the geometry file.
    """
    with h5py.File(geom_file, 'r') as f:
        if 'longitude' in f and 'latitude' in f:
            lon = np.array(f['longitude'], dtype=np.float64)
            lat = np.array(f['latitude'], dtype=np.float64)
            return lon, lat
        attrs = dict(f.attrs)

    required = ('X_FIRST', 'Y_FIRST', 'X_STEP', 'Y_STEP')
    if not all(k in attrs for k in required):
        raise ValueError(
            "Geometry file has neither longitude/latitude datasets nor the "
            f"geocoding attributes {required}; cannot build a lon/lat grid.")
    x_first = float(attrs['X_FIRST'])
    y_first = float(attrs['Y_FIRST'])
    x_step = float(attrs['X_STEP'])
    y_step = float(attrs['Y_STEP'])
    cols = x_first + x_step * np.arange(n_cols)
    rows = y_first + y_step * np.arange(n_rows)
    lon, lat = np.meshgrid(cols, rows)
    return lon, lat


def lonlat_to_local_enu_km(lon, lat, lat0, lon0):
    """
    Convert lon/lat (deg) to local ENU East/North in km, centred on (lat0, lon0).

    Uses an azimuthal-equidistant projection centred on the origin, so the origin
    maps to (0, 0) and distances near it are accurate. Same origin must be used
    for the source-parameter ranges.
    """
    aeqd = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    transformer = pyproj.Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    east_m, north_m = transformer.transform(lon, lat)
    return east_m / 1000.0, north_m / 1000.0  # m -> km


def date_bytes_to_iso(date_array):
    """MintPy 'date' dataset (bytes 'YYYYMMDD') -> list of 'YYYY-MM-DD' strings."""
    iso = []
    for d in date_array:
        s = d.decode() if isinstance(d, (bytes, bytearray)) else str(d)
        iso.append(f"{s[0:4]}-{s[4:6]}-{s[6:8]}")
    return iso


# --- Main pipeline ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare MintPy LOS time series for PILA InSAR inversion.")
    parser.add_argument('--timeseries', required=True, help='MintPy LOS timeseries .h5 (m)')
    parser.add_argument('--geometry', required=True, help='MintPy geometry .h5 (incidence/azimuth, lat/lon)')
    parser.add_argument('--mask', default=None, help='MintPy mask .h5 (e.g. maskTempCoh.h5); optional')
    parser.add_argument('--lat0', type=float, required=True, help='source-origin latitude (deg)')
    parser.add_argument('--lon0', type=float, required=True, help='source-origin longitude (deg)')
    parser.add_argument('--n_points', type=int, default=300, help='number of downsampled observation points N')
    parser.add_argument('--coh_threshold', type=float, default=0.0,
                        help='if no mask file, keep pixels with coherence>=this (needs coherence dataset)')
    parser.add_argument('--ref_date', default=None,
                        help='YYYYMMDD: re-reference the whole cube to this (pre-event) epoch')
    parser.add_argument('--ref_lalo', type=float, nargs=2, default=None, metavar=('LAT', 'LON'),
                        help='far-field reference pixel lat lon: subtract its LOS per epoch')
    parser.add_argument('--episode_start_date', default=None, help='YYYYMMDD inclusive (Stage A window start)')
    parser.add_argument('--episode_end_date', default=None, help='YYYYMMDD inclusive (Stage A window end)')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='valid+test fraction of epochs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', required=True, help='output dir for CSVs + standardization .npy')
    parser.add_argument('--points_json', required=True, help='output path for the points-info JSON')
    args = parser.parse_args()

    np.random.seed(args.seed)
    t_start = time.time()

    for path in (args.timeseries, args.geometry):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- [1/8] Load LOS time-series cube -----------------------------------
    print(f"[1/8] Loading time series from {args.timeseries} ...")
    with h5py.File(args.timeseries, 'r') as f:
        ts_cube_m = np.array(f['timeseries'], dtype=np.float32)  # [n_epoch, L, W], metres
        date_iso = date_bytes_to_iso(np.array(f['date']))
    n_epoch, n_rows, n_cols = ts_cube_m.shape
    print(f"  Loaded: shape={ts_cube_m.shape}, dtype={ts_cube_m.dtype}, n_epoch={n_epoch}")

    # --- [2/8] Coherent-pixel mask -----------------------------------------
    print("[2/8] Building coherent-pixel mask ...")
    finite_mask = np.isfinite(ts_cube_m).all(axis=0)  # finite at every epoch
    if args.mask is not None and os.path.exists(args.mask):
        with h5py.File(args.mask, 'r') as f:
            mask_key = 'mask' if 'mask' in f else list(f.keys())[0]
            keep_mask = np.array(f[mask_key], dtype=bool)
        print(f"  Using mask file dataset '{mask_key}'")
    elif args.coh_threshold > 0.0:
        # Fall back to a coherence dataset alongside the timeseries (if present).
        with h5py.File(args.timeseries, 'r') as f:
            if 'coherence' in f:
                coh = np.array(f['coherence'], dtype=np.float32)
                keep_mask = coh >= args.coh_threshold
            else:
                keep_mask = np.ones((n_rows, n_cols), dtype=bool)
                print("  WARNING: no mask file and no coherence dataset; keeping all finite pixels")
    else:
        keep_mask = np.ones((n_rows, n_cols), dtype=bool)
    keep_mask = keep_mask & finite_mask
    n_coherent = int(keep_mask.sum())
    print(f"  Coherent pixels: {n_coherent} / {n_rows * n_cols}")
    if n_coherent < args.n_points:
        raise ValueError(f"Only {n_coherent} coherent pixels < requested n_points={args.n_points}")

    # --- [3/8] Downsample to N points (random over coherent pixels) --------
    # Random (without replacement) sampling is spatially unbiased and returns
    # EXACTLY n_points distinct pixels. A linspace over np.where's row-major
    # ordering would oversample the top/bottom rows and np.unique could silently
    # drop points; random.choice avoids both. Reproducible via the fixed seed.
    print(f"[3/8] Downsampling to {args.n_points} points (random) ...")
    coherent_rows, coherent_cols = np.where(keep_mask)
    pick = np.sort(np.random.choice(n_coherent, args.n_points, replace=False))
    sel_rows = coherent_rows[pick]
    sel_cols = coherent_cols[pick]
    n_points = len(sel_rows)
    print(f"  Selected {n_points} points")

    # --- [4/8] Coordinates + LOS unit vectors at selected points -----------
    print("[4/8] Computing local ENU coords and LOS unit vectors ...")
    lon_grid, lat_grid = read_lonlat_grid(args.geometry, n_rows, n_cols)
    sel_lon = lon_grid[sel_rows, sel_cols]
    sel_lat = lat_grid[sel_rows, sel_cols]
    east_km, north_km = lonlat_to_local_enu_km(sel_lon, sel_lat, args.lat0, args.lon0)

    with h5py.File(args.geometry, 'r') as f:
        inc_full = np.array(f['incidenceAngle'], dtype=np.float64)
        az_full = np.array(f['azimuthAngle'], dtype=np.float64)
    sel_inc = inc_full[sel_rows, sel_cols]
    sel_az = az_full[sel_rows, sel_cols]
    los_E, los_N, los_U = enu_to_los_unit_vectors(sel_inc, sel_az)
    print(f"  Sanity: mean e_U={los_U.mean():.3f} (expect ~0.7-0.85, positive), "
          f"mean e_E={los_E.mean():.3f}, mean e_N={los_N.mean():.3f}")
    # e_U should be strongly positive (so uplift -> positive LOS = toward
    # satellite). The horizontal signs depend on track geometry: ascending vs
    # descending flip e_E (and e_N), so we do NOT assert their sign here — verify
    # them against the known look direction of your Akutan acquisitions.
    if los_U.mean() < 0.5:
        print("  WARNING: mean e_U < 0.5 — unexpected for Sentinel-1 geometry. "
              "Check the incidence/azimuth datasets and the enu2los convention.")

    # --- [5/8] Reference handling + extract per-epoch LOS at points (mm) ----
    print("[5/8] Referencing and extracting per-epoch LOS (mm) ...")
    if args.ref_date is not None:
        ref_iso = f"{args.ref_date[0:4]}-{args.ref_date[4:6]}-{args.ref_date[6:8]}"
        if ref_iso not in date_iso:
            raise ValueError(f"--ref_date {args.ref_date} not in time series dates")
        ref_idx = date_iso.index(ref_iso)
        ts_cube_m = ts_cube_m - ts_cube_m[ref_idx:ref_idx + 1, :, :]
        print(f"  Re-referenced whole cube to {ref_iso}")

    # LOS at selected points: [n_epoch, n_points], m -> mm
    los_points_mm = ts_cube_m[:, sel_rows, sel_cols] * 1000.0

    if args.ref_lalo is not None:
        ref_lat, ref_lon = args.ref_lalo
        # nearest coherent pixel to the requested far-field reference
        dist2 = (lat_grid - ref_lat) ** 2 + (lon_grid - ref_lon) ** 2
        dist2[~keep_mask] = np.inf
        r_ref, c_ref = np.unravel_index(np.argmin(dist2), dist2.shape)
        ref_series_mm = ts_cube_m[:, r_ref, c_ref] * 1000.0  # [n_epoch], mm
        los_points_mm = los_points_mm - ref_series_mm[:, None]
        # Report how far the nearest coherent pixel is from the requested point,
        # so a far-field reference landing on the wrong area is caught.
        found_lat = float(lat_grid[r_ref, c_ref])
        found_lon = float(lon_grid[r_ref, c_ref])
        offset_km = np.sqrt(((found_lat - ref_lat) * 111.0) ** 2
                            + ((found_lon - ref_lon) * 111.0 * np.cos(np.deg2rad(ref_lat))) ** 2)
        print(f"  Subtracted far-field reference pixel at row={r_ref}, col={c_ref} "
              f"(lat={found_lat:.4f}, lon={found_lon:.4f}, {offset_km:.2f} km from requested)")
        if offset_km > 5.0:
            print("  WARNING: nearest coherent reference pixel is >5 km from the "
                  "requested location — check --ref_lalo.")

    # --- [6/8] Build per-epoch table + train/valid/test split --------------
    print("[6/8] Building per-epoch table and splitting ...")
    point_ids = [f"p{idx:04d}" for idx in range(n_points)]
    los_cols = [f"los_{pid}" for pid in point_ids]
    df_all = pd.DataFrame(los_points_mm, columns=los_cols)
    df_all['date'] = date_iso

    epoch_idx = np.arange(n_epoch)
    train_idx, test_idx = train_test_split(epoch_idx, test_size=args.split_ratio, random_state=args.seed)
    train_idx, valid_idx = train_test_split(train_idx, test_size=args.split_ratio, random_state=args.seed)

    # --- [7/8] Standardize (per-point) using TRAIN epochs ------------------
    print("[7/8] Standardizing (per-point) on the train split ...")
    scaler = preprocessing.StandardScaler().fit(df_all.iloc[train_idx][los_cols].values)
    # Near-zero per-point std (e.g. a point near the reference pixel with almost
    # no LOS variation) would blow up after division and dominate the MSE.
    n_tiny_scale = int(np.sum(scaler.scale_ < 1e-3))
    if n_tiny_scale > 0:
        print(f"  WARNING: {n_tiny_scale} points have train std < 1e-3 mm; "
              f"consider dropping them or raising the coherence threshold.")

    def standardize(df):
        df = df.copy()
        df[los_cols] = scaler.transform(df[los_cols].values)
        return df

    df_train = standardize(df_all.iloc[train_idx])
    df_valid = standardize(df_all.iloc[valid_idx])
    df_test = standardize(df_all.iloc[test_idx])

    df_train.to_csv(os.path.join(args.out_dir, 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(args.out_dir, 'valid.csv'), index=False)
    df_test.to_csv(os.path.join(args.out_dir, 'test.csv'), index=False)
    np.save(os.path.join(args.out_dir, 'train_x_mean.npy'), scaler.mean_.astype(np.float32))
    np.save(os.path.join(args.out_dir, 'train_x_scale.npy'), scaler.scale_.astype(np.float32))
    print(f"  Wrote train/valid/test ({len(df_train)}/{len(df_valid)}/{len(df_test)} epochs) + standardization")

    # Stage A episode.csv: high-SNR window of the SAME (referenced) cumulative
    # series, standardized with the SAME scaler so it shares the decoder's space.
    if args.episode_start_date and args.episode_end_date:
        start_iso = f"{args.episode_start_date[0:4]}-{args.episode_start_date[4:6]}-{args.episode_start_date[6:8]}"
        end_iso = f"{args.episode_end_date[0:4]}-{args.episode_end_date[4:6]}-{args.episode_end_date[6:8]}"
        in_window = (pd.to_datetime(df_all['date']) >= pd.to_datetime(start_iso)) & \
                    (pd.to_datetime(df_all['date']) <= pd.to_datetime(end_iso))
        df_episode = standardize(df_all[in_window])
    else:
        # Default: the single max-amplitude epoch (largest mean |LOS|).
        peak_epoch = int(np.argmax(np.nanmean(np.abs(los_points_mm), axis=1)))
        df_episode = standardize(df_all.iloc[[peak_epoch]])
        print(f"  No episode window given; using peak epoch {date_iso[peak_epoch]}")
    df_episode.to_csv(os.path.join(args.out_dir, 'episode.csv'), index=False)
    print(f"  Wrote episode.csv ({len(df_episode)} epochs)")

    # --- [8/8] Write points-info JSON --------------------------------------
    print(f"[8/8] Writing points config to {args.points_json} ...")
    points_info = OrderedDict()
    for i, pid in enumerate(point_ids):
        points_info[pid] = {
            "xE": float(east_km[i]),
            "yN": float(north_km[i]),
            "los_E": float(los_E[i]),
            "los_N": float(los_N[i]),
            "los_U": float(los_U[i]),
        }
    with open(args.points_json, 'w') as f:
        json.dump(points_info, f, indent=2)

    print(f"\nDone in {time.time() - t_start:.1f}s. N points = {n_points}.")
    print(f"IMPORTANT: set arch.args.input_dim = {n_points} in the InSAR configs.")


if __name__ == '__main__':
    main()
