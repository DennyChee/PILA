#!/usr/bin/env python
# Usage:       python classical_comsol/build_comsol_iokada_shims.py [--surface cone] [--tracks asc desc]
#                  [--out-dir exported_comsol/iokada_shims]
# Description: Write the COMSOL wk6 scenes as the HDF5 'los' shims read by the existing MATLAB
#              topography-capable inversion iokada_agung_residuals.m ('Model','mogi',
#              'TopoCorrection','varying_depth'). The points, LOS and LOS unit vectors are the
#              EXACT ones in the iclassic_insar exports (exported_comsol/single/*.mat), so the
#              classical with/without-topography runs see the same data as the earlier
#              classical baseline and as PILA. Elevation comes from PILA's own loader on the
#              topography scenes (data/comsol/wk6_per_chamber_topo/, key height_m).
# Date:        2026-10-12
#
# Shim layout (as build_ic_inversion_shims.write_los_shim):
#   /Residual_m [N, 1] LOS, metres, + toward satellite   /dates_proj_yyyymmdd ['20000101']
#   /llh [N, 3] lon, lat (deg), height (m)                /los_enu [N, 3] E, N, U unit vectors
#
# Assumptions (READ):
#   * iokada_agung_residuals.m geocodes lon/lat to local metres with a flat-earth formula about
#     the Agung summit (lat0 -8.343, lon0 115.508). The COMSOL frame is local metres, so lon/lat
#     here are a CARRIER: the exact inverse of that formula, so MATLAB recovers x/y to ~1e-9 m.
#     They are not real geographic positions.
#   * Height datum = COMSOL base z = 0, so iokada's depth m(3) (below the h = 0 datum) compares
#     directly with -z_centroid_m and with PILA (h_ref_m = 0).
#   * --tracks joint (2026-10-12): asc + desc stacked from exported_comsol/joint/ (asc points
#     first, each point its own LOS vector), as the iclassic joint baseline. iokada bins pixels
#     by location, which would AVERAGE an asc and a desc pixel at the same spot into one bin
#     with a meaningless mean LOS vector. So the desc half is shifted JOINT_DESC_SHIFT_M east
#     and the joint runs use BinSize_m = JOINT_BIN_SIZE_M: every pixel then has its own bin.
#     The 9.4 cm shift changes the predicted LOS by ~gradient x 0.09 m (< 0.01 mm here), i.e.
#     negligible against the ~10-30 mm signal; it is a bookkeeping device, not physics.

import argparse
import os
import re
import time

import h5py
import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
import sys                                                                    # noqa: E402
sys.path.insert(0, PROJECT_ROOT)
from datasets.preprocessing.insar_mintpy import load_insar_mintpy             # noqa: E402

EXPORT_DIR = 'exported_comsol/single'
TOPO_SCENE_DIR = 'data/comsol/wk6_per_chamber_topo'
# iokada_agung_residuals.m geocoding constants (lines 197-198 and [3/7]); must match exactly.
IOKADA_LAT0_DEG, IOKADA_LON0_DEG = -8.343, 115.508
IOKADA_DEG2M_LAT = 110540.0
IOKADA_DEG2M_LON = 111320.0 * np.cos(np.deg2rad(IOKADA_LAT0_DEG))
EXPORT_PATTERN = re.compile(r'comsol_wk6_cid(\d+)_(flat|cone)_(asc|desc|joint)_mogi\.mat')
JOINT_EXPORT_DIR = 'exported_comsol/joint'
# Binary-exact bin (1/16 m) and a 1.5-bin desc shift: asc pixel k -> bin 1600k, desc -> 1600k+1,
# even with the ~1e-9 m lon/lat round-trip error. Must match run_comsol_iokada_group.m.
JOINT_BIN_SIZE_M = 0.0625
JOINT_DESC_SHIFT_M = 1.5 * JOINT_BIN_SIZE_M


def scene_heights_m(cid, surface, track, x_export_m, y_export_m):
    """
    Per-point elevation (m) from PILA's loader on the topography scene, in export order.

    The exports were written from the same loader call (same npz grid, multilook 1, no mask),
    so the point order is identical; the coordinates are checked to be sure.
    """
    scene_path = os.path.join(TOPO_SCENE_DIR, f'wk6_cid{cid}_{surface}.npz')
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"{scene_path} (run build_comsol_topo_npz.py)")
    scene = load_insar_mintpy(scene_path, track, None, 0.0, 0.0, multilook=1, verbose=False)
    x_scene_m = scene.xE_pts.astype(np.float64) * 1000.0
    y_scene_m = scene.yN_pts.astype(np.float64) * 1000.0
    if x_scene_m.size != x_export_m.size:
        raise ValueError(f"point count {x_scene_m.size} != export {x_export_m.size}")
    max_offset_m = max(np.abs(x_scene_m - x_export_m).max(), np.abs(y_scene_m - y_export_m).max())
    if max_offset_m > 1.0:
        raise ValueError(f"loader and export points differ by {max_offset_m:.3f} m (order mismatch?)")
    return scene.height_pts_m.astype(np.float64)


def export_dir_of(track):
    """Export folder for a track token ('joint' exports live in their own folder)."""
    return JOINT_EXPORT_DIR if track == 'joint' else EXPORT_DIR


def write_shim(out_path, x_m, y_m, height_m, los_m, los_enu):
    """Write one iokada 'los' shim (see header for the layout)."""
    # Exact inverse of iokada's flat-earth geocoding, so MATLAB gets x/y back unchanged.
    lon_deg = IOKADA_LON0_DEG + x_m / IOKADA_DEG2M_LON
    lat_deg = IOKADA_LAT0_DEG + y_m / IOKADA_DEG2M_LAT
    with h5py.File(out_path, 'w') as fh:
        fh.create_dataset('/Residual_m', data=los_m[:, None].astype(np.float64))
        fh.create_dataset('/llh', data=np.column_stack([lon_deg, lat_deg, height_m]).astype(np.float64))
        fh.create_dataset('/dates_proj_yyyymmdd', data=np.array(['20000101'], dtype='S8'))
        fh.create_dataset('/los_enu', data=los_enu.astype(np.float64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface', default='cone', choices=['cone', 'flat'])
    parser.add_argument('--tracks', nargs='+', default=['asc', 'desc'], choices=['asc', 'desc', 'joint'])
    parser.add_argument('--out-dir', default='exported_comsol/iokada_shims')
    args = parser.parse_args()
    t0 = time.time()

    # --- [1/2] Write one shim per (chamber, track) ---
    export_files = sorted(f for f in os.listdir(EXPORT_DIR) if EXPORT_PATTERN.fullmatch(f))
    if 'joint' in args.tracks:
        export_files += sorted(f for f in os.listdir(JOINT_EXPORT_DIR) if EXPORT_PATTERN.fullmatch(f))
    selected = [f for f in export_files
                if EXPORT_PATTERN.fullmatch(f)[2] == args.surface
                and EXPORT_PATTERN.fullmatch(f)[3] in args.tracks]
    if not selected:
        raise FileNotFoundError(f"no {args.surface} exports for {args.tracks} in {EXPORT_DIR}")
    print(f"[1/2] Writing {len(selected)} shims to {args.out_dir} ...")
    os.makedirs(args.out_dir, exist_ok=True)
    n_written = 0
    for export_file in selected:
        match = EXPORT_PATTERN.fullmatch(export_file)
        cid, surface, track = int(match[1]), match[2], match[3]
        out_path = os.path.join(args.out_dir, export_file.replace('.mat', '.h5'))
        if os.path.exists(out_path):
            print(f"  exists, skipped: {out_path}")     # never overwrite
            continue
        export = loadmat(os.path.join(export_dir_of(track), export_file))
        x_m = export['x_east_m'].ravel().astype(np.float64)
        y_m = export['y_north_m'].ravel().astype(np.float64)
        los_m = export['los_obs_m'].ravel().astype(np.float64)
        los_enu = np.column_stack([export[key].ravel() for key in ('losE', 'losN', 'losU')])
        if track == 'joint':
            # Stacked asc (first half) then desc (second half), as export_comsol_points.py.
            n_half = x_m.size // 2
            if x_m.size != 2 * n_half or not (los_enu[:n_half, 0] < 0).all() or not (los_enu[n_half:, 0] > 0).all():
                raise ValueError(f"{export_file}: expected asc (losE<0) then desc (losE>0) halves")
            height_m = np.concatenate([
                scene_heights_m(cid, surface, 'asc', x_m[:n_half], y_m[:n_half]),
                scene_heights_m(cid, surface, 'desc', x_m[n_half:], y_m[n_half:])])
            x_m = x_m.copy()
            x_m[n_half:] += JOINT_DESC_SHIFT_M          # keep asc/desc pixels in separate bins
        else:
            height_m = scene_heights_m(cid, surface, track, x_m, y_m)
        write_shim(out_path, x_m, y_m, height_m, los_m, los_enu)
        n_written += 1
    print(f"  wrote {n_written}")

    # --- [2/2] Sanity checks on the first shim ---
    print("[2/2] Sanity checks ...")
    check_file = selected[0]
    check_track = EXPORT_PATTERN.fullmatch(check_file)[3]
    check_path = os.path.join(args.out_dir, check_file.replace('.mat', '.h5'))
    export = loadmat(os.path.join(export_dir_of(check_track), check_file))
    x_expected_m = export['x_east_m'].ravel().astype(np.float64)
    if check_track == 'joint':
        x_expected_m[x_expected_m.size // 2:] += JOINT_DESC_SHIFT_M
    with h5py.File(check_path, 'r') as fh:
        llh = fh['/llh'][()]
        residual_m = fh['/Residual_m'][()]
        print(f"  Loaded: llh shape={llh.shape}, Residual_m shape={residual_m.shape}")
    x_back_m = (llh[:, 0] - IOKADA_LON0_DEG) * IOKADA_DEG2M_LON
    y_back_m = (llh[:, 1] - IOKADA_LAT0_DEG) * IOKADA_DEG2M_LAT
    checks = {
        'x/y round trip < 1e-6 m': max(np.abs(x_back_m - x_expected_m).max(),
                                       np.abs(y_back_m - export['y_north_m'].ravel()).max()) < 1e-6,
        'LOS identical to iclassic export': np.array_equal(residual_m[:, 0], export['los_obs_m'].ravel()),
        'height in [0, 1000] m, summit present': (llh[:, 2].min() >= 0 and abs(llh[:, 2].max() - 1000) < 1),
    }
    if check_track == 'joint':
        # Reproduce iokada's binning: every pixel must land in its own bin.
        ix = np.floor((x_back_m - x_back_m.min()) / JOINT_BIN_SIZE_M)
        iy = np.floor((y_back_m - y_back_m.min()) / JOINT_BIN_SIZE_M)
        checks['joint: one pixel per bin at JOINT_BIN_SIZE_M'] = (
            np.unique(np.column_stack([ix, iy]), axis=0).shape[0] == x_back_m.size)
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise SystemExit("sanity checks failed")
    print(f"Done in {time.time() - t0:.1f}s. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
