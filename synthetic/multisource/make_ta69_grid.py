#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.make_ta69_grid
# Description: Build an idealized 128x128 synthetic base grid that MATCHES the real Marapi
#              Sentinel-1 ASCENDING track 69 stack after x4 multilooking, so networks trained on
#              it can be applied to real TA69 windows without resampling. Writes MintPy-format
#              geo_geometryRadar.h5 (constant incidence/azimuth = TA69 medians) and
#              geo_maskTempCoh.h5 (all valid) to synthetic/marapi_ta69_grid/.
# Scientific notes:
#   * Marapi (W. Sumatra, -0.38, 100.47) -- NOT Merapi (Java).
#   * Pixel spacing = 4 x TA69 geocoded spacing (X_STEP 0.000852 deg, Y_STEP -0.000740 deg)
#     -> ~380 m E x ~328 m N, so the grid spans ~48.6 x 42.0 km.
#   * Constant LOS geometry (medians over valid TA69 pixels); real TA69 varies ~+-3 deg in
#     incidence across the frame -- a small mismatch for 40-km windows.
# Date:        2026-10-05

import os

import h5py
import numpy as np

TA69_GEOMETRY_H5 = '/eos-rs/INSAR_processing/denny/Marapi/S1_TA69/mintpy/geo/geo_geometryRadar.h5'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'marapi_ta69_grid')
GRID_PX = 128                       # pixels per side
MULTILOOK = 4                       # TA69 geocoded pixels per grid pixel
LAT0_DEG, LON0_DEG = -0.38, 100.47  # Marapi summit = grid centre (local ENU origin)


def main():
    if not os.path.exists(TA69_GEOMETRY_H5):
        raise FileNotFoundError(TA69_GEOMETRY_H5)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ['geo_geometryRadar.h5', 'geo_maskTempCoh.h5']:
        if os.path.exists(os.path.join(OUT_DIR, name)):
            raise FileExistsError(f"{os.path.join(OUT_DIR, name)} exists; not overwriting")

    print(f"[1/3] Reading real TA69 geometry: {TA69_GEOMETRY_H5}")
    with h5py.File(TA69_GEOMETRY_H5, 'r') as f:
        attrs = dict(f.attrs)
        inc_deg = f['incidenceAngle'][:]
        az_deg = f['azimuthAngle'][:]
    valid = (inc_deg != 0) & np.isfinite(inc_deg)
    inc_med_deg = float(np.median(inc_deg[valid]))
    az_med_deg = float(np.median(az_deg[valid]))
    print(f"  incidence median {inc_med_deg:.2f} deg (range {inc_deg[valid].min():.1f}-{inc_deg[valid].max():.1f}), "
          f"azimuth median {az_med_deg:.2f} deg")

    print("[2/3] Building 128x128 grid at 4x TA69 spacing, centred on Marapi...")
    x_step_deg = MULTILOOK * float(attrs['X_STEP'])
    y_step_deg = MULTILOOK * float(attrs['Y_STEP'])                 # negative (north-up)
    x_first_deg = LON0_DEG - x_step_deg * GRID_PX / 2
    y_first_deg = LAT0_DEG - y_step_deg * GRID_PX / 2
    km_per_deg = 111.32
    print(f"  pixel ~{abs(x_step_deg) * km_per_deg * np.cos(np.radians(LAT0_DEG)) * 1e3:.0f} m E x "
          f"{abs(y_step_deg) * 110.57 * 1e3:.0f} m N")
    geo_attrs = {'FILE_TYPE': 'geometry', 'WIDTH': str(GRID_PX), 'LENGTH': str(GRID_PX),
                 'X_FIRST': str(x_first_deg), 'Y_FIRST': str(y_first_deg),
                 'X_STEP': str(x_step_deg), 'Y_STEP': str(y_step_deg),
                 'X_UNIT': 'degrees', 'Y_UNIT': 'degrees', 'ORBIT_DIRECTION': 'ASCENDING',
                 'SOURCE': f'idealized from {TA69_GEOMETRY_H5} (medians, x{MULTILOOK} spacing)'}

    print(f"[3/3] Writing {OUT_DIR}/ ...")
    with h5py.File(os.path.join(OUT_DIR, 'geo_geometryRadar.h5'), 'w') as f:
        f.create_dataset('incidenceAngle', data=np.full((GRID_PX, GRID_PX), inc_med_deg, np.float32))
        f.create_dataset('azimuthAngle', data=np.full((GRID_PX, GRID_PX), az_med_deg, np.float32))
        f.attrs.update(geo_attrs)
    with h5py.File(os.path.join(OUT_DIR, 'geo_maskTempCoh.h5'), 'w') as f:
        f.create_dataset('mask', data=np.ones((GRID_PX, GRID_PX), bool))
        f.attrs.update({**geo_attrs, 'FILE_TYPE': 'mask'})
    print(f"Done. Output saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
