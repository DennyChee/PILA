#!/usr/bin/env python
# Usage:       python -m synthetic.make_marapi_grid
# Description: Build the idealized "Marapi" synthetic base scene (Mode B) so the
#              precomputed 128x128 / 76-epoch Marapi noise cube can be added with
#              NO regridding. Writes three MintPy-format HDF5 files the production
#              loader + generator read:
#                geo_geometryRadar.h5  : constant incidence/azimuth (descending S1),
#                                        geocoding attrs spanning 40 km.
#                geo_maskTempCoh.h5    : all-coherent mask.
#                base_timeseries.h5    : zero cube + 76 synthesized dates + attrs
#                                        (only its dates/attrs are reused; the signal
#                                         is written by generate_timeseries).
# Conventions: incidence/azimuth taken from a real descending S1 scene (Sierra Negra
#              TD128 medians: inc 33.14 deg, az -101.98 deg) so the LOS vector is
#              physically representative (losU ~ cos(33) ~ 0.837). Grid 128x128 over
#              40 km (~312.5 m px) centred on Marapi, W. Sumatra (-0.38, 100.47).
# Date:        2026-06-22

import os
import numpy as np
import h5py

OUT_DIR = os.path.join(os.path.dirname(__file__), 'marapi_grid')

GRID = 128                 # pixels per side (matches the noise cube)
EXTENT_M = 40000.0         # 40 km span
N_EPOCH = 76               # matches the noise cube
INC_DEG = 33.14            # descending S1 incidence (Sierra Negra median)
AZ_DEG = -101.98           # descending S1 azimuthAngle (MintPy convention)
LAT0, LON0 = -0.38, 100.47  # Marapi, W. Sumatra (local-ENU origin = grid centre)


def _geocoding_attrs():
    """Geocoding affine for a GRID x GRID box of EXTENT_M centred on (LAT0, LON0)."""
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * np.cos(np.deg2rad(LAT0)))
    x_step = (EXTENT_M / GRID) * deg_per_m_lon       # +east
    y_step = -(EXTENT_M / GRID) * deg_per_m_lat      # -north (rows go south)
    x_first = LON0 - x_step * (GRID / 2.0)
    y_first = LAT0 - y_step * (GRID / 2.0)
    return {
        'WIDTH': GRID, 'LENGTH': GRID,
        'X_FIRST': x_first, 'Y_FIRST': y_first,
        'X_STEP': x_step, 'Y_STEP': y_step,
        'X_UNIT': 'degrees', 'Y_UNIT': 'degrees',
        'WAVELENGTH': 0.05546576,
    }


def build(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    attrs = _geocoding_attrs()

    # --- geometry ---
    geom_h5 = os.path.join(out_dir, 'geo_geometryRadar.h5')
    inc = np.full((GRID, GRID), INC_DEG, dtype=np.float32)
    az = np.full((GRID, GRID), AZ_DEG, dtype=np.float32)
    with h5py.File(geom_h5, 'w') as f:
        f.create_dataset('incidenceAngle', data=inc)
        f.create_dataset('azimuthAngle', data=az)
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs['FILE_TYPE'] = 'geometry'
        f.attrs['UNIT'] = '1'

    # --- mask (all coherent) ---
    mask_h5 = os.path.join(out_dir, 'geo_maskTempCoh.h5')
    with h5py.File(mask_h5, 'w') as f:
        f.create_dataset('mask', data=np.ones((GRID, GRID), dtype=bool))
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs['FILE_TYPE'] = 'mask'

    # --- base timeseries (dates + attrs only; signal written later) ---
    base_h5 = os.path.join(out_dir, 'base_timeseries.h5')
    start = np.datetime64('2020-01-01')
    dates = [(start + np.timedelta64(12 * i, 'D')) for i in range(N_EPOCH)]  # 12-day S1
    date_bytes = np.array([str(d).replace('-', '').encode() for d in dates], dtype='|S8')
    with h5py.File(base_h5, 'w') as f:
        f.create_dataset('timeseries', data=np.zeros((N_EPOCH, GRID, GRID), dtype=np.float32))
        f.create_dataset('date', data=date_bytes)
        f.create_dataset('bperp', data=np.zeros(N_EPOCH, dtype=np.float32))
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs['FILE_TYPE'] = 'timeseries'
        f.attrs['UNIT'] = 'm'
        f.attrs['REF_DATE'] = date_bytes[0]

    print(f"Wrote Marapi idealized grid to {out_dir}")
    print(f"  geometry : {geom_h5}  ({GRID}x{GRID}, inc={INC_DEG} az={AZ_DEG})")
    print(f"  mask     : {mask_h5}  (all coherent)")
    print(f"  base ts  : {base_h5}  ({N_EPOCH} epochs, 12-day cadence)")
    print(f"  origin   : lat0={LAT0} lon0={LON0}; losU~{np.cos(np.deg2rad(INC_DEG)):.3f}")
    return geom_h5, mask_h5, base_h5


if __name__ == '__main__':
    build()
