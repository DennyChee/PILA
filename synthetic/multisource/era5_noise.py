#!/usr/bin/env python
# Usage:       from synthetic.multisource.era5_noise import CALIBRATION_TRACKS, Era5Track
#              track = Era5Track(CALIBRATION_TRACKS[0]); win = track.volcano_window(...)
#              noise_mm = track.noise_stack_mm(win, ref_day_idx, acq_day_idx, inc_target_deg)
# Description: Real ERA5 tropospheric-delay noise for synthetic PILA albums, taken from the
#              inputs/ERA5.h5 files of the user's OTHER MintPy runs (read-only). One album's
#              ERA5 noise = delay(day_t) - delay(ref_day) over a 48.6 x 42.0 km window cropped at
#              TRUE ground scale, area-resampled to the 128 x 128 TA69 grid.
# Scientific notes:
#   * MintPy applies the correction as  ts_corrected = ts - ERA5, so ERA5.h5 is already in the
#     displacement sign convention (LOS, + toward satellite, metres). Noise = (E_t - E_ref) * 1e3
#     mm, no extra sign flip.
#   * Slant delay scales ~1/cos(incidence); the track's delay is converted to TA69-equivalent
#     slant delay by cos(inc_track) / cos(inc_target) (centre incidences, a ~+-3 deg approximation).
#   * Radar-coordinate windows: range ~ east-west, azimuth ~ north-south (heading rotation
#     ~10-15 deg and descending image flips are ignored; noise statistics are ~isotropic).
#   * Ground spacing: range = RANGE_PIXEL_SIZE / sin(inc_centre), azimuth = AZIMUTH_PIXEL_SIZE
#     (MintPy attributes already include the multilook factor).
#   * Zero = no data in ERA5.h5 (outside swath / masked); a window is valid only if >= 99 % of
#     its pixels are non-zero and finite on the probe dates.
# Date:        2026-10-06

import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

DENNY_ROOT = '/eos-rs/INSAR_processing/denny'

# Target window = TA69 synthetic grid extent (synthetic/multisource/make_ta69_grid.py):
# 128 px x ~380 m east, 128 px x ~328 m north.
WINDOW_EAST_M = 48.6e3
WINDOW_NORTH_M = 42.0e3
GRID_SHAPE = (128, 128)
MIN_VALID_FRAC = 0.99

# --- Track lists --------------------------------------------------------------------------
# Calibration pool: other volcanoes, tracks with >= 31 ERA5 dates (1 ref + 30 acquisitions).
# Volcano summit lat/lon (deg) used to centre one window per track on the edifice, so the
# calibration includes topography-correlated stratified delay like the Marapi evaluation set.
CALIBRATION_TRACKS = [
    dict(name='Agung_TA156',        rel='Agung/S1_TA156/mintpy',           lat=-8.343, lon=115.508),
    dict(name='Agung_TD32',         rel='Agung/S1_TD32/mintpy',            lat=-8.343, lon=115.508),
    dict(name='Awu_TD163',          rel='Awu/S1_TD163/mintpy',             lat=3.689,  lon=125.447),
    dict(name='Etna_TA44',          rel='Etna/S1_TA44/mintpy',             lat=37.751, lon=14.993),
    dict(name='Etna_TD124',         rel='Etna/S1_TD124/mintpy',            lat=37.751, lon=14.993),
    dict(name='Kirishima_TD163',    rel='Kirishima/S1_TD163/mintpy',       lat=31.909, lon=130.886),
    dict(name='LaPalma_TA60',       rel='LaPalma/S1_TA60_archive/mintpy',  lat=28.613, lon=-17.866),
    dict(name='LaPalma_TD169',      rel='LaPalma/S1_TD169/mintpy',         lat=28.613, lon=-17.866),
    dict(name='Lewotobi_TA112',     rel='Lewatobi/S1_TA112/mintpy',        lat=-8.542, lon=122.775),
    dict(name='Lewotobi_TA39',      rel='Lewatobi/S1_TA39/mintpy',         lat=-8.542, lon=122.775),
    dict(name='Lewotobi_TD163',     rel='Lewatobi/S1_TD163/mintpy',        lat=-8.542, lon=122.775),
    dict(name='Mayon_TA69',         rel='Mayon/S1_TA69/mintpy',            lat=13.257, lon=123.685),
    dict(name='Nyiragongo_TA174',   rel='Nyiragongo/S1_TA174/mintpy',      lat=-1.520, lon=29.250),
]
# Evaluation pool: Marapi's own track (no shared days/places with the calibration pool).
EVALUATION_TRACKS = [
    dict(name='Marapi_TA69',        rel='Marapi/S1_TA69/mintpy',           lat=-0.381, lon=100.473),
    dict(name='NoiseArea_TA69',     rel='Noise/S1_TA69/mintpy',            lat=None,   lon=None),
]


class Era5Track:
    """Read-only access to one track's inputs/ERA5.h5 plus window geometry."""

    def __init__(self, spec, verbose=True):
        self.name = spec['name']
        self.mintpy_dir = os.path.join(DENNY_ROOT, spec['rel'])
        self.era5_h5 = os.path.join(self.mintpy_dir, 'inputs', 'ERA5.h5')
        self.geometry_h5 = os.path.join(self.mintpy_dir, 'inputs', 'geometryRadar.h5')
        self.volcano_lat_deg, self.volcano_lon_deg = spec.get('lat'), spec.get('lon')
        for path in (self.era5_h5, self.geometry_h5):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        with h5py.File(self.era5_h5, 'r') as f:
            attrs = dict(f.attrs)
            self.n_dates, self.n_rows, self.n_cols = f['timeseries'].shape
            self.dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f['date'][:]]
            # Probe dates (first, middle, last) define the valid-data footprint.
            probe_idx = sorted({0, self.n_dates // 2, self.n_dates - 1})
            probe = f['timeseries'][probe_idx]
        self.inc_centre_deg = float(attrs['CENTER_INCIDENCE_ANGLE'])
        self.orbit = str(attrs.get('ORBIT_DIRECTION', '?'))
        # Ground pixel spacing (m); MintPy attrs already include multilooking.
        self.range_ground_m = float(attrs['RANGE_PIXEL_SIZE']) / np.sin(np.radians(self.inc_centre_deg))
        self.azimuth_m = float(attrs['AZIMUTH_PIXEL_SIZE'])
        # Window size in this track's pixels (range = columns ~ east, azimuth = rows ~ north).
        self.win_cols = int(round(WINDOW_EAST_M / self.range_ground_m))
        self.win_rows = int(round(WINDOW_NORTH_M / self.azimuth_m))
        self.valid = np.all(np.isfinite(probe) & (probe != 0), axis=0)          # [rows, cols] bool
        # Zero is treated as no-data, which is only safe for ABSOLUTE delays (|delay| ~ 2-4 m).
        # A ref-date/ref-pixel-referenced ERA5.h5 would have legitimate zeros -> refuse it.
        median_abs_m = float(np.median(np.abs(probe[:, self.valid])))
        if median_abs_m < 1.0 or attrs.get('REF_DATE') is not None:
            raise ValueError(f"{self.era5_h5}: ERA5 looks referenced (median |delay| {median_abs_m:.2f} m, "
                             f"REF_DATE {attrs.get('REF_DATE')}); zero-as-nodata would be wrong")
        # Integral image -> valid fraction of any window in O(1).
        self._valid_cumsum = np.pad(self.valid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        if verbose:
            print(f"  {self.name:18s} {self.orbit[:4]} dates={self.n_dates:3d} frame={self.n_rows}x{self.n_cols} px "
                  f"({self.range_ground_m:.0f} x {self.azimuth_m:.0f} m) window={self.win_rows}x{self.win_cols} px "
                  f"valid={self.valid.mean():.2f} inc={self.inc_centre_deg:.1f} deg")
        if self.win_rows > self.n_rows or self.win_cols > self.n_cols:
            raise ValueError(f"{self.name}: frame smaller than the 48.6 x 42 km window")

    def window_valid_frac(self, row0, col0):
        """Fraction of valid pixels in the window with top-left corner (row0, col0)."""
        c = self._valid_cumsum
        r1, c1 = row0 + self.win_rows, col0 + self.win_cols
        total = c[r1, c1] - c[row0, c1] - c[r1, col0] + c[row0, col0]
        return total / (self.win_rows * self.win_cols)

    def valid_windows(self, step_px=None):
        """All top-left corners (row0, col0) on a coarse lattice with >= 99 % valid pixels."""
        step_px = step_px or max(self.win_rows // 8, 1)
        corners = []
        for row0 in range(0, self.n_rows - self.win_rows + 1, step_px):
            for col0 in range(0, self.n_cols - self.win_cols + 1, step_px):
                if self.window_valid_frac(row0, col0) >= MIN_VALID_FRAC:
                    corners.append((row0, col0))
        return corners

    def volcano_pixel(self):
        """(row, col) of the radar pixel nearest the volcano summit, or None if not set."""
        if self.volcano_lat_deg is None:
            return None
        with h5py.File(self.geometry_h5, 'r') as f:
            lat_deg = f['latitude'][:]
            lon_deg = f['longitude'][:]
        # Small-angle distance is fine for picking the nearest pixel.
        dist2 = (lat_deg - self.volcano_lat_deg) ** 2 + \
                ((lon_deg - self.volcano_lon_deg) * np.cos(np.radians(self.volcano_lat_deg))) ** 2
        dist2[~np.isfinite(dist2)] = np.inf
        row, col = np.unravel_index(np.argmin(dist2), dist2.shape)
        if np.sqrt(dist2[row, col]) * 111.0 > 2.0:      # > ~2 km away -> volcano not in frame
            return None
        return int(row), int(col)

    def volcano_window(self):
        """
        Valid window whose centre is closest to the volcano summit.

        Returns (row0, col0, offset_km) or None if the volcano is not in the frame; offset_km is
        the distance between the window centre and the summit (0 if the centred window is valid).
        """
        pix = self.volcano_pixel()
        if pix is None:
            return None
        corners = self.valid_windows(step_px=max(self.win_rows // 16, 1))
        if not corners:
            return None
        centres = np.array([(r + self.win_rows / 2, c + self.win_cols / 2) for r, c in corners])
        off_km = np.hypot((centres[:, 0] - pix[0]) * self.azimuth_m,
                          (centres[:, 1] - pix[1]) * self.range_ground_m) / 1e3
        best = int(np.argmin(off_km))
        return corners[best][0], corners[best][1], float(off_km[best])

    def read_window_m(self, day_idx, row0, col0):
        """ERA5 delay (m, displacement sign) for the given day indices over one window: [n, rows, cols]."""
        order = np.argsort(day_idx)                       # h5py fancy indexing needs increasing order
        sorted_idx = np.asarray(day_idx)[order]
        with h5py.File(self.era5_h5, 'r') as f:
            block = f['timeseries'][sorted_idx.tolist(), row0:row0 + self.win_rows, col0:col0 + self.win_cols]
        out = np.empty_like(block)
        out[order] = block                                # restore caller's day order
        return out.astype(np.float64)

    def noise_stack_mm(self, row0, col0, ref_day_idx, acq_day_idx, inc_target_deg):
        """
        MintPy-style ERA5 noise for one album: [n_acq, 128, 128] mm, LOS, + toward satellite.

        Acquisition t = (E(day_t) - E(ref_day)) * 1e3 * cos(inc_track) / cos(inc_target).
        No spatial referencing here (the album adds ramp/turbulence, then one reference pixel).
        """
        days = [ref_day_idx] + list(acq_day_idx)
        block_m = self.read_window_m(days, row0, col0)
        diff_mm = (block_m[1:] - block_m[:1]) * 1e3
        # Fill the <= 1 % invalid pixels with the window median so resampling stays finite.
        bad = ~np.isfinite(block_m[1:]) | (block_m[1:] == 0) | ~np.isfinite(block_m[:1]) | (block_m[:1] == 0)
        if bad.any():
            for t in range(diff_mm.shape[0]):
                diff_mm[t][bad[t]] = np.median(diff_mm[t][~bad[t]])
        # Area-average (anti-aliased) resampling to the target grid; ERA5 is smooth anyway.
        resampled_mm = F.interpolate(torch.from_numpy(diff_mm)[:, None], size=GRID_SHAPE,
                                     mode='area')[:, 0].numpy()
        incidence_scale = np.cos(np.radians(self.inc_centre_deg)) / np.cos(np.radians(inc_target_deg))
        return resampled_mm * incidence_scale
