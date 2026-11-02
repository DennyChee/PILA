#!/usr/bin/env python
# Usage:       from synthetic.multisource.turbulence import fit_turbulence, sample_turbulence_mm
#              model = fit_turbulence()                       # from the Marapi noise cube
#              fields_mm = sample_turbulence_mm(rng, 31, model)
# Description: Power-law ("fractal") turbulent-atmosphere fields for synthetic PILA albums. ERA5
#              only resolves the large-scale/stratified delay, so the small-scale turbulence that
#              can mimic a point source is added as random fields whose radial power spectrum
#              (empirical median, with a k^-beta tail) and RMS are taken ONCE from the 'turbulent'
#              component of the Marapi noise cube (noise_cube_v2.mat). Also returns the white-noise
#              sigma of the cube's 'white' component.
# Scientific notes:
#   * Assumption (flagged): the cube's turbulent component (MATLAB sim_atm) was simulated
#     directly on the 128 x 128 grid, so its pixel spacing is taken as the TA69 grid spacing
#     (~380 x 328 m). Unlike the cube's 'trop' component it was not squeezed from a whole frame.
#   * Fields are generated PERIODIC on the 128 x 128 grid (no padding), exactly like the cube
#     (verified: the cube's edge-to-edge jump equals its neighbour-pixel difference). Padding
#     + cropping would add scales larger than the grid and, after RMS normalisation, starve
#     the small scales (Hann-tapered check: 2.5-6x less power at k > 0.05 cyc/px).
#   * The filter is scaled so the EXPECTED RMS is 6 mm (noise_cube_v2.mat sets every turbulent
#     epoch to exactly 6.00 mm by construction); fields are NOT individually renormalised, so
#     they keep the natural RMS scatter of a red-spectrum random field (reviewer, 2026-10-06).
#   * Units: mm, LOS. Fields have zero spatial mean (referencing is done by the album builder).
# Date:        2026-10-06

import numpy as np

from synthetic.noise import load_marapi_noise

GRID_SHAPE = (128, 128)


def radial_power_spectrum(field):
    """Azimuthally averaged power spectrum of one 2-D field: (k [cycles/px], power)."""
    n_rows, n_cols = field.shape
    spec = np.abs(np.fft.fft2(field - field.mean())) ** 2
    k_rows = np.fft.fftfreq(n_rows)[:, None]
    k_cols = np.fft.fftfreq(n_cols)[None, :]
    k = np.sqrt(k_rows ** 2 + k_cols ** 2)
    edges = np.linspace(0, 0.5, 33)                      # 32 radial bins up to Nyquist
    idx = np.digitize(k.ravel(), edges) - 1
    centres, power = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any() and edges[b] > 0:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            power.append(spec.ravel()[sel].mean())
    return np.array(centres), np.array(power)


def fit_turbulence(k_fit_range=(0.01, 0.2), verbose=True):
    """
    Fit P(k) ~ k^-beta and the RMS distribution to the Marapi cube's turbulent component.

    k_fit_range : (k_min, k_max) in cycles/pixel used for the log-log slope fit (avoids the
                  largest scales, which are few, and the near-Nyquist roll-off).
    Returns dict: beta (dimensionless), rms_mm (array of per-epoch RMS, mm), white_sigma_mm,
                  k, power (median spectrum, for the QC figure).
    """
    turbulent_mm = load_marapi_noise(('turbulent',)) * 1e3          # [76, 128, 128] mm
    white_mm = load_marapi_noise(('white',)) * 1e3
    if verbose:
        print(f"  Turbulent cube: shape={turbulent_mm.shape}, dtype={turbulent_mm.dtype}")
    spectra = [radial_power_spectrum(field)[1] for field in turbulent_mm]
    k, _ = radial_power_spectrum(turbulent_mm[0])
    median_power = np.median(np.array(spectra), axis=0)
    sel = (k >= k_fit_range[0]) & (k <= k_fit_range[1])
    slope, _ = np.polyfit(np.log(k[sel]), np.log(median_power[sel]), 1)
    beta = float(-slope)
    rms_mm = np.sqrt(np.mean((turbulent_mm - turbulent_mm.mean(axis=(1, 2), keepdims=True)) ** 2, axis=(1, 2)))
    white_sigma_mm = float(np.mean(np.std(white_mm, axis=(1, 2))))
    if verbose:
        print(f"  Turbulence fit: beta={beta:.2f}, RMS median {np.median(rms_mm):.2f} mm "
              f"(p5-p95 {np.percentile(rms_mm, 5):.2f}-{np.percentile(rms_mm, 95):.2f}); "
              f"white sigma {white_sigma_mm:.2f} mm")
    return {'beta': beta, 'rms_mm': rms_mm, 'white_sigma_mm': white_sigma_mm,
            'k': k, 'power': median_power}


def sample_turbulence_mm(rng, n_fields, model, shape=GRID_SHAPE):
    """
    Draw n independent fields [n, H, W] (mm) with the cube's spectrum; expected RMS = cube RMS,
    individual fields keep their natural RMS scatter.
    """
    n_rows, n_cols = shape
    k_rows = np.fft.fftfreq(n_rows)[:, None]
    k_cols = np.fft.fftfreq(n_cols)[None, :]
    k = np.sqrt(k_rows ** 2 + k_cols ** 2)
    # Amplitude filter = sqrt of the cube's EMPIRICAL median radial spectrum (log-log
    # interpolated). A single fitted power law under-predicts small-scale power ~2.5x after
    # RMS normalisation (the cube spectrum flattens at large scales). Corners beyond the
    # last radial bin (k > ~0.49 cyc/px) are extrapolated with the fitted slope beta.
    log_k_tab, log_p_tab = np.log(model['k']), np.log(model['power'])
    k_safe = np.where(k > 0, k, model['k'][0])
    log_power = np.interp(np.log(k_safe), log_k_tab, log_p_tab)
    beyond = k_safe > model['k'][-1]
    log_power[beyond] = log_p_tab[-1] - model['beta'] * (np.log(k_safe[beyond]) - log_k_tab[-1])
    amplitude = np.sqrt(np.exp(log_power))
    amplitude[0, 0] = 0.0                                 # zero the DC term (no mean offset)
    # Normalise the FILTER once so the EXPECTED field RMS equals the cube's RMS; individual
    # fields keep their natural RMS scatter (a red spectrum has few large-scale modes, so
    # day-to-day amplitude varies). Forcing every field to exactly 6 mm would suppress the
    # heavy tails that drive false alarms. For white noise w and filter A, E[RMS^2] =
    # sum(A^2) / (n_rows * n_cols) per unit-variance white sample (numpy ifft2 convention).
    target_rms = float(np.median(model['rms_mm']))
    amplitude = amplitude * target_rms / np.sqrt(np.sum(amplitude ** 2) / (n_rows * n_cols))
    fields = np.empty((n_fields,) + tuple(shape))
    for i in range(n_fields):
        white = rng.standard_normal((n_rows, n_cols))
        field = np.real(np.fft.ifft2(np.fft.fft2(white) * amplitude))
        fields[i] = field - field.mean()
    return fields
