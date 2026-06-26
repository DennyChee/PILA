#!/usr/bin/env python
# Usage:       from synthetic.noise import load_marapi_noise, compute_snr_db, scale_to_target_snr
# Description: Noise sources for the PILA synthetic-test framework.
#              synthetic ("Marapi"): reads the precomputed component cube
#              synthetic_noise/noise_cube.mat (struct: trop=stratified/topo-correlated,
#              atm=turbulent, orbit=ramp, noise=white, combined=sum). Each component
#              is individually toggleable. No MatAPS toolbox needed (precomputed).
#              SNR helpers follow the MATLAB convention in mogi_snr_vbica_bench.m:
#                  SNR_dB = 20*log10( signal_RMS / noise_RMS ).
# Scientific notes:
#   * Cubes are LOS atmospheric delay in METRES on the native 128x128 / 40 km Marapi
#     grid, 76 epochs. Used in Mode B (idealized grid) with NO regridding.
#   * noise_RMS is a single scalar (RMS over valid pixels, averaged across active
#     epochs); the same scale is applied to every epoch when hitting a target SNR.
# Date:        2026-06-22

import os
import numpy as np
import scipy.io as sio

DEFAULT_NOISE_MAT = ('/eos-rs/INSAR_processing/denny/matlab/synthetic_deformation/'
                     'synthetic_noise/noise_cube_v2.mat')  # v2: stripe-free orbital ramp

# Orientation fix for MATLAB->Python import. The MATLAB grid stores rows with y
# increasing upward (row 0 = south); MintPy/geocoded convention is row 0 = north.
# A vertical flip (np.flip on the rows axis) puts the data north-up, which places the
# Marapi summit top-left (validated against the DEM s01_e100_3arc tif: flipud gives the
# only positive DEM correlation with the summit top-left). Set to None to disable.
NOISE_ORIENTATION = 'flipud'

# Map the user's physical taxonomy to the noise_cube.mat struct fields.
NOISE_COMPONENTS = {
    'trop': 'trop',        # topo-correlated / stratified (ERA5/MatAPS)
    'turbulent': 'atm',    # turbulent atmosphere (sim_atm_denny)
    'orbit': 'orbit',      # orbital ramp
    'white': 'noise',      # white
    'combined': 'combined',
}


def load_marapi_noise(components=('combined',), noise_mat=DEFAULT_NOISE_MAT):
    """
    Load and sum the requested Marapi noise components.

    Parameters
    ----------
    components : sequence of str   any of NOISE_COMPONENTS keys. 'combined' is the
                                   pre-summed total; otherwise pass individual
                                   components (e.g. ('trop','turbulent','white')).
    noise_mat  : str               path to noise_cube.mat.

    Returns
    -------
    noise_cube_m : np.ndarray [n_epoch, Ny, Nx]   summed noise, metres
                   (axis order matched to MintPy [epoch, rows, cols]).
    """
    if not os.path.exists(noise_mat):
        raise FileNotFoundError(noise_mat)
    m = sio.loadmat(noise_mat)['noise_cube']
    fields = m.dtype.names

    total = None
    for c in components:
        if c not in NOISE_COMPONENTS:
            raise ValueError(f"Unknown noise component '{c}'; choose from "
                             f"{list(NOISE_COMPONENTS)}")
        fld = NOISE_COMPONENTS[c]
        if fld not in fields:
            raise KeyError(f"noise_cube.mat has no field '{fld}' (have {fields})")
        cube = np.asarray(m[fld][0, 0], dtype=np.float64)     # [Ny, Nx, Nt]
        total = cube if total is None else total + cube

    # MATLAB stores [Ny, Nx, Nt]; move epoch axis first -> [Nt, Ny, Nx].
    out = np.moveaxis(total, 2, 0).astype(np.float64)

    # Orientation fix: flip the rows axis (axis=1) so the data is north-up.
    if NOISE_ORIENTATION == 'flipud':
        out = np.flip(out, axis=1)
    elif NOISE_ORIENTATION not in (None, 'identity'):
        raise ValueError(f"Unknown NOISE_ORIENTATION '{NOISE_ORIENTATION}'")
    return out


def _valid_rms(field_2d, mask):
    """RMS over valid (mask True, finite) pixels of a 2-D field."""
    vals = field_2d[mask] if mask is not None else field_2d.ravel()
    vals = vals[np.isfinite(vals)]
    return float(np.sqrt(np.mean(vals ** 2))) if vals.size else 0.0


def _valid_peak(field_2d, mask):
    """Peak |value| over valid (mask True, finite) pixels of a 2-D field.

    This is the maximum absolute LOS displacement actually present on the surface
    at that epoch (a more physical 'how big is the deformation' measure than RMS,
    which is diluted by the large quiet area around a compact source).
    """
    vals = field_2d[mask] if mask is not None else field_2d.ravel()
    vals = vals[np.isfinite(vals)]
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def signal_peak_per_epoch(signal_cube, mask=None):
    """Per-epoch peak |LOS| (same units as the cube) over valid pixels.

    Returns a [n_epoch] array; pairs with the per-epoch signal RMS used for SNR.
    """
    return np.array([_valid_peak(signal_cube[i], mask)
                     for i in range(signal_cube.shape[0])])


def noise_rms_scalar(noise_cube, mask=None, active_epochs=None):
    """
    Single noise-RMS scalar: RMS per epoch over valid pixels, averaged across
    active epochs (MATLAB mogi_snr_vbica_bench convention).
    """
    n_epoch = noise_cube.shape[0]
    idx = range(n_epoch) if active_epochs is None else active_epochs
    per = [_valid_rms(noise_cube[i], mask) for i in idx]
    return float(np.mean(per)) if per else 0.0


def compute_snr_db(signal_cube, noise_cube, mask=None, epoch=None, active_epochs=None):
    """
    SNR_dB = 20*log10(signal_RMS / noise_RMS).

    signal_RMS is taken at `epoch` (default: the peak-RMS epoch); noise_RMS is the
    scalar above. Returns (snr_db, signal_rms_m, noise_rms_m, epoch_used).
    """
    n_rms = noise_rms_scalar(noise_cube, mask, active_epochs)
    sig_rms_per = np.array([_valid_rms(signal_cube[i], mask)
                            for i in range(signal_cube.shape[0])])
    ep = int(np.argmax(sig_rms_per)) if epoch is None else int(epoch)
    s_rms = float(sig_rms_per[ep])
    snr = 20.0 * np.log10(s_rms / n_rms) if (n_rms > 0 and s_rms > 0) else float('-inf')
    return snr, s_rms, n_rms, ep


def scale_to_target_snr(signal_cube, noise_cube, target_snr_db, mask=None,
                        epoch=None, active_epochs=None):
    """
    Scale factor for the noise cube so the (peak-epoch) SNR equals target_snr_db.

    required_noise_rms = signal_rms / 10^(target_dB/20);  scale = required / current.
    Returns (scale, info_dict).
    """
    _, s_rms, n_rms, ep = compute_snr_db(signal_cube, noise_cube, mask, epoch, active_epochs)
    if n_rms <= 0 or s_rms <= 0:
        return 1.0, {'signal_rms_m': s_rms, 'noise_rms_m': n_rms, 'epoch': ep}
    required = s_rms / (10.0 ** (target_snr_db / 20.0))
    scale = required / n_rms
    return scale, {'signal_rms_m': s_rms, 'noise_rms_m_before': n_rms,
                   'noise_rms_m_after': required, 'epoch': ep,
                   'target_snr_db': target_snr_db}
