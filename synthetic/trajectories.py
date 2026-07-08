#!/usr/bin/env python
# Usage:       from synthetic.trajectories import build_trajectory, profile_linear
#                  names, params = build_trajectory('sun69', n_epochs=25,
#                                                   start={...}, end={...},
#                                                   profile='sigmoid')
# Description: Build per-epoch PHYSICAL source-parameter trajectories for the
#              synthetic generator. Two scenario styles:
#                * static_buildup : source geometry fixed, amplitude (dV / opening)
#                                   ramps from ~0 to a peak  -> a cumulative
#                                   inflation time-series (epoch 0 ~ quiescent).
#                * moving_source  : any/all parameters vary epoch-to-epoch (lateral
#                                   migration, deepening, growing volume).
#              All parameters are returned in SI units (m, m^3, degrees), in the
#              order PILA's rescale() expects, so the generator and evaluator agree.
# Scientific note: the real MintPy time-series is CUMULATIVE LOS referenced to
#              epoch 0, so a trajectory whose amplitude starts near zero at epoch 0
#              mimics deformation accumulating from a quiescent baseline.
# Date:        2026-06-22

import numpy as np

from synthetic.forward_numpy import FORWARD_MODELS


# --- Interpolation profiles (fraction 0->1 across n epochs) -----------------
def profile_linear(n_epochs):
    """Linear ramp from 0 (epoch 0) to 1 (last epoch)."""
    if n_epochs == 1:
        return np.array([1.0])
    return np.linspace(0.0, 1.0, n_epochs)


def profile_sigmoid(n_epochs, steepness=8.0):
    """
    Smooth S-shaped ramp from ~0 to ~1, normalized to hit exactly [0, 1].

    Mimics an eruption that accelerates then saturates. `steepness` controls how
    sharp the onset is (larger = more step-like).
    """
    if n_epochs == 1:
        return np.array([1.0])
    t = np.linspace(-1.0, 1.0, n_epochs)
    s = 1.0 / (1.0 + np.exp(-steepness * t))
    s = (s - s[0]) / (s[-1] - s[0])               # renormalize to [0, 1]
    return s


PROFILES = {'linear': profile_linear, 'sigmoid': profile_sigmoid}


def _resolve_profile_name(profile, param_name):
    """
    Pick the profile name for one parameter from the `profile` argument.

    `profile` may be a single string (applied to every parameter) or a dict
    {param_name: profile_name} with an optional 'default' key as the fallback.
    """
    if isinstance(profile, str):
        name = profile                                   # global profile for all params
    elif isinstance(profile, dict):
        name = profile.get(param_name, profile.get('default'))  # per-param, then default
    else:
        raise TypeError(f"profile must be str or dict, got {type(profile).__name__}")
    if name is None:
        raise ValueError(f"No profile for parameter '{param_name}' and no 'default' key")
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'; choose from {list(PROFILES)}")
    return name


# --- Trajectory builder -----------------------------------------------------
def build_trajectory(model_name, n_epochs, start, end, profile='sigmoid'):
    """
    Interpolate each parameter from `start` to `end` over n_epochs.

    Parameters
    ----------
    model_name : str          'mogi' | 'sun69' | 'okada'.
    n_epochs   : int          number of epochs in the synthetic stack.
    start, end : dict          physical SI params at epoch 0 and the last epoch.
                               Keys must be the model's parameter names
                               (FORWARD_MODELS[model_name][1]). For a static-buildup
                               scenario set geometry keys equal in start and end and
                               only ramp the amplitude (dV / opening).
    profile    : str | dict    'linear' or 'sigmoid'. A single string applies to every
                               parameter (backward-compatible). A dict maps individual
                               parameter names to profile names, with an optional
                               'default' key as the fallback for unlisted params. This
                               lets geometry migrate (e.g. linearly) while amplitude
                               ramps on a different curve (e.g. sigmoid).

                               SCIENTIFIC CONSTRAINT (Okada fixed-tip dike): if you want a
                               propagating dike whose trailing tip stays fixed while the
                               length grows, `xoff`, `yoff`, and `length` MUST share the
                               same profile. The fixed-tip relation
                               centroid = tip + (length/2) * strike_dir is affine, so it
                               only holds at every epoch when those three interpolate
                               together (any profile is fine as long as it is the SAME
                               one). The amplitude (`opening`) may use a different profile.

    Returns
    -------
    param_names : list[str]                 physical parameter names, in order.
    params_per_epoch : np.ndarray [n_epoch, n_param]  physical SI values per epoch.
    """
    if model_name not in FORWARD_MODELS:
        raise ValueError(f"Unknown model '{model_name}'")
    param_names = FORWARD_MODELS[model_name][1]

    missing = [k for k in param_names if k not in start or k not in end]
    if missing:
        raise ValueError(f"start/end missing parameter(s) {missing} for {model_name}; "
                         f"need all of {param_names}")

    # Build the 0->1 ramp PER PARAMETER so geometry and amplitude can ramp differently.
    params = np.zeros((n_epochs, len(param_names)), dtype=np.float64)
    for j, name in enumerate(param_names):
        prof_name = _resolve_profile_name(profile, name)
        frac = PROFILES[prof_name](n_epochs)           # [n_epoch], 0->1
        v0, v1 = float(start[name]), float(end[name])
        params[:, j] = v0 + frac * (v1 - v0)           # per-epoch interpolation
    return param_names, params


def static_buildup(model_name, n_epochs, geometry, amplitude_key,
                   amp_start, amp_end, profile='sigmoid'):
    """
    Convenience wrapper: fixed geometry, single amplitude parameter ramped.

    Parameters
    ----------
    model_name    : str         'mogi' | 'sun69' | 'okada'.
    n_epochs      : int
    geometry      : dict         fixed physical params (everything except the
                                 amplitude key).
    amplitude_key : str          parameter to ramp ('dV' for mogi/sun69,
                                 'opening' for okada).
    amp_start, amp_end : float   amplitude at epoch 0 and last epoch (SI).
    profile       : str

    Returns
    -------
    param_names, params_per_epoch  (see build_trajectory).
    """
    start = dict(geometry); start[amplitude_key] = amp_start
    end = dict(geometry);   end[amplitude_key] = amp_end
    return build_trajectory(model_name, n_epochs, start, end, profile)


# --- Bounds check against a PILA *_paras.json -------------------------------
def to_paras_value(model_name, param_name, value_si, dv_scale=1e5, dv_shift=-1e7):
    """
    Convert a physical SI parameter value to its PILA *_paras.json (z) value.

    Inverse of model_phys_smpl rescale(): km-distance params are divided by 1000;
    dV uses dV_paras = (dV_si - dv_shift) / dv_scale; angles/opening are unchanged.

    dv_scale / dv_shift default to the legacy Mogi/Sun69 map (1e5, -1e7). Pass the
    'scale'/'shift' from the paras-JSON dV entry to match a custom mapping (e.g. the
    symmetric-about-0 range in configs/mogi_paras_symdV.json, scale=1, shift=0).
    """
    km_params = {
        'mogi':  ['xcen', 'ycen', 'd'],
        'sun69': ['xcen', 'ycen', 'depth', 'radius'],
        'okada': ['xoff', 'yoff', 'depth', 'length', 'width'],
    }[model_name]
    if param_name == 'dV':
        return (value_si - dv_shift) / dv_scale   # rescale(): dV_si = z*dv_scale + dv_shift
    if param_name in km_params:
        return value_si / 1000.0                  # m -> km
    return value_si                               # strike/dip (deg), opening (m)


def check_within_bounds(model_name, param_names, params_per_epoch, paras_ranges,
                        margin=1e-6):
    """
    Verify every per-epoch parameter lies inside the PILA paras bounds.

    PILA can only represent sources whose paras-value is in [min, max]; a ground
    truth outside the box is structurally unrecoverable (NOT a PILA failure).

    Parameters
    ----------
    model_name       : str
    param_names      : list[str]
    params_per_epoch : np.ndarray [n_epoch, n_param]  physical SI.
    paras_ranges     : dict   loaded *_paras.json: {name: {'min':.., 'max':..}}.
    margin           : float  tolerance on the bound (paras-value units).

    Returns
    -------
    ok          : bool
    violations  : list[str]   human-readable descriptions of any out-of-bounds.
    """
    violations = []
    for j, name in enumerate(param_names):
        if name not in paras_ranges:
            violations.append(f"'{name}' not in paras_ranges")
            continue
        lo = paras_ranges[name]['min']
        hi = paras_ranges[name]['max']
        # dV entry may carry a custom affine ('scale'/'shift'); default to legacy map.
        dv_scale = float(paras_ranges[name].get('scale', 1e5))
        dv_shift = float(paras_ranges[name].get('shift', -1e7))
        z = np.array([to_paras_value(model_name, name, v, dv_scale, dv_shift)
                      for v in params_per_epoch[:, j]])
        if np.any(z < lo - margin) or np.any(z > hi + margin):
            violations.append(
                f"'{name}': paras-value range [{z.min():.4g}, {z.max():.4g}] "
                f"outside bounds [{lo}, {hi}]")
    return (len(violations) == 0), violations
