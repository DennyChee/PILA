#!/usr/bin/env python
# Usage:       from synthetic.forward_numpy import mogi, sun69, okada_dike
#                  uE_mm, uN_mm, uU_mm = sun69(x_east_m, y_north_m,
#                                              xcen_m, ycen_m, depth_m, radius_m, dV_m3)
# Description: INDEPENDENT NumPy reimplementations of the Mogi (1958) point source,
#              the Sun (1969) penny-shaped crack, and the opening-mode Okada (1985)
#              rectangular dislocation. These deliberately live OUTSIDE PILA's torch
#              physics decoders (physics/{mogi,okada,sun69}) so that synthetic data
#              generated here and inverted by PILA does not constitute a trivial
#              inverse crime: the code path, the library (numpy vs torch), and the
#              evaluation grid (full-resolution pixels here, then mask-aware multilook
#              by the loader, vs coarse-cell centres in the decoder) all differ.
#              The closed-form equations are the same physics, so the residual is
#              small-but-nonzero, which is exactly what a robustness test needs.
# Conventions: All inputs in SI (metres, m^3, degrees). Coordinates x_east, y_north
#              are observation-point East/North in metres (local ENU). Source position
#              (xcen/ycen or xoff/yoff) is also in metres. Output is East/North/Up
#              surface displacement in MILLIMETRES (to match PILA's decoders).
#              Up positive = upward. LOS projection is done by the caller.
# Date:        2026-06-22

import numpy as np


# --- Mogi (1958) point source ----------------------------------------------
def mogi(x_east_m, y_north_m, xcen_m, ycen_m, depth_m, dV_m3, nu=0.25):
    """
    Surface displacement from a Mogi (1958) point pressure source.

    Parameters
    ----------
    x_east_m, y_north_m : np.ndarray [N]   observation-point East/North, metres.
    xcen_m, ycen_m      : float            source East/North offset, metres.
    depth_m             : float            depth to point source (>0), metres.
    dV_m3               : float            volume change, m^3 (positive = inflation).
    nu                  : float            Poisson's ratio (default 0.25).

    Returns
    -------
    uE_mm, uN_mm, uU_mm : np.ndarray [N]   East/North/Up displacement, millimetres.
    """
    # --- Centre the observation grid on the source epicentre ---
    dx = x_east_m - xcen_m                       # [N], m
    dy = y_north_m - ycen_m                      # [N], m

    rho = np.sqrt(dx ** 2 + dy ** 2)             # horizontal distance, m
    theta = np.arctan2(dy, dx)                   # surface azimuth, rad
    R = np.sqrt(depth_m ** 2 + rho ** 2)         # 3-D radial distance from source, m

    # Mogi closed form (VMOD parameterization): C = (1-nu)/pi * dV
    C = ((1.0 - nu) / np.pi) * dV_m3
    ur = C * rho / R ** 3                         # radial horizontal displacement, m
    uU = C * depth_m / R ** 3                     # vertical displacement, m

    uE = ur * np.cos(theta)                       # rotate radial -> East, m
    uN = ur * np.sin(theta)                       # rotate radial -> North, m

    return uE * 1e3, uN * 1e3, uU * 1e3           # m -> mm


# --- Sun (1969) penny-shaped horizontal crack -------------------------------
def sun69(x_east_m, y_north_m, xcen_m, ycen_m, depth_m, radius_m, dV_m3, nu=0.25):
    """
    Surface displacement from a Sun (1969) penny-shaped (horizontal circular) crack.

    Independent NumPy port of Sun (1969) eqs (16)/(17), p.6001, with the volume
    parameterization B = 3V/(2*pi*A^2). Valid for H/A > 2 (best for H/A > 5).

    Parameters
    ----------
    x_east_m, y_north_m : np.ndarray [N]   observation-point East/North, metres.
    xcen_m, ycen_m      : float            crack-centre East/North offset, metres.
    depth_m             : float            depth H to crack centre, metres.
    radius_m            : float            crack radius A, metres.
    dV_m3               : float            injected volume V, m^3.
    nu                  : float            Poisson's ratio (unused in volume form;
                                           kept for signature parity).

    Returns
    -------
    uE_mm, uN_mm, uU_mm : np.ndarray [N]   East/North/Up displacement, millimetres.
    """
    # --- Centre observation grid on the crack epicentre; polar coordinates ---
    dx = x_east_m - xcen_m
    dy = y_north_m - ycen_m
    th = np.arctan2(dy, dx)                       # surface azimuth, rad
    r = np.sqrt(dx ** 2 + dy ** 2)               # radial distance, m

    h = depth_m                                   # crack depth H, m
    a = radius_m                                  # crack radius A, m

    # Maximum crack opening B = 3V / (2*pi*A^2)  (Sun 1969 eq 9/10a), metres
    B = 3.0 * dV_m3 / (2.0 * np.pi * a ** 2)

    # Intermediate quantities (Sun 1969, p.6001); branch-safe via atan2
    D = r ** 2 + h ** 2 - a ** 2
    theta = np.arctan2(2.0 * a * h, D)
    k = np.sqrt((D / a ** 2) ** 2 + (2.0 * h / a) ** 2)
    srk = np.sqrt(k)
    akcos2 = a * srk * np.cos(theta / 2.0)
    aksin2 = a * srk * np.sin(theta / 2.0)
    hkcos2 = h * srk * np.cos(theta / 2.0)
    ak2cos = a * k * np.cos(theta)

    # Vertical displacement, Up positive (Sun 1969 eq 16); finite at r = 0
    uU = B * (srk * np.sin(theta / 2.0)
              - (h / (a * srk)) * np.cos(theta / 2.0))     # m

    # Radial (horizontal) displacement (Sun 1969 eq 17); -> 0 at epicentre
    term1 = (a + aksin2) / ((h + akcos2) ** 2 + (a + aksin2) ** 2)
    term2_num = hkcos2 - aksin2 + ak2cos
    term2_den = ((hkcos2 - aksin2 + ak2cos) ** 2
                 + (akcos2 + h * srk * np.sin(theta / 2.0)
                    + a * k * np.sin(theta)) ** 2)
    ur = (B * r * h / a) * (term1 - term2_num / term2_den)  # m

    uE = ur * np.cos(th)                          # rotate radial -> East, m
    uN = ur * np.sin(th)                          # rotate radial -> North, m

    return uE * 1e3, uN * 1e3, uU * 1e3           # m -> mm


# --- Okada (1985) opening-mode rectangular dislocation ----------------------
def _okada_tensile_kernel(xi, eta, q, sin_d, cos_d, nu):
    """
    Opening-mode displacement kernel at one Chinnery corner (Okada 1985 eq.7, z=0).

    NumPy port of the tensile I1/I3/I5 sub-functions with the vertical-fault
    (cos dip -> 0) limiting forms and the q=0 / xi=0 singularity guards.
    Returns raw fault-frame ux, uy, uz; the U3/(2*pi) prefactor is applied by
    the caller.
    """
    k = 1.0 - 2.0 * nu
    eps = 1e-15

    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2 + eps)
    X = np.sqrt(xi ** 2 + q ** 2 + eps)
    db = eta * sin_d - q * cos_d                  # d-tilde
    yb = eta * cos_d + q * sin_d                  # y-tilde

    Rpe = R + eta + eps
    Rpx = R + xi + eps
    Rpd = R + db + eps

    is_vert = abs(cos_d) < 1e-9
    cos_d_safe = 1.0 if is_vert else cos_d

    # I5 (eq.28)
    I5_num = eta * (X + q * cos_d) + X * (R + X) * sin_d
    I5_den = xi * (R + X) * cos_d_safe
    I5_gen = k * 2.0 / cos_d_safe * np.arctan(I5_num / (I5_den + eps))
    I5_gen = np.where(xi == 0, 0.0, I5_gen)
    I5_ver = -k * xi * sin_d / Rpd
    I5 = I5_ver if is_vert else I5_gen

    # I4 (eq.27), used by general I3
    I4 = k / cos_d_safe * (np.log(Rpd) - sin_d * np.log(Rpe))

    # I3 (eq.26)
    I3_gen = k * (yb / (cos_d_safe * Rpd) - np.log(Rpe)) + sin_d / cos_d_safe * I4
    I3_ver = k / 2.0 * (eta / Rpd + yb * q / Rpd ** 2 - np.log(Rpe))
    I3 = I3_ver if is_vert else I3_gen

    # I1 (eq.24)
    I1_gen = k * (-xi / (cos_d_safe * Rpd)) - sin_d / cos_d_safe * I5
    I1_ver = -k / 2.0 * xi * q / Rpd ** 2
    I1 = I1_ver if is_vert else I1_gen

    # atan(xi*eta/(q*R)) term, only where q != 0
    q_safe = np.where(q == 0, 1.0, q)
    atan_term = np.arctan(xi * eta / (q_safe * R + eps))
    atan_term = np.where(q == 0, 0.0, atan_term)

    sin2 = sin_d ** 2
    ux = q ** 2 / (R * Rpe) - I3 * sin2
    uy = (-db * q / (R * Rpx) - sin_d * xi * q / (R * Rpe)
          - I1 * sin2 + sin_d * atan_term)
    uz = (yb * q / (R * Rpx) + cos_d * xi * q / (R * Rpe)
          - I5 * sin2 - cos_d * atan_term)
    return ux, uy, uz


def okada_dike(x_east_m, y_north_m, xoff_m, yoff_m, depth_m,
               strike_deg, dip_deg, length_m, width_m, opening_m, nu=0.25):
    """
    Surface displacement from an opening-mode Okada (1985) rectangular dislocation.

    CENTROID reference convention: (xoff,yoff,depth) is the fault centroid.
    strike CW from North (fault dips to the RIGHT of the trace); dip in [0,90].

    Parameters
    ----------
    x_east_m, y_north_m : np.ndarray [N]   observation-point East/North, metres.
    xoff_m, yoff_m      : float            fault-centroid East/North, metres.
    depth_m             : float            fault-centroid depth (>0), metres.
    strike_deg, dip_deg : float            strike/dip, degrees.
    length_m, width_m   : float            along-strike length / down-dip width, m.
    opening_m           : float            tensile opening, m (positive = inflation).
    nu                  : float            Poisson's ratio (default 0.25).

    Returns
    -------
    uE_mm, uN_mm, uU_mm : np.ndarray [N]   East/North/Up displacement, millimetres.
    """
    strike_rad = np.deg2rad(strike_deg)
    dip_rad = np.deg2rad(dip_deg)
    sin_s, cos_s = np.sin(strike_rad), np.cos(strike_rad)
    sin_d, cos_d = np.sin(dip_rad), np.cos(dip_rad)

    L, W, U3 = length_m, width_m, opening_m

    # Station offsets from the centroid
    e1 = x_east_m - xoff_m                         # [N]
    n1 = y_north_m - yoff_m                        # [N]

    # Centroid-referenced (E,N,depth) -> Okada (x,y,d) front-end (okadaMod.m)
    d = depth_m + sin_d * W / 2.0                  # deeper-edge depth
    ec = e1 + cos_s * cos_d * W / 2.0
    nc = n1 - sin_s * cos_d * W / 2.0
    x = cos_s * nc + sin_s * ec + L / 2.0
    y = sin_s * nc - cos_s * ec + cos_d * W
    p = y * cos_d + d * sin_d
    q = y * sin_d - d * cos_d                      # constant over the 4 corners

    # Chinnery 4-corner sum: f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    uz = np.zeros_like(x)
    for xi, eta, sign in [(x, p, 1.0), (x, p - W, -1.0),
                          (x - L, p, -1.0), (x - L, p - W, 1.0)]:
        kx, ky, kz = _okada_tensile_kernel(xi, eta, q, sin_d, cos_d, nu)
        ux = ux + sign * kx
        uy = uy + sign * ky
        uz = uz + sign * kz

    scale = U3 / (2.0 * np.pi)
    ux, uy, uz = scale * ux, scale * uy, scale * uz

    # Fault-frame -> geographic East/North (okadaMod.m back-end)
    uE = sin_s * ux - cos_s * uy
    uN = cos_s * ux + sin_s * uy

    return uE * 1e3, uN * 1e3, uz * 1e3           # m -> mm


# --- Dispatch helper --------------------------------------------------------
# Maps a model name to (function, ordered physical-parameter names). The names
# match PILA's rescale() output keys so the generator and evaluator agree.
FORWARD_MODELS = {
    'mogi':  (mogi,       ['xcen', 'ycen', 'd', 'dV']),
    'sun69': (sun69,      ['xcen', 'ycen', 'depth', 'radius', 'dV']),
    'okada': (okada_dike, ['xoff', 'yoff', 'depth', 'strike', 'dip',
                           'length', 'width', 'opening']),
}


def run_model(model_name, x_east_m, y_north_m, params_si):
    """
    Evaluate a named forward model at observation points for one parameter set.

    Parameters
    ----------
    model_name : str                       'mogi' | 'sun69' | 'okada'.
    x_east_m, y_north_m : np.ndarray [N]   observation-point East/North, metres.
    params_si  : dict or sequence          physical params in SI units, in the
                                           order given by FORWARD_MODELS[name][1].

    Returns
    -------
    uE_mm, uN_mm, uU_mm : np.ndarray [N]   East/North/Up displacement, millimetres.
    """
    if model_name not in FORWARD_MODELS:
        raise ValueError(f"Unknown model '{model_name}'; choose from "
                         f"{list(FORWARD_MODELS.keys())}")
    func, names = FORWARD_MODELS[model_name]
    if isinstance(params_si, dict):
        args = [params_si[n] for n in names]
    else:
        args = list(params_si)
    return func(x_east_m, y_north_m, *args)
