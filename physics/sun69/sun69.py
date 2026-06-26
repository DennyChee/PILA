# Usage:       from physics.sun69.sun69 import Sun69
#                  model = Sun69(x_east_m, y_north_m)
#                  enu_mm = model.run(xcen, ycen, depth, radius, dV)
# Description: PyTorch batched, differentiable forward model for the Sun (1969)
#              penny-shaped (horizontal circular) crack in an elastic half-space.
#              Mirrors physics/mogi/mogi.py so it drops straight into the PILA
#              physics-informed VAE inversion as an alternative volcanic source.
# Date:        2026-06-17

import numpy as np
import torch


class Sun69():
    """
    Surface deformation from a horizontal penny-shaped crack (Sun, 1969).

    A penny-shaped crack is a flat, circular crack of radius `A` buried at depth
    `H`, inflated by an injected volume `V` (m^3). It is the sill-like / flat-body
    counterpart to the Mogi point source and is appropriate when the deforming
    body is better approximated by a horizontal disk than by a point. The PyTorch
    implementation here mirrors `physics/mogi/mogi.py`: station coordinates are
    fixed at construction, source parameters are batched, and the output is the
    concatenated [East | North | Up] surface displacement in millimetres.

    Scientific caveats
    ------------------
    - This is the Sun (1969) approximate solution, valid for H/A > 2 (error
      ~2-3%) and effectively exact for H/A > 5. Below H/A ~ 2 the closed-form
      expressions below STILL evaluate (no NaN), but the physical accuracy
      degrades. We deliberately do NOT return NaN for H < 2A (unlike the MATLAB
      reference `sun69.m`), because NaN would break autograd during inversion;
      instead keep H/A reasonable via the parameter bounds in *_paras.json.
    - Sign / unit conventions match Mogi exactly: inputs in metres, volume in
      m^3, Up (uz) positive = upward, output displacements in millimetres.
    - Maximum crack opening B = 3*V / (2*pi*A^2)  [Sun 1969, eqs (9)/(10a),
      p. 5999], the volume parameterization.

    Provenance
    ----------
    Ported from `matlab/synthetic_deformation/Volcano_Deformation/modified
    models/sun69.m` (F. Beauducel, BSD), which implements Sun (1969) eqs (16)
    and (17), p. 6001.

    Reference
    ---------
    Sun, R. J. (1969). Theoretical size of hydraulically induced horizontal
    fractures and corresponding surface uplift in an idealized medium,
    J. Geophys. Res., 74, 5995-6011.

    Attributes
    ----------
    x : Tensor
        East coordinates of observation points / stations (m), 1D tensor.
    y : Tensor
        North coordinates of observation points / stations (m), 1D tensor.
    nu : float
        Poisson's ratio (dimensionless). Unused in the volume form (kept for
        signature parity with Mogi and for a future pressure parameterization).
    """

    def __init__(self, x, y, nu=0.25):
        super(Sun69, self).__init__()
        """
        Initialize the Sun (1969) penny-shaped crack model.

        Parameters:
            x (Tensor): East-coordinate of observation points (m), 1D tensor
            y (Tensor): North-coordinate of observation points (m), 1D tensor
            nu (float): Poisson's ratio for the medium (default 0.25)
        """
        self.x = x
        self.y = y
        self.nu = nu

    def cart2pol(self, x, y):
        """
        Converts cartesian coordinates to polar coordinates.

        Parameters:
            x (Tensor): x-coordinate (m)
            y (Tensor): y-coordinate (m)

        Returns:
            theta (Tensor): polar angle (radians)
            r (Tensor): polar radius (m)
        """
        theta = torch.atan2(y, x)
        r = torch.sqrt(x**2 + y**2)
        return theta, r

    def pol2cart(self, theta, r):
        """
        Converts polar coordinates to cartesian coordinates.

        Parameters:
            theta (Tensor): polar angle (radians)
            r (Tensor): polar radius (m)

        Returns:
            x (Tensor): x-coordinate (m)
            y (Tensor): y-coordinate (m)
        """
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return x, y

    def run(self, xcen, ycen, depth, radius, dV):
        """
        Forward model of Sun (1969) penny-shaped crack for batched data.
        3D surface displacement field from a horizontal circular crack.

        Parameters:
            xcen (Tensor):   East-offset of crack centre (m), shape [batch]
            ycen (Tensor):   North-offset of crack centre (m), shape [batch]
            depth (Tensor):  depth H to crack centre (m), shape [batch]
            radius (Tensor): crack radius A (m), shape [batch]
            dV (Tensor):     injected volume V (m^3), shape [batch]

        Returns:
            output (Tensor): concatenated [East | North | Up] surface
                             displacements in millimetres, shape [batch, 3*N].
        """
        # --- Centre the observation grid on the crack epicentre ---
        # self.x, self.y are [N]; the unsqueezed params are [batch, 1] so the
        # subtraction broadcasts to [batch, N] (one row per batch item).
        x_adjusted = self.x - xcen.unsqueeze(1)   # [batch, N], m
        y_adjusted = self.y - ycen.unsqueeze(1)   # [batch, N], m

        # --- Surface polar coordinates: angle (th) and radial distance (r) ---
        th, r = self.cart2pol(x_adjusted, y_adjusted)   # th [rad], r [m]

        # --- Broadcast source scalars to [batch, 1] for elementwise math ---
        h = depth.unsqueeze(1)    # depth H (m)
        a = radius.unsqueeze(1)   # radius A (m)

        # --- Maximum crack opening B = 3V / (2*pi*A^2)  (Sun 1969 eq 9/10a) ---
        B = 3.0 * dV.unsqueeze(1) / (2.0 * torch.pi * a**2)   # m

        # --- Intermediate quantities (Sun 1969, p. 6001) ---
        D = r**2 + h**2 - a**2                      # may be negative; atan2 below handles it
        theta = torch.atan2(2.0 * a * h, D)         # branch-safe angle
        k = torch.sqrt((D / a**2)**2 + (2.0 * h / a)**2)
        srk = torch.sqrt(k)                         # sqrt(k); k > 0 for a,h > 0
        akcos2 = a * srk * torch.cos(theta / 2.0)
        aksin2 = a * srk * torch.sin(theta / 2.0)
        hkcos2 = h * srk * torch.cos(theta / 2.0)
        ak2cos = a * k * torch.cos(theta)

        # --- Vertical displacement, Up positive (Sun 1969 eq 16) ---
        # No 1/r term, so this is finite everywhere (including r = 0).
        uz = B * (srk * torch.sin(theta / 2.0)
                  - (h / (a * srk)) * torch.cos(theta / 2.0))   # m

        # --- Radial (horizontal) displacement (Sun 1969 eq 17) ---
        # The leading factor r makes ur -> 0 at the epicentre, so no singularity.
        term1 = (a + aksin2) / ((h + akcos2)**2 + (a + aksin2)**2)
        term2_num = hkcos2 - aksin2 + ak2cos
        term2_den = (hkcos2 - aksin2 + ak2cos)**2 \
            + (akcos2 + h * srk * torch.sin(theta / 2.0) + a * k * torch.sin(theta))**2
        ur = (B * r * h / a) * (term1 - term2_num / term2_den)   # m

        # --- Rotate radial displacement back to East/North components ---
        ux, uy = self.pol2cart(th, ur)   # East, North (m)

        # --- Concatenate [E | N | U] and convert metres -> millimetres ---
        output = torch.cat((ux, uy, uz), dim=1) * 1e3   # [batch, 3*N], mm

        return output
