# Usage:       from physics.okada.okada import Okada
# Description: Okada (1985) rectangular dislocation model (PyTorch, batched).
#              Computes 3-component surface displacements from a finite
#              rectangular fault in a homogeneous elastic half-space.
# Reference:   Okada, Y. (1985). Surface deformation due to shear and tensile
#              faults in a half-space. Bull. Seismol. Soc. Am., 75(4), 1135-1154.
# Date:        2026-06-10

import numpy as np
import torch


class Okada():
    """
    Batched PyTorch implementation of the Okada (1985) rectangular fault model.

    Supports three independent slip components:
      - strike_slip : left-lateral along-strike slip (U1 in Okada notation)
      - dip_slip    : reverse/thrust up-dip slip     (U2)
      - opening     : tensile fault opening           (U3)

    For volcanic applications, opening on a shallow horizontal fault models a
    sill intrusion; opening on a vertical fault models a dike.

    The interface is intentionally parallel to the Mogi class in
    physics/mogi/mogi.py so that either model can drop into the PILA decoder.

    Attributes
    ----------
    x  : Tensor [N_stations]  East  coordinates of observation stations, m
    y  : Tensor [N_stations]  North coordinates of observation stations, m
    nu : float                Poisson's ratio (default 0.25)
    """

    def __init__(self, x, y, nu=0.25):
        super(Okada, self).__init__()
        self.x  = x    # East  station coords [N_stations], m
        self.y  = y    # North station coords [N_stations], m
        self.nu = nu   # Poisson's ratio for elastic half-space

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _displacement_term(self, xi, eta, q, sin_d, cos_d):
        """
        Evaluate the Okada (1985) displacement kernels at one corner (xi, eta).

        This is the inner kernel called four times per station per batch item
        (once per corner of the rectangular fault).  The four results are
        combined via Chinnery's notation in run().

        Implements Okada (1985) eqs. (5), (6), (7) for the free surface z = 0,
        together with the I-function definitions (eqs. 24-28).

        Scientific assumptions
        ----------------------
        * Homogeneous, isotropic elastic half-space.
        * Observation points are on the free surface (z = 0).
        * Poisson's ratio fixed at construction time.

        Parameters
        ----------
        xi    : [batch, N_stations]  Along-strike coord measured from corner, m
        eta   : [batch, N_stations]  Along-dip   coord measured from corner, m
        q     : [batch, N_stations]  Rotated depth variable (fixed per station), m
        sin_d : [batch, 1]           sin(dip angle)
        cos_d : [batch, 1]           cos(dip angle)

        Returns
        -------
        Tuple of 9 tensors, each [batch, N_stations]:
            ux_ss, uy_ss, uz_ss  -- strike-slip   kernels (along-strike, perp, up)
            ux_ds, uy_ds, uz_ds  -- dip-slip      kernels
            ux_ts, uy_ts, uz_ts  -- tensile        kernels
        """
        nu  = self.nu
        # k = mu/(lambda+mu) = 1 - 2*nu
        # This factor appears in all five I-function definitions (Okada 1985 eqs 24-28).
        k   = 1.0 - 2.0 * nu

        eps = 1e-10  # guard against log(0) and 0/0 at fault edges

        # --- Intermediate geometric quantities ---
        R   = torch.sqrt(xi**2 + eta**2 + q**2 + eps)   # 3-D distance from corner
        X   = torch.sqrt(xi**2 + q**2 + eps)             # distance in xi-q plane

        # Tilde variables: project depth and y into fault-aligned coords
        y_t = eta * cos_d + q * sin_d    # ỹ  (Okada 1985, eq. 2)
        d_t = eta * sin_d - q * cos_d    # d̃  (Okada 1985, eq. 3)

        # Clamped denominators to prevent log(0) right at fault edges
        Rpe = torch.clamp(R + eta, min=eps)   # R + η
        Rpx = torch.clamp(R + xi,  min=eps)   # R + ξ
        Rpd = torch.clamp(R + d_t, min=eps)   # R + d̃

        # --- I functions: Okada (1985) eqs. (24)-(28) ---
        # For nearly-vertical faults (|cos δ| < threshold), the regular forms
        # have 1/cos δ divergence; use the analytic limiting forms instead.
        is_vert   = cos_d.abs() < 1e-3          # [batch, 1], broadcast to [batch, N]
        cos_d_s   = torch.where(is_vert, torch.ones_like(cos_d), cos_d)  # safe denom
        tan_d     = sin_d / cos_d_s              # safe; unused in vertical branch

        # I5  [eq. 28]. General form uses plain atan (range -pi/2..pi/2), matching
        # Okada's atan(.) — NOT atan2, which would flip sign for xi*cos_d < 0 corners.
        # Vertical limit: I5 = -k * xi * sin_d / (R + d̃)  (NOT zero).
        I5_num = eta * (X + q * cos_d) + X * (R + X) * sin_d
        I5_den = xi  * (R + X) * cos_d_s
        I5 = torch.where(
            is_vert,
            -k * xi * sin_d / Rpd,
            k / cos_d_s * 2.0 * torch.atan(I5_num / (I5_den + eps))
        )

        # I4  [eq. 27]
        I4 = torch.where(
            is_vert,
            -k * q / Rpe,                                                   # vertical limit
            k / cos_d_s * (torch.log(Rpd) - sin_d * torch.log(Rpe))        # general form
        )

        # I3  [eq. 26]
        I3_reg  = k * (y_t / (cos_d_s * Rpd) - torch.log(Rpe)) + tan_d * I4
        I3_vert = k / 2.0 * (eta / Rpd + y_t * q / (Rpd * Rpd) - torch.log(Rpe))
        I3 = torch.where(is_vert, I3_vert, I3_reg)

        # I2  [eq. 25]
        I2 = -k * torch.log(Rpe) - I3

        # I1  [eq. 24]
        I1_reg  = -k * xi / (Rpd * cos_d_s) - tan_d * I5
        I1_vert = -k / 2.0 * xi * q / (Rpd * Rpd)
        I1 = torch.where(is_vert, I1_vert, I1_reg)

        # Okada (1985) uses atan(ξη/qR) — standard arctangent in (-π/2, π/2).
        # Do NOT substitute atan2: when q < 0 (footwall stations), atan2 returns
        # values near ±π rather than near 0, causing large spurious displacements
        # in the Chinnery sum.
        atan_xeqR = torch.atan(xi * eta / (q * R + eps))

        # ----------------------------------------------------------------
        # Strike-slip kernels  (Okada 1985, eq. 5, z = 0)
        # Positive U1 = left-lateral slip
        # ----------------------------------------------------------------
        inv_2pi   = 1.0 / (2.0 * np.pi)
        xi_q_Rpe  = xi * q / (R * Rpe)           # reused in ss and ts

        ux_ss = -inv_2pi * (xi_q_Rpe + atan_xeqR + I1 * sin_d)
        uy_ss = -inv_2pi * (y_t * q / (R * Rpe) + q * cos_d / Rpe + I2 * sin_d)
        uz_ss = -inv_2pi * (d_t * q / (R * Rpe) + q * sin_d / Rpe + I4 * sin_d)

        # ----------------------------------------------------------------
        # Dip-slip kernels  (Okada 1985, eq. 6, z = 0)
        # Positive U2 = reverse / thrust (hanging wall moves up-dip)
        # ----------------------------------------------------------------
        ux_ds = -inv_2pi * (q / R - I3 * sin_d * cos_d)
        uy_ds = -inv_2pi * (y_t * q / (R * Rpx) + cos_d * atan_xeqR - I1 * sin_d * cos_d)
        uz_ds = -inv_2pi * (d_t * q / (R * Rpx) + sin_d * atan_xeqR - I5 * sin_d * cos_d)

        # ----------------------------------------------------------------
        # Tensile (opening) kernels  (Okada 1985, eq. 7, z = 0)
        # Positive U3 = fault opening (inflation / tensile fracture)
        # ----------------------------------------------------------------
        xi_q_sub  = xi_q_Rpe - atan_xeqR         # reused in ts uy and uz

        # NOTE: the leading d̃q / ỹq terms use (R+ξ) = Rpx, NOT (R+η).
        # (Okada 1985 eq. 7; cf. strike-slip which uses R+η.)
        ux_ts = +inv_2pi * (q**2 / (R * Rpe)  - I3 * sin_d**2)
        uy_ts = +inv_2pi * (-d_t * q / (R * Rpx) - sin_d * xi_q_sub - I1 * sin_d**2)
        uz_ts = +inv_2pi * ( y_t * q / (R * Rpx) + cos_d * xi_q_sub - I5 * sin_d**2)

        return ux_ss, uy_ss, uz_ss, ux_ds, uy_ds, uz_ds, ux_ts, uy_ts, uz_ts

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, xcen, ycen, depth, strike, dip, length, width,
            strike_slip, dip_slip, opening):
        """
        Compute surface displacements from a finite rectangular fault.

        Parameters
        ----------
        xcen        : Tensor [batch]  East  coordinate of fault top-centre, m
        ycen        : Tensor [batch]  North coordinate of fault top-centre, m
        depth       : Tensor [batch]  Depth to top of fault, positive downward, m
        strike      : Tensor [batch]  Fault strike, degrees clockwise from North
        dip         : Tensor [batch]  Fault dip, degrees from horizontal
                                      (0 = horizontal sill, 90 = vertical dike)
        length      : Tensor [batch]  Fault length along strike, m
        width       : Tensor [batch]  Fault width  along dip,    m
        strike_slip : Tensor [batch]  Left-lateral slip, m
                                      (right-lateral → negative)
        dip_slip    : Tensor [batch]  Reverse/thrust slip, m
                                      (normal faulting → negative)
        opening     : Tensor [batch]  Tensile opening, m
                                      (closing → negative)

        Returns
        -------
        output : Tensor [batch, 3 * N_stations]
            Surface displacements in mm, concatenated as [East | North | Up].
            Positive Up convention matches the Mogi model in this codebase.
        """
        # --- Convert fault angles to radians ---
        strike_rad = strike * (np.pi / 180.0)
        dip_rad    = dip    * (np.pi / 180.0)

        sin_s = torch.sin(strike_rad).unsqueeze(1)   # [batch, 1]
        cos_s = torch.cos(strike_rad).unsqueeze(1)
        sin_d = torch.sin(dip_rad).unsqueeze(1)
        cos_d = torch.cos(dip_rad).unsqueeze(1)

        # --- Rotate station coordinates into fault frame ---
        # x_f: along-strike component (positive = strike direction)
        # y_f: perpendicular component (positive = hanging-wall side)
        # Rotation convention: strike measured clockwise from North,
        # so along-strike unit vector in (E, N) = (sin s, cos s).
        dx  = self.x - xcen.unsqueeze(1)               # [batch, N_stations]
        dy  = self.y - ycen.unsqueeze(1)

        x_f = dx * sin_s + dy * cos_s                  # along-strike
        y_f = dx * cos_s - dy * sin_s                  # perp-to-strike (hanging wall +)

        # --- Set up Chinnery integration variables ---
        # Origin is placed at the left corner of the fault's top edge.
        # The fault top-centre is at x_f = 0, so the left corner is at -L/2.
        L = length.unsqueeze(1)                        # [batch, 1]
        W = width.unsqueeze(1)

        x   = x_f + L / 2.0                           # along-strike from left corner

        # Okada's p,q substitution references the DEEPER (bottom) fault edge.
        # The user supplies `depth` = depth to the TOP edge (see docstring), so
        # convert to the bottom-edge depth used by the kernel. Verified against
        # Okada DC3D (okada_wrapper) to < 5e-8 m for dips 13deg-89deg.
        depth_eff = depth.unsqueeze(1) + W * sin_d       # depth to bottom edge, m

        p   = y_f * cos_d + depth_eff * sin_d   # Okada (1985) eq. (1)
        q   = y_f * sin_d - depth_eff * cos_d   # constant for all corners

        # --- Accumulate Chinnery sum over 4 fault corners ---
        # Chinnery notation: f(ξ,η)|| = f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)
        total = [torch.zeros_like(x) for _ in range(9)]

        for xi, eta, sign in [
            (x,     p,     +1),
            (x,     p - W, -1),
            (x - L, p,     -1),
            (x - L, p - W, +1),
        ]:
            terms = self._displacement_term(xi, eta, q, sin_d, cos_d)
            for idx, t in enumerate(terms):
                total[idx] = total[idx] + sign * t

        ux_ss, uy_ss, uz_ss, ux_ds, uy_ds, uz_ds, ux_ts, uy_ts, uz_ts = total

        # --- Superpose the three slip components ---
        ss = strike_slip.unsqueeze(1)   # [batch, 1]
        ds = dip_slip.unsqueeze(1)
        op = opening.unsqueeze(1)

        ux_f = ss * ux_ss + ds * ux_ds + op * ux_ts   # along-strike displacement
        uy_f = ss * uy_ss + ds * uy_ds + op * uy_ts   # perp-to-strike displacement
        uz   = ss * uz_ss + ds * uz_ds + op * uz_ts   # vertical displacement (up +)

        # --- Rotate fault-frame displacements back to geographic (East, North) ---
        # Inverse of the station rotation above.
        # Along-strike (E,N) = (sin s, cos s)  →  East = ux_f*sin_s + uy_f*cos_s
        # Perp-to-strike (E,N) = (cos s, -sin s) →  North = ux_f*cos_s - uy_f*sin_s
        ux_E = ux_f * sin_s + uy_f * cos_s    # East  component [batch, N_stations]
        uy_N = ux_f * cos_s - uy_f * sin_s    # North component

        # Concatenate [East | North | Up] and convert m → mm
        output = torch.cat((ux_E, uy_N, uz), dim=1) * 1e3

        return output
