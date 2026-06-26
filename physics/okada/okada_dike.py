# Usage:       from physics.okada.okada_dike import OkadaDike
# Description: Opening-only Okada (1985) rectangular dislocation (dike/sill) in
#              PyTorch, batched and differentiable. Faithful port of the user's
#              okadaMod.m (Beauducel okada85, modified opening-only), using the
#              CENTROID reference convention. Built as a drop-in PILA physics
#              decoder, mirroring physics/mogi/mogi.py.
# Reference:   Okada, Y. (1985). BSSA 75(4):1135-1154.
#              Beauducel okada85; okadaMod.m (L.W.J. Daniel, opening-only).
# Convention:  (xoff,yoff,depth) reference the fault CENTROID. strike CW from
#              North (fault dips to the RIGHT of the trace), dip in [0,90].
#              Output [East | North | Up] in mm. nu = 0.25.
# Date:        2026-06-11

import numpy as np
import torch


class OkadaDike():
    """
    Batched, differentiable opening-mode Okada (1985) dislocation.

    Parallel to the Mogi class so either can drive the PILA decoder.
    Station coordinates are fixed at construction; source parameters are passed
    (batched) to run().

    Attributes
    ----------
    x  : Tensor [N_stations]  East  coordinates of observation points, m
    y  : Tensor [N_stations]  North coordinates of observation points, m
    nu : float                Poisson's ratio (default 0.25)
    """

    def __init__(self, x, y, nu=0.25):
        super(OkadaDike, self).__init__()
        self.x = x      # East station coords [N_stations], m
        self.y = y      # North station coords [N_stations], m
        self.nu = nu    # Poisson's ratio

    # ------------------------------------------------------------------
    # Tensile (opening) per-corner kernel  -- Okada (1985) eq. (7), z = 0
    # ------------------------------------------------------------------
    def _tensile_kernel(self, xi, eta, q, sin_d, cos_d):
        """
        Opening-mode displacement kernel at one Chinnery corner.

        Verbatim port of okadaMod.m's ux_tf / uy_tf / uz_tf and the
        I1/I3/I5 subfunctions, including the cos(dip)->0 (vertical) limiting
        forms and the q=0 / xi=0 singularity guards.

        Parameters
        ----------
        xi, eta : [batch, N_stations]  Okada integration coordinates, m
        q       : [batch, N_stations]  perpendicular distance to fault plane, m
        sin_d   : [batch, 1]           sin(dip)
        cos_d   : [batch, 1]           cos(dip)

        Returns
        -------
        ux, uy, uz : [batch, N_stations]  fault-frame displacements; raw kernel
                     values -- the U3 and 1/(2*pi) prefactors are applied in run()
        """
        nu = self.nu
        k = 1.0 - 2.0 * nu               # = mu/(lambda+mu) for the I-functions
        eps = 1e-15                      # guards exact-edge denominators only

        # eps inside sqrt guards the autograd path (d/du sqrt(u) = 1/(2 sqrt(u))
        # is inf at u=0) for a station sitting exactly at a Chinnery corner.
        R = torch.sqrt(xi**2 + eta**2 + q**2 + eps)
        X = torch.sqrt(xi**2 + q**2 + eps)
        db = eta * sin_d - q * cos_d     # d-tilde
        yb = eta * cos_d + q * sin_d     # y-tilde

        Rpe = R + eta + eps              # R + eta
        Rpx = R + xi + eps               # R + xi
        Rpd = R + db + eps               # R + d-tilde

        # Vertical-fault mask (cos(dip) ~ 0); safe cos(dip) for the general branch
        is_vert = cos_d.abs() < 1e-9
        cos_d_safe = torch.where(is_vert, torch.ones_like(cos_d), cos_d)

        # --- I5 (eq. 28) ---
        I5_num = eta * (X + q * cos_d) + X * (R + X) * sin_d
        I5_den = xi * (R + X) * cos_d_safe
        I5_gen = k * 2.0 / cos_d_safe * torch.atan(I5_num / (I5_den + eps))
        I5_gen = torch.where(xi == 0, torch.zeros_like(xi), I5_gen)   # okadaMod guard
        I5_ver = -k * xi * sin_d / Rpd
        I5 = torch.where(is_vert, I5_ver, I5_gen)

        # --- I4 (eq. 27); only used by the general I3 ---
        I4 = k / cos_d_safe * (torch.log(Rpd) - sin_d * torch.log(Rpe))

        # --- I3 (eq. 26) ---
        I3_gen = k * (yb / (cos_d_safe * Rpd) - torch.log(Rpe)) + sin_d / cos_d_safe * I4
        I3_ver = k / 2.0 * (eta / Rpd + yb * q / Rpd**2 - torch.log(Rpe))
        I3 = torch.where(is_vert, I3_ver, I3_gen)

        # --- I1 (eq. 24) ---
        I1_gen = k * (-xi / (cos_d_safe * Rpd)) - sin_d / cos_d_safe * I5
        I1_ver = -k / 2.0 * xi * q / Rpd**2
        I1 = torch.where(is_vert, I1_ver, I1_gen)

        # atan(xi*eta/(q*R)) term, added only where q != 0 (okadaMod's find(q~=0))
        q_safe = torch.where(q == 0, torch.ones_like(q), q)
        atan_term = torch.atan(xi * eta / (q_safe * R + eps))
        atan_term = torch.where(q == 0, torch.zeros_like(q), atan_term)

        sin2 = sin_d**2
        ux = q**2 / (R * Rpe) - I3 * sin2
        uy = (-db * q / (R * Rpx) - sin_d * xi * q / (R * Rpe)
              - I1 * sin2 + sin_d * atan_term)
        uz = (yb * q / (R * Rpx) + cos_d * xi * q / (R * Rpe)
              - I5 * sin2 - cos_d * atan_term)
        return ux, uy, uz

    # ------------------------------------------------------------------
    # Public forward model
    # ------------------------------------------------------------------
    def run(self, xoff, yoff, depth, strike, dip, length, width, opening):
        """
        Surface displacement from an opening rectangular dislocation.

        Parameters (each Tensor [batch])
        --------------------------------
        xoff, yoff : East, North of the fault CENTROID, m
        depth      : depth of the fault CENTROID (>0), m
        strike     : degrees CW from North (fault dips to the right of trace)
        dip        : degrees from horizontal, [0, 90]
        length     : along-strike length, m
        width      : down-dip width, m
        opening    : tensile opening, m (positive = inflation)

        Returns
        -------
        output : Tensor [batch, 3 * N_stations]
            [East | North | Up] surface displacement in mm.
        """
        # Angles -> radians, shaped [batch, 1] for broadcasting over stations
        strike_rad = (strike * (np.pi / 180.0)).unsqueeze(1)
        dip_rad = (dip * (np.pi / 180.0)).unsqueeze(1)
        sin_s, cos_s = torch.sin(strike_rad), torch.cos(strike_rad)
        sin_d, cos_d = torch.sin(dip_rad), torch.cos(dip_rad)

        L = length.unsqueeze(1)          # [batch, 1]
        W = width.unsqueeze(1)
        U3 = opening.unsqueeze(1)

        # Station offsets from centroid (broadcast [batch,1] vs [N] -> [batch,N])
        e1 = self.x.unsqueeze(0) - xoff.unsqueeze(1)
        n1 = self.y.unsqueeze(0) - yoff.unsqueeze(1)

        # okadaMod.m front-end: centroid-referenced (E,N,depth) -> Okada (x,y,d)
        d = depth.unsqueeze(1) + sin_d * W / 2.0          # deeper-edge depth
        ec = e1 + cos_s * cos_d * W / 2.0
        nc = n1 - sin_s * cos_d * W / 2.0
        x = cos_s * nc + sin_s * ec + L / 2.0
        y = sin_s * nc - cos_s * ec + cos_d * W
        p = y * cos_d + d * sin_d
        q = y * sin_d - d * cos_d                          # const over corners

        # Chinnery 4-corner sum: f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)
        ux = torch.zeros_like(x)
        uy = torch.zeros_like(x)
        uz = torch.zeros_like(x)
        for xi, eta, sign in [(x, p, 1.0), (x, p - W, -1.0),
                              (x - L, p, -1.0), (x - L, p - W, 1.0)]:
            kx, ky, kz = self._tensile_kernel(xi, eta, q, sin_d, cos_d)
            ux = ux + sign * kx
            uy = uy + sign * ky
            uz = uz + sign * kz

        scale = U3 / (2.0 * np.pi)
        ux, uy, uz = scale * ux, scale * uy, scale * uz

        # okadaMod.m back-end: fault-frame -> geographic (East, North)
        uE = sin_s * ux - cos_s * uy
        uN = cos_s * ux + sin_s * uy

        # [East | North | Up], metres -> millimetres
        return torch.cat((uE, uN, uz), dim=1) * 1e3
