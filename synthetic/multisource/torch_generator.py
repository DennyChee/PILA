#!/usr/bin/env python
# Usage:       from synthetic.multisource.torch_generator import TorchSceneGenerator
#              gen = TorchSceneGenerator(meta); batch = gen.sample(batch_size=128, seed=step)
# Description: Batched (vectorised) PyTorch version of the Stage-1 multi-source scene sampler
#              in generate_multisource.py, for fast on-the-fly TRAINING data. It draws the
#              same prior (K ~ U{0..K_max}; xcen, ycen ~ U(+-15 km); depth ~ U(1, 10 km);
#              peak ratio log-uniform; P(inflation); Pascal geometry rejection), evaluates
#              Mogi LOS fields for all sources at once, and returns standardized inputs and
#              targets in the same format as dataset.py. NOISE-FREE ONLY (noise='none').
#              Validation/test sets stay NumPy-generated (independent implementation), and
#              this module is checked against generate_multisource.mogi_los_mm numerically.
# Scientific conventions: identical to generate_multisource.py -- Mogi point source,
#   nu = 0.25, LOS positive toward satellite, fields in mm, params in km / m^3.
# Date:        2026-10-05

import numpy as np
import torch

from synthetic.multisource import generate_multisource as gen

POISSON_RATIO = 0.25


def mogi_los_mm_torch(xE_m, yN_m, los_enu, xcen_m, ycen_m, depth_m, dV_m3, nu=POISSON_RATIO):
    """
    Mogi LOS displacement for many sources at once.

    Parameters
    ----------
    xE_m, yN_m : Tensor [N]          observation East/North, m.
    los_enu    : Tensor [3, N]       LOS unit vector (E, N, U) per pixel.
    xcen_m, ycen_m, depth_m, dV_m3 : Tensor [B, S]   source params (m, m^3).

    Returns
    -------
    los_mm : Tensor [B, S, N]   LOS displacement, mm, positive toward satellite.
    """
    dx = xE_m - xcen_m[..., None]                       # [B, S, N], m
    dy = yN_m - ycen_m[..., None]
    rho_sq = dx ** 2 + dy ** 2
    R_cubed = (depth_m[..., None] ** 2 + rho_sq) ** 1.5  # 3-D distance^3, m^3
    C = ((1.0 - nu) / np.pi) * dV_m3[..., None]          # Mogi strength, m^3
    # ur * cos(theta) = C * dx / R^3 (radial -> East), same for North; vertical C*d/R^3.
    uE = C * dx / R_cubed
    uN = C * dy / R_cubed
    uU = C * depth_m[..., None] / R_cubed
    return 1e3 * (los_enu[0] * uE + los_enu[1] * uN + los_enu[2] * uU)   # m -> mm


class TorchSceneGenerator:
    """Vectorised noise-free multi-Mogi scene sampler matching generate_multisource's prior."""

    def __init__(self, meta, dtype=torch.float32):   # float32: ~1e-6 rel. error, 2.5x faster
        if meta['noise'] != 'none':
            raise NotImplementedError("TorchSceneGenerator is noise-free only (meta['noise']='none')")
        self.prior = meta['prior']
        self.sigma_ref_mm = float(meta['sigma_ref_mm'])
        self.scale_mm = float(meta['input_scaler']['std_mm'])
        self.dtype = dtype
        grid = gen.load_grid()
        self.shape = grid['shape']
        self.xE_m = torch.tensor(grid['xE_m'], dtype=dtype)
        self.yN_m = torch.tensor(grid['yN_m'], dtype=dtype)
        self.los_enu = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]), dtype=dtype)

    def _draw_positions(self, gen_torch, n):
        """Draw n candidate (x, y, depth) in km."""
        p = self.prior
        u = torch.rand(n, 3, generator=gen_torch, dtype=self.dtype)
        x_km = p['xcen_km'][0] + u[:, 0] * (p['xcen_km'][1] - p['xcen_km'][0])
        y_km = p['ycen_km'][0] + u[:, 1] * (p['ycen_km'][1] - p['ycen_km'][0])
        d_km = p['depth_km'][0] + u[:, 2] * (p['depth_km'][1] - p['depth_km'][0])
        return torch.stack([x_km, y_km, d_km], dim=-1)

    def sample(self, batch_size, seed):
        """
        Draw one batch of scenes.

        Returns dict with x [B,1,H,W] float32 (obs_mm / scale), params [B,K_max,4] float32
        (xcen_km, ycen_km, depth_km, dV_m3; 0 in empty slots), exist [B,K_max] bool, k [B]
        int64, extra [B,K_max,2] (peak_ratio, nan for scene-RMS SNR), obs_mm [B,H,W].
        """
        p = self.prior
        k_max = p['k_max']
        g = torch.Generator().manual_seed(int(seed))
        k = torch.randint(0, k_max + 1, (batch_size,), generator=g)
        exist = torch.arange(k_max)[None, :] < k[:, None]                       # [B, S]

        # --- Positions with sequential Pascal rejection (slot j vs slots < j) ---
        pos_km = self._draw_positions(g, batch_size * k_max).view(batch_size, k_max, 3)
        for slot in range(1, k_max):
            for _ in range(p['max_placement_tries']):
                prev = pos_km[:, :slot]                                          # [B, j, 3]
                cur = pos_km[:, slot:slot + 1]                                   # [B, 1, 3]
                dist_km = torch.linalg.norm(cur - prev, dim=-1)                  # [B, j]
                d_shallow = torch.minimum(cur[..., 2], prev[..., 2])
                bad = (dist_km < p['sep_reject_dshallow'] * d_shallow).any(dim=1) & exist[:, slot]
                if not bad.any():
                    break
                pos_km[bad, slot] = self._draw_positions(g, int(bad.sum()))
            else:
                raise RuntimeError("Pascal rejection did not converge")

        # --- Amplitudes: log-uniform peak ratio, random sign, exact dV solve ---
        lo, hi = np.log(p['peak_ratio'][0]), np.log(p['peak_ratio'][1])
        peak_ratio = torch.exp(lo + torch.rand(batch_size, k_max, generator=g, dtype=self.dtype) * (hi - lo))
        sign = torch.where(torch.rand(batch_size, k_max, generator=g, dtype=self.dtype) < p['p_inflation'],
                           1.0, -1.0).to(self.dtype)
        unit_mm = mogi_los_mm_torch(self.xE_m, self.yN_m, self.los_enu, pos_km[..., 0] * 1e3,
                                    pos_km[..., 1] * 1e3, pos_km[..., 2] * 1e3,
                                    torch.ones(batch_size, k_max, dtype=self.dtype))     # [B,S,N]
        dV_m3 = sign * peak_ratio * self.sigma_ref_mm / unit_mm.abs().amax(dim=-1)
        fields_mm = unit_mm * dV_m3[..., None] * exist[..., None]                 # empty -> 0

        # --- Canonical slot order: existing sources by descending peak ratio, empties last ---
        order_key = torch.where(exist, peak_ratio, torch.full_like(peak_ratio, -1.0))
        order = torch.argsort(order_key, dim=1, descending=True)
        take = lambda t: torch.gather(t, 1, order)                                # noqa: E731
        params = torch.stack([take(pos_km[..., 0]), take(pos_km[..., 1]),
                              take(pos_km[..., 2]), take(dV_m3)], dim=-1)
        exist_sorted = torch.gather(exist, 1, order)
        params = torch.where(exist_sorted[..., None], params, torch.zeros_like(params))
        peak_sorted = torch.where(exist_sorted, take(peak_ratio), torch.full_like(peak_ratio, float('nan')))

        obs_mm = fields_mm.sum(dim=1).view(batch_size, *self.shape)
        extra = torch.stack([peak_sorted, torch.full_like(peak_sorted, float('nan'))], dim=-1)
        return {'x': (obs_mm / self.scale_mm).float()[:, None],
                'params': params.float(), 'exist': exist_sorted, 'k': k,
                'extra': extra.float(), 'obs_mm': obs_mm.float()}
