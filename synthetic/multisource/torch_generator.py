#!/usr/bin/env python
# Usage:       from synthetic.multisource.torch_generator import TorchSceneGenerator
#              gen = TorchSceneGenerator(meta); batch = gen.sample(batch_size=128, seed=step)
# Description: Batched (vectorised) PyTorch version of the Stage-1 multi-source scene sampler
#              in generate_multisource.py, for fast on-the-fly training data and for building
#              noisy val/test sets. It draws the same source prior (K ~ U{0..K_max};
#              xcen, ycen, depth uniform; peak ratio log-uniform; P(inflation); Pascal
#              geometry rejection), evaluates Mogi LOS fields for all sources at once, and
#              optionally adds realistic noise (meta['noise'] == 'marapi'):
#                * Marapi APS components (trop, turbulent, orbit, white), each the difference
#                  of two random epochs from the TRAIN or held-out TEST epoch pool;
#                * a random planar ramp (gradients ~ N(0, ramp_sigma) mm/km in E and N);
#                * MintPy-style referencing: the whole observed map (sources + noise + ramp)
#                  minus its value at one random reference pixel per scene.
#              Returns standardized inputs and targets in the same format as dataset.py.
# Scientific conventions: identical to generate_multisource.py -- Mogi point source,
#   nu = 0.25, LOS positive toward satellite, fields in mm, params in km / m^3. Labels are
#   the TRUE (un-referenced) source parameters; referencing only changes the data.
# Assumptions (flagged): ramp_sigma default 0.3 mm/km (~+-15 mm over 48 km) on top of the
#   cube's orbital component; the 128x128 noise cube (made on 312 m pixels) is used as-is on
#   other grids, i.e. its spatial scale is stretched to the grid's pixel spacing.
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
    """Vectorised multi-Mogi scene sampler (noise-free or Marapi APS + ramp + reference)."""

    def __init__(self, meta, dtype=torch.float32):   # float32: ~1e-6 rel. error, 2.5x faster
        self.prior = meta['prior']
        self.noise_mode = meta['noise']
        self.sigma_ref_mm = float(meta['sigma_ref_mm'])
        self.scale_mm = float(meta['input_scaler']['std_mm'])
        self.dtype = dtype
        grid_cfg = meta.get('grid', {})
        grid = gen.load_grid(grid_cfg.get('geometry_h5'), grid_cfg.get('lat0_deg', gen.LAT0_DEG),
                             grid_cfg.get('lon0_deg', gen.LON0_DEG))
        self.shape = grid['shape']
        self.xE_m = torch.tensor(grid['xE_m'], dtype=dtype)
        self.yN_m = torch.tensor(grid['yN_m'], dtype=dtype)
        self.los_enu = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]), dtype=dtype)

        if self.noise_mode == 'marapi':
            noise_cfg = meta['noise_cfg']
            pool = gen.NoisePool(self.prior['n_holdout_epochs'])
            if pool.cubes_mm['trop'].shape[1:] != tuple(self.shape):
                raise ValueError(f"Noise cube {pool.cubes_mm['trop'].shape[1:]} != grid {self.shape}")
            for split in ('train', 'test'):     # split must match the data dir's meta
                if pool.epochs[split].tolist() != meta['noise_epochs'][split]:
                    raise ValueError(f"Noise {split} epochs differ from meta.json")
            self.noise_cubes = {c: torch.tensor(cube, dtype=dtype) for c, cube in pool.cubes_mm.items()}
            self.noise_epochs = {s: torch.tensor(e) for s, e in pool.epochs.items()}
            self.ramp_sigma_mm_per_km = float(noise_cfg['ramp_sigma_mm_per_km'])
            self.reference = noise_cfg['reference']
        elif self.noise_mode != 'none':
            raise ValueError(f"Unknown noise mode {self.noise_mode}")

    def _draw_positions(self, gen_torch, n):
        """Draw n candidate (x, y, depth) in km."""
        p = self.prior
        u = torch.rand(n, 3, generator=gen_torch, dtype=self.dtype)
        x_km = p['xcen_km'][0] + u[:, 0] * (p['xcen_km'][1] - p['xcen_km'][0])
        y_km = p['ycen_km'][0] + u[:, 1] * (p['ycen_km'][1] - p['ycen_km'][0])
        d_km = p['depth_km'][0] + u[:, 2] * (p['depth_km'][1] - p['depth_km'][0])
        return torch.stack([x_km, y_km, d_km], dim=-1)

    def sample_noise(self, batch_size, g, split='train'):
        """APS noise [B, H, W] mm: per component, epoch-pair difference from the split's pool."""
        pool = self.noise_epochs[split]
        n_pool = len(pool)
        noise_mm = torch.zeros(batch_size, *self.shape, dtype=self.dtype)
        for cube in self.noise_cubes.values():
            i = torch.randint(0, n_pool, (batch_size,), generator=g)
            j = (i + torch.randint(1, n_pool, (batch_size,), generator=g)) % n_pool   # j != i
            noise_mm += cube[pool[i]] - cube[pool[j]]
        return noise_mm

    def sample_ramp_and_reference(self, obs_mm, g):
        """Add a random planar ramp, then subtract the value at a random reference pixel."""
        batch_size = obs_mm.shape[0]
        grad = torch.randn(batch_size, 2, generator=g, dtype=self.dtype) * self.ramp_sigma_mm_per_km
        ramp_mm = (grad[:, 0, None] * self.xE_m[None] / 1e3 +
                   grad[:, 1, None] * self.yN_m[None] / 1e3).view(batch_size, *self.shape)
        obs_mm = obs_mm + ramp_mm
        if self.reference == 'ref_pixel':
            ref_idx = torch.randint(0, obs_mm[0].numel(), (batch_size,), generator=g)
            ref_val = obs_mm.view(batch_size, -1).gather(1, ref_idx[:, None])          # [B,1]
            obs_mm = obs_mm - ref_val[..., None]
        else:
            raise ValueError(f"Unknown reference mode {self.reference}")
        return obs_mm

    def sample(self, batch_size, seed, split='train'):
        """
        Draw one batch of scenes (split selects the noise epoch pool: 'train' or 'test').

        Returns dict with x [B,1,H,W] float32 (obs_mm / scale), params [B,K_max,4] float32
        (xcen_km, ycen_km, depth_km, dV_m3; 0 in empty slots), exist [B,K_max] bool, k [B]
        int64, extra [B,K_max,2] (peak_ratio, nan), obs_mm [B,H,W], clean_mm [B,H,W].
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

        clean_mm = fields_mm.sum(dim=1).view(batch_size, *self.shape)
        if self.noise_mode == 'marapi':
            obs_mm = self.sample_ramp_and_reference(clean_mm + self.sample_noise(batch_size, g, split), g)
        else:
            obs_mm = clean_mm
        extra = torch.stack([peak_sorted, torch.full_like(peak_sorted, float('nan'))], dim=-1)
        return {'x': (obs_mm / self.scale_mm).float()[:, None],
                'params': params.float(), 'exist': exist_sorted, 'k': k,
                'extra': extra.float(), 'obs_mm': obs_mm.float(), 'clean_mm': clean_mm.float()}
