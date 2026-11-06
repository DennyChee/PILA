#!/usr/bin/env python
# Usage:       from synthetic.multisource.dataset import (
#                  OnTheFlySceneDataset, FixedSceneDataset, load_stage1_meta)
# Description: PyTorch datasets for Stage-1 supervised multi-source PILA. Training scenes
#              are generated ON THE FLY (unlimited, nothing on disk) with the same sampler
#              as generate_multisource.py; validation/test scenes are FIXED sets stored as
#              .npz by make_stage1_data.py. Every item is standardized with ONE global scaler
#              computed from training scenes:  x_std = (obs_mm - mean) / std  with mean = 0
#              (scale-only; zero deformation -> zero input) and std = training pixel RMS (mm).
#              This differs from PILA's 'global' mode only in not subtracting a mean.
# Scientific notes:
#   * A fixed global scaler (not per-scene) keeps absolute amplitude in the input, which
#     the network needs to recover dV. Per-scene normalisation would make dV unidentifiable.
#   * Targets are returned in PHYSICAL units (xcen_km, ycen_km, depth_km, dV_m3) with NaN in
#     empty slots plus a boolean `exist` mask; target normalisation belongs to the model.
#   * Noise-free mode: obs_mm is exactly the sum of source fields (no spatial re-reference).
# Date:        2026-10-05

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic.multisource import generate_multisource as gen


def load_stage1_meta(data_dir):
    """Read meta.json (prior, scaler, noise mode, sigma_ref) from a Stage-1 data directory."""
    meta_path = os.path.join(data_dir, 'meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path) as f:
        return json.load(f)


def _to_item(obs_mm, params, extra, exist, k, scaler):
    """Pack one scene into the tensor dict the training loop consumes."""
    x_std = (obs_mm - scaler['mean_mm']) / scaler['std_mm']
    return {'x': torch.from_numpy(np.ascontiguousarray(x_std, np.float32))[None],  # [1, H, W]
            'params': torch.from_numpy(params.astype(np.float32)),                # [K_max, 4]
            'extra': torch.from_numpy(extra.astype(np.float32)),                  # [K_max, 2]
            'exist': torch.from_numpy(exist.astype(bool)),                        # [K_max]
            'k': torch.tensor(int(k))}


class OnTheFlySceneDataset(Dataset):
    """
    Infinite-style training set: each __getitem__ draws a brand-new scene.

    K is drawn uniformly in 0..K_max. The RNG is seeded from (base_seed, epoch, index) so a
    given (epoch, index) is reproducible, while every epoch sees new scenes. Call
    set_epoch(e) at the start of each epoch. The epoch lives in a SHARED-memory tensor so
    the update also reaches DataLoader workers (including persistent_workers=True), which
    hold their own copy of this object. Uses only the TRAIN noise pool.
    """

    def __init__(self, meta, scenes_per_epoch, base_seed=1000):
        self.meta = meta
        self.prior = meta['prior']
        self.noise_mode = meta['noise']
        self.sigma_ref_mm = meta['sigma_ref_mm']
        self.scaler = meta['input_scaler']
        self.scenes_per_epoch = int(scenes_per_epoch)
        self.base_seed = int(base_seed)
        # Shared memory: workers forked from this process see set_epoch() updates.
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()
        self.grid = gen.load_grid()
        # Noise cubes are only needed when noise is on (keeps the clean path light).
        self.noise_pool = None
        if self.noise_mode != 'none':
            self.noise_pool = gen.NoisePool(self.prior['n_holdout_epochs'])
            # Train/test disjointness relies on the same fixed-seed epoch split as the data dir.
            if self.noise_pool.epochs['train'].tolist() != meta['noise_epochs']['train']:
                raise ValueError("Noise TRAIN epochs differ from meta.json: train/test split broken")

    def set_epoch(self, epoch):
        """Change the scene stream (new scenes every epoch); visible to all workers."""
        self._epoch[0] = int(epoch)

    def __len__(self):
        return self.scenes_per_epoch

    def __getitem__(self, index):
        rng = np.random.default_rng([self.base_seed, int(self._epoch[0]), int(index)])
        k = int(rng.integers(0, self.prior['k_max'] + 1))
        scene = gen.make_scene(rng, self.grid, self.noise_pool, self.sigma_ref_mm, k,
                               'train', self.noise_mode, self.prior)
        return _to_item(scene['obs_mm'], scene['params'], scene['extra'],
                        scene['exist'], k, self.scaler)


class FixedSceneDataset(Dataset):
    """Fixed validation / test scenes read from an .npz written by make_stage1_data.py."""

    def __init__(self, npz_path, meta):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        with np.load(npz_path) as npz:
            self.arrays = {key: npz[key] for key in npz.files}
        self.scaler = meta['input_scaler']

    def __len__(self):
        return len(self.arrays['k'])

    def __getitem__(self, index):
        a = self.arrays
        return _to_item(a['obs_mm'][index], a['params'][index], a['extra'][index],
                        a['exist'][index], a['k'][index], self.scaler)
