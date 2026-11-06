#!/usr/bin/env python
# Usage:       python train_pila_delayedpen.py -c <config.json>     (same arguments as train_pila.py)
# Description: Launcher for the DELAYED-PENALTY arms. Runs train_pila.py UNCHANGED, except that
#              PhysVAETrainerSMPL._train_epoch and PHYS_VAE_SMPL.draw are wrapped (in memory only) so the standard
#              penalties follow an epoch schedule read from config trainer.phys_vae.delayed_penalty =
#              {"off_epochs": N, "ramp_epochs": M, "kl_stays_off": false}:
#                epochs 1..N        : penalties OFF (curvature 0, dV-L1 0, z_phy KL off -> deterministic z_phy,
#                                     i.e. exactly the penalties-off recipe) so the two sources can separate first
#                epochs N+1..N+M    : weights ramp linearly 0 -> full config value; the z_phy SAMPLING NOISE ramps
#                                     with them (u = mean + factor*std*eps), so switching the KL on is a true ramp,
#                                     not a full-strength noise step from the still-untrained lnvar head
#                epochs > N+M       : full standard penalties (curvature, dV-L1, KL beta_max_z_phy)
#              kl_stays_off=true (isolating arm): curvature and dV-L1 follow the schedule, but KL and sampling stay
#              off for the whole run (deterministic z_phy) -> separates "penalty terms" from "sampled-z noise".
#              off_epochs >= epochs gives the penalties-off arm through the same code path.
#              Every epoch writes one row to <run models dir>/per_epoch_sources.csv: schedule factor, train rec loss,
#              and per-slot median xcen/ycen/depth (km) and dV (m^3) of the encoder MEAN over that epoch's training
#              batches, plus the median sampling std in u-space (sigma_u).
#              The switch is a fixed epoch, never a truth-based criterion (honest inversion).
#              No existing file is edited.
# Assumptions: epochs_pretrain = 0 (standard recipe); the KL linear warm-up (kl_warmup_epochs, 30) has already
#              reached 1 by epoch N, so the ramp factor alone sets the KL weight. The final epoch is always saved
#              as model_best.pth (base_trainer). Per-epoch values are medians over batches seen while the weights
#              were still being updated within that epoch (a running summary, not an end-of-epoch evaluation).
# Date:        2026-10-19

import csv
import os
import runpy
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from trainer import PhysVAETrainerSMPL  # noqa: E402
from model.model_phys_smpl import PHYS_VAE_SMPL  # noqa: E402
from utils.utils import draw_normal  # noqa: E402

_original_train_epoch = PhysVAETrainerSMPL._train_epoch
_original_draw = PHYS_VAE_SMPL.draw
PARAMETERS_PER_SOURCE = 4      # MultiMogi block order: xcen, ycen, d, dV


def noise_scaled_draw(self, z_phy_stat, z_aux_stat, hard_z_phy=False, hard_z_aux=False):
    """
    PHYS_VAE_SMPL.draw with the z_phy noise scaled by self._z_phy_noise_scale (default 1 = unchanged):
    u = mean + scale*(sample - mean). Deterministic (hard) draws and z_aux are untouched.
    While training, also stashes the encoder mean and lnvar (detached) for the per-epoch source CSV.
    """
    stash = getattr(self, '_epoch_stash', None)
    if stash is not None and self.training and not self.no_phy:
        stash['mean'].append(z_phy_stat['mean'].detach().reshape(-1, z_phy_stat['mean'].shape[-1]).cpu())
        stash['lnvar'].append(z_phy_stat['lnvar'].detach().reshape(-1, z_phy_stat['lnvar'].shape[-1]).cpu())
    scale = getattr(self, '_z_phy_noise_scale', 1.0)
    if hard_z_phy or scale >= 1.0 or self.no_phy:
        return _original_draw(self, z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
    # z_aux exactly as the original; z_phy sampled here in u-space (no lossy sigmoid->logit round trip at the rails)
    _z_phy_mean_only, z_aux = _original_draw(self, z_phy_stat, z_aux_stat, hard_z_phy=True, hard_z_aux=hard_z_aux)
    u_sampled = draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])        # same clamp/reparam as the original
    u_scaled = z_phy_stat['mean'] + scale * (u_sampled - z_phy_stat['mean'])
    return torch.sigmoid(u_scaled), z_aux


def penalty_ramp_factor(epoch, off_epochs, ramp_epochs):
    """
    Fraction (0-1) of the full penalty weights to apply at this epoch.

    Inputs:  epoch (int, 1-based as in base_trainer), off_epochs (int) penalty-free epochs,
             ramp_epochs (int) linear ramp length (0 = switch on at once).
    Output:  float in [0, 1].
    """
    if epoch <= off_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - off_epochs) / float(ramp_epochs))


def write_source_row(trainer, epoch, factor, result):
    """
    Append this epoch's per-slot source summary to <save_dir>/per_epoch_sources.csv.

    Inputs:  trainer (PhysVAETrainerSMPL) whose model holds the epoch stash; epoch (int); factor (float);
             result (dict) returned by _train_epoch (for the train rec loss).
    Output:  none (file append). Units: xcen/ycen/depth km, dV m^3, sigma_u dimensionless (u-space std).
    """
    stash = trainer.model._epoch_stash
    if not stash['mean']:
        return
    mean_u = torch.cat(stash['mean'])                       # [n_samples, 4K], pre-sigmoid
    lnvar_u = torch.cat(stash['lnvar'])
    physics = trainer.model.physics_model
    n_sources = mean_u.shape[1] // PARAMETERS_PER_SOURCE
    with torch.no_grad():
        # rescale maps z in (0,1) -> physical (m, m^3), exactly as dump_multisource_fit.py does
        rescaled = physics.rescale(torch.sigmoid(mean_u).reshape(-1, PARAMETERS_PER_SOURCE))   # dict of m, m^3
    params = torch.stack([rescaled['xcen'], rescaled['ycen'], rescaled['d'], rescaled['dV']], dim=-1)
    params = params.reshape(mean_u.shape[0], n_sources, PARAMETERS_PER_SOURCE).median(dim=0).values
    row = {'epoch': epoch, 'factor': round(factor, 4), 'rec_loss': result.get('rec_loss', float('nan')),
           'sigma_u_median': float(torch.exp(0.5 * lnvar_u).median())}
    for source_index in range(n_sources):
        xcen_m, ycen_m, depth_m, dv_m3 = [float(value) for value in params[source_index]]
        row.update({f's{source_index + 1}_x_km': xcen_m / 1e3, f's{source_index + 1}_y_km': ycen_m / 1e3,
                    f's{source_index + 1}_depth_km': depth_m / 1e3, f's{source_index + 1}_dV_m3': dv_m3})
    csv_path = os.path.join(str(trainer.config.save_dir), 'per_epoch_sources.csv')
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def delayed_penalty_train_epoch(self, epoch):
    """
    Set this epoch's penalty weights from the schedule, run the unchanged _train_epoch, log per-slot sources.

    The full weights are captured once from the config on the first call (attributes are overwritten afterwards).
    """
    if not hasattr(self, '_delayed_full_weights'):
        schedule = self.config['trainer']['phys_vae'].get('delayed_penalty')
        if not schedule:
            raise ValueError("train_pila_delayedpen.py needs trainer.phys_vae.delayed_penalty in the config")
        if self.epochs_pretrain != 0:
            raise ValueError(f"delayed penalty assumes epochs_pretrain=0 (got {self.epochs_pretrain})")
        self._delayed_schedule = (int(schedule['off_epochs']), int(schedule['ramp_epochs']))
        self._kl_stays_off = bool(schedule.get('kl_stays_off', False))
        self._delayed_full_weights = dict(
            temporal_smoothness_weight=self.temporal_smoothness_weight, dv_l1_weight=self.dv_l1_weight,
            use_kl_term_z_phy=self.use_kl_term_z_phy, beta_max_z_phy=self.beta_max_z_phy)
        print(f"[delayed_penalty] schedule off={self._delayed_schedule[0]} ramp={self._delayed_schedule[1]} "
              f"kl_stays_off={self._kl_stays_off}, full weights {self._delayed_full_weights}")
    factor = penalty_ramp_factor(epoch, *self._delayed_schedule)
    full = self._delayed_full_weights
    self.temporal_smoothness_weight = full['temporal_smoothness_weight'] * factor
    self.dv_l1_weight = full['dv_l1_weight'] * factor
    # KL fully OFF (deterministic z_phy, as in the penalties-off arm) until the ramp starts; then scale beta.
    # Isolating arm: KL and sampling never switch on.
    kl_factor = 0.0 if self._kl_stays_off else factor
    self.use_kl_term_z_phy = full['use_kl_term_z_phy'] and kl_factor > 0.0
    self.beta_max_z_phy = full['beta_max_z_phy'] * kl_factor
    self.model._z_phy_noise_scale = kl_factor       # sampling noise ramps with the KL weight (see noise_scaled_draw)
    self.logger.info(f"[delayed_penalty] epoch {epoch}: factor={factor:.3f} curvature={self.temporal_smoothness_weight:.1f} "
                     f"dv_l1={self.dv_l1_weight:.1f} kl_on={self.use_kl_term_z_phy} beta_max={self.beta_max_z_phy:.3f}")
    self.model._epoch_stash = {'mean': [], 'lnvar': []}
    result = _original_train_epoch(self, epoch)
    write_source_row(self, epoch, factor, result)
    self.model._epoch_stash = None
    return result


PhysVAETrainerSMPL._train_epoch = delayed_penalty_train_epoch    # in memory only; train_pila uses this class
PHYS_VAE_SMPL.draw = noise_scaled_draw
if __name__ == '__main__':
    runpy.run_path(os.path.join(PROJECT_ROOT, 'train_pila.py'), run_name='__main__')
