# Description: Per-epoch source logging + noise-scaled z_phy draw (from legacy train_pila_delayedpen.py /
#              train_pila_logsources.py, logic copied verbatim). Installed on the MODEL / TRAINER INSTANCE by
#              pila.extensions.install_extensions instead of patching the classes at import time.
#              Writes one row per epoch to <run models dir>/per_epoch_sources.csv: schedule factor, train rec loss,
#              per-slot median xcen/ycen/depth (km) and dV (m^3) of the encoder MEAN over that epoch's training
#              batches, and the median u-space sampling std (sigma_u).
# Date:        2026-11-23
import csv
import os
import types

import torch

from pila.utils.utils import draw_normal

PARAMETERS_PER_SOURCE = 4      # MultiMogi block order: xcen, ycen, d, dV


def _noise_scaled_draw(self, z_phy_stat, z_aux_stat, hard_z_phy=False, hard_z_aux=False):
    """
    PHYS_VAE_SMPL.draw with the z_phy noise scaled by self._z_phy_noise_scale (default 1 = unchanged):
    u = mean + scale*(sample - mean). Deterministic (hard) draws and z_aux are untouched.
    While training, also stashes the encoder mean and lnvar (detached) for the per-epoch source CSV.
    """
    original_draw = self._pila_original_draw
    stash = getattr(self, '_epoch_stash', None)
    if stash is not None and self.training and not self.no_phy:
        stash['mean'].append(z_phy_stat['mean'].detach().reshape(-1, z_phy_stat['mean'].shape[-1]).cpu())
        stash['lnvar'].append(z_phy_stat['lnvar'].detach().reshape(-1, z_phy_stat['lnvar'].shape[-1]).cpu())
    scale = getattr(self, '_z_phy_noise_scale', 1.0)
    if hard_z_phy or scale >= 1.0 or self.no_phy:
        return original_draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
    # z_aux exactly as the original; z_phy sampled here in u-space (no lossy sigmoid->logit round trip at the rails)
    _z_phy_mean_only, z_aux = original_draw(z_phy_stat, z_aux_stat, hard_z_phy=True, hard_z_aux=hard_z_aux)
    u_sampled = draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])        # same clamp/reparam as the original
    u_scaled = z_phy_stat['mean'] + scale * (u_sampled - z_phy_stat['mean'])
    return torch.sigmoid(u_scaled), z_aux


def install_draw_wrapper(model):
    """Wrap model.draw (instance only) with the stash + noise-scale draw. Idempotent."""
    if getattr(model, '_pila_original_draw', None) is None:
        model._pila_original_draw = model.draw                     # bound original method
        model.draw = types.MethodType(_noise_scaled_draw, model)


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
        # rescale maps z in (0,1) -> physical (m, m^3), exactly as dump_fit.py does
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


def make_logging_epoch(trainer, inner_epoch):
    """Unchanged epoch plus the per-epoch source row (factor recorded as 1 = full standard weights)."""
    def logging_train_epoch(epoch):
        trainer.model._epoch_stash = {'mean': [], 'lnvar': []}
        result = inner_epoch(epoch)
        write_source_row(trainer, epoch, 1.0, result)
        trainer.model._epoch_stash = None
        return result
    return logging_train_epoch
