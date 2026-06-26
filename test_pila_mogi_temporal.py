#!/usr/bin/env python3
# Usage:       python test_pila_mogi_temporal.py --resume saved/mogi/.../model_best.pth \
#                  --alpha 0.3 [--alpha_spatial 0.4] [--alpha_dv 0.1]
# Description: Temporal inference for Mogi inversion using PILA. Processes epochs
#              sequentially in chronological order. Each epoch's encoder output is
#              blended with the previous epoch's result via a weighted average in
#              u-space (logit-space). This constrains source parameters to evolve
#              smoothly in time without relying on untrained encoder variances.
# Date:        2026-06-25

import argparse
import os
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model import PHYS_VAE_SMPL
from utils import read_json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Temporal blending utility ---

def temporal_blend(mu_encoder, mu_prior, alpha):
    """
    Blend encoder output with previous epoch's result via weighted average in u-space.

    mu_combined = (1 - alpha) * mu_encoder + alpha * mu_prior

    Args:
        mu_encoder: Current epoch's encoder mean in u-space, shape (1, dim_z_phy)
        mu_prior: Previous epoch's result in u-space (logit of prev z_phy), shape (1, dim_z_phy)
        alpha: Blending weight per dimension, shape (dim_z_phy,)
               0 = ignore prior (original behavior)
               1 = ignore encoder (fully trust previous epoch)

    Returns:
        mu_combined: Blended mean in u-space, shape (1, dim_z_phy)
    """
    # alpha shape (dim_z_phy,) broadcasts against (1, dim_z_phy)
    mu_combined = (1.0 - alpha) * mu_encoder + alpha * mu_prior
    return mu_combined


def build_alpha(dim_z_phy, alpha_default, alpha_spatial, alpha_dv, device):
    """
    Build per-dimension alpha vector for Mogi parameters [xcen, ycen, d, dV].

    Physical reasoning:
        - Source position (xcen, ycen, d) changes slowly between epochs.
          Higher alpha = more temporal smoothing = more physical.
        - Volume change (dV) can vary rapidly (episodic inflation/deflation).
          Lower alpha = let the encoder track fast changes.

    Args:
        dim_z_phy: Number of physics dimensions (4 for Mogi)
        alpha_default: Default alpha for all dimensions
        alpha_spatial: Override alpha for xcen, ycen, d (first 3 dims)
        alpha_dv: Override alpha for dV (4th dim)
        device: torch device

    Returns:
        alpha: Per-dimension blending weights, shape (dim_z_phy,)
    """
    alpha = torch.full((dim_z_phy,), alpha_default, device=device)

    # Override spatial dimensions (xcen, ycen, d) if specified
    if alpha_spatial is not None and dim_z_phy >= 3:
        alpha[:3] = alpha_spatial

    # Override dV dimension if specified
    if alpha_dv is not None and dim_z_phy >= 4:
        alpha[3] = alpha_dv

    return alpha


def setup_test_logging(checkpoint_path):
    """
    Setup logging to save test logs in the checkpoint's directory.
    """
    checkpoint_path = Path(checkpoint_path)
    experiment_dir = checkpoint_path.parent.parent
    log_dir = experiment_dir / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    log_filename = f'info_test_temporal_{timestamp}.log'
    log_file_path = log_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger('test_temporal')
    logger.info(f"Temporal test logging setup. Log file: {log_file_path}")
    return logger


def main(config, args):
    logger = setup_test_logging(args.resume)

    # --- Parse temporal inference hyperparameters ---
    alpha_default = args.alpha
    alpha_spatial = args.alpha_spatial
    alpha_dv = args.alpha_dv

    logger.info(f"[1/5] Temporal inference configuration:")
    logger.info(f"  Default alpha: {alpha_default}")
    if alpha_spatial is not None:
        logger.info(f"  Spatial (xcen, ycen, d) alpha: {alpha_spatial}")
    if alpha_dv is not None:
        logger.info(f"  Volume change (dV) alpha: {alpha_dv}")

    # --- Load data (full test set, will sort by date) ---
    logger.info("[2/5] Loading test data...")
    t0 = time.time()

    data_dir = config.config['data_loader']['data_dir_test']
    data_df = pd.read_csv(os.path.join(CURRENT_DIR, data_dir))

    # Sort by date for sequential processing
    data_df = data_df.sort_values(by='date').reset_index(drop=True)
    num_epochs = len(data_df)
    logger.info(f"  Loaded {num_epochs} epochs, date range: {data_df['date'].iloc[0]} to {data_df['date'].iloc[-1]}")
    logger.info(f"  Done in {time.time() - t0:.1f}s")

    # --- Build model and load checkpoint ---
    logger.info("[3/5] Loading model and checkpoint...")
    t0 = time.time()

    model = PHYS_VAE_SMPL(config.config)
    checkpoint = torch.load(os.path.join(CURRENT_DIR, config.resume), weights_only=False)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    no_phy = config.config['arch']['phys_vae']['no_phy']
    dim_z_phy = config.config['arch']['phys_vae']['dim_z_phy']
    dim_z_aux = config.config['arch']['phys_vae']['dim_z_aux']

    # Load tau/r values for inference
    tau_r_values = checkpoint.get('tau_r_values', None)
    if tau_r_values is not None and not no_phy:
        model.dec.set_tau_r_from_checkpoint(tau_r_values)
        logger.info(f"  Loaded tau/r from checkpoint: tau={tau_r_values['tau']:.3f}, r={tau_r_values['r']:.3f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    logger.info(f"  Model loaded on {device}. Done in {time.time() - t0:.1f}s")

    # Determine hard_z settings from config (same logic as standard test)
    if 'use_kl_term' in config.config['trainer']['phys_vae']:
        use_kl_term_z_phy = config.config['trainer']['phys_vae']['use_kl_term']
        use_kl_term_z_aux = config.config['trainer']['phys_vae']['use_kl_term']
    else:
        use_kl_term_z_phy = config.config['trainer']['phys_vae'].get('use_kl_term_z_phy', False)
        use_kl_term_z_aux = config.config['trainer']['phys_vae'].get('use_kl_term_z_aux', False)

    hard_z_phy = not use_kl_term_z_phy
    hard_z_aux = not use_kl_term_z_aux

    # Build per-dimension alpha vector
    alpha = build_alpha(dim_z_phy, alpha_default, alpha_spatial, alpha_dv, device)
    logger.info(f"  Alpha per dim [xcen, ycen, d, dV]: {alpha.cpu().numpy()}")

    # --- Extract column names ---
    data_key = config.config['trainer']['input_key']
    input_const_keys = config.config['trainer'].get('input_const_keys', None)

    station_info = {}
    try:
        station_info = read_json(os.path.join(CURRENT_DIR, 'configs/station_info.json'))
    except Exception:
        logger.warning("Could not load station_info.json")

    GPS = []
    for direction in ['ux', 'uy', 'uz']:
        for station in station_info.keys():
            GPS.append(f'{direction}_{station}')

    ATTRS = ['xcen', 'ycen', 'd', 'dV']

    # --- Sequential temporal inference ---
    logger.info("[4/5] Running sequential temporal inference...")
    t0 = time.time()

    # Storage for results
    all_z_phy = []          # inferred z_phy in (0,1) for each epoch (with temporal blending)
    all_z_phy_raw = []      # z_phy from encoder alone (no temporal blending)
    all_z_aux = []          # auxiliary latent
    all_x_PB = []           # corrected output (physics + residual)
    all_x_P = []            # raw physics output
    all_targets = []        # observation targets
    all_dates = []          # epoch dates

    prev_z_phy = None  # previous epoch's result in (0,1) space

    with torch.no_grad():
        for epoch_idx in range(num_epochs):
            row = data_df.iloc[epoch_idx]

            # --- Prepare input observation (displacement vector) ---
            displacement = torch.tensor(
                row.iloc[:36].values.astype('float32')
            ).unsqueeze(0).to(device)  # shape (1, 36)

            # --- Prepare time features ---
            date_str = row['date']
            from datasets.displacementGPS import time_feats as compute_time_feats
            time_features = compute_time_feats(date_str, time_feat_dim=4)
            time_feats_tensor = torch.tensor(time_features).unsqueeze(0).to(device)  # shape (1, 4)

            # --- Step 1: Encode to get encoder mean in u-space ---
            z_phy_stat, z_aux_stat = model.encode(displacement, time_feats_tensor)
            # z_phy_stat['mean']: (1, 4) — encoder's best guess in u-space
            # z_phy_stat['lnvar']: (1, 4) — NOT used (untrained, KL was disabled)

            mu_encoder = z_phy_stat['mean']  # (1, 4) — what the encoder says for this epoch

            # Store raw encoder result (before blending) for comparison
            z_phy_raw = torch.sigmoid(mu_encoder.clone())  # (1, 4) in (0,1)

            # --- Step 2: Blend with previous epoch if available ---
            if prev_z_phy is not None and not no_phy:
                # Map previous epoch's z_phy (0,1) back to u-space
                eps = 1e-6
                prev_z_clamped = prev_z_phy.clamp(eps, 1.0 - eps)
                mu_prior = torch.logit(prev_z_clamped)  # (1, 4) in u-space

                # Weighted average: (1-alpha)*encoder + alpha*prior
                mu_blended = temporal_blend(mu_encoder, mu_prior, alpha)
            else:
                # First epoch: no prior available, use encoder as-is
                mu_blended = mu_encoder

            # --- Step 3: Draw z_phy from blended mean ---
            # Override z_phy_stat with blended mean (lnvar is unused with hard=True)
            z_phy_stat_blended = {
                'mean': mu_blended,
                'lnvar': z_phy_stat['lnvar']  # passed through but ignored by draw()
            }
            z_phy, z_aux = model.draw(z_phy_stat_blended, z_aux_stat,
                                      hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)

            # --- Step 4: Decode ---
            x_PB, x_P, y, delta, c = model.decode(
                z_phy, z_aux,
                epoch=0, epochs_pretrain=0,
                full=True, const=None,
                use_inference_values=True,
                detach_x_P_for_bias=model.detach_x_P_for_bias
            )

            # --- Store current z_phy as prior for next epoch ---
            prev_z_phy = z_phy.clone()

            # --- Accumulate results ---
            all_z_phy.append(z_phy.cpu())
            all_z_phy_raw.append(z_phy_raw.cpu())
            all_z_aux.append(z_aux.cpu())
            all_x_PB.append(x_PB.cpu())
            all_x_P.append(x_P.cpu())
            all_targets.append(displacement.cpu())
            all_dates.append(date_str)

            # Log progress every 100 epochs
            if (epoch_idx + 1) % 100 == 0 or epoch_idx == 0:
                z_vals = z_phy.cpu().numpy().flatten()
                z_raw_vals = z_phy_raw.cpu().numpy().flatten()
                logger.info(
                    f"  Epoch {epoch_idx + 1}/{num_epochs} [{date_str}] "
                    f"z_phy_blended=[{z_vals[0]:.3f}, {z_vals[1]:.3f}, {z_vals[2]:.3f}, {z_vals[3]:.3f}] "
                    f"z_phy_raw=[{z_raw_vals[0]:.3f}, {z_raw_vals[1]:.3f}, {z_raw_vals[2]:.3f}, {z_raw_vals[3]:.3f}]"
                )

    logger.info(f"  Sequential inference done in {time.time() - t0:.1f}s")

    # --- Concatenate and rescale results ---
    all_z_phy = torch.cat(all_z_phy, dim=0)           # (N, dim_z_phy)
    all_z_phy_raw = torch.cat(all_z_phy_raw, dim=0)   # (N, dim_z_phy)
    all_z_aux = torch.cat(all_z_aux, dim=0)            # (N, dim_z_aux)
    all_x_PB = torch.cat(all_x_PB, dim=0)             # (N, 36)
    all_x_P = torch.cat(all_x_P, dim=0)               # (N, 36)
    all_targets = torch.cat(all_targets, dim=0)        # (N, 36)

    # Rescale z_phy to physical units
    if not no_phy:
        latent_phy = model.physics_model.rescale(all_z_phy.to(device))
        latent = torch.stack([latent_phy[k] for k in latent_phy.keys()], dim=1).cpu()

        # Also rescale raw encoder results for comparison
        latent_phy_raw = model.physics_model.rescale(all_z_phy_raw.to(device))
        latent_raw = torch.stack([latent_phy_raw[k] for k in latent_phy_raw.keys()], dim=1).cpu()
    else:
        latent = all_z_aux
        latent_raw = all_z_aux

    # Compute reconstruction loss
    loss_fn = getattr(module_loss, config.config['loss_test'])
    total_loss = loss_fn(all_x_PB, all_targets).item()
    logger.info(f"  Total reconstruction loss (MSE): {total_loss:.6f}")

    # --- Save results ---
    logger.info("[5/5] Saving results...")

    # Build output DataFrame
    columns = []
    columns += ['output_' + b for b in GPS]
    columns += ['target_' + b for b in GPS]
    columns += ['latent_temporal_' + b for b in ATTRS]
    columns += ['latent_raw_' + b for b in ATTRS]

    output_data = torch.hstack((
        all_x_PB,
        all_targets,
        latent,
        latent_raw,
    ))

    if not no_phy:
        bias = all_x_PB - all_x_P
        columns += ['init_output_' + b for b in GPS]
        columns += ['bias_' + b for b in GPS]
        output_data = torch.hstack((output_data, all_x_P, bias))

        if dim_z_aux > 0:
            columns += ['latent_aux_' + str(b + 1) for b in range(dim_z_aux)]
            output_data = torch.hstack((output_data, all_z_aux))

    # Convert to DataFrame
    df = pd.DataFrame(columns=columns, data=output_data.numpy())
    df['date'] = all_dates

    # Add normalized z_phy for both temporal and raw
    for i, attr in enumerate(ATTRS):
        df[f'z_phy_temporal_{attr}'] = all_z_phy[:, i].numpy()
        df[f'z_phy_raw_{attr}'] = all_z_phy_raw[:, i].numpy()

    # Add alpha values used for this run
    df['alpha_xcen'] = alpha[0].item()
    df['alpha_ycen'] = alpha[1].item()
    df['alpha_d'] = alpha[2].item()
    df['alpha_dv'] = alpha[3].item()

    # Save CSV
    alpha_str = f"a{alpha_default}"
    if alpha_spatial is not None:
        alpha_str += f"_spatial{alpha_spatial}"
    if alpha_dv is not None:
        alpha_str += f"_dv{alpha_dv}"
    output_path = os.path.join(
        CURRENT_DIR,
        str(config.resume).split('.pth')[0] + f'_temporal_{alpha_str}.csv'
    )
    df.to_csv(output_path, index=False)
    logger.info(f"  Results saved to {output_path}")

    # --- Summary statistics ---
    logger.info("--- Temporal vs Raw comparison ---")
    logger.info(f"  Alpha: {alpha.cpu().numpy()}")
    for i, attr in enumerate(ATTRS):
        raw_std = latent_raw[:, i].std().item()
        temporal_std = latent[:, i].std().item()
        reduction_pct = (1.0 - temporal_std / max(raw_std, 1e-8)) * 100.0
        logger.info(
            f"  {attr}: raw_std={raw_std:.4f}, temporal_std={temporal_std:.4f}, "
            f"variance_reduction={reduction_pct:.1f}%"
        )

    logger.info("Done. Temporal inference complete.")


class TestConfigParser:
    """
    Custom config parser for testing that doesn't create new experiment directories.
    """
    def __init__(self, config_dict, resume_path):
        self._config = config_dict
        self.resume = resume_path

    @classmethod
    def from_args(cls, args):
        """Initialize from command line arguments."""
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            raise ValueError("Resume path must be specified for testing.")

        config = read_json(cfg_fname)
        return cls(config, args.resume)

    @property
    def config(self):
        return self._config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PILA Mogi temporal inference: sequential epoch processing '
                    'with weighted-average blending from previous epoch in u-space.'
    )
    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='GPU device index (default: auto)')

    # --- Temporal blending hyperparameters ---
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Default blending weight for all Mogi params. '
                             '0 = no temporal smoothing (original behavior). '
                             '1 = fully trust previous epoch. '
                             '(default: 0.3)')
    parser.add_argument('--alpha_spatial', type=float, default=None,
                        help='Override alpha for spatial params (xcen, ycen, d). '
                             'Source position changes slowly, so higher alpha is physical. '
                             '(default: uses --alpha)')
    parser.add_argument('--alpha_dv', type=float, default=None,
                        help='Override alpha for volume change (dV). '
                             'dV can vary rapidly, so lower alpha lets encoder track changes. '
                             '(default: uses --alpha)')

    args = parser.parse_args()
    config = TestConfigParser.from_args(args)
    main(config, args)
