#!/usr/bin/env python3
# Usage:       python run_iterative_prior_refit.py \
#                  --config configs/phys_smpl/PILA_Mogi_C_iterprior.json \
#                  --resume saved/mogi/<EXP>/<RUN>/models/model_best.pth \
#                  [--max-rounds 3] [-d 0]
# Description: Sequential "previous-epoch-as-prior" encoder retraining for PILA.
#              This is the TRAINING-TIME analogue of the inference-time temporal
#              blend in test_pila_mogi_temporal.py: instead of blending epoch t
#              with epoch t-1 in u-space at inference, we set the u_phy KL prior
#              MEAN to epoch (t-1)'s inferred u-space guess and RETRAIN the encoder
#              on the full time series (warm-started from the previous round), then
#              advance. One retraining round per epoch. The prior VARIANCE (u-space
#              prior_std) is the temporal-smoothing strength: small std => strong
#              pull to the previous epoch => smoother parameter track.
# Date:        2026-07-01
#
# Scientific assumptions / caveats (see plan file for the full list):
#   * Prior source is the RAW per-epoch encoder mean (no blending/refit).
#   * Warm-start accumulates state across rounds -> order dependence / drift; a
#     poor early prior can bias later rounds. Use --fresh-each-round to ablate.
#   * Too-tight prior_std over-smooths genuine episodic change (e.g. Mogi dV);
#     mirror the loc-high / amp-low intuition via --loc-std / --amp-std.
#   * LOS convention: positive = motion toward satellite. GPS/MintPy metres are
#     converted to the decoder's mm at load; parameters are rescaled from the
#     bounded (0,1) u-space via configs/<physics>_paras.json.

import argparse
import copy
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model import PHYS_VAE_SMPL
from parse_config import ConfigParser
from trainer import PhysVAETrainerSMPL
from utils import prepare_device, read_json
from train_pila import _propagate_insar_uq, set_seed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Per-parameter grouping for the u-space prior variance (smoothing strength) ---
# Mirrors synthetic/evaluate.py::LOC_GROUPS. Location/geometry parameters evolve
# SLOWLY between epochs (safe to smooth hard -> small std). Amplitude parameters
# (dV / opening) are genuinely time-varying (episodic inflation/deflation), so they
# want a LOOSER prior (larger std) that lets the encoder track fast change per epoch.
LOC_NAMES = {'xcen', 'ycen', 'd', 'depth', 'radius',
             'xoff', 'yoff', 'strike', 'dip', 'length', 'width'}
AMP_NAMES = {'dV', 'opening'}


# --- Prior-variance construction ---

def build_prior_lnvar(attrs, prior_std_default, loc_std, amp_std):
    """
    Build the per-parameter u-space prior log-variance vector, in `attrs` order.

    Args:
        attrs             : physical-parameter names in z_phy/rescale column order
                            (e.g. ['xcen','ycen','d','dV'] for Mogi).
        prior_std_default : u-space prior std applied to every parameter unless a
                            group override is given. Smaller => tighter prior =>
                            more temporal smoothing.
        loc_std           : override std for the location/geometry group (or None).
        amp_std           : override std for the amplitude group (or None).

    Returns:
        np.ndarray shape (len(attrs),), dtype float32, of prior log-variances
        (lnvar = 2 * ln(std)).
    """
    std_vec = np.full(len(attrs), float(prior_std_default), dtype=np.float64)
    for i, name in enumerate(attrs):
        if loc_std is not None and name in LOC_NAMES:
            std_vec[i] = float(loc_std)
        if amp_std is not None and name in AMP_NAMES:
            std_vec[i] = float(amp_std)
    # Guard against a zero/negative std that would make ln blow up.
    std_vec = np.clip(std_vec, 1e-3, None)
    lnvar = 2.0 * np.log(std_vec)
    # kldiv_normal_normal (utils/utils.py) clamps BOTH lnvars to [-9, 5]; a requested
    # std outside exp([-4.5, 2.5]) = [~0.011, ~3.49] would be silently floored/capped
    # there. Clamp here too so the printed prior std matches what training actually uses.
    lnvar_clamped = np.clip(lnvar, -9.0, 5.0)
    if not np.allclose(lnvar, lnvar_clamped):
        print(f"  WARNING: prior std {np.exp(0.5*lnvar).round(4).tolist()} clamped to "
              f"{np.exp(0.5*lnvar_clamped).round(4).tolist()} to match the KL lnvar range [-9,5].")
    return lnvar_clamped.astype(np.float32)


# --- Single-epoch inference (raw encoder, u-space) ---

def load_epoch_series(config_dict, device, input_dim, time_feat_dim):
    """
    Load the per-epoch displacement series in CHRONOLOGICAL order for either the GPS
    CSV path or the InSAR h5 path. Each returned epoch is a dict with pre-built,
    device-resident tensors so the sequential loop can encode on demand.

    Returns (epochs, dates) where epochs is a list of
        {'data': (1, input_dim) tensor, 'tfeat': (1, time_feat_dim) tensor|None, 'date': str}
    """
    dl_type = config_dict['data_loader']['type']
    is_insar = ('InSAR' in dl_type) or ('insar' in config_dict['arch']['args'])
    encoder_type = config_dict['arch']['args'].get('encoder_type', 'mlp')
    if is_insar and encoder_type == 'cnn':
        raise NotImplementedError(
            "run_iterative_prior_refit currently supports the MLP/point LOS path only; "
            "the CNN image path needs different per-epoch handling.")

    epochs, dates = [], []
    if is_insar:
        # Build the test loader exactly as synthetic/evaluate.py does (shuffle=False so
        # epoch order matches the truth sidecar). Each sample = one epoch's LOS vector.
        import data_loader.data_loaders as module_data
        dl = getattr(module_data, config_dict['data_loader']['type_test'])(
            insar=config_dict['data_loader']['data_dir_test'],
            batch_size=512, shuffle=False, validation_split=0.0, num_workers=0,
            with_const=config_dict['data_loader']['args'].get('with_const', False))
        for batch in dl:
            disp = batch['displacement']                 # (b, N)
            tfeat = batch.get('time_feats', None)         # (b, 4) or None
            bdates = list(batch['date'])
            for i in range(disp.shape[0]):
                epochs.append({
                    'data': disp[i:i + 1].to(device),
                    'tfeat': (tfeat[i:i + 1, :time_feat_dim].to(device)
                              if tfeat is not None else None),
                    'date': bdates[i]})
                dates.append(bdates[i])
    else:
        # GPS CSV path: first input_dim columns are the standardized displacement, a
        # 'date' column supplies the time features (mirrors test_pila_mogi_temporal.py).
        from datasets.displacementGPS import time_feats as compute_time_feats
        data_dir = config_dict['data_loader']['data_dir_test']
        df = pd.read_csv(os.path.join(CURRENT_DIR, data_dir)).sort_values(by='date').reset_index(drop=True)
        for _, row in df.iterrows():
            tf = compute_time_feats(row['date'], time_feat_dim=time_feat_dim)
            epochs.append({
                'data': torch.tensor(row.iloc[:input_dim].values.astype('float32')).unsqueeze(0).to(device),
                'tfeat': torch.tensor(tf).unsqueeze(0).to(device),
                'date': row['date']})
            dates.append(row['date'])
    return epochs, dates


def infer_epoch_uspace(model, epoch_item):
    """
    Encode ONE epoch's displacement field and return the raw encoder u-space guess.

    No temporal blending and no sampling: we take the encoder's posterior MEAN in
    u-space (this is exactly the 'raw' branch of test_pila_mogi_temporal.py).

    Args:
        epoch_item : dict from load_epoch_series with 'data' (1, input_dim) and
                     'tfeat' (1, time_feat_dim) or None, already on the model's device.

    Returns:
        (u_mean, u_lnvar, phys_dict) where u_mean/u_lnvar are np.ndarray (dim_z_phy,)
        in u-space and phys_dict maps each attr name -> rescaled physical value.
    """
    model.eval()
    with torch.no_grad():
        z_phy_stat, _z_aux_stat = model.encode(epoch_item['data'], epoch_item['tfeat'])
        u_mean = z_phy_stat['mean'].squeeze(0)      # (dim_z_phy,) u-space mean
        u_lnvar = z_phy_stat['lnvar'].squeeze(0)    # (dim_z_phy,) u-space log-var
        # Rescale to physical units for reporting: sigmoid(u) -> (0,1) -> [min,max].
        z01 = torch.sigmoid(u_mean).unsqueeze(0)    # (1, dim_z_phy)
        resc = model.physics_model.rescale(z01)     # dict attr -> (1,) tensor

    attrs = list(model.physics_model.z_phy_ranges.keys())
    phys_dict = {name: float(resc[name].squeeze().cpu()) for name in attrs}
    return u_mean.cpu().numpy(), u_lnvar.cpu().numpy(), phys_dict


# --- Checkpoint I/O ---

def load_checkpoint_into_model(model, checkpoint_path, device):
    """Load a PILA checkpoint's weights (+ tau/r) into an existing model, in place."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tau_r_values = checkpoint.get('tau_r_values', None)
    if tau_r_values is not None and not model.no_phy:
        model.dec.set_tau_r_from_checkpoint(tau_r_values)
    return model


# --- One warm-start retraining round (reuses the real PILA trainer) ---

def train_one_round(model, base_config_dict, run_id, device):
    """
    Fine-tune `model` (warm weights, informed prior already set) for one round,
    reusing PhysVAETrainerSMPL exactly. The informed prior is carried on the model
    object, so no prior file is needed on this in-process path.

    Returns (checkpoint_path, train_sec, n_samples, n_train_epochs). NOTE: the base
    trainer writes model_best.pth as the FINAL-epoch weights at end of train() (it
    overwrites the monitored-best on normal exit), so the warm-start chain carries
    the last epoch of each round, not the val-best epoch. See base/base_trainer.py.
    """
    # Fresh ConfigParser per round -> deterministic, non-colliding output dir.
    # resume=None: we warm-start via the persistent model object, NOT the trainer's
    # own resume path (which would also restore optimizer/epoch state).
    round_cfg = copy.deepcopy(base_config_dict)  # isolate per-round mutations
    round_cfg['arch']['phys_vae']['informed_prior']['enabled'] = True  # document intent
    _propagate_insar_uq(round_cfg)  # no-op for GPS; keeps InSAR blocks consistent

    config = ConfigParser(round_cfg, resume=None, run_id=run_id)

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['data_dir_valid'],
        batch_size=64, shuffle=True, validation_split=0.0, num_workers=2,
        with_const=config['data_loader']['args'].get('with_const', False),
    )

    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = None
    if 'lr_scheduler' in config.config:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = PhysVAETrainerSMPL(
        model, criterion, metrics, optimizer, config=config, device=device,
        data_loader=data_loader, valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )
    n_samples = len(data_loader.dataset)         # time-series epochs used as training samples
    n_train_epochs = config['trainer']['epochs']  # gradient epochs this round
    t_train0 = time.time()
    trainer.train()
    train_sec = time.time() - t_train0

    best_path = os.path.join(str(config.save_dir), 'model_best.pth')
    return best_path, train_sec, n_samples, n_train_epochs


def main():
    parser = argparse.ArgumentParser(description='PILA sequential previous-epoch-as-prior retraining')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='iterprior config template (JSON), e.g. configs/phys_smpl/PILA_Mogi_C_iterprior.json')
    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='base fully-trained checkpoint (E0) to bootstrap and warm-start from')
    parser.add_argument('-d', '--device', default=None, type=str, help='GPU index (default: auto)')
    parser.add_argument('--max-rounds', type=int, default=None,
                        help='cap on number of epochs/rounds to process (default: all epochs)')
    parser.add_argument('--fresh-each-round', action='store_true',
                        help='ablation: reload the base checkpoint before each round (no warm-start drift)')
    parser.add_argument('--no-refit', action='store_true',
                        help='ablation: NO temporal prior at all. Pure independent per-epoch '
                             'inference with the fixed E0 (no retraining, no informed prior, no '
                             'weight carryover). The true amortized baseline for A/B comparison.')
    parser.add_argument('--settle', action='store_true',
                        help='burn-in: run one throwaway retraining round on epoch 0 before the '
                             'main loop, then re-infer epoch 0 with the settled weights. Absorbs '
                             'the one-time E0->single-epoch-refit weight adaptation into epoch 0 '
                             'so it does not contaminate epoch 1 (removes the round-1 transient). '
                             'Ignored with --no-refit.')
    parser.add_argument('--out', type=str, default=None,
                        help='output CSV of the per-epoch parameter series (default: next to --resume)')
    args = parser.parse_args()

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device, _ = prepare_device(1)

    # The PILA trainer calls wandb.log() unconditionally; init once in disabled mode
    # (matches train_pila.py) so every round's trainer.train() is a no-op to wandb.
    wandb.init(project="PILA", name="iterprior", mode="disabled")

    print("[1/6] Loading config + base checkpoint...")
    base_config_dict = read_json(os.path.join(CURRENT_DIR, args.config))
    run_seed = base_config_dict.get('seed', 123)
    set_seed(run_seed)
    print(f"  RNG seed = {run_seed}")

    # Read informed-prior policy (consumed HERE by the driver, not by the model).
    informed_cfg = base_config_dict['arch']['phys_vae'].get('informed_prior', {})
    prior_std_default = informed_cfg.get('prior_std', 1.0)
    loc_std = informed_cfg.get('loc_std', None)
    amp_std = informed_cfg.get('amp_std', None)
    use_encoder_var = informed_cfg.get('use_encoder_var', False)
    input_dim = base_config_dict['arch']['args']['input_dim']
    time_feat_dim = base_config_dict['arch']['args'].get('time_feat_dim', 4)

    base_ckpt = os.path.join(CURRENT_DIR, args.resume) if not os.path.isabs(args.resume) else args.resume
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"base checkpoint not found: {base_ckpt}")

    print("[2/6] Loading epoch time series (chronological)...")
    epochs, all_dates = load_epoch_series(base_config_dict, device, input_dim, time_feat_dim)
    n_total_epochs = len(epochs)
    num_epochs = n_total_epochs if args.max_rounds is None else min(n_total_epochs, args.max_rounds)
    print(f"  {n_total_epochs} epochs available, processing {num_epochs}: "
          f"{all_dates[0]} .. {all_dates[num_epochs-1]}")

    print("[3/6] Building model + bootstrap encoder (E0)...")
    model = PHYS_VAE_SMPL(base_config_dict).to(device)
    load_checkpoint_into_model(model, base_ckpt, device)
    attrs = list(model.physics_model.z_phy_ranges.keys())
    print(f"  Physics = {base_config_dict['arch']['args']['physics']}, params (u-space order) = {attrs}")

    fixed_prior_lnvar = build_prior_lnvar(attrs, prior_std_default, loc_std, amp_std)
    print(f"  Fixed prior std per param = {np.exp(0.5 * fixed_prior_lnvar).round(3).tolist()} "
          f"(use_encoder_var={use_encoder_var})")

    # Floor for the encoder-variance prior path: without it a confident encoder
    # (tiny posterior lnvar) becomes a near-delta prior next round -> a self-tightening
    # ratchet toward variance collapse. Mirror the EMA path's variance floor idea.
    ENCVAR_LNVAR_FLOOR = float(2.0 * np.log(0.05))  # std >= 0.05 in u-space

    # --- Bootstrap: epoch 0 guess from the base encoder (seeds round-1's prior) ---
    t_infer0 = time.time()
    u_mean_prev, u_lnvar_prev, phys0 = infer_epoch_uspace(model, epochs[0])
    infer_sec0 = time.time() - t_infer0
    n_obs_points = input_dim  # observation channels per epoch (GPS components / LOS cells)
    records = [{'round': 0, 'date': all_dates[0], 'source': 'bootstrap',
                **{f'u_{a}': u_mean_prev[i] for i, a in enumerate(attrs)},
                **{a: phys0[a] for a in attrs}}]
    # Per-round timing + scale + drift statistics (written to a stats CSV; see below).
    stats = [{'round': 0, 'date': all_dates[0], 'train_sec': 0.0,
              'infer_sec': round(infer_sec0, 3), 'n_train_epochs': 0,
              'n_samples': 0, 'n_obs_points': n_obs_points,
              'u_step_l2': 0.0, 'max_abs_u_mean': float(np.max(np.abs(u_mean_prev)))}]
    print(f"  [round 0 / epoch 0] {all_dates[0]}  " +
          "  ".join(f"{a}={phys0[a]:.3f}" for a in attrs) +
          f"   (bootstrap infer {infer_sec0:.2f}s)")

    run_tag = f"iterprior_{datetime.now().strftime('%m%d_%H%M%S')}"

    # --- Optional burn-in: absorb the one-time E0->refit weight adaptation on epoch 0 ---
    # The FIRST retraining round from the pooled/KL-off base encoder E0 overshoots (biggest
    # single weight adjustment; shows up in the ill-conditioned dV-d direction). Left in the
    # main loop it contaminates epoch 1. Here we spend that adjustment on a throwaway round
    # over epoch 0, then RE-INFER epoch 0 with the settled weights and overwrite the bootstrap
    # record + seed. Round 1 then warm-starts from already-adapted weights and no longer jumps.
    if args.settle and not args.no_refit:
        prior_mean = torch.tensor(u_mean_prev, dtype=torch.float32)
        if use_encoder_var:
            prior_lnvar = torch.tensor(np.clip(u_lnvar_prev, ENCVAR_LNVAR_FLOOR, 5.0),
                                       dtype=torch.float32)
        else:
            prior_lnvar = torch.tensor(fixed_prior_lnvar, dtype=torch.float32)
        model.set_informed_prior(prior_mean, prior_lnvar)
        settle_best, settle_sec, _, settle_epochs = train_one_round(
            model, base_config_dict, f"{run_tag}/settle_epoch0", device)
        load_checkpoint_into_model(model, settle_best, device)
        # Warm the WEIGHTS only. We deliberately do NOT re-infer or overwrite epoch 0: the
        # settling round IS the first-retrain overshoot, so its epoch-0 estimate is the very
        # transient we are trying to remove. Keeping epoch 0's record as the clean pure-E0
        # bootstrap AND keeping the bootstrap u as round 1's prior seed (more accurate than
        # the settled overshoot) ELIMINATES the transient rather than relocating it to epoch 0.
        # Round 1 then warm-starts from these settled weights and no longer jumps.
        stats[0]['train_sec'] = round(settle_sec, 2)
        print(f"  [settle / epoch 0] warmed encoder ({settle_sec:.1f}s, {settle_epochs} ep); "
              f"epoch-0 record kept as bootstrap; round 1 warm-starts from settled weights")

    print(f"[4/6] Sequential retraining ({num_epochs-1} rounds), output tag = {run_tag}")
    print(f"  Scale: {n_obs_points} obs points/epoch, processing {num_epochs} epochs.")
    t_all = time.time()

    for t in range(1, num_epochs):
        # Prior for round t = epoch (t-1)'s guess. Mean = previous u-mean; variance =
        # encoder's own lnvar (if use_encoder_var, floored) else the fixed per-param std.
        u_mean_at_prior = u_mean_prev.copy()  # save for the drift metric below

        if args.no_refit:
            # No-temporal-prior baseline: skip the informed prior AND the retraining.
            # The model stays the fixed E0 for every epoch, so each epoch is inverted
            # independently (pure amortized inference). Nothing couples epoch t to t-1.
            train_sec, n_samples, n_train_epochs = 0.0, 0, 0
        else:
            # Prior for round t = epoch (t-1)'s guess. Mean = previous u-mean; variance =
            # encoder's own lnvar (if use_encoder_var, floored) else the fixed per-param std.
            prior_mean = torch.tensor(u_mean_prev, dtype=torch.float32)
            if use_encoder_var:
                enc_lnvar = np.clip(u_lnvar_prev, ENCVAR_LNVAR_FLOOR, 5.0)  # floor -> no ratchet
                prior_lnvar = torch.tensor(enc_lnvar, dtype=torch.float32)
            else:
                prior_lnvar = torch.tensor(fixed_prior_lnvar, dtype=torch.float32)

            # Saturated-bound diagnostic: |u|>6 => sigmoid within ~0.0025 of a bound, where
            # the KL pull fights vanishing reconstruction gradients (railing risk).
            if float(np.max(np.abs(u_mean_prev))) > 6.0:
                railed = [attrs[i] for i in range(len(attrs)) if abs(u_mean_prev[i]) > 6.0]
                print(f"  WARNING round {t}: prior mean near a bound for {railed} "
                      f"(|u|>6); informed prior may lock these (railing).")

            if args.fresh_each_round:
                load_checkpoint_into_model(model, base_ckpt, device)  # ablation: no weight drift

            model.set_informed_prior(prior_mean, prior_lnvar)  # activates use_informed_prior

            run_id = f"{run_tag}/round{t:04d}"
            best_path, train_sec, n_samples, n_train_epochs = train_one_round(
                model, base_config_dict, run_id, device)
            load_checkpoint_into_model(model, best_path, device)  # infer from the round's final weights

        # Infer epoch t with the retrained encoder -> seeds the NEXT round's prior.
        t_infer0 = time.time()
        u_mean_prev, u_lnvar_prev, phys_t = infer_epoch_uspace(model, epochs[t])
        infer_sec = time.time() - t_infer0

        # Drift metric: how far this epoch's u-guess moved from the prior mean it was
        # pulled toward. Large => encoder overrode the prior; ~0 => prior dominated.
        u_step_l2 = float(np.linalg.norm(u_mean_prev - u_mean_at_prior))

        records.append({'round': t, 'date': all_dates[t], 'source': 'retrained',
                        **{f'u_{a}': u_mean_prev[i] for i, a in enumerate(attrs)},
                        **{a: phys_t[a] for a in attrs}})
        stats.append({'round': t, 'date': all_dates[t],
                      'train_sec': round(train_sec, 2), 'infer_sec': round(infer_sec, 3),
                      'n_train_epochs': n_train_epochs, 'n_samples': n_samples,
                      'n_obs_points': n_obs_points, 'u_step_l2': round(u_step_l2, 4),
                      'max_abs_u_mean': round(float(np.max(np.abs(u_mean_prev))), 3)})
        print(f"  [round {t} / epoch {t}] {all_dates[t]}  " +
              "  ".join(f"{a}={phys_t[a]:.3f}" for a in attrs) +
              f"   (train {train_sec:.1f}s, infer {infer_sec:.2f}s, u_step={u_step_l2:.3f})")

    total_sec = time.time() - t_all
    n_rounds = max(num_epochs - 1, 1)
    total_train_sec = sum(s['train_sec'] for s in stats)
    n_train_samples = stats[-1]['n_samples']  # training-split size (NOT the inferred-epoch count)
    print(f"  All {n_rounds} rounds done in {total_sec:.1f}s "
          f"(train {total_train_sec:.1f}s, mean {total_train_sec/n_rounds:.1f}s/round; "
          f"per round = {n_obs_points} obs pts x {n_train_samples} training epochs x "
          f"{stats[-1]['n_train_epochs']} grad-epochs).")

    print("[5/6] Writing per-epoch parameter series + timing/scale stats...")
    out_csv = args.out or os.path.join(os.path.dirname(base_ckpt),
                                       f'iterprior_series_{run_tag}.csv')
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"  Series: {out_csv}  (shape={len(records)}x{len(records[0])})")

    # Timing/scale/drift stats: per-round wall time, training-set size, obs points,
    # gradient epochs, and the u-space drift metric. Lets us size the moving-Mogi job.
    stats_csv = os.path.splitext(out_csv)[0].replace('iterprior_series_', 'iterprior_stats_') + '.csv'
    if 'iterprior_stats_' not in stats_csv:  # --out without the series prefix
        stats_csv = os.path.splitext(out_csv)[0] + '_stats.csv'
    pd.DataFrame(stats).to_csv(stats_csv, index=False)
    print(f"  Stats:  {stats_csv}")

    print(f"[6/6] Done. Parameter time series written for {num_epochs} epochs "
          f"({n_rounds} retraining rounds).")
    wandb.finish()


if __name__ == '__main__':
    main()
