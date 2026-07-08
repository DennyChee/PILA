# Training script for PILA (our method)
import argparse
import collections
import json
import os
import resource
import subprocess
import sys
import time
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model import PHYS_VAE_SMPL  # PILA model
from parse_config import ConfigParser
from trainer import PhysVAETrainerSMPL  # PILA trainer
from utils import prepare_device
import wandb

# Fix random seeds for reproducibility. SEED is the historical default (123); it can be
# overridden per run via the config key "seed" or the CLI flag --seed (see main()), which
# is how multi-start ensembles escape null-source local minima on noisy real scenes.
SEED = 123


def set_seed(seed):
    """Set torch + numpy RNG seeds (deterministic cuDNN) for a reproducible run."""
    torch.manual_seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(int(seed))


set_seed(SEED)  # default; main() re-seeds from config['seed'] if provided


def log_compute_usage(config, logger, data_loader, model, device, data_load_sec, train_sec):
    """
    Record compute/scalability stats for the run and write them to compute_usage.json.

    Captures the problem size (observation points/grid cells, time-series samples), the
    MintPy load+multilook time, the training time and throughput, and peak memory — so
    runs of different scene sizes / multilook factors can be compared for scalability.
    """
    args = config['arch']['args']
    n_train_epochs = config['trainer']['epochs']
    n_samples = len(data_loader.dataset)
    usage = {
        'device': str(device),
        'physics': args.get('physics'),
        'encoder_type': args.get('encoder_type', 'mlp'),
        'n_obs_points': args.get('input_dim'),     # LOS cells (MLP) or grid cells (CNN)
        'n_samples_epochs': n_samples,             # time-series epochs used as samples
        'n_train_epochs': n_train_epochs,
        'batch_size': config['data_loader']['args'].get('batch_size'),
        'model_params': int(sum(p.numel() for p in model.parameters())),
        'data_load_sec': round(data_load_sec, 2),  # MintPy read + multilook (+model build)
        'train_sec': round(train_sec, 2),
        'sec_per_train_epoch': round(train_sec / max(n_train_epochs, 1), 3),
        'samples_per_sec': round(n_samples * n_train_epochs / max(train_sec, 1e-9), 1),
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        'gpu_peak_mb': round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0,
    }
    # InSAR multilook geometry (memoized loader -> instant) for the scene/grid context.
    ins = args.get('insar')
    if ins is not None:
        try:
            from datasets.preprocessing.insar_mintpy import load_insar_mintpy
            d = load_insar_mintpy(
                ins['timeseries'], ins['geometry'], ins.get('mask'),
                ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 20),
                coh_valid_frac=ins.get('coh_valid_frac', 0.5), bbox=ins.get('bbox'),
                ref_date=ins.get('ref_date'), verbose=False)
            usage.update({'multilook': ins.get('multilook', 20),
                          'coarse_grid': list(d.mask_d.shape),
                          'coherent_cells': int(d.mask_d.sum()),
                          'bbox': ins.get('bbox')})
        except Exception as exc:  # telemetry must never crash training
            logger.warning(f"compute-usage: could not read InSAR geometry: {exc}")

    out_path = os.path.join(str(config.save_dir), 'compute_usage.json')
    with open(out_path, 'w') as f:
        json.dump(usage, f, indent=2)
    logger.info("=== Compute usage (scalability) ===")
    for key, val in usage.items():
        logger.info(f"  {key}: {val}")
    logger.info(f"  written to {out_path}")


def generate_insar_figures(config, logger):
    """
    End-of-run verification figures for InSAR inversions: observed LOS, model fit,
    and residual maps + a 1:1 scatter, for visual inspection.

    Delegates to the standalone plot_insar_results.py (the tested MLP/point LOS
    plotting path) as an isolated subprocess so a plotting failure can never fail
    an otherwise-completed training run. Only applies to the h5-native InSAR LOS
    MLP path (physics '*_LOS', encoder_type != 'cnn', with an 'insar' block); the
    CNN/image path is skipped (different I/O) and GPS configs have no insar field.
    """
    args = config['arch']['args']
    physics = args.get('physics', '')
    encoder_type = args.get('encoder_type', 'mlp')
    if not (physics.endswith('_LOS') and encoder_type != 'cnn' and 'insar' in args):
        logger.info(f"Skipping auto figures (physics={physics}, encoder_type={encoder_type}): "
                    f"plot_insar_results.py only supports the InSAR LOS MLP path.")
        return

    # config.save_dir is the run's models/ dir; model_best.pth is written there.
    resume_path = os.path.join(str(config.save_dir), 'model_best.pth')
    if not os.path.exists(resume_path):
        logger.warning(f"Auto figures: model_best.pth not found at {resume_path}; skipping.")
        return

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_insar_results.py')
    cmd = [sys.executable, script, '--resume', resume_path]
    logger.info(f"Generating verification figures: {' '.join(cmd)}")
    try:
        # Plotting must never crash a finished run; capture and log on failure only.
        subprocess.run(cmd, check=True, cwd=os.path.dirname(script))
        figures_dir = os.path.join(os.path.dirname(str(config.save_dir)), 'figures')
        logger.info(f"Verification figures written to {figures_dir}")
    except subprocess.CalledProcessError as exc:
        logger.warning(f"Auto figure generation failed (exit {exc.returncode}); "
                       f"run plot_insar_results.py -r {resume_path} manually.")


def _propagate_insar_uq(cfg):
    """
    Mirror the UQ/decimation keys (multilook, stride, offset) from the canonical
    arch.args.insar block into every OTHER insar block in the config (the data_loader
    train block and the valid/test data_dir blocks). PILA derives points from each block
    independently via the memoized loader, so all blocks MUST carry identical
    multilook/stride/offset or the dataset and decoder would invert different subsets.

    Mutates `cfg` (a plain dict) in place. No-op when there is no arch.args.insar block
    (e.g. GNSS runs) or when none of the keys are set.
    """
    arch_ins = cfg.get('arch', {}).get('args', {}).get('insar')
    if not arch_ins:
        return
    # ref_date MUST propagate too: if the canonical block re-references to a chosen
    # epoch but the train/valid/test blocks don't, the dataset and decoder would
    # invert different temporal baselines for the same nominal epoch.
    uq_keys = {k: arch_ins[k] for k in
               ('multilook', 'stride', 'offset', 'ref_date',
                'bootstrap_k', 'bootstrap_block', 'bootstrap_seed') if k in arch_ins}
    if not uq_keys:
        return
    dl = cfg.get('data_loader', {})
    # Two block shapes exist: data_loader.args wraps the insar dict under an 'insar' key,
    # whereas data_dir_valid / data_dir_test ARE the insar dict directly (they hold
    # 'timeseries'/'geometry' at top level and are passed as the loader's `insar` arg).
    candidates = [dl.get('args', {}), dl.get('data_dir_valid', {}), dl.get('data_dir_test', {})]
    for block in candidates:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get('insar'), dict):       # wrapped: ...args.insar
            block['insar'].update(uq_keys)
        elif 'timeseries' in block:                    # bare insar dict (valid/test)
            block.update(uq_keys)


def main(config):
    logger = config.get_logger('train')

    # Re-seed from the config (set via --seed / config key "seed"); default = SEED (123).
    # The module-level set_seed(SEED) above ran at import with the default; this lets a
    # multi-start ensemble vary the RNG init per run to escape null-source local minima.
    run_seed = config.config.get('seed', SEED)
    set_seed(run_seed)
    logger.info(f"RNG seed = {run_seed}")

    # Keep all insar blocks consistent before any loader/model reads them (UQ decimation).
    _propagate_insar_uq(config.config)
    ins_dbg = config.config.get('arch', {}).get('args', {}).get('insar', {})
    print(f"  InSAR UQ config: multilook={ins_dbg.get('multilook', 1)}, "
          f"stride={ins_dbg.get('stride', 1)}, offset={ins_dbg.get('offset', 0)}")

    # Setup data_loader instances (time the MintPy read + multilook for h5 datasets)
    t_data0 = time.time()
    data_loader = config.init_obj('data_loader', module_data)

    valid_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['data_dir_valid'],
        batch_size=64,
        shuffle=True,
        validation_split=0.0,
        num_workers=2,
        with_const=config['data_loader']['args']['with_const'] if 'with_const' in config['data_loader']['args'] else False
    )

    data_load_sec = time.time() - t_data0

    # Build model architecture and log
    model = PHYS_VAE_SMPL(config)
    logger.info(model)

    # Prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles for loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # Build optimizer and learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # CHANGED: make scheduler optional
    lr_scheduler = None
    if 'lr_scheduler' in config.config:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Initialize Phys-VAE Trainer
    trainer = PhysVAETrainerSMPL(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler
    )

    t_train0 = time.time()
    trainer.train()
    train_sec = time.time() - t_train0

    # Record compute/scalability usage for this run.
    log_compute_usage(config, logger, data_loader, model, device, data_load_sec, train_sec)

    # Generate observed / model-fit / residual figures for visual inspection.
    generate_insar_figures(config, logger)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PILA training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
        # PhysVAE specific arguments TODO a more elegant way to do this, currently it is hardcoded in parse_config.py
        CustomArgs(['--use_kl_term_z_phy'], type=str,
                   target='trainer;phys_vae;use_kl_term_z_phy'),
        CustomArgs(['--beta_max_z_phy'], type=float,
                   target='trainer;phys_vae;beta_max_z_phy'),
        CustomArgs(['--use_kl_term_z_aux'], type=str,
                   target='trainer;phys_vae;use_kl_term_z_aux'),
        CustomArgs(['--beta_max_z_aux'], type=float,
                   target='trainer;phys_vae;beta_max_z_aux'),
        CustomArgs(['--kl_warmup_epochs'], type=int,
                   target='trainer;phys_vae;kl_warmup_epochs'),
        CustomArgs(['--epochs_pretrain'], type=int,
                   target='trainer;phys_vae;epochs_pretrain'),
        CustomArgs(['--tau_warmup_epochs'], type=int,
                   target='arch;phys_vae;tau_warmup_epochs'),
        CustomArgs(['--tau_init'], type=float,
                   target='arch;phys_vae;tau_init'),
        CustomArgs(['--r_warmup_epochs'], type=int,
                   target='arch;phys_vae;r_warmup_epochs'),
        CustomArgs(['--r_init'], type=float,
                   target='arch;phys_vae;r_init'),
        CustomArgs(['--dim_z_aux'], type=int,
                   target='arch;phys_vae;dim_z_aux'),
        CustomArgs(['--residual_rank'], type=int,
                   target='arch;phys_vae;residual_rank'),
        CustomArgs(['--detach_x_P_for_bias'], type=str,
                   target='arch;phys_vae;detach_x_P_for_bias'),
        # NEW: Capacity control arguments
        CustomArgs(['--use_capacity_control'], type=str,
                   target='trainer;phys_vae;use_capacity_control'),
        CustomArgs(['--C_max'], type=float,
                   target='trainer;phys_vae;C_max'),
        CustomArgs(['--C_gamma'], type=float,
                   target='trainer;phys_vae;C_gamma'),
        CustomArgs(['--beta_aux'], type=float,
                   target='trainer;phys_vae;beta_aux'),
        CustomArgs(['--coeff_penalty_weight'], type=float,
            target='trainer;phys_vae;coeff_penalty_weight'),
        CustomArgs(['--delta_penalty_weight'], type=float,
            target='trainer;phys_vae;delta_penalty_weight'),
        # NEW: Edge penalty arguments
        CustomArgs(['--edge_penalty_weight'], type=float,
                   target='trainer;phys_vae;edge_penalty_weight'),
        CustomArgs(['--edge_penalty_power'], type=float,
                   target='trainer;phys_vae;edge_penalty_power'),
        # NEW: EMA prior arguments
        CustomArgs(['--use_ema_prior'], type=str,
                   target='trainer;phys_vae;use_ema_prior'),
        CustomArgs(['--ema_momentum'], type=float,
                   target='trainer;phys_vae;ema_momentum'),
        # NEW: data_loader type
        CustomArgs(['--data_loader_type'], type=str,
                   target='data_loader;type'),
        # NEW: Time feature arguments
        CustomArgs(['--time_feat_dim'], type=int,
                   target='arch;args;time_feat_dim'),
        CustomArgs(['--use_time_in_residual'], type=str,
                   target='arch;args;use_time_in_residual'),
        CustomArgs(['--temporal_smoothness_weight'], type=float,
                   target='trainer;phys_vae;temporal_smoothness_weight'),
        CustomArgs(['--loss'], type=str,
                   target='loss'),
        # UQ strided-decimation ensemble: multilook (default 1 = full res), stride n, and
        # this member's phase offset. These set the CANONICAL arch.args.insar block;
        # _propagate_insar_uq() then mirrors them into the data_loader / valid / test
        # insar blocks so every path inverts the SAME point subset.
        CustomArgs(['--multilook'], type=int,
                   target='arch;args;insar;multilook'),
        CustomArgs(['--stride'], type=int,
                   target='arch;args;insar;stride'),
        CustomArgs(['--offset'], type=int,
                   target='arch;args;insar;offset'),
        # Multi-start: override the RNG seed (top-level config key) so the same scene can be
        # re-fit from several inits; keep the best-reconstruction-loss run to escape
        # null-source local minima on noisy real data.
        CustomArgs(['--seed'], type=int, target='seed'),
    ]
    config = ConfigParser.from_args(args, options)
    
    # Debug: Print updated config values
    print(f"Updated config values:")
    print(f"  use_kl_term_z_phy: {config.config['trainer']['phys_vae'].get('use_kl_term_z_phy', 'NOT_SET')}")
    print(f"  beta_max_z_phy: {config.config['trainer']['phys_vae'].get('beta_max_z_phy', 'NOT_SET')}")
    print(f"  use_kl_term_z_aux: {config.config['trainer']['phys_vae'].get('use_kl_term_z_aux', 'NOT_SET')}")
    print(f"  beta_max_z_aux: {config.config['trainer']['phys_vae'].get('beta_max_z_aux', 'NOT_SET')}")
    print(f"  kl_warmup_epochs: {config.config['trainer']['phys_vae'].get('kl_warmup_epochs', 'NOT_SET')}")
    print(f"  epochs_pretrain: {config.config['trainer']['phys_vae'].get('epochs_pretrain', 'NOT_SET')}")
    print(f"  use_ema_prior: {config.config['trainer']['phys_vae'].get('use_ema_prior', 'NOT_SET')}")
    print(f"  ema_momentum: {config.config['trainer']['phys_vae'].get('ema_momentum', 'NOT_SET')}")
    print(f"  edge_penalty_weight: {config.config['trainer']['phys_vae'].get('edge_penalty_weight', 'NOT_SET')}")
    print(f"  edge_penalty_power: {config.config['trainer']['phys_vae'].get('edge_penalty_power', 'NOT_SET')}")
    print(f"  detach_x_P_for_bias: {config.config['arch']['phys_vae'].get('detach_x_P_for_bias', 'NOT_SET')}")

    wandb.init(
        project="PILA",
        name=f"{config['name']}",
        config=config,
        mode="online" if config['trainer']['wandb'] else "disabled",
    )

    main(config)

    wandb.finish()

# NOTE: consider CosineAnnealingLR or ReduceLROnPlateau; see config.
