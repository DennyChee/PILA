#!/usr/bin/env python
# Usage:       from sequential_configs import make_sequential_config
#              cfg = make_sequential_config(album_npz, n_sources=2, save_dir='saved/multisource_seq/E1_S3/K2',
#                                           init_from='saved/.../K1/<run>/model_best.pth', seed=1)
#              python sequential_configs.py --album <npz> --k 1 --save-dir <dir> --out <cfg.json>
# Description: Builds PILA configs for the sequential multi-source runs (one album, K sources, optional
#              warm start from the K-1 run). One recipe for every album and K so differences between runs
#              come only from the data and K. Recipe = the current synthetic Mogi recipe
#              (configs/synth/marapi_mogi_*_dvl1u300.json): MLP encoder, symmetric dV range
#              (configs/mogi_paras_symdV.json: x, y +-15 km, depth 0.5-12 km, dV +-2e8 m^3, dV = 0 at
#              z = 0.5), dV-L1 penalty 300 in u-space (pushes unneeded sources to dV = 0), z_phy KL with
#              beta 0.5 (30-epoch warm-up), 150 epochs, Adam 3e-4 + cosine, rank-4 low-rank residual.
#              No temporal smoothness: albums are static, and each acquisition is an independent sample.
# Notes:       save_period = epochs, so only the final checkpoint + model_best are kept (~27 MB each);
#              per-10-epoch checkpoints would cost ~0.4 GB per run.
# Date:        2026-10-06

import argparse
import copy
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_RECIPE = {
    "n_gpu": 1,
    "arch": {
        "type": "PHYS_VAE_SMPL",
        "args": {
            "physics": "MultiMogi_LOS", "encoder_type": "mlp", "input_dim": 16384, "hidden_dim": 8,
            "time_feat_dim": 4, "use_time_in_residual": False, "mogi_paras": "configs/mogi_paras_symdV.json",
        },
        "phys_vae": {
            "no_phy": False, "dim_z_aux": 4, "activation": "elu", "num_units_feat": 128,
            "hidlayers_feat": [128], "hidlayers_z_aux": [128], "hidlayers_z_phy": [128],
            "residual_rank": 4, "tau_init": 3.0, "tau_final": 1.0, "tau_warmup_epochs": 20,
            "r_init": 0.0, "r_final": 1.0, "r_warmup_epochs": 30, "ortho_penalty_weight": 0.1,
            "coeff_penalty_weight": 0.0, "delta_penalty_weight": 0.0, "detach_x_P_for_bias": True,
        },
    },
    "optimizer": {"type": "Adam", "args": {"lr": 0.0003, "weight_decay": 0.0001, "amsgrad": True}},
    "loss": "mse_loss", "loss_test": "mse_loss", "metrics": [],
    "lr_scheduler": {"type": "CosineAnnealingLR", "args": {"T_max": 150}},
    "trainer": {
        "epochs": 150, "save_period": 150, "log_step": 5, "verbosity": 2,
        "monitor": "min val_rec_loss", "early_stop": 150, "input_key": "displacement",
        "output_key": "displacement", "stablize_grad": True, "grad_clip_norm": 1.0,
        "tensorboard": False, "wandb": False, "timing": {"enabled": True},
        "phys_vae": {
            "epochs_pretrain": 0, "kl_warmup_epochs": 30, "balance_gate": 0.001, "balance_data_aug": 1.0,
            "balance_residual": 0.001, "use_kl_term_z_phy": True, "beta_max_z_phy": 0.5,
            "use_kl_term_z_aux": False, "beta_max_z_aux": 1.0, "use_capacity_control": False,
            "C_max": 4.0, "C_gamma": 10.0, "beta_aux": 1.0, "edge_penalty_weight": 0.0,
            "edge_penalty_power": 1.0, "temporal_smoothness_weight": 0.0, "use_ema_prior": False,
            "ema_momentum": 0.999, "dv_l1_weight": 300.0, "dv_l2_weight": 0.0, "dv_penalty_space": "u",
        },
    },
}


# ORIGINAL = the user's standard PILA InSAR Mogi volcano-run recipe (generate_volcano_jobs.py):
# no z_phy KL, no dV sparsity penalty, legacy dV map (configs/<volcano>_mogi_paras.json:
# x, y +-15 km, depth 0.5-12 km, dV = v*1e5 - 1e7 for v in [-2000, 4000] -> -2.1e8 .. 3.9e8 m^3),
# early_stop 100, seed 123 (train_pila default), no nuisance offset/ramp. User decision 2026-10-06:
# "do not change anything else from the original pila runs" -- only the K-source decoder and the
# K-1 -> K warm start are added. save_period is irrelevant to results (the final model is always
# saved as model_best) and is kept = epochs to avoid ~90 GB of intermediate checkpoints.
ORIGINAL_OVERRIDES = {
    'mogi_paras': 'configs/marapi_mogi_paras.json',
    'trainer.phys_vae': {'use_kl_term_z_phy': False, 'beta_max_z_phy': 1.0, 'kl_warmup_epochs': 50,
                         'dv_l1_weight': 0.0, 'dv_l2_weight': 0.0},
    'early_stop': 100,
    'nuisance_ramp': False,
    'seed': 123,
}


# STANDARD (default from 2026-10-06, user decision): the PILA configuration of the PILA-vs-vbICA
# campaign (generate_combined_curvdvl1u_campaign.py, cell curv15000_dvl1u300), loaded VERBATIM from
# one of its generated configs so nothing drifts: temporal-curvature smoothing 15000 (include_dv) with
# the InSARSeqh5DataLoader (seq_len 7), dV-L1 300 in u-space, symmetric dV prior, KL beta 0.5, 150
# epochs, batch 8, early_stop 100, no seed key (train_pila default 123), no nuisance offset/ramp.
# Only the multi-source pieces (physics MultiMogi_LOS, n_sources, dim_z_phy, init_from) and the data
# paths / run name / save_dir are changed. save_period is set to epochs (results unaffected: the final
# model is always saved as model_best) to avoid ~90 GB of intermediate checkpoints.
STANDARD_TEMPLATE = 'configs/synth/marapi_mogi_move_loc_curv15000_dvl1u300.json'


def make_standard_config(album_npz, n_sources, save_dir, init_from=None, seed=None, epochs=None,
                         name=None, insar_extra=None, seed_init=None, init_sources=None, trainer_overrides=None,
                         lr_scheduler=None):
    """Campaign template + multi-source changes + this album's npz paths (see STANDARD_TEMPLATE)."""
    with open(os.path.join(PROJECT_ROOT, STANDARD_TEMPLATE)) as f:
        cfg = json.load(f)
    album = os.path.basename(album_npz)[:-4]
    insar = {"timeseries": album_npz, "geometry": "asc", "lat0": 0.0, "lon0": 0.0, "multilook": 1,
             "coh_valid_frac": 0.5, "verbose": False, **(insar_extra or {})}
    cfg['name'] = name or f'{album}_K{n_sources}'
    cfg['arch']['args']['physics'] = 'MultiMogi_LOS'
    cfg['arch']['args']['n_sources'] = int(n_sources)
    cfg['arch']['args']['insar'] = dict(insar)
    cfg['arch']['phys_vae']['dim_z_phy'] = 4 * int(n_sources)
    if init_from:
        cfg['arch']['phys_vae']['init_from'] = init_from
    if seed_init:
        cfg['arch']['phys_vae']['seed_init'] = dict(seed_init)
    if init_sources and (init_from or seed_init):
        raise ValueError("init_sources cannot be combined with init_from or seed_init (only one start option)")
    if init_sources:
        if len(init_sources) != int(n_sources) or any(len(row) != 4 for row in init_sources):
            raise ValueError(f"init_sources needs {n_sources} rows of [xcen_km, ycen_km, depth_km, dV_m3]")
        cfg['arch']['phys_vae']['init_sources'] = [[float(v) for v in row] for row in init_sources]
    # Loader blocks keep the template's types and args (Seq train loader, seq_len 7); only paths change.
    cfg['data_loader']['args']['insar'] = dict(insar)
    cfg['data_loader']['data_dir_valid'] = dict(insar)
    cfg['data_loader']['data_dir_test'] = dict(insar)
    if seed is not None:
        cfg['seed'] = int(seed)
    if epochs is not None and epochs != cfg['trainer']['epochs']:
        cfg['trainer']['epochs'] = int(epochs)
        cfg['lr_scheduler']['args']['T_max'] = int(epochs)
    if lr_scheduler:        # e.g. {"type": "CosineAnnealingWarmRestarts", "args": {"T_0": 50}}; replaces the template's cosine
        cfg['lr_scheduler'] = {"type": lr_scheduler['type'], "args": dict(lr_scheduler.get('args', {}))}
    cfg['trainer']['save_period'] = int(cfg['trainer']['epochs'])
    cfg['trainer']['save_dir'] = save_dir
    cfg['trainer']['timing'] = {"enabled": True}                     # instrumentation only
    if trainer_overrides:       # e.g. penalties-off control: {'use_kl_term_z_phy': False, 'dv_l1_weight': 0.0, ...}
        unknown = [k for k in trainer_overrides if k not in cfg['trainer']['phys_vae']]
        if unknown:
            raise ValueError(f"unknown trainer.phys_vae keys {unknown}")
        cfg['trainer']['phys_vae'].update(trainer_overrides)
    return cfg


def make_sequential_config(album_npz, n_sources, save_dir, init_from=None, seed=None, epochs=150,
                           name=None, insar_extra=None, residual_rank=4, nuisance_ramp=None,
                           recipe='standard', seed_init=None, init_sources=None, trainer_overrides=None,
                           from_scratch=False, lr_scheduler=None):
    """
    PILA config dict for one album and K = n_sources Mogi sources.

    album_npz  : str   absolute path of the album (.npz, ascending TA69 grid)
    n_sources  : int   K (dim_z_phy = 4K)
    save_dir   : str   run output root (train_pila adds <name>/<MMDD_HHMMSS>/)
    init_from  : str   (K-1)-source checkpoint for the warm start (None for K = 1)
    from_scratch: bool allow K > 1 with NO checkpoint and NO explicit sources: a fresh K-source network from the
                       default random init (no K-1 stage, no truth); standard recipe only
    init_sources: list explicit start [[xcen_km, ycen_km, depth_km, dV_m3], ...] (K rows) for a fresh
                       K-source network (standard recipe only); replaces init_from for K > 1
    seed       : int   RNG seed (train_pila --seed / top-level 'seed')
    epochs     : int   training epochs (also cosine T_max and save_period)
    insar_extra: dict  extra insar keys (e.g. bootstrap_k/block/seed for ensemble members)
    recipe     : str   'standard' (PILA-vs-vbICA campaign config, DEFAULT), 'original' (the older
                       volcano-run settings) or 'synth_v1'
                       (the first sequential test: KL beta 0.5, dV-L1 300 in u-space, symmetric dV,
                       nuisance offset + ramp on, seed 1)
    residual_rank: int rank of PILA's learnable low-rank residual; 0 = physics only (dim_z_aux = 0)
    nuisance_ramp: bool fit a per-acquisition offset + planar ramp (projected out of the loss).
                   Needed for MintPy-style data: without it the source is dragged to a deep corner
                   position to mimic reference offsets / tilts (found 2026-10-06 on album E1).
    """
    if not os.path.isabs(album_npz):
        album_npz = os.path.join(PROJECT_ROOT, album_npz)
    if not os.path.exists(album_npz):
        raise FileNotFoundError(album_npz)
    if n_sources > 1 and init_from is None and not init_sources and not from_scratch:
        raise ValueError("K > 1 runs are warm-started from the K-1 run: pass init_from (or init_sources / from_scratch)")
    if recipe not in ('standard', 'original', 'synth_v1'):
        raise ValueError(f"unknown recipe {recipe!r}")
    if n_sources > 1 and init_from is None and not init_sources and not from_scratch:
        raise ValueError("K > 1 runs are warm-started from the K-1 run: pass init_from (or init_sources / from_scratch)")
    if recipe == 'standard':
        if residual_rank != 4 or nuisance_ramp:
            raise ValueError("the standard recipe is used verbatim (rank 4, no nuisance ramp)")
        if not os.path.isabs(album_npz):
            album_npz = os.path.join(PROJECT_ROOT, album_npz)
        if not os.path.exists(album_npz):
            raise FileNotFoundError(album_npz)
        return make_standard_config(album_npz, n_sources, save_dir, init_from, seed,
                                    None if epochs == 150 else epochs, name, insar_extra, seed_init, init_sources,
                                    trainer_overrides, lr_scheduler)
    if init_sources or trainer_overrides or lr_scheduler:
        raise ValueError("init_sources / trainer_overrides / lr_scheduler are supported by the standard recipe only")
    if seed is None:
        seed = ORIGINAL_OVERRIDES['seed'] if recipe == 'original' else 1
    if nuisance_ramp is None:
        nuisance_ramp = ORIGINAL_OVERRIDES['nuisance_ramp'] if recipe == 'original' else True
    cfg = copy.deepcopy(BASE_RECIPE)
    if recipe == 'original':
        cfg['arch']['args']['mogi_paras'] = ORIGINAL_OVERRIDES['mogi_paras']
        for key in ('dv_penalty_space',):
            cfg['trainer']['phys_vae'].pop(key, None)
        cfg['trainer']['phys_vae'].update(ORIGINAL_OVERRIDES['trainer.phys_vae'])
    album = os.path.basename(album_npz)[:-4]
    cfg['name'] = name or f'{album}_K{n_sources}'
    cfg['seed'] = int(seed)
    insar = {"timeseries": album_npz, "geometry": "asc", "lat0": 0.0, "lon0": 0.0, "multilook": 1,
             "coh_valid_frac": 0.5, "verbose": False, **(insar_extra or {})}
    cfg['arch']['args']['n_sources'] = int(n_sources)
    cfg['arch']['args']['insar'] = dict(insar)
    cfg['arch']['phys_vae']['dim_z_phy'] = 4 * int(n_sources)
    if init_from:
        cfg['arch']['phys_vae']['init_from'] = init_from
    # A static source is common to every acquisition, so a learnable spatial basis can absorb it
    # completely; residual_rank = 0 forces the physics to explain all coherent deformation.
    cfg['arch']['phys_vae']['residual_rank'] = int(residual_rank)
    cfg['arch']['phys_vae']['dim_z_aux'] = int(residual_rank)
    cfg['trainer']['phys_vae']['nuisance_ramp'] = bool(nuisance_ramp)
    cfg['data_loader'] = {
        "type": "InSARh5DataLoader", "type_test": "InSARh5DataLoader",
        "args": {"insar": dict(insar), "batch_size": 8, "shuffle": True, "validation_split": 0.0,
                 "num_workers": 0, "with_const": False},
        "data_dir_valid": dict(insar), "data_dir_test": dict(insar),
    }
    cfg['trainer']['epochs'] = int(epochs)
    cfg['trainer']['save_period'] = int(epochs)
    cfg['trainer']['early_stop'] = ORIGINAL_OVERRIDES['early_stop'] if recipe == 'original' else int(epochs)
    cfg['lr_scheduler']['args']['T_max'] = int(epochs)
    cfg['trainer']['save_dir'] = save_dir
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--init-from', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} exists; not overwriting")
    cfg = make_sequential_config(args.album, args.k, args.save_dir, args.init_from, args.seed, args.epochs)
    with open(args.out, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Done. Config saved to {args.out}")


if __name__ == '__main__':
    main()
