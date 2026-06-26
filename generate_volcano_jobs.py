#!/usr/bin/env python
# Usage:       python generate_volcano_jobs.py
# Description: Generate PILA physics-VAE (Stage A, MLP) InSAR configs, source-parameter
#              prior files, and PBS submit scripts for La Palma, Etna, Nyiragongo and
#              Krasheninnikov, for the Mogi, Sun69 (penny-crack) and Okada sources,
#              on both ascending and descending tracks. Templated on the existing
#              SierraNegra_*_InSAR_A.json configs so behaviour matches the paper runs.
# Date:        2026-06-17
#
# Scientific assumptions (flagged for the user to verify):
#   * lat0/lon0 is the LOCAL ENU ORIGIN (the (0,0) km peg) that the +/-15 km source
#     search box is centred on. We set it to each volcano's summit / deformation source.
#     These are approximate published summit coordinates -- confirm against your data.
#   * Source-parameter prior RANGES are reused from SierraNegra (generic: +/-15 km in
#     map plane, 0.5-12 km depth, etc.). Tune per-volcano depth/volume ranges later.
#   * Each MintPy timeseries is already referenced/corrected
#     (geo_timeseries_SET_ERA5_ramp_demErr.h5); no extra re-referencing is done here.
#   * input_dim is a placeholder; the model auto-derives it at runtime from the
#     multilooked, coherence-masked grid (_maybe_set_insar_input_dim).

import json
import os

# --- Paths -----------------------------------------------------------------
REPO_ROOT = "/eos-rs/INSAR_processing/denny/PILA"
DATA_ROOT = "/eos-rs/INSAR_processing/denny"
CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "phys_smpl")
PARAS_DIR = os.path.join(REPO_ROOT, "configs")
TS_NAME = "geo_timeseries_SET_ERA5_ramp_demErr.h5"   # note: _ramp_ variant for these datasets
GEOM_NAME = "geo_geometryRadar.h5"
MASK_NAME = "geo_maskTempCoh.h5"
MULTILOOK = 20                                        # matches SierraNegra runs

# --- Per-volcano definitions ----------------------------------------------
# lat0/lon0 = summit (local ENU origin); track label -> on-disk track directory.
VOLCANOES = {
    "LaPalma": {
        "abbr": "LP",
        "lat0": 28.61, "lon0": -17.87,
        "tracks": {"TA60": "S1_TA60_archive", "TD169": "S1_TD169"},
    },
    "Etna": {
        "abbr": "ET",
        "lat0": 37.751, "lon0": 14.993,
        "tracks": {"TA44": "S1_TA44", "TD124": "S1_TD124"},
    },
    "Nyiragongo": {
        "abbr": "NY",
        "lat0": -1.52, "lon0": 29.25,
        "tracks": {"TA174": "S1_TA174", "TD21": "S1_TD21"},
    },
    "Krasheninnikov": {
        "abbr": "KR",
        "lat0": 54.593, "lon0": 160.273,
        "tracks": {"TA140": "S1_TA140", "TD162": "S1_TD162"},
    },
}

# --- Per-model definitions -------------------------------------------------
# dim_z_phy / C_max mirror the SierraNegra configs; paras_key is the arch arg name.
MODELS = {
    "Mogi":  {"physics": "Mogi_LOS",  "paras_key": "mogi_paras",  "dim_z_phy": 4, "C_max": 4.0},
    "Sun69": {"physics": "Sun69_LOS", "paras_key": "sun69_paras", "dim_z_phy": 5, "C_max": 4.0},
    "Okada": {"physics": "Okada_LOS", "paras_key": "okada_paras", "dim_z_phy": 8, "C_max": 7.0},
}

# --- Source-parameter prior ranges (reused from SierraNegra; km / deg / Mm^3 etc.) ---
PARAS = {
    # NOTE: dV here is the PRE-transform range. The decoder applies a hardcoded affine
    # map dV_physical = dV_prior*1e5 - 1e7 (model_phys_smpl.py rescale), so the range
    # [-2000, 4000] -> physical [-210e6, +390e6] m^3 (deflation AND inflation headroom).
    # The previous [0, 2000] mapped to [-10e6, +190e6] m^3, which railed: deflating
    # volcanoes pinned at the -10e6 floor and Nyiragongo's +172e6 neared the ceiling.
    "Mogi": {
        "xcen": {"min": -15, "max": 15},
        "ycen": {"min": -15, "max": 15},
        "d":    {"min": 0.5, "max": 12},
        "dV":   {"min": -2000, "max": 4000},
    },
    "Sun69": {
        "xcen":   {"min": -15, "max": 15},
        "ycen":   {"min": -15, "max": 15},
        "depth":  {"min": 1,   "max": 12},
        "radius": {"min": 0.1, "max": 8},     # widened: railed at both 0.1 and 4 km before
        "dV":     {"min": -2000, "max": 4000},
    },
    "Okada": {
        "xoff":    {"min": -15, "max": 15},
        "yoff":    {"min": -15, "max": 15},
        "depth":   {"min": 0.5, "max": 12},
        "strike":  {"min": 0,   "max": 360},
        "dip":     {"min": 1,   "max": 90},
        "length":  {"min": 0.1, "max": 15},
        "width":   {"min": 0.1, "max": 15},
        "opening": {"min": 0,   "max": 20},
    },
}


def insar_block(ts, geom, mask, lat0, lon0):
    """Return the shared 'insar' dict used in arch.args and the data_loader."""
    return {
        "timeseries": ts, "geometry": geom, "mask": mask,
        "lat0": lat0, "lon0": lon0,
        "multilook": MULTILOOK, "coh_valid_frac": 0.5, "verbose": False,
    }


def build_config(volcano, vinfo, track_label, track_dir, model, minfo):
    """Build one PILA Stage-A InSAR config dict for (volcano, track, model)."""
    geo_dir = os.path.join(DATA_ROOT, volcano, track_dir, "mintpy", "geo")
    ts = os.path.join(geo_dir, TS_NAME)
    geom = os.path.join(geo_dir, GEOM_NAME)
    mask = os.path.join(geo_dir, MASK_NAME)
    name = f"{volcano}_{model}_{track_label}_A"
    insar = insar_block(ts, geom, mask, vinfo["lat0"], vinfo["lon0"])

    arch_args = {
        "physics": minfo["physics"],
        "encoder_type": "mlp",
        "input_dim": 1000,                # placeholder; auto-derived at runtime
        "hidden_dim": 8,
        "time_feat_dim": 4,
        "use_time_in_residual": False,
        minfo["paras_key"]: f"configs/{volcano.lower()}_{model.lower()}_paras.json",
        "insar": insar,
    }

    cfg = {
        "name": name,
        "n_gpu": 1,
        "arch": {
            "type": "PHYS_VAE_SMPL",
            "args": arch_args,
            "phys_vae": {
                "no_phy": False,
                "dim_z_aux": 4,
                "dim_z_phy": minfo["dim_z_phy"],
                "activation": "elu",
                "num_units_feat": 128,
                "hidlayers_feat": [128],
                "hidlayers_z_aux": [128],
                "hidlayers_z_phy": [128],
                "residual_rank": 4,
                "tau_init": 3.0, "tau_final": 1.0, "tau_warmup_epochs": 20,
                "r_init": 0.0, "r_final": 1.0, "r_warmup_epochs": 30,
                "ortho_penalty_weight": 0.1,
                "coeff_penalty_weight": 0.0,
                "delta_penalty_weight": 0.0,
                "detach_x_P_for_bias": True,
            },
        },
        "data_loader": {
            "type": "InSARh5DataLoader",
            "type_test": "InSARh5DataLoader",
            "args": {
                "insar": insar,
                "batch_size": 8,
                "shuffle": True,
                "validation_split": 0.0,
                "num_workers": 0,
                "with_const": False,
            },
            "data_dir_valid": insar,
            "data_dir_test": insar,
        },
        "optimizer": {"type": "Adam", "args": {"lr": 0.0003, "weight_decay": 0.0001, "amsgrad": True}},
        "loss": "mse_loss",
        "loss_test": "mse_loss",
        "metrics": [],
        "lr_scheduler": {"type": "CosineAnnealingLR", "args": {"T_max": 150}},
        "trainer": {
            "epochs": 150,
            "save_dir": f"saved/full_scene/{volcano.lower()}_{model.lower()}_{track_label.lower()}_A",
            "save_period": 10,
            "log_step": 5,
            "verbosity": 2,
            "monitor": "min val_rec_loss",
            "early_stop": 100,
            "input_key": "displacement",
            "output_key": "displacement",
            "stablize_grad": True,
            "grad_clip_norm": 1.0,
            "tensorboard": False,
            "wandb": False,
            "phys_vae": {
                "epochs_pretrain": 0,
                "kl_warmup_epochs": 50,
                "balance_gate": 0.001,
                "balance_data_aug": 1.0,
                "balance_residual": 0.001,
                "use_kl_term_z_phy": False,
                "beta_max_z_phy": 1.0,
                "use_kl_term_z_aux": False,
                "beta_max_z_aux": 1.0,
                "use_capacity_control": False,
                "C_max": minfo["C_max"],
                "C_gamma": 10.0,
                "beta_aux": 1.0,
                "edge_penalty_weight": 0.0,
                "edge_penalty_power": 1.0,
                "temporal_smoothness_weight": 0.0,
                "use_ema_prior": False,
                "ema_momentum": 0.999,
            },
        },
    }
    return name, cfg


def build_pbs(volcano, vinfo, track_label, model, config_name):
    """Build one PBS submit script string for (volcano, track, model)."""
    job = f"{vinfo['abbr']}_{model}_{track_label}"        # short PBS -N (<=15 chars)
    return f"""#!/bin/bash
# Usage:       qsub submit_{config_name.lower()}.pbs
# Description: Submit the PILA {volcano} {model} InSAR (Stage A, MLP) training job to PBS,
#              track {track_label}. Trains the physics-VAE on the whole MintPy LOS time
#              series (each epoch an independent sample) to invert for the {model} source.
# Date:        2026-06-17
#PBS -N {job}
#PBS -P eos_btaisne
#PBS -q qintel_wfly
#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -l mem=16gb
#PBS -j oe
#PBS -o logs/{job}_pbs.log

# --- Move to the directory the job was submitted from (the PILA repo root) ---
cd "$PBS_O_WORKDIR"

# --- Activate the PILA conda environment (torch 2.0.0, CPU-only on this node) ---
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila

# --- Make the repo importable (data_loader, model, datasets, ... are top-level pkgs) ---
export PYTHONPATH="${{PYTHONPATH}}:$PBS_O_WORKDIR"

echo "[{job}] Host: $(hostname)  Env: $CONDA_DEFAULT_ENV  Workdir: $PBS_O_WORKDIR"
echo "[{job}] Starting training at $(date)"

# --- Train {model} InSAR A (whole-timeseries, epochs independent) ---
python train_pila.py --config configs/phys_smpl/{config_name}.json

echo "[{job}] Finished at $(date)"
"""


def main():
    print("[1/4] Ensuring output directories exist ...")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)

    print("[2/4] Writing source-parameter prior files (per volcano per model) ...")
    n_paras = 0
    for volcano in VOLCANOES:
        for model in MODELS:
            path = os.path.join(PARAS_DIR, f"{volcano.lower()}_{model.lower()}_paras.json")
            with open(path, "w") as fh:
                json.dump(PARAS[model], fh, indent=4)
            n_paras += 1
    print(f"  Wrote {n_paras} paras files.")

    print("[3/4] Writing configs ...")
    n_cfg = 0
    submit_cmds = []
    for volcano, vinfo in VOLCANOES.items():
        for track_label, track_dir in vinfo["tracks"].items():
            for model, minfo in MODELS.items():
                name, cfg = build_config(volcano, vinfo, track_label, track_dir, model, minfo)
                cfg_path = os.path.join(CONFIG_DIR, f"{name}.json")
                with open(cfg_path, "w") as fh:
                    json.dump(cfg, fh, indent=4)
                n_cfg += 1
    print(f"  Wrote {n_cfg} configs.")

    print("[4/4] Writing PBS submit scripts ...")
    n_pbs = 0
    for volcano, vinfo in VOLCANOES.items():
        for track_label, track_dir in vinfo["tracks"].items():
            for model, minfo in MODELS.items():
                name = f"{volcano}_{model}_{track_label}_A"
                pbs = build_pbs(volcano, vinfo, track_label, model, name)
                pbs_path = os.path.join(REPO_ROOT, f"submit_{name.lower()}.pbs")
                with open(pbs_path, "w") as fh:
                    fh.write(pbs)
                submit_cmds.append(f"qsub submit_{name.lower()}.pbs")
                n_pbs += 1
    print(f"  Wrote {n_pbs} PBS scripts.")

    print("\nDone. To submit (review first!):")
    for c in submit_cmds:
        print("  " + c)


if __name__ == "__main__":
    main()
