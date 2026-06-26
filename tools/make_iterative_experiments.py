#!/usr/bin/env python
# Usage:       python tools/make_iterative_experiments.py [--dry-run]
# Description: Generate iterative-refinement experiment variants for PILA. For each base
#              config it writes a sibling "<base>_iterative.json" (with the iterative_refinement
#              block enabled and a distinct save_dir) plus a matching PBS submit script
#              "submit_<base>_iterative.pbs" so the job can be qsub'd unchanged on the HPC.
# Date:        2026-06-26
#
# This script ONLY writes generated artifacts (configs/phys_smpl/*_iterative.json and
# submit_*_iterative.pbs). It never modifies base configs, data, or pretrained models.
# It is idempotent: re-running regenerates the same outputs.

import os
import copy
import json
import argparse

# --- Paths (relative to the repo root, which is this file's parent's parent) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "phys_smpl")

# --- The iterative-refinement block injected under arch.phys_vae ---
# All fields are read with safe .get() defaults by the model/trainer, so this block is the
# single source of truth for the feature's hyper-parameters.
ITERATIVE_BLOCK = {
    "enabled": True,
    "num_passes_train": 3,      # encoder refinement passes during training
    "num_passes_infer": 5,      # max passes at inference (stops early on convergence)
    "convergence_tol": 1e-4,    # early-stop tolerance on the bounded z_phy change
    "initial_u_noise_std": 0.1, # jitter added to the zero init estimate (train only)
    "iter_loss_weight": 1.0,    # weight on the per-pass physics reconstruction loss
}

# --- Base experiments to derive iterative variants from (config stems, no .json) ---
# Exactly: GPS synthetic Mogi + all 3 physics across Etna / Nyiragongo / LaPalma, plus
# SierraNegra InSAR & CNN. Krasheninnikov is intentionally excluded.
BASE_EXPERIMENTS = [
    # GPS synthetic (MLP path)
    "PILA_Mogi_C",
    # Etna (MLP path, InSAR LOS)
    "Etna_Mogi_TA44_A", "Etna_Mogi_TD124_A",
    "Etna_Okada_TA44_A", "Etna_Okada_TD124_A",
    "Etna_Sun69_TA44_A", "Etna_Sun69_TD124_A",
    # Nyiragongo
    "Nyiragongo_Mogi_TA174_A", "Nyiragongo_Mogi_TD21_A",
    "Nyiragongo_Okada_TA174_A", "Nyiragongo_Okada_TD21_A",
    "Nyiragongo_Sun69_TA174_A", "Nyiragongo_Sun69_TD21_A",
    # La Palma
    "LaPalma_Mogi_TA60_A", "LaPalma_Mogi_TD169_A",
    "LaPalma_Okada_TA60_A", "LaPalma_Okada_TD169_A",
    "LaPalma_Sun69_TA60_A", "LaPalma_Sun69_TD169_A",
    # Sierra Negra (InSAR = MLP path, CNN = conv path)
    "SierraNegra_Mogi_InSAR_A", "SierraNegra_Okada_InSAR_A", "SierraNegra_Sun69_InSAR_A",
    "SierraNegra_Mogi_CNN_A", "SierraNegra_Okada_CNN_A", "SierraNegra_Sun69_CNN_A",
]

# --- Volcano -> short code, for compact PBS job names (the -o log path keeps the full stem) ---
VOLCANO_ABBR = {
    "SierraNegra": "SN", "Nyiragongo": "NY", "Krasheninnikov": "KR",
    "LaPalma": "LP", "Etna": "ET", "PILA": "GPS",
}

# --- PBS submit-script template (mirrors the existing submit_*.pbs structure) ---
# Resources/queue/project copied from the curated InSAR submit scripts. Adjust on the HPC
# if a given run needs more walltime/memory.
PBS_TEMPLATE = """#!/bin/bash
# Usage:       qsub {pbs_name}
# Description: Submit the PILA iterative-refinement training job for experiment
#              "{base}". Identical to the baseline run except the encoder iteratively
#              refines its source-parameter estimate (arch.phys_vae.iterative_refinement).
# Date:        2026-06-26
#PBS -N {job_name}
#PBS -P eos_btaisne
#PBS -q qintel_wfly
#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -l mem=16gb
#PBS -j oe
#PBS -o logs/{base}_IT_pbs.log

# Exit immediately if any command fails (e.g. a failed conda activate) so the job does
# not silently waste walltime running in the wrong environment.
set -e

# --- Move to the directory the job was submitted from (the PILA repo root) ---
cd "$PBS_O_WORKDIR"

# --- Activate the PILA conda environment ---
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila

# --- Make the repo importable (data_loader, model, datasets, ... are top-level pkgs) ---
export PYTHONPATH="${{PYTHONPATH}}:$PBS_O_WORKDIR"

echo "[{job_name}] Host: $(hostname)  Env: $CONDA_DEFAULT_ENV  Workdir: $PBS_O_WORKDIR"
echo "[{job_name}] Starting iterative-refinement training at $(date)"

# --- Train (iterative refinement enabled via the config) ---
python train_pila.py --config configs/phys_smpl/{base}_iterative.json

echo "[{job_name}] Finished at $(date)"
"""


def make_job_name(stem):
    """Build a compact PBS job name (target <= 15 chars) with an _IT suffix.

    Input:  config stem, e.g. 'Etna_Mogi_TA44_A'
    Output: short label, e.g. 'ET_Mogi_TA44_IT'
    The full stem is preserved in the config path and the -o log path, so any truncation
    here only affects the human-facing job label, not the outputs.
    """
    name = stem
    for full, abbr in VOLCANO_ABBR.items():
        name = name.replace(full, abbr)
    name = name.replace("__", "_").strip("_")
    return name[:12].rstrip("_") + "_IT"


def build_iterative_config(base_cfg):
    """Return a deep copy of base_cfg with the iterative block injected and save_dir/name
    suffixed by '_iterative' so iterative runs never clobber baseline outputs.

    Inputs:  base_cfg (dict) - parsed base config JSON
    Output:  new config dict (the input is not mutated)
    """
    cfg = copy.deepcopy(base_cfg)

    # Inject the iterative block under arch.phys_vae.
    cfg["arch"]["phys_vae"]["iterative_refinement"] = dict(ITERATIVE_BLOCK)

    # Suffix the experiment name (best-effort; some configs may omit 'name').
    if "name" in cfg:
        cfg["name"] = f"{cfg['name']}_iterative"

    # Suffix the save_dir leaf so baseline and iterative runs land in separate folders.
    save_dir = cfg["trainer"].get("save_dir")
    if save_dir is not None:
        cfg["trainer"]["save_dir"] = save_dir.rstrip("/") + "_iterative"

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate iterative-refinement configs + PBS submit scripts for PILA.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be written without writing any files.")
    args = parser.parse_args()

    n = len(BASE_EXPERIMENTS)
    print(f"[1/3] Generating iterative variants for {n} base experiments")
    print(f"  config dir : {CONFIG_DIR}")
    print(f"  pbs dir    : {REPO_ROOT}")
    print(f"  dry-run    : {args.dry_run}")

    written_configs, written_pbs, missing = [], [], []

    print(f"[2/3] Processing base configs...")
    for idx, stem in enumerate(BASE_EXPERIMENTS, start=1):
        base_path = os.path.join(CONFIG_DIR, f"{stem}.json")
        if not os.path.exists(base_path):
            print(f"  ({idx}/{n}) MISSING base config, skipping: {stem}.json")
            missing.append(stem)
            continue

        with open(base_path, "r") as f:
            base_cfg = json.load(f)

        # --- Build the iterative config ---
        iter_cfg = build_iterative_config(base_cfg)
        iter_cfg_path = os.path.join(CONFIG_DIR, f"{stem}_iterative.json")

        # --- Build the matching PBS submit script ---
        pbs_name = f"submit_{stem}_iterative.pbs"
        pbs_path = os.path.join(REPO_ROOT, pbs_name)
        pbs_text = PBS_TEMPLATE.format(base=stem, pbs_name=pbs_name, job_name=make_job_name(stem))

        if args.dry_run:
            print(f"  ({idx}/{n}) would write: configs/phys_smpl/{stem}_iterative.json  +  {pbs_name}")
        else:
            with open(iter_cfg_path, "w") as f:
                json.dump(iter_cfg, f, indent=4)
                f.write("\n")
            with open(pbs_path, "w") as f:
                f.write(pbs_text)
            print(f"  ({idx}/{n}) wrote: {stem}_iterative.json  +  {pbs_name}  "
                  f"(save_dir={iter_cfg['trainer'].get('save_dir')})")

        written_configs.append(iter_cfg_path)
        written_pbs.append(pbs_path)

    print(f"[3/3] Done. {len(written_configs)} configs + {len(written_pbs)} PBS scripts"
          f"{' (dry-run, nothing written)' if args.dry_run else ''}.")
    if missing:
        print(f"  WARNING: {len(missing)} base config(s) were missing and skipped: {missing}")


if __name__ == "__main__":
    main()
