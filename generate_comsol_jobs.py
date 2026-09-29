#!/usr/bin/env python
# Usage:       python generate_comsol_jobs.py [--topo flat cone] [--track desc]
#                  [--epochs 3000] [--submit]
#              (omit --submit for a dry run: writes per-chamber npz + configs + PBS
#               scripts and prints the qsub commands / submit-all script)
# Description: One PILA inversion per COMSOL wk6 chamber. Splits the catalogue maps
#              (40 chambers x {flat, cone}) into single-chamber npz scenes, templates a
#              config per chamber from configs/comsol/wk6_flat_desc_mogi.json, and writes
#              one PBS job per chamber (train_pila.py; figures are generated at the end of
#              training). Chambers are independent scenes, so each gets its own job.
# Date:        2026-09-28
#
# Assumptions (READ):
#   * Each job trains on ONE map (1 sample -> 1 gradient step per epoch), so the epoch
#     count is much larger than the 40-sample catalogue config; the cosine schedule and
#     early stopping are rescaled to it. Warm-ups (tau/r/KL, in epochs) are unchanged.
#   * Maps are noise-free COMSOL output (metres, + = toward satellite); single LOS track.
#   * Truth for comparison: metadata/wk6_truth.csv (z_centroid_m, dV_cavity_m3), keyed
#     by (cid, surface).

import argparse
import copy
import json
import os
import subprocess
import time

import numpy as np

# --- Project-root-relative paths (run from the repo root) ---
CATALOGUE_DIR = 'data/comsol/wk6_chambers_inversion_20260928'
MAPS_TEMPLATE = os.path.join(CATALOGUE_DIR, 'maps', 'wk6_surface_maps_{topo}.npz')
PER_CHAMBER_DIR = 'data/comsol/wk6_per_chamber'          # derived single-chamber scenes
TEMPLATE_CONFIG = 'configs/comsol/wk6_flat_desc_mogi.json'
CONFIG_DIR = 'configs/comsol/per_chamber'
JOB_DIR = 'jobs/comsol'
LOG_DIR = 'logs/comsol'
SAVE_ROOT = 'saved/comsol_wk6'


def write_chamber_npz(maps_file, topo, out_dir):
    """
    Split one catalogue maps file into single-chamber npz scenes (same keys, 1 sample).

    Inputs : maps_file str (catalogue npz), topo str ('flat'/'cone'), out_dir str.
    Output : list of (cid, npz_path). Existing files are left untouched (never overwritten).
    """
    if not os.path.exists(maps_file):
        raise FileNotFoundError(maps_file)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    with np.load(maps_file) as maps:
        cid_list = [int(cid) for cid in maps['cid']]
        # 3-D arrays are per-chamber stacks [n_chamber, H, W]; everything else is shared.
        stacked_keys = [key for key in maps.files if key != 'cid' and maps[key].ndim == 3]
        shared = {key: maps[key] for key in maps.files if key != 'cid' and key not in stacked_keys}
        stacks = {key: maps[key] for key in stacked_keys}
    print(f"  {maps_file}: {len(cid_list)} chambers, per-chamber keys {stacked_keys}")
    for index, cid in enumerate(cid_list):
        npz_path = os.path.join(out_dir, f'wk6_cid{cid}_{topo}.npz')
        if not os.path.exists(npz_path):
            np.savez(npz_path, cid=np.array([cid]),
                     **{key: stack[index:index + 1] for key, stack in stacks.items()}, **shared)
        written.append((cid, npz_path))
    return written


def make_chamber_config(template, npz_path, track, name, epochs):
    """Template config pointed at one chamber scene, with the epoch schedule rescaled."""
    config = copy.deepcopy(template)
    config['name'] = name
    insar_blocks = [config['arch']['args']['insar'], config['data_loader']['args']['insar'],
                    config['data_loader']['data_dir_valid'], config['data_loader']['data_dir_test']]
    for block in insar_blocks:
        block['timeseries'] = os.path.abspath(npz_path)
        block['geometry'] = track
    config['data_loader']['args']['batch_size'] = 1       # one map per scene
    config['trainer']['epochs'] = epochs
    config['trainer']['save_dir'] = f'{SAVE_ROOT}/{name}'
    config['trainer']['save_period'] = epochs              # only the final + best checkpoints
    config['trainer']['early_stop'] = epochs               # noise-free single map: run full schedule
    config['trainer']['log_step'] = 1
    config['lr_scheduler']['args']['T_max'] = epochs
    return config


def pbs_script(name, job_name, config_path):
    """PBS script text for one chamber inversion (mirrors the jobs/ submit scripts)."""
    return f"""#!/bin/bash
# Usage:       qsub {JOB_DIR}/submit_{name}.pbs
# Description: PILA inversion of one COMSOL wk6 chamber scene ({name}); figures are
#              generated automatically at the end of train_pila.py.
# Date:        {time.strftime('%Y-%m-%d')}
#PBS -N {job_name}
#PBS -P eos_btaisne
#PBS -q qintel_wfly
#PBS -l nodes=1:ppn=4
#PBS -l walltime=04:00:00
#PBS -l mem=8gb
#PBS -j oe
#PBS -o {LOG_DIR}/{name}_pbs.log

set -e
cd "$PBS_O_WORKDIR"

# --- Environment (project env is 'pila', NOT base) ---
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila
export PYTHONPATH="${{PYTHONPATH}}:$PBS_O_WORKDIR"

CONFIG="{config_path}"

echo "[{name}] Host: $(hostname)  Env: $CONDA_DEFAULT_ENV  Start: $(date)"
echo "[{name}] Step 1/1: training + figures ({config_path}) ..."
python train_pila.py --config "$CONFIG"
echo "[{name}] Finished: $(date)"
"""


def main():
    parser = argparse.ArgumentParser(description='One PBS job per COMSOL wk6 chamber inversion')
    parser.add_argument('--topo', nargs='+', default=['flat', 'cone'], choices=['flat', 'cone'])
    parser.add_argument('--track', default='desc', choices=['asc', 'desc'])
    parser.add_argument('--epochs', type=int, default=3000,
                        help='training epochs per chamber (1 gradient step each)')
    parser.add_argument('--submit', action='store_true', help='qsub every job (else dry run)')
    args = parser.parse_args()

    if not os.path.exists(TEMPLATE_CONFIG):
        raise FileNotFoundError(TEMPLATE_CONFIG)
    with open(TEMPLATE_CONFIG) as fh:
        template = json.load(fh)
    for directory in (CONFIG_DIR, JOB_DIR, LOG_DIR):
        os.makedirs(directory, exist_ok=True)

    # --- [1/3] Split catalogue into single-chamber scenes ---
    print(f"[1/3] Writing single-chamber npz scenes to {PER_CHAMBER_DIR} ...")
    chambers = []
    for topo in args.topo:
        for cid, npz_path in write_chamber_npz(MAPS_TEMPLATE.format(topo=topo), topo, PER_CHAMBER_DIR):
            chambers.append((cid, topo, npz_path))
    print(f"  {len(chambers)} chamber scenes")

    # --- [2/3] Per-chamber config + PBS script ---
    print(f"[2/3] Writing configs to {CONFIG_DIR} and PBS scripts to {JOB_DIR} ...")
    job_paths = []
    for cid, topo, npz_path in chambers:
        name = f'comsol_wk6_cid{cid}_{topo}_{args.track}_mogi'
        config_path = os.path.join(CONFIG_DIR, f'{name}.json')
        with open(config_path, 'w') as fh:
            json.dump(make_chamber_config(template, npz_path, args.track, name, args.epochs), fh, indent=4)
        job_path = os.path.join(JOB_DIR, f'submit_{name}.pbs')
        with open(job_path, 'w') as fh:
            # PBS names cap at 15 chars; keep them unique and readable in qstat.
            fh.write(pbs_script(name, f'wk6_{cid}_{topo}', config_path))
        job_paths.append(job_path)
    submit_all = os.path.join(JOB_DIR, 'submit_all_comsol_wk6.sh')
    with open(submit_all, 'w') as fh:
        fh.write("#!/bin/bash\n# Submit every COMSOL wk6 chamber job (run from the repo root on the login node)\n"
                 "set -e\n" + "".join(f'qsub "{path}"\n' for path in job_paths))
    os.chmod(submit_all, 0o755)
    print(f"  {len(job_paths)} jobs; submit-all script: {submit_all}")

    # --- [3/3] Submit or print ---
    if args.submit:
        print("[3/3] Submitting ...")
        for job_path in job_paths:
            subprocess.run(['qsub', job_path], check=True)
    else:
        print(f"[3/3] Dry run. Submit with:  bash {submit_all}")
    print("Done.")


if __name__ == '__main__':
    main()
