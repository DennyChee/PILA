#!/usr/bin/env python3
# Usage:       python generate_combined_curvdvl1u_jobs.py [--dry-run]
# Description: Generate the config JSONs + PBS submit scripts for the COMBINED
#              temporal-curvature x dV-L1(u-space) 2D grid on the synthetic cube
#              marapi_mogi_rising_combined. Curvature smoothing removes the
#              incoherent per-epoch jitter; the u-space dV-L1 penalty drags the
#              residual flat dV level the rest of the way to ~0 in the quiescent
#              epochs while preserving the genuine late rise. This script clones
#              the validated single-penalty template and only flips the two
#              penalty weights (+ dv_penalty_space) per grid cell, so the loader
#              pair (Seq train loader for curvature, InSARh5 test loader for the
#              per-epoch dump) and every other hyper-parameter stay identical.
# Date:        2026-07-28

import argparse
import json
import os

# --- Repo-root-relative paths (this script lives at the repo root) ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Validated single-penalty template: curvature=15000 config already has the
# correct loader pair (InSARSeqh5DataLoader train / InSARh5DataLoader test),
# temporal_smoothness_mode="curvature", temporal_smoothness_include_dv=true,
# and mogi_paras=configs/mogi_paras_symdV.json. We clone it and flip 2 knobs.
TEMPLATE_CONFIG = os.path.join(
    REPO_ROOT, "configs/synth/marapi_mogi_rising_combined_tscurv_lam15000.json")

# Ground-truth trajectory for the recovered-vs-truth overlay (per epoch params).
TRUTH_JSON = "synthetic/cubes/marapi_mogi_rising_combined/marapi_mogi_rising_combined_truth.json"

# --- The 2D grid ------------------------------------------------------------
# curvature axis: 5000 vs 15000 directly probes the main risk of over-smoothing
#   (lagging the genuine late dV onset). dV-L1(u) axis: 30/100/300 brackets the
#   middle of the already-run standalone dvl1u sweep {10,30,100,300,1000}.
CURVATURE_WEIGHTS = [5000, 15000]      # temporal_smoothness_weight
DVL1U_WEIGHTS = [30, 100, 300]         # dv_l1_weight, acting in pre-sigmoid u-space

# --- Output locations -------------------------------------------------------
CONFIG_DIR = os.path.join(REPO_ROOT, "configs/synth")
JOBS_DIR = os.path.join(REPO_ROOT, "jobs")
# Grid root under saved/. compare_prior_variants.py takes explicit LABEL=path
# --series pairs (no auto-discovery), so this predictable layout just makes each
# cell's uspace_series.csv path easy to hand to a later manual comparison call.
SAVE_ROOT = "saved/combined_curvdvl1u_marapi_mogi_rising_combined"

# --- PBS script template (mirrors submit_tscurv_marapi_mogi_rising_combined_*) ---
# 3-step body: train -> dump per-epoch u-space series -> plot recovered vs truth.
PBS_TEMPLATE = """\
#!/bin/bash
# Usage:       qsub jobs/{pbs_name}
# Description: COMBINED prior on the synthetic cube marapi_mogi_rising_combined:
#              temporal curvature smoothing (weight={curv}) + dV-L1 in u-space
#              (lambda={dvl1u}). train -> dump per-epoch u-space series ->
#              plot recovered vs truth.
# Date:        2026-07-28
#PBS -N {job_tag}
#PBS -P eos_btaisne
#PBS -q qintel_wfly
#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -l mem=16gb
#PBS -j oe
#PBS -o logs/{job_tag}_marapi_mogi_rising_combined_pbs.log

set -e
cd "$PBS_O_WORKDIR"

# --- Environment (project env is 'pila', NOT base) ---
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila
export PYTHONPATH="${{PYTHONPATH}}:$PBS_O_WORKDIR"

echo "[{job_tag}] Host: $(hostname)  Env: $CONDA_DEFAULT_ENV  Workdir: $PBS_O_WORKDIR"
echo "[{job_tag}] Start: $(date)  curvature={curv}  dV-L1(u) lambda={dvl1u}  cube=marapi_mogi_rising_combined"

# --- Named paths (combined-prior config + ground-truth trajectory) ---
CONFIG="{config_rel}"
SAVE_DIR="{save_dir}"
TRUTH_JSON="{truth_json}"
SERIES_CSV="$SAVE_DIR/uspace_series.csv"
OUT_STEM="$SAVE_DIR/param_recovery_{job_tag}"

# --- Step 1/3: single-run training (curvature + u-space dV-L1 both active) ---
echo "[{job_tag}] Step 1/3: training..."
python train_pila.py --config "$CONFIG"

# --- Step 2/3: locate the freshly-trained checkpoint (newest model_best.pth) ---
CKPT="$(find "$SAVE_DIR" -name model_best.pth -printf '%T@ %p\\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$CKPT" ]; then
  echo "[{job_tag}] ERROR: no model_best.pth under $SAVE_DIR" >&2
  exit 1
fi
echo "[{job_tag}] Checkpoint: $CKPT"

# --- Step 2b: dump the per-epoch u-space series (uses InSARh5DataLoader test loader) ---
echo "[{job_tag}] Step 2/3: dumping per-epoch u-space series..."
python dump_uspace_series.py --config "$CONFIG" --resume "$CKPT" --out "$SERIES_CSV"

# --- Step 3/3: overlay recovered per-epoch parameters vs synthetic ground truth ---
echo "[{job_tag}] Step 3/3: plotting recovered vs truth..."
python plot_iterprior_vs_truth.py \\
  --series "$SERIES_CSV" \\
  --truth "$TRUTH_JSON" \\
  --out "$OUT_STEM" \\
  --label "Curvature w={curv} + dV-L1(u) lambda={dvl1u}, marapi_mogi_rising_combined"

echo "[{job_tag}] Param-recovery figure: $PBS_O_WORKDIR/${{OUT_STEM}}.png"
echo "[{job_tag}] Finished: $(date)"
"""


def build_config(template, curvature_weight, dvl1u_weight, name, save_dir):
    """Return a config dict cloned from `template` with only the combined-prior
    knobs changed.

    Inputs:
      template        : dict, parsed template config (curvature-only lam15000)
      curvature_weight: int, temporal_smoothness_weight (curvature mode)
      dvl1u_weight    : int, dv_l1_weight acting in pre-sigmoid u-space
      name            : str, config "name" field
      save_dir        : str, trainer.save_dir (grid-cell output dir)
    Output:
      dict, ready to json.dump. Loader pair + all other hyper-params untouched.
    """
    # Deep copy via round-trip so we never mutate the shared template object.
    cfg = json.loads(json.dumps(template))

    cfg["name"] = name

    phys = cfg["trainer"]["phys_vae"]
    # Curvature smoothing weight (mode/include_dv already set in template).
    phys["temporal_smoothness_weight"] = float(curvature_weight)
    # Turn the dV-L1 penalty back on, in pre-sigmoid u-space so the pull does
    # NOT vanish at the rail. dv_l2 stays off. Cast to float so the emitted JSON
    # matches the template's float type ("dv_l1_weight": 0.0) byte-for-byte.
    phys["dv_l1_weight"] = float(dvl1u_weight)
    phys["dv_l2_weight"] = 0.0
    phys["dv_penalty_space"] = "u"

    cfg["trainer"]["save_dir"] = save_dir
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined curvature x dV-L1(u) grid configs + PBS scripts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without touching disk.")
    args = parser.parse_args()

    # --- Load the validated template config ---
    print(f"[1/3] Loading template config: {TEMPLATE_CONFIG}")
    if not os.path.exists(TEMPLATE_CONFIG):
        raise FileNotFoundError(f"Template config not found: {TEMPLATE_CONFIG}")
    with open(TEMPLATE_CONFIG, "r") as f:
        template = json.load(f)
    print(f"  Loaded template name='{template.get('name')}', "
          f"loader='{template['data_loader']['type']}' / "
          f"test='{template['data_loader']['type_test']}'")

    n_cells = len(CURVATURE_WEIGHTS) * len(DVL1U_WEIGHTS)
    print(f"[2/3] Emitting {n_cells} grid cells "
          f"(curvature {CURVATURE_WEIGHTS} x dvl1u {DVL1U_WEIGHTS})...")

    qsub_lines = []  # collected for the final submission cheat-sheet
    for curvature_weight in CURVATURE_WEIGHTS:
        for dvl1u_weight in DVL1U_WEIGHTS:
            cell_tag = f"curv{curvature_weight}_dvl1u{dvl1u_weight}"
            name = f"marapi_mogi_rising_combined_{cell_tag}"
            job_tag = f"combined_{cell_tag}"

            config_rel = f"configs/synth/{name}.json"
            config_abs = os.path.join(REPO_ROOT, config_rel)
            save_dir = f"{SAVE_ROOT}/{cell_tag}"
            pbs_name = f"submit_{job_tag}.pbs"
            pbs_abs = os.path.join(JOBS_DIR, pbs_name)

            cfg = build_config(template, curvature_weight, dvl1u_weight,
                               name, save_dir)
            pbs_text = PBS_TEMPLATE.format(
                pbs_name=pbs_name, job_tag=job_tag,
                curv=curvature_weight, dvl1u=dvl1u_weight,
                config_rel=config_rel, save_dir=save_dir, truth_json=TRUTH_JSON)

            if args.dry_run:
                print(f"  [dry-run] would write {config_rel}  "
                      f"(tsw={curvature_weight}, dv_l1={dvl1u_weight}, space=u)")
                print(f"  [dry-run] would write jobs/{pbs_name}")
            else:
                with open(config_abs, "w") as f:
                    json.dump(cfg, f, indent=2)
                with open(pbs_abs, "w") as f:
                    f.write(pbs_text)
                print(f"  Wrote {config_rel}  and  jobs/{pbs_name}")

            qsub_lines.append(f"qsub jobs/{pbs_name}")

    # --- Emit the submission cheat-sheet (submit from the LOGIN node) ---
    print(f"[3/3] Done. Submit these {n_cells} jobs from the LOGIN node:")
    for line in qsub_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
