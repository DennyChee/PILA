#!/usr/bin/env python3
# Usage:       python generate_combined_curvdvl1u_campaign.py [--dry-run]
#              python generate_combined_curvdvl1u_campaign.py --cell-suffix _pathfix --jobs-subdir combined_curvdvl1u_rerun --rerun-complete
#              python generate_combined_curvdvl1u_campaign.py --cell-suffix _pathfix_dvbox --amp-box --jobs-subdir combined_curvdvl1u_rerun --rerun-complete
#              then submit the printed qsub lines from the LOGIN node.
# Description: Generate config JSONs + one PBS script per cube for the 12-cube
#              PILA campaign at a SINGLE fixed combined_curvdvl1u cell
#              (temporal-curvature smoothing 15000 + pull-to-zero amplitude-L1 in
#              pre-sigmoid u-space, lambda 300). One fixed cell on the PILA side
#              matches the single fixed setting on the vbICA side (N=5, whitening
#              ON), so the head-to-head becomes untuned-vs-untuned and the old
#              "tuned PILA vs untuned vbICA" caveat no longer applies.
# Date:        2026-09-07 (2026-10-22: decoder-path fix, --cell-suffix / --amp-box / --jobs-subdir)
#
# 2026-10-22 DECODER-PATH FIX: the original version re-pointed only the three data_loader cube paths, so
# arch.args.insar.timeseries (which the physics decoder uses for its mean/std) stayed on the template cube
# marapi_mogi_rising_combined.h5 for the other 11 cubes. build_config now sets all four.
# --amp-box: data-scaled amplitude box (utils/amp_box.py); Sun69 then starts from the symmetric
# configs/sierranegra_sun69_paras_symdV.json so zero volume stays at z = 0.5.
#
# WHY A SEPARATE SCRIPT: generate_combined_curvdvl1u_jobs.py is hard-wired to the
# single cube marapi_mogi_rising_combined and is the provenance record for the
# published 2x3 grid (EXPERIMENTS_2026-06-23.md). It is deliberately left alone.
#
# WHY ONE SHARED TEMPLATE: each cube also has its own *_base.json, but those
# differ in epochs (120 vs 150), kl_warmup_epochs and beta_max_z_phy. Cloning one
# validated template for all 12 is what makes this column genuinely uniform. The
# template additionally carries the InSARSeqh5DataLoader train loader that the
# curvature term requires (it needs epoch SEQUENCES); the Okada and Sun69 cube
# configs use the plain per-epoch loader, so taking the loader block from the
# template is what supplies seq_len=7 and mode="train" for them.

import argparse
import json
import os

# --- Repo-root-relative paths (this script lives at the repo root) ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs/synth")
JOBS_DIR = os.path.join(REPO_ROOT, "jobs")
CUBES_DIR = os.path.join(REPO_ROOT, "synthetic/cubes")

# Validated template: curvature=15000 on marapi_mogi_rising_combined. Already has
# the correct loader pair, temporal_smoothness_mode="curvature",
# temporal_smoothness_include_dv=true and mogi_paras=configs/mogi_paras_symdV.json.
TEMPLATE_CONFIG = os.path.join(
    CONFIG_DIR, "marapi_mogi_rising_combined_tscurv_lam15000.json")

# --- The one fixed cell (winner of the rising_combined 2x3 grid) ---
CURVATURE_WEIGHT = 15000.0     # temporal_smoothness_weight, mode="curvature"
DVL1U_WEIGHT = 300.0           # dv_l1_weight, acting in pre-sigmoid u-space
CELL_TAG = "curv15000_dvl1u300"

# --- Per-cube physics overrides -------------------------------------------
# The 10 Mogi cubes inherit the template's physics untouched. Okada and Sun69
# need the decoder swapped, the latent width changed, and their own paras file.
#   Okada uses the SYMMETRIC-opening paras file: the amplitude pull-to-zero acts
#   on |u - u0| with u0 = logit(z0), and with the stock opening range [0, 20] the
#   physical zero sits at z0 = 0 -> u0 = -13.82, a huge constant pull that would
#   swamp the reconstruction loss. A range symmetric about zero puts u0 = 0,
#   exactly as configs/mogi_paras_symdV.json does for Mogi dV. Negative opening
#   is closing, which is physically meaningful for a deflating dike.
PHYSICS_OVERRIDES = {
    "marapi_okada_dike_combined": {
        "physics": "Okada_LOS",
        "paras_key": "okada_paras",
        "paras_file": "configs/marapi_okada_dike_symopen_paras.json",
        "dim_z_phy": 8,
    },
    "marapi_sun69_sill_combined": {
        "physics": "Sun69_LOS",
        "paras_key": "sun69_paras",
        # Stock file is already fine: dV range [0, 2000] with the default
        # scale=1e5 / shift=-1e7 puts physical dV=0 at z0=0.05, well inside
        # [0, 1], so no symmetric variant is needed here.
        "paras_file": "configs/sierranegra_sun69_paras.json",
        "dim_z_phy": 5,
    },
}
SUN69_SYMDV_PARAS = "configs/sierranegra_sun69_paras_symdV.json"
# Paras keys that must be removed when the physics is swapped away from Mogi.
MOGI_PARAS_KEY = "mogi_paras"

# --- The 12 ground-truth synthetic cubes -----------------------------------
CUBES = [
    "marapi_mogi_move_loc",
    "marapi_mogi_move_loc_clean",
    "marapi_mogi_move_loc_hisnr",
    "marapi_mogi_move_locdepth",
    "marapi_mogi_rising_clean",
    "marapi_mogi_rising_combined",
    "marapi_mogi_rising_lineardV",
    "marapi_mogi_rising_lineardV_hisnr",
    "marapi_mogi_staticloc_lineardV",
    "marapi_mogi_staticloc_sigmoiddV",
    "marapi_okada_dike_combined",
    "marapi_sun69_sill_combined",
]

# --- PBS template ----------------------------------------------------------
# One cube = one script = one qsub = one job id. The $PBS_JOBID goes into the log
# filename so a rerun never overwrites an earlier run's log, and the body is the
# validated 3 steps. Emitting uspace_series.csv at exactly this path is MANDATORY:
# score_pila_variants.py discovers runs by that filename, which is why no
# iterprior run was ever scored.
PBS_TEMPLATE = """\
#!/bin/bash
# Usage:       qsub jobs/{pbs_name}
# Description: combined_curvdvl1u PILA run on the synthetic cube {cube}:
#              temporal curvature smoothing (weight={curv:g}) + amplitude-L1 in
#              pre-sigmoid u-space (lambda={dvl1u:g}). One fixed cell for all 12
#              cubes, for the untuned-vs-untuned head-to-head against vbICA.
# Date:        2026-09-07
#PBS -N {job_tag}
#PBS -P eos_btaisne
#PBS -q qintel_wfly
#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -l mem=16gb
#PBS -j oe
#PBS -o logs/{job_tag}_pbs.log

set -e
cd "$PBS_O_WORKDIR"

# --- Environment (project env is 'pila', NOT base) ---
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila
export PYTHONPATH="${{PYTHONPATH}}:$PBS_O_WORKDIR"

# --- Named paths ---
CUBE="{cube}"
CONFIG="{config_rel}"
SAVE_DIR="{save_dir}"
TRUTH_JSON="{truth_json}"
SERIES_CSV="$SAVE_DIR/uspace_series.csv"
OUT_STEM="$SAVE_DIR/param_recovery_{job_tag}"
# Job id in the log name so reruns never clobber an earlier run's log.
RUN_LOG="logs/{job_tag}_${{PBS_JOBID%%.*}}.log"

mkdir -p logs "$SAVE_DIR"
# Truncating tee: this run owns this file.
exec > >(tee "$RUN_LOG") 2>&1

echo "[{job_tag}] Host: $(hostname)  Env: $CONDA_DEFAULT_ENV  Job: $PBS_JOBID"
echo "[{job_tag}] Start: $(date)"
echo "[{job_tag}] cube=$CUBE  curvature={curv:g}  amplitude-L1(u) lambda={dvl1u:g}"

# --- Step 1/3: training (curvature + u-space amplitude-L1 both active) ---
echo "[{job_tag}] Step 1/3: training..."
python train_pila.py --config "$CONFIG"

# --- Step 2/3: locate the freshly-trained checkpoint, dump per-epoch u-space series ---
CKPT="$(find "$SAVE_DIR" -name model_best.pth -printf '%T@ %p\\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$CKPT" ]; then
  echo "[{job_tag}] ERROR: no model_best.pth under $SAVE_DIR" >&2
  exit 1
fi
echo "[{job_tag}] Checkpoint: $CKPT"
echo "[{job_tag}] Step 2/3: dumping per-epoch u-space series..."
python dump_uspace_series.py --config "$CONFIG" --resume "$CKPT" --out "$SERIES_CSV"

# --- Step 3/3: overlay recovered per-epoch parameters vs synthetic ground truth ---
echo "[{job_tag}] Step 3/3: plotting recovered vs truth..."
python plot_recovered_vs_truth.py \\
  --series "$SERIES_CSV" \\
  --truth "$TRUTH_JSON" \\
  --out "$OUT_STEM" \\
  --label "combined_curvdvl1u curv={curv:g} lambda={dvl1u:g}, $CUBE"

echo "[{job_tag}] Series: $PBS_O_WORKDIR/$SERIES_CSV"
echo "[{job_tag}] Figure: $PBS_O_WORKDIR/${{OUT_STEM}}.png"
echo "[{job_tag}] Finished: $(date)"
"""


def build_config(template, cube, name, save_dir, amp_box_dir=None):
    """Clone `template` and retarget it at one cube at the fixed combined cell.

    Inputs:
      template : dict, parsed template config (Mogi curvature-only lam15000)
      cube     : str, cube stem, e.g. 'marapi_okada_dike_combined'
      name     : str, config "name" field
      save_dir : str, trainer.save_dir (repo-relative)
      amp_box_dir : str or None, if given write a data-scaled amplitude paras JSON there and use it
    Output:
      dict, ready to json.dump.
    """
    # Deep copy via round-trip so the shared template object is never mutated.
    cfg = json.loads(json.dumps(template))
    cfg["name"] = name
    cfg["trainer"]["save_dir"] = save_dir

    # --- Retarget every h5 path at this cube ---
    h5_path = os.path.join(CUBES_DIR, cube, f"{cube}.h5")
    # arch.args.insar feeds the physics decoder's mean/std; it MUST be this cube too (2026-10-22 fix).
    cfg["arch"]["args"]["insar"]["timeseries"] = h5_path
    cfg["data_loader"]["args"]["insar"]["timeseries"] = h5_path
    cfg["data_loader"]["data_dir_valid"]["timeseries"] = h5_path
    cfg["data_loader"]["data_dir_test"]["timeseries"] = h5_path

    # --- The fixed combined cell ---
    # Curvature mode and include_dv already come from the template; only the two
    # weights and the penalty space are set here, mirroring the original grid
    # generator so the emitted JSON is directly comparable to it.
    phys = cfg["trainer"]["phys_vae"]
    phys["temporal_smoothness_weight"] = float(CURVATURE_WEIGHT)
    phys["dv_l1_weight"] = float(DVL1U_WEIGHT)
    phys["dv_l2_weight"] = 0.0
    phys["dv_penalty_space"] = "u"

    # --- Non-Mogi physics swap (Okada / Sun69 only) ---
    ov = PHYSICS_OVERRIDES.get(cube)
    if ov is not None:
        arch_args = cfg["arch"]["args"]
        arch_args["physics"] = ov["physics"]
        arch_args.pop(MOGI_PARAS_KEY, None)
        arch_args[ov["paras_key"]] = ov["paras_file"]
        # dim_z_phy is the decoder's latent width and MUST match the number of
        # keys in the paras file (4 Mogi / 5 Sun69 / 8 Okada), else rescale()
        # enumerates past the end of z_phy.
        cfg["arch"]["phys_vae"]["dim_z_phy"] = ov["dim_z_phy"]

    # --- Data-scaled amplitude box (the box width follows this cube's own LOS std) ---
    if amp_box_dir is not None:
        from utils.amp_box import write_data_scaled_paras
        arch_args = cfg["arch"]["args"]
        if "sun69_paras" in arch_args:
            # The legacy Sun69 file is asymmetric (dV 0 at z = 0.05); the box rule needs the symmetric map.
            arch_args["sun69_paras"] = SUN69_SYMDV_PARAS
        # One sub-dir per cube: the box differs per cube, so a shared file would be overwritten by the last cube.
        paras_path, info = write_data_scaled_paras(cfg, os.path.join(amp_box_dir, cube), project_root=REPO_ROOT)
        paras_key = [key for key in ("mogi_paras", "sun69_paras", "okada_paras") if key in arch_args][0]
        arch_args[paras_key] = os.path.relpath(paras_path, REPO_ROOT)
        print(f"    amp box {info['param']} +-{info['box']:.4g} (std {info['std_mm']:.2f} mm)"
              + (f", opening_ref {info['opening_ref_m']:.4g} m" if 'opening_ref_m' in info else ""))

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate the 12-cube combined_curvdvl1u campaign at one fixed cell.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without touching disk.")
    parser.add_argument("--rerun-complete", action="store_true",
                        help=("Also emit qsub lines for cubes whose target run already "
                              "exists with a byte-identical config. Off by default: "
                              "marapi_mogi_rising_combined was already trained at this "
                              "exact cell for the published 2x3 grid, so re-running it "
                              "would burn 12 h to reproduce a file we already have."))
    parser.add_argument("--cell-suffix", default="",
                        help="Appended to the cell tag, e.g. _pathfix -> new configs/save dirs, nothing overwritten.")
    parser.add_argument("--amp-box", action="store_true",
                        help="Data-scaled amplitude box (utils/amp_box.py) written to configs/synth/ampbox/.")
    parser.add_argument("--jobs-subdir", default="",
                        help="Write PBS scripts into jobs/<subdir>/ instead of jobs/.")
    args = parser.parse_args()
    cell_tag = CELL_TAG + args.cell_suffix
    jobs_dir = os.path.join(JOBS_DIR, args.jobs_subdir)
    jobs_rel = os.path.join("jobs", args.jobs_subdir) if args.jobs_subdir else "jobs"
    os.makedirs(jobs_dir, exist_ok=True)
    amp_box_dir = os.path.join(CONFIG_DIR, "ampbox") if args.amp_box else None

    # --- [1/4] Load the validated template ---
    print(f"[1/4] Loading template config: {os.path.relpath(TEMPLATE_CONFIG, REPO_ROOT)}")
    if not os.path.exists(TEMPLATE_CONFIG):
        raise FileNotFoundError(f"Template config not found: {TEMPLATE_CONFIG}")
    with open(TEMPLATE_CONFIG, "r") as f:
        template = json.load(f)
    print(f"  Loaded name='{template.get('name')}', "
          f"loader='{template['data_loader']['type']}' / "
          f"test='{template['data_loader']['type_test']}', "
          f"mode='{template['trainer']['phys_vae'].get('temporal_smoothness_mode')}'")

    # --- [2/4] Check every cube and truth file exists BEFORE writing anything ---
    print(f"[2/4] Checking {len(CUBES)} cubes and truth files...")
    truth_for = {}
    for cube in CUBES:
        h5 = os.path.join(CUBES_DIR, cube, f"{cube}.h5")
        truth = os.path.join("synthetic/cubes", cube, f"{cube}_truth.json")
        if not os.path.exists(h5):
            raise FileNotFoundError(f"Cube h5 missing: {h5}")
        if not os.path.exists(os.path.join(REPO_ROOT, truth)):
            raise FileNotFoundError(f"Truth json missing: {truth}")
        truth_for[cube] = truth
    print(f"  All {len(CUBES)} cubes present.")

    # --- [3/4] Emit one config + one PBS per cube ---
    print(f"[3/4] Emitting {len(CUBES)} cells at {cell_tag} "
          f"(curvature={CURVATURE_WEIGHT:g}, dvl1u={DVL1U_WEIGHT:g})...")
    qsub_lines = []
    for cube in CUBES:
        name = f"{cube}_{cell_tag}"
        job_tag = f"combined_{cell_tag}_{cube}"
        config_rel = f"configs/synth/{name}.json"
        config_abs = os.path.join(REPO_ROOT, config_rel)
        save_dir = f"saved/combined_curvdvl1u_{cube}/{cell_tag}"
        pbs_name = f"submit_{job_tag}.pbs"
        pbs_abs = os.path.join(jobs_dir, pbs_name)

        cfg = build_config(template, cube, name, save_dir, amp_box_dir)
        pbs_text = PBS_TEMPLATE.format(
            pbs_name=os.path.join(args.jobs_subdir, pbs_name), job_tag=job_tag, cube=cube,
            curv=CURVATURE_WEIGHT, dvl1u=DVL1U_WEIGHT,
            config_rel=config_rel, save_dir=save_dir, truth_json=truth_for[cube])

        physics = cfg["arch"]["args"]["physics"]

        # --- Is this cell already trained, from an identical config? ---------
        # A completed run is identified by uspace_series.csv at the target
        # save_dir (the same filename score_pila_variants.py discovers runs by).
        # We only treat it as reusable if the config on disk is byte-equivalent
        # to what we would write -- otherwise the saved weights came from
        # different settings and must not be silently adopted.
        series_abs = os.path.join(REPO_ROOT, save_dir, "uspace_series.csv")
        already_done = False
        if os.path.exists(series_abs) and os.path.exists(config_abs):
            with open(config_abs, "r") as f:
                on_disk = json.load(f)
            already_done = (on_disk == cfg)

        if already_done:
            print(f"  SKIP  {config_rel}  (physics={physics}) -- already trained at "
                  f"this exact cell, config byte-identical, series present:")
            print(f"          {save_dir}/uspace_series.csv")
            if not args.rerun_complete:
                continue
            print("        --rerun-complete given: queueing it anyway.")

        if args.dry_run:
            print(f"  [dry-run] {config_rel}  (physics={physics}, "
                  f"dim_z_phy={cfg['arch']['phys_vae']['dim_z_phy']})")
            print(f"  [dry-run] {jobs_rel}/{pbs_name}")
        else:
            # Never silently clobber a hand-edited config.
            for path in (config_abs, pbs_abs):
                if os.path.exists(path):
                    os.replace(path, path + ".bak")
            with open(config_abs, "w") as f:
                json.dump(cfg, f, indent=2)
            with open(pbs_abs, "w") as f:
                f.write(pbs_text)
            print(f"  Wrote {config_rel}  and  {jobs_rel}/{pbs_name}   "
                  f"(physics={physics})")

        qsub_lines.append(f"qsub {jobs_rel}/{pbs_name}")

    # --- [4/4] Submission cheat-sheet: 12 separate qsubs, never a PBS array ---
    n_skipped = len(CUBES) - len(qsub_lines)
    if n_skipped:
        print(f"  ({n_skipped} cube(s) already complete at this cell and were skipped; "
              f"pass --rerun-complete to queue them anyway.)")
    print(f"[4/4] Done. Submit these {len(qsub_lines)} jobs from the LOGIN node "
          f"(one qsub per run -> {len(qsub_lines)} distinct job ids):")
    for line in qsub_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
