#!/usr/bin/env python
# Usage:       python -m synthetic.generate_dvpenalty_jobs
#                  [--lambdas 10 30 100 300 1000] [--submit]
#              (omit --submit for a dry run: writes configs + PBS scripts, prints qsub cmds)
# Description: Generate one PBS job per u-space dV-L1 lambda on the marapi_mogi_rising_combined
#              (sigmoid-ramp dV) synthetic cube. Re-runs the dV-sparsity sweep in PRE-sigmoid
#              (u) space via the new trainer flag `dv_penalty_space: "u"`, which gives the dV
#              penalty a constant, non-vanishing pull even at the sigmoid rail (the z-space
#              form's gradient vanishes there, which is why the old lambda sweep was flat).
#              For each lambda it writes (a) a config JSON derived from the z-space dvl1
#              template with only dv_l1_weight / dv_l2_weight / dv_penalty_space / name /
#              save_dir changed, and (b) a PBS script mirroring the existing dvl1 submit
#              scripts (train -> dump u-space series -> plot recovered-vs-truth).
# Date:        2026-07-27

import argparse
import json
import os
import subprocess

# --- Project-root-relative paths (this generator is run from the repo root) ---
TEMPLATE_CONFIG = 'configs/synth/marapi_mogi_rising_combined_dvl1_lam0p5.json'
CONFIG_DIR = 'configs/synth'
JOB_DIR = 'jobs'
LOG_DIR = 'logs'
# The cube stem that appears in the TEMPLATE config's InSAR paths. We swap this for the
# target --cube-stem so a clean/hisnr cube can reuse the same structural template.
TEMPLATE_CUBE_STEM = 'marapi_mogi_rising_combined'
DEFAULT_CUBE_STEM = 'marapi_mogi_rising_combined'   # target cube (override with --cube-stem)
# Per-penalty defaults: structural template, its cube stem, save/naming prefix, and plot label.
# The curvature template carries the SEQUENCE loader (InSARSeqh5DataLoader, seq_len>=3) the
# temporal-smoothness penalty requires; the dvl1 template is the per-epoch loader.
PENALTY_DEFAULTS = {
    'dvl1': {
        'template': 'configs/synth/marapi_mogi_rising_combined_dvl1_lam0p5.json',
        'template_cube_stem': 'marapi_mogi_rising_combined',
        'prefix': 'dvl1u',
        'label': 'dV-L1 (u-space)',
    },
    'curvature': {
        'template': 'configs/synth/marapi_mogi_rising_lineardV_temporalsmoothcurv.json',
        'template_cube_stem': 'marapi_mogi_rising_lineardV',
        'prefix': 'tscurv',
        'label': 'Temporal smooth (curvature)',
    },
}

# Default u-space L1 lambda grid. In u-space a railed |u - u0| ~ 7 (vs <=0.5 in z-space),
# so these span roughly 1-30% of rec_loss (a SUM over ~16384 px, so O(1e4)) at the rail.
DEFAULT_LAMBDAS = [10, 30, 100, 300, 1000]

# 3-step PBS body, mirroring jobs/submit_dvl1_marapi_mogi_rising_combined_lam0p1.pbs but with
# the u-space penalty active (dv_penalty_space="u" lives in the config, not the command line).
PBS_TEMPLATE = """#!/bin/bash
# Usage:       qsub {job_path}
# Description: {pen_label} (weight={lam}) on the synthetic cube {cube_stem}.
#              train -> dump per-epoch u-space series -> plot recovered vs truth.
# Date:        2026-07-27
#PBS -N {jobname}
#PBS -P eos_btaisne
#PBS -q qintel_wfly
#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -l mem=16gb
#PBS -j oe
#PBS -o {log_dir}/{prefix}_{tag}_{cube_stem}_pbs.log

set -e
cd "$PBS_O_WORKDIR"

# --- Environment (project env is 'pila', NOT base) ---
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila
export PYTHONPATH="${{PYTHONPATH}}:$PBS_O_WORKDIR"

echo "[{prefix}_{tag}] Host: $(hostname)  Env: $CONDA_DEFAULT_ENV  Workdir: $PBS_O_WORKDIR"
echo "[{prefix}_{tag}] Start: $(date)  u-space dV-L1 lambda={lam}  cube={cube_stem}"

# --- Named paths (dV cube + its ground-truth trajectory) ---
CONFIG="{config_path}"
SAVE_DIR="{save_dir}"
TRUTH_JSON="{truth_json}"
SERIES_CSV="$SAVE_DIR/uspace_series.csv"
OUT_STEM="$SAVE_DIR/param_recovery_{prefix}_{tag}"

# --- Step 1/3: single-run training (u-space dV-L1 penalty via dv_l1_weight + dv_penalty_space) ---
echo "[{prefix}_{tag}] Step 1/3: training..."
python train_pila.py --config "$CONFIG"

# --- Step 2/3: locate the freshly-trained checkpoint (newest model_best.pth) ---
CKPT="$(find "$SAVE_DIR" -name model_best.pth -printf '%T@ %p\\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$CKPT" ]; then
  echo "[{prefix}_{tag}] ERROR: no model_best.pth under $SAVE_DIR" >&2
  exit 1
fi
echo "[{prefix}_{tag}] Checkpoint: $CKPT"

# --- Step 2b: dump the per-epoch u-space series ---
echo "[{prefix}_{tag}] Step 2/3: dumping per-epoch u-space series..."
python dump_uspace_series.py --config "$CONFIG" --resume "$CKPT" --out "$SERIES_CSV"

# --- Step 3/3: overlay recovered per-epoch parameters vs synthetic ground truth ---
echo "[{prefix}_{tag}] Step 3/3: plotting recovered vs truth..."
python plot_iterprior_vs_truth.py \\
  --series "$SERIES_CSV" \\
  --truth "$TRUTH_JSON" \\
  --out "$OUT_STEM" \\
  --label "{pen_label} w={lam}, {cube_stem}"

echo "[{prefix}_{tag}] Param-recovery figure: $PBS_O_WORKDIR/${{OUT_STEM}}.png"
echo "[{prefix}_{tag}] Finished: $(date)"
"""


def tag_for_lambda(lam):
    """Map a lambda value to a filesystem tag: 10 -> 'lam10', 0.5 -> 'lam0p5'."""
    # Integer lambdas (our grid) render cleanly as lam<int>; keep a decimal fallback.
    if float(lam).is_integer():
        return f"lam{int(lam)}"
    return "lam" + str(lam).replace('.', 'p')


def _swap_stem(obj, old_stem, new_stem):
    """Recursively replace old_stem with new_stem in every string value of a nested dict/list.
    Used to retarget the template's InSAR cube paths (which all contain the template cube stem)
    at a different cube. Geometry/mask paths use the shared 'marapi_grid' and physics 'paras'
    use their own names, so neither contains the cube stem and neither is touched."""
    if isinstance(obj, dict):
        return {k: _swap_stem(v, old_stem, new_stem) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_swap_stem(v, old_stem, new_stem) for v in obj]
    if isinstance(obj, str):
        return obj.replace(old_stem, new_stem)
    return obj


def build_config(template, lam, tag, cube_stem, save_prefix, penalty, template_cube_stem):
    """Return a config for this weight: copy the template, retarget its cube paths to cube_stem,
    and set the sweep field for the chosen penalty. weight/lam=0 => baseline (penalty inert).

      penalty='dvl1'      -> u-space dV-L1 magnitude penalty (per-epoch loader template).
      penalty='curvature' -> temporal 2nd-difference smoothness on the SEQUENCE-loader template
                             (needs a seq data_loader + seq_len>=3, which the curvature template
                             already carries; weight is O(1e3-1e4) vs rec_loss O(2e3))."""
    cfg = json.loads(json.dumps(template))                    # deep copy via round-trip
    cfg = _swap_stem(cfg, template_cube_stem, cube_stem)       # retarget InSAR timeseries paths
    name = f"{cube_stem}_{save_prefix}_{tag}"
    cfg['name'] = name
    cfg['trainer']['save_dir'] = f"saved/{save_prefix}_{cube_stem}/{tag}"
    phys_vae = cfg['trainer']['phys_vae']
    if penalty == 'dvl1':
        phys_vae['dv_l1_weight'] = lam        # u-space L1 weight (0 => KL-only baseline)
        phys_vae['dv_l2_weight'] = 0.0        # L1-only sweep
        phys_vae['dv_penalty_space'] = 'u'    # pre-sigmoid penalty (non-vanishing at rail); inert when lam=0
    else:  # curvature
        phys_vae['temporal_smoothness_weight'] = lam      # 0 => seq-loader baseline (penalty inert)
        phys_vae['temporal_smoothness_mode'] = 'curvature'
        phys_vae['temporal_smoothness_include_dv'] = True  # smooth dV too, not just geometry
        phys_vae['dv_l1_weight'] = 0.0        # magnitude penalty OFF for the curvature sweep
        phys_vae['dv_l2_weight'] = 0.0
    return name, cfg


def main():
    ap = argparse.ArgumentParser(
        description="Generate/submit u-space dV-L1 sweep PBS jobs on marapi_mogi_rising_combined.")
    ap.add_argument('--lambdas', type=float, nargs='+', default=DEFAULT_LAMBDAS,
                    help='u-space L1 lambda grid (default: 10 30 100 300 1000). Include 0 for a KL-only baseline.')
    ap.add_argument('--cube-stem', default=DEFAULT_CUBE_STEM,
                    help='Target synthetic cube stem (default: marapi_mogi_rising_combined). '
                         'e.g. marapi_mogi_rising_clean retargets the template at the clean cube.')
    ap.add_argument('--penalty', choices=['dvl1', 'curvature'], default='dvl1',
                    help="Which penalty to sweep: 'dvl1' (u-space magnitude, per-epoch loader) or "
                         "'curvature' (temporal 2nd-diff smoothness, sequence loader).")
    ap.add_argument('--save-prefix', default=None,
                    help="Naming/save-dir prefix (default: dvl1u for dvl1, tscurv for curvature).")
    ap.add_argument('--template', default=None,
                    help='Structural template config. Defaults per --penalty.')
    ap.add_argument('--template-cube-stem', default=TEMPLATE_CUBE_STEM,
                    help='Cube stem present in --template paths, to be swapped for --cube-stem. '
                         'Default marapi_mogi_rising_combined; use marapi_mogi_rising_lineardV for '
                         'the curvature template.')
    ap.add_argument('--submit', action='store_true',
                    help='Actually qsub each job (default: dry run; qsub must run on a login node).')
    args = ap.parse_args()

    # Resolve per-penalty defaults (CLI overrides win).
    pen_def = PENALTY_DEFAULTS[args.penalty]
    template_path = args.template or pen_def['template']
    save_prefix = args.save_prefix or pen_def['prefix']
    pen_label = pen_def['label']
    # Use the per-penalty template cube stem unless the user explicitly overrode --template-cube-stem.
    template_cube_stem = (args.template_cube_stem
                          if args.template_cube_stem != TEMPLATE_CUBE_STEM
                          else pen_def['template_cube_stem'])

    cube_stem = args.cube_stem
    truth_json = f'synthetic/cubes/{cube_stem}/{cube_stem}_truth.json'
    # The clean cube must exist before we point jobs at it.
    if not os.path.exists(truth_json):
        raise FileNotFoundError(
            f"Truth sidecar not found: {truth_json} — generate the '{cube_stem}' cube first "
            f"(python -m synthetic.generate_timeseries --spec synthetic/specs/{cube_stem}.json).")

    # --- [1/3] Load the structural template config ---
    print(f"[1/3] penalty={args.penalty}  template={template_path}  cube={cube_stem}")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template config not found: {template_path}")
    with open(template_path) as f:
        template = json.load(f)

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- [2/3] Write one config + one PBS script per lambda ---
    print(f"[2/3] Writing {len(args.lambdas)} config + PBS pairs (u-space dV-L1) for cube '{cube_stem}'...")
    planned = []
    for lam in args.lambdas:
        tag = tag_for_lambda(lam)
        name, cfg = build_config(template, lam, tag, cube_stem, save_prefix,
                                 args.penalty, template_cube_stem)

        config_path = os.path.join(CONFIG_DIR, f"{name}.json")
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

        save_dir = cfg['trainer']['save_dir']
        jobname = f"{save_prefix}_{tag}"[:15]  # PBS -N limit is 15 chars
        job_path = os.path.join(JOB_DIR, f"submit_{save_prefix}_{cube_stem}_{tag}.pbs")
        script = PBS_TEMPLATE.format(
            job_path=job_path, jobname=jobname, tag=tag, lam=lam, cube_stem=cube_stem,
            prefix=save_prefix, pen_label=pen_label, log_dir=LOG_DIR, config_path=config_path,
            save_dir=save_dir, truth_json=truth_json)
        with open(job_path, 'w') as f:
            f.write(script)

        planned.append((lam, tag, config_path, job_path, save_dir))
        print(f"    lambda={lam:>7g}  ->  {config_path}")
        print(f"                     {job_path}")

    # --- [3/3] Submit (login node only) or print the qsub commands for a dry run ---
    qsubs = [f"qsub {job_path}" for _, _, _, job_path, _ in planned]
    if args.submit:
        print(f"[3/3] Submitting {len(qsubs)} jobs via qsub...")
        for lam, tag, _, job_path, _ in planned:
            result = subprocess.run(['qsub', job_path], capture_output=True, text=True)
            print(f"    {tag}: {result.stdout.strip() or result.stderr.strip()}")
    else:
        print(f"[3/3] DRY RUN — wrote {len(planned)} config + PBS pairs. "
              f"qsub from a LOGIN node (qsub fails from a compute-node shell):")
        for q in qsubs:
            print(f"    {q}")

    print("Done. Configs in configs/synth/, PBS scripts in jobs/.")


if __name__ == '__main__':
    main()
