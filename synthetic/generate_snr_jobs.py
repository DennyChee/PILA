#!/usr/bin/env python
# Usage:       python -m synthetic.generate_snr_jobs --snr-min -1 --snr-max 8 --snr-step 0.5
#                  [--submit]      # omit --submit for a dry run (writes scripts, prints qsub)
# Description: Generate (and optionally qsub) one PBS job per SNR-sweep point to fan the
#              no-multilook Marapi-Mogi sweep across the cluster. dV is calibrated from
#              the target SNR via dV = DV0 * 10^(SNR/20) (DV0 fixed so 0 dB -> 2e7 m^3,
#              6 dB -> 4e7 m^3, matching the measured points). Each job runs
#              synthetic/run_one_snr_point.py (generate -> train -> eval -> full plots).
#              Aggregate afterwards with synthetic/aggregate_sweep.py.
# Date:        2026-06-22

import argparse
import os
import subprocess

DV0 = 2.0e7              # dV at SNR 0 dB (calibrated: 2e7->0 dB, 4e7->6 dB, 8e7->12 dB)
BASE_SPEC = 'synthetic/specs/marapi_mogi_buildup_combined.json'
TEMPLATE = 'configs/phys_smpl/SierraNegra_Mogi_InSAR_A.json'
OUT = 'synthetic/snr_sweep_ml1'
JOB_DIR = 'synthetic/snr_jobs'

PBS_TEMPLATE = """#!/bin/bash
#PBS -N {jobname}
#PBS -P eos_btaisne
#PBS -q qintel_wfly
#PBS -l nodes=1:ppn=8
#PBS -l walltime=01:00:00
#PBS -l mem=8gb
#PBS -j oe
#PBS -o {job_dir}/logs/{tag}.log
cd "$PBS_O_WORKDIR"
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila
export PYTHONPATH="${{PYTHONPATH}}:$PBS_O_WORKDIR"
export MKL_THREADING_LAYER=GNU
echo "[{tag}] host $(hostname) start $(date)  target SNR {snr:+.1f} dB  dV {dv:.3e}"
python -m synthetic.run_one_snr_point \\
    --base-spec {base_spec} --template {template} \\
    --dv {dv:.6e} --multilook {multilook} --epochs {epochs} --out {out} \\
    --std-mode {std_mode} --ff-radius {ff_radius} --tag-suffix "{tag_suffix}"
echo "[{tag}] done $(date)"
"""


def dv_for_snr(snr):
    return DV0 * 10.0 ** (snr / 20.0)


def tag_for_dv(dv, base_name='marapi_mogi_buildup_combined', amp_key='dV'):
    return f"{base_name}_{amp_key}{dv:.2e}".replace('+', '').replace('.', 'p')


def main():
    ap = argparse.ArgumentParser(description="Generate/submit PBS jobs for a finer SNR sweep.")
    ap.add_argument('--snr-min', type=float, default=-1.0)
    ap.add_argument('--snr-max', type=float, default=8.0)
    ap.add_argument('--snr-step', type=float, default=0.5)
    ap.add_argument('--multilook', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--base-spec', default=BASE_SPEC)
    ap.add_argument('--template', default=TEMPLATE)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--std-mode', default='global', choices=['global', 'far_field_robust'])
    ap.add_argument('--ff-radius', type=float, default=8.0)
    ap.add_argument('--tag-suffix', default='', help='Isolate a variant (e.g. "ff").')
    ap.add_argument('--submit', action='store_true', help='Actually qsub (default: dry run).')
    args = ap.parse_args()

    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(os.path.join(JOB_DIR, 'logs'), exist_ok=True)

    n = int(round((args.snr_max - args.snr_min) / args.snr_step)) + 1
    targets = [round(args.snr_min + i * args.snr_step, 3) for i in range(n)]

    suffix = args.tag_suffix
    planned, skipped = [], []
    for snr in targets:
        dv = dv_for_snr(snr)
        tag = tag_for_dv(dv)
        if suffix:
            tag = f"{tag}_{suffix}"
        metrics = os.path.join(args.out, 'eval', tag, f'{tag}_metrics.json')
        if os.path.exists(metrics):
            skipped.append((snr, dv, tag)); continue
        jn = f"snr{snr:+.1f}{('_' + suffix) if suffix else ''}"
        jobname = jn.replace('+', 'p').replace('-', 'm').replace('.', '')[:15]
        script = PBS_TEMPLATE.format(jobname=jobname, tag=tag, job_dir=JOB_DIR, snr=snr, dv=dv,
                                     base_spec=args.base_spec, template=args.template,
                                     multilook=args.multilook, epochs=args.epochs, out=args.out,
                                     std_mode=args.std_mode, ff_radius=args.ff_radius,
                                     tag_suffix=suffix)
        path = os.path.join(JOB_DIR, f'{tag}.pbs')
        with open(path, 'w') as f:
            f.write(script)
        planned.append((snr, dv, tag, path))

    print(f"Target SNRs: {targets[0]}..{targets[-1]} dB step {args.snr_step}  "
          f"({len(targets)} pts; {len(skipped)} already done, {len(planned)} to run)")
    print(f"\n{'SNR dB':>8}{'dV m^3':>14}   job script")
    for snr, dv, tag, path in planned:
        print(f"{snr:>+8.1f}{dv:>14.3e}   {path}")
    if skipped:
        print(f"\nSkipped (cached): " + ", ".join(f"{s:+.1f}dB" for s, _, _ in skipped))

    qsubs = [f"qsub {p}" for _, _, _, p in planned]
    if args.submit:
        print(f"\nSubmitting {len(qsubs)} jobs...")
        for _, _, tag, path in planned:
            r = subprocess.run(['qsub', path], capture_output=True, text=True)
            print(f"  {tag}: {r.stdout.strip() or r.stderr.strip()}")
    else:
        print(f"\nDRY RUN — wrote {len(planned)} PBS scripts to {JOB_DIR}/. To submit:")
        print(f"  python -m synthetic.generate_snr_jobs --snr-min {args.snr_min} "
              f"--snr-max {args.snr_max} --snr-step {args.snr_step} --submit")
        print("Or qsub individually:")
        for q in qsubs[:3]:
            print(f"  {q}")
        if len(qsubs) > 3:
            print(f"  ... ({len(qsubs)} total)")


if __name__ == '__main__':
    main()
