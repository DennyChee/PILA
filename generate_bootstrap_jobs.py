#!/usr/bin/env python
# Usage:       conda activate pila
#              python generate_bootstrap_jobs.py --base-config configs/phys_smpl/uq/<base>.json \
#                      --members 20 --k 819 --block 3 [--multilook 1]
# Description: Build a SPATIAL BLOCK-BOOTSTRAP uncertainty-quantification ensemble for PILA --
#              an alternative to the strided-decimation ensemble (generate_uq_jobs.py). Given
#              ONE base InSAR config, writes B derived configs (one per member). Every member
#              draws ~k coherent cells by randomly keeping whole `block`-square tiles (member
#              seed = its index), so: (1) member DENSITY k is fixed and decoupled from the
#              ensemble SIZE B, and (2) contiguous-tile resampling respects spatially-
#              correlated atmospheric noise that independent-pixel/stride sampling treats as
#              independent. The spread of inverted parameters across the B members is the UQ.
#              GENERATES ONLY -- never trains. A sequential CPU runner is also written (each
#              synthetic member is ~90 s on CPU), plus the aggregate command.
# Date:        2026-06-24
#
# Scientific assumptions / caveats (flagged):
#   * Block subsampling is WITHOUT replacement (m-out-of-n subsampling, Politis-Romano-Wolf),
#     so members never contain duplicate pixels; the spread is a subsampling variance proxy,
#     still NOT a calibrated Bayesian posterior.
#   * `block` is the tile side in COARSE pixels; pick it ~ the noise correlation length. Small
#     blocks keep whole-scene coverage with local clumping (short-range correlation); large
#     blocks give contiguous holes (long-range) but risk members that miss the source.
#   * Standardization stays PER-MEMBER (each member's own points), matching the strided scheme.

import argparse
import copy
import json
import os

REPO_ROOT = "/eos-rs/INSAR_processing/denny/PILA"


def set_insar_bootstrap(cfg, multilook, k, block, seed, offset):
    """
    Stamp block-bootstrap keys into EVERY insar block of a config dict (in place).

    Mirrors generate_uq_jobs.set_insar_uq but writes the bootstrap triplet instead of stride.
    `offset` is kept as the member's unique id so aggregate_uq.find_members (which dedups by
    arch.args.insar.offset) treats the B members as distinct. stride is forced to 1.
    """
    bk = {"multilook": int(multilook), "stride": 1, "offset": int(offset),
          "bootstrap_k": int(k), "bootstrap_block": int(block), "bootstrap_seed": int(seed)}

    arch_ins = cfg.get("arch", {}).get("args", {}).get("insar")
    if not isinstance(arch_ins, dict):
        raise ValueError("base config has no arch.args.insar block -- not an InSAR config?")
    arch_ins.update(bk)

    dl = cfg.get("data_loader", {})
    for block_dict in (dl.get("args", {}), dl.get("data_dir_valid", {}), dl.get("data_dir_test", {})):
        if not isinstance(block_dict, dict):
            continue
        if isinstance(block_dict.get("insar"), dict):    # wrapped (data_loader.args.insar)
            block_dict["insar"].update(bk)
        elif "timeseries" in block_dict:                 # bare insar dict (valid/test)
            block_dict.update(bk)


def main():
    ap = argparse.ArgumentParser(description="Generate a PILA block-bootstrap UQ ensemble")
    ap.add_argument("--base-config", required=True, help="existing InSAR config JSON to template from")
    ap.add_argument("--members", type=int, required=True, help="ensemble size B (>=2)")
    ap.add_argument("--k", type=int, required=True, help="target coherent cells per member")
    ap.add_argument("--block", type=int, default=3, help="tile side in coarse px (default 3)")
    ap.add_argument("--multilook", type=int, default=1, help="multilook factor (default 1)")
    ap.add_argument("--out-config-dir", default=os.path.join(REPO_ROOT, "configs", "phys_smpl", "bootstrap"))
    ap.add_argument("--runner", default=os.path.join(REPO_ROOT, "run_bootstrap_ensemble.sh"))
    args = ap.parse_args()

    if args.members < 2:
        ap.error(f"--members must be >= 2 to form an ensemble (got {args.members})")

    base_path = args.base_config if os.path.isabs(args.base_config) else os.path.join(REPO_ROOT, args.base_config)
    if not os.path.exists(base_path):
        raise FileNotFoundError(base_path)

    print(f"[1/3] Loading base config: {base_path}")
    with open(base_path) as fh:
        base_cfg = json.load(fh)
    # Strip any inherited stride/offset from the base so only bootstrap keys drive decimation.
    base_name = base_cfg.get("name", os.path.splitext(os.path.basename(base_path))[0])
    # Normalize the base name: drop a trailing _uqstrN_offM token if templating off a stride member.
    import re
    base_name = re.sub(r'_uqstr\d+_off\d+$', '', base_name)
    tag = f"{base_name}_bootk{args.k}_b{args.block}"
    print(f"  Base name: {base_name}; B={args.members}, k={args.k}, block={args.block}px, ml={args.multilook}")

    os.makedirs(args.out_config_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
    boot_root = f"saved/bootstrap_uq/{tag.lower()}"

    print(f"[2/3] Writing {args.members} member configs ...")
    cfg_rels = []
    for member in range(args.members):
        cfg = copy.deepcopy(base_cfg)
        member_name = f"{tag}_off{member}"
        set_insar_bootstrap(cfg, args.multilook, args.k, args.block, seed=member, offset=member)
        cfg["name"] = member_name
        cfg["trainer"]["save_dir"] = boot_root
        cfg_path = os.path.join(args.out_config_dir, f"{member_name}.json")
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=4)
        cfg_rels.append(os.path.relpath(cfg_path, REPO_ROOT))
    print(f"  Wrote {args.members} configs to {args.out_config_dir}")

    # Sequential CPU runner (members are ~90 s each on CPU; no PBS needed for the synthetic).
    with open(args.runner, "w") as fh:
        fh.write("#!/bin/bash\n")
        fh.write("# Train one PILA block-bootstrap UQ ensemble sequentially on CPU.\n")
        fh.write(f"# Generated by generate_bootstrap_jobs.py -- {tag}, B={args.members}.\n")
        fh.write("set -e\n")
        fh.write('cd "$(dirname "$0")"\n')
        fh.write("source /eos-rs/miniconda3/etc/profile.d/conda.sh\n")
        fh.write("conda activate pila\n")
        fh.write('export PYTHONPATH="${PYTHONPATH}:$(pwd)"\n')
        for i, rel in enumerate(cfg_rels):
            fh.write(f'echo "[bootstrap {i+1}/{len(cfg_rels)}] training {rel}"\n')
            fh.write(f"python train_pila.py --config {rel}\n")
        fh.write('echo "All bootstrap members trained."\n')
    os.chmod(args.runner, 0o755)
    print(f"  Wrote sequential runner: {args.runner}")

    print("[3/3] Done. NOT trained for you. To run the ensemble (CPU, sequential):")
    print(f"  bash {os.path.relpath(args.runner, REPO_ROOT)}")
    print(f"\nAfter all members finish, aggregate with:")
    print(f"  python aggregate_uq.py --runs-glob '{boot_root}/*' --name {tag}")


if __name__ == "__main__":
    main()
