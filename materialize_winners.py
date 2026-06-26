#!/usr/bin/env python
# Usage:       conda activate pila
#              python materialize_winners.py
# Description: After select_multistart.py, take the BEST (non-null) seed per config from
#              comparison_out/multistart/multistart_scores.csv and materialize it into a unified
#              tree so export + comparison can use a single --saved-root:
#                * copy the winning seed's ml5 config -> configs/phys_smpl/definitive/<Name>.json
#                  (config_name == file stem == config['name'], multilook=5 baked in)
#                * symlink the EXACT winning checkpoint's <timestamp> run dir to
#                  saved/full_scene_def/<name_lower>/<Name>/<ts>  (points at one run, so the
#                  saved_root/*/<Name>/*/figures glob in export/compare resolves uniquely).
#              Some (config, seed) have >1 timestamp (re-queued), so we link the scored ts dir,
#              not the seed dir. Errors if any config has no non-null winner.
# Date:        2026-06-25

import os
import re
import csv
import json
import shutil
import argparse
import collections

REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    # Defaults reproduce the original ml5 behavior exactly; the args let the same
    # script materialize an ml20 (or any multilook) run into parallel directories.
    ap = argparse.ArgumentParser(description="Materialize best multi-start seed per config.")
    ap.add_argument('--scores', default=os.path.join(REPO, 'comparison_out', 'multistart',
                                                      'multistart_scores.csv'))
    ap.add_argument('--ms-cfg', default=os.path.join(REPO, 'configs', 'phys_smpl', 'multistart'),
                    help="Dir of the per-seed multi-start configs (source of the winning config).")
    ap.add_argument('--def-cfg', default=os.path.join(REPO, 'configs', 'phys_smpl', 'definitive'),
                    help="Dir to write the winning <Name>.json definitive configs.")
    ap.add_argument('--def-saved', default=os.path.join(REPO, 'saved', 'full_scene_def'),
                    help="Unified tree to symlink each winning run dir into.")
    ap.add_argument('--multilook', type=int, default=5,
                    help="Expected multilook in the winning config (sanity assert).")
    args = ap.parse_args()
    SCORES, MS_CFG = args.scores, args.ms_cfg
    DEF_CFG, DEF_SAVED = args.def_cfg, args.def_saved

    if not os.path.exists(SCORES):
        raise FileNotFoundError(f"scores CSV not found: {SCORES} (run select_multistart.py first)")
    rows = list(csv.DictReader(open(SCORES)))
    by = collections.defaultdict(list)
    for r in rows:
        by[r['config']].append(r)

    os.makedirs(DEF_CFG, exist_ok=True)
    os.makedirs(DEF_SAVED, exist_ok=True)

    no_winner, done = [], []
    print(f"[0/{len(by)}] Materializing winners from {len(rows)} scored runs...")
    for cfg in sorted(by):
        cands = [r for r in by[cfg] if str(r['null_source']).lower() != 'true']
        if not cands:
            no_winner.append(cfg)
            print(f"  {cfg}: NO non-null fit (all seeds collapsed) -- SKIPPED")
            continue
        best = max(cands, key=lambda r: float(r['r2_phys']))
        ckpt = best['checkpoint']                                   # .../<Name>/<ts>/models/model_best.pth
        ts_dir = os.path.dirname(os.path.dirname(ckpt))            # .../<Name>/<ts>
        ts_name = os.path.basename(ts_dir)
        seed = best['seed']
        name_lower = cfg.lower()

        # --- copy winning seed's ml5 config -> definitive/<Name>.json ---
        src_cfg = os.path.join(MS_CFG, f'{cfg}_s{seed}.json')
        if not os.path.exists(src_cfg):
            raise FileNotFoundError(f"{cfg}: winning config not found: {src_cfg}")
        # sanity: config name must equal the file stem so export's config_name matches the run dir
        with open(src_cfg) as fh:
            jc = json.load(fh)
        assert jc.get('name') == cfg, f"{cfg}: config['name']={jc.get('name')} != {cfg}"
        assert jc['arch']['args']['insar'].get('multilook') == args.multilook, \
            f"{cfg}: winning config multilook != {args.multilook}"
        shutil.copyfile(src_cfg, os.path.join(DEF_CFG, f'{cfg}.json'))

        # --- symlink the exact winning <ts> run dir into the unified tree ---
        link_parent = os.path.join(DEF_SAVED, name_lower, cfg)
        os.makedirs(link_parent, exist_ok=True)
        link = os.path.join(link_parent, ts_name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(ts_dir), link)

        print(f"  {cfg:28s} seed {seed:>2}  R2={float(best['r2_phys']):+.3f}  "
              f"RMSE={float(best['rmse_mm']):5.1f}  opening={float(best['opening_m']):6.2f}  -> {ts_name}")
        done.append(cfg)

    print(f"\nMaterialized {len(done)}/{len(by)} configs into {DEF_SAVED}/ and {DEF_CFG}/")
    if no_winner:
        print(f"WARNING: {len(no_winner)} config(s) had NO non-null winner: {no_winner}")
        print("  -> investigate before exporting/comparing these (Okada non-uniqueness).")


if __name__ == '__main__':
    main()
