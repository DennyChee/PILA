#!/usr/bin/env python3
# Usage:       conda activate pila
#              python regen_param_recovery_noshade.py [--dry-run]
# Description: Regenerate EVERY staged PILA parameter-recovery figure with the grey
#              unresolved-epoch shading removed (see plot_iterprior_vs_truth.py, edited
#              2026-09-08). Walks PILA/saved/ for the benchmark cubes' variant runs,
#              backs each existing PNG up, then replots it in place at the same stem so
#              vbICA's stage_figures.py picks it up with no path changes.
# Date:        2026-09-08
#
# WHY A DRIVER: these 63 figures were produced over months by ~78 different job scripts.
# Re-running those jobs would retrain the models; only the PLOT needs redoing, and every
# run already has the per-epoch `uspace_series.csv` the plotter reads.
#
# ON LABELS: the legend/title label is a free-text argument that lived in each historical
# job script, and it is not recoverable reliably (e.g. the stem
# "param_recovery_original_lineardV" is a SUBSTRING of the *_hisnr one, so grepping the
# job scripts silently mismatches). Labels are therefore derived from the run path in the
# same canonical form score_pila_variants.py uses for its CSV rows ("family/cell, cube"),
# with ONE exception: the 12 pinned-cell runs keep the exact wording their campaign
# generator used, since that is the column in the head-to-head. Legend text may therefore
# differ from the historical figure for the older variants; the plotted data does not.
#
# GATING IS UNCHANGED: no job ever passed --dv-eps-frac (verified by grep over jobs/ and
# the repo scripts), so the plotter's default 0.05 reproduces each figure's masking
# exactly. Only the shaded overlay disappears; the track breaks stay.

import argparse
import os
import subprocess
import sys

PILA_ROOT  = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR  = os.path.join(PILA_ROOT, "saved")
CUBES_DIR  = os.path.join(PILA_ROOT, "synthetic", "cubes")
PLOTTER    = os.path.join(PILA_ROOT, "plot_recovered_vs_truth.py")
EXCLUDED_FAMILIES = ("iterprior",)      # deprecated; never staged, so never replotted
PINNED_CELL = "curv15000_dvl1u300"      # the head-to-head cell, label wording preserved
BACKUP_SUFFIX = ".bak_20260908"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dry-run", action="store_true",
                    help="list what would be replotted, touch nothing")
args = parser.parse_args()

# --- Cube tags, longest-first so rising_lineardV never swallows rising_lineardV_hisnr ---
cube_tags = sorted([d for d in os.listdir(CUBES_DIR) if d.startswith("marapi_")],
                   key=len, reverse=True)


def tag_of(top_dir_name):
    """Which benchmark cube a saved/ top-level run directory belongs to (None if not one)."""
    for tag in cube_tags:
        if top_dir_name.endswith(tag):
            return tag
    return None


print("[1/3] Discovering staged parameter-recovery figures...")
jobs = []           # (tag, family, rel_cell, out_stem_abs, png_abs, series_csv_abs)
for dirpath, _dirnames, filenames in os.walk(SAVED_DIR):
    rel = os.path.relpath(dirpath, SAVED_DIR)
    top = rel.split(os.sep)[0]
    if top.startswith("_") or top == ".":
        continue
    if any(top.startswith(fam) for fam in EXCLUDED_FAMILIES):
        continue
    tag = tag_of(top)
    if tag is None:
        continue
    series_csv = os.path.join(dirpath, "uspace_series.csv")
    if not os.path.exists(series_csv):
        continue                        # e.g. the perepoch montage subfolder: not ours
    for fig_name in sorted(f for f in filenames
                           if f.startswith("param_recovery_") and f.endswith(".png")):
        family = top[: len(top) - len(tag)].rstrip("_") or "run"
        cell   = "/".join(s for s in rel.split(os.sep)[1:] if s != "run")
        jobs.append((tag, family, cell, os.path.join(dirpath, fig_name[:-4]),
                     os.path.join(dirpath, fig_name), series_csv))
print(f"  Found: {len(jobs)} figure(s) across {len({j[0] for j in jobs})} cube(s)")

print("[2/3] Replotting each figure in place (backup first)...")
n_done = n_fail = 0
failures = []
for tag, family, cell, out_stem, png_path, series_csv in jobs:
    truth_json = os.path.join(CUBES_DIR, tag, f"{tag}_truth.json")
    if not os.path.exists(truth_json):
        print(f"  MISSING truth sidecar, skipped: {truth_json}")
        n_fail += 1
        failures.append((png_path, "no truth json"))
        continue
    # Label: preserve the campaign generator's exact wording for the pinned cell.
    if family == "combined_curvdvl1u" and cell == PINNED_CELL:
        label = f"combined_curvdvl1u curv=15000 lambda=300, {tag}"
    else:
        label = f"{family}/{cell}, {tag}" if cell else f"{family}, {tag}"

    rel_png = os.path.relpath(png_path, SAVED_DIR)
    if args.dry_run:
        print(f"  WOULD replot {rel_png}\n            label: {label}")
        continue

    # Back up before overwriting -- the replot is reproducible, but the original is not
    # regenerable once the shading code is gone.
    backup = png_path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        os.replace(png_path, backup)
    cmd = [sys.executable, PLOTTER,
           "--series", series_csv, "--truth", truth_json,
           "--out", out_stem, "--label", label]
    proc = subprocess.run(cmd, cwd=PILA_ROOT, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(png_path):
        print(f"  FAILED {rel_png}\n{proc.stdout[-800:]}{proc.stderr[-800:]}")
        n_fail += 1
        failures.append((png_path, f"exit {proc.returncode}"))
        # Put the original back so a failed replot never leaves a hole in the tree.
        if not os.path.exists(png_path) and os.path.exists(backup):
            import shutil
            shutil.copy2(backup, png_path)
        continue
    n_done += 1
    print(f"  [{n_done}/{len(jobs)}] {rel_png}")

print("[3/3] Summary")
print(f"  Replotted : {n_done}")
print(f"  Failed    : {n_fail}")
for path, why in failures:
    print(f"    {os.path.relpath(path, SAVED_DIR)}: {why}")
if n_fail:
    sys.exit(1)
print("Done. Restage with vbICA's stage_figures.py to refresh the review tree.")
