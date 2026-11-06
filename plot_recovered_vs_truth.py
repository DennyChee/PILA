#!/usr/bin/env python3
# Usage:       python plot_recovered_vs_truth.py \
#                  --series saved/combined_curvdvl1u_<cube>/curv15000_dvl1u300/uspace_series.csv \
#                  --truth  synthetic/cubes/<cube>/<cube>_truth.json \
#                  --out    saved/combined_curvdvl1u_<cube>/curv15000_dvl1u300/param_recovery \
#                  --label  "combined_curvdvl1u curv=15000 lambda=300, <cube>"
# Description: Neutral entry point for the per-epoch recovered-vs-truth parameter
#              overlay. Identical behaviour to plot_iterprior_vs_truth.py, which it
#              delegates to; only the NAME differs.
# Date:        2026-09-07
#
# WHY THIS EXISTS: the plotter is generic -- it reads param_names from the truth
# sidecar, so it serves Mogi (4 params), Sun69 (5) and Okada (8) equally -- but its
# original filename refers to the DEPRECATED iterative-prior ('iterprior') family.
# That family is no longer part of the benchmark and must not be referenced by new
# work, so new job scripts call this name instead. The original file is deliberately
# left in place and unchanged: ~119 existing PBS scripts invoke it, and renaming it
# would break the reproducibility of every archived run for no benefit.

import runpy

if __name__ == "__main__":
    # runpy, not `from plot_iterprior_vs_truth import main`: the target guards its
    # entry point with `if __name__ == "__main__"`, and run_module executes it under
    # that name, so argparse sees this process's argv unchanged and the two scripts
    # cannot drift apart.
    #
    # Do NOT wrap this in sys.exit(): run_module returns the executed module's
    # GLOBALS DICT, and sys.exit(<dict>) prints the whole dict and exits 1 -- which
    # under `set -e` kills the calling PBS job even though the plot was written
    # successfully. Let the delegated module's own exceptions set the exit status.
    runpy.run_module("plot_iterprior_vs_truth", run_name="__main__")
