# vbICA (adapter)

vbICA stays in its own repo (`${VBICA_REPO}`). Nothing is written into it: `export.py` builds a self-contained
working directory inside the run directory (vbICA resolves every path relative to `pwd`), with `Code/` symlinked.

```bash
python -m methods.vbica.export --scenario-id <id> --out runs/vbica/<id>/<run_id>
methods/vbica/run_vbica.sh runs/vbica/<id>/<run_id> <N> 1                  # decomposition + IC fields
python -m methods.vbica.perepoch export --run runs/vbica/<id>/<run_id> --n-comp <N> --rule overlap_cluster --offset fit
matlab -batch "addpath('methods/mcmc'); run_iclassic_perdate('runs/vbica/<id>/<run_id>/group_<g>', '<MCMC_REPO>')"   # each group
python -m methods.vbica.perepoch import --run runs/vbica/<id>/<run_id>
```
Steps: (1) export (north-first rows, ascending axes, metres; orientation round trip asserted); (2) vbICA's own
driver; (3) `export_ic_fields.m` rebuilds every IC field with vbICA's own `calc_displ_lincomp` (checked against
`ICA.ts_comp`); (4) group ICs (`grouping.py`), sum each group, invert every date with the SAME per-epoch MCMC
(K=1 per group, free offset, shared prior box); (5) import to the common output.

Removed from the legacy route because they read the truth: IC choice by peak correlation with the clean field,
the unit check against the truth RMS, and the truth gates in `export_ic_group.m`.

OPEN for Phase 4 (must be automatic, truth-free, identical for every scenario): the IC-grouping rule
(`overlap_cluster` is only a candidate) and the choice of N. Real data: vbICA reads MintPy through its own
scenario setup (not yet wired into this adapter). Convergence (`max_steps`) is recorded per run.
