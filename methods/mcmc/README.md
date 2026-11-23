# Per-epoch classical SA+MCMC (adapter)

MCMC code stays in its own repo: `iclassic_insar.m` (original, untouched) and `iclassic_multi.m` (K sources +
optional free offset; added 2026-11-23) in `${MCMC_REPO}` (github.com/DennyChee/insar-classical-inversion).

```bash
python -m methods.mcmc.export --scenario-id <id> --out runs/mcmc/<id>/<run_id> --offset fit --n-starts 3
matlab -batch "addpath('methods/mcmc'); run_iclassic_perdate('runs/mcmc/<id>/<run_id>', '<MCMC_REPO>')"
python -m methods.mcmc.import_results --run runs/mcmc/<id>/<run_id>
```
* Same pixels as PILA (export reads every scenario through PILA's loader); same prior box (`methods/priors.py`,
  tested equal to PILA's decoder range by `tests/unit/test_priors_match_pila.py`).
* `gate_iclassic_multi.m`: `iclassic_multi` with K=1, no offset == `iclassic_insar` BIT-FOR-BIT (both start
  modes); okadaMod / mogiMod == `scenarios/forward_numpy.py` (centroid depth) to 3e-8 relative.
* Noise scale = std(data) as the original (user decision 2026-11-23; option `NoiseScale` reserved).
* Best estimate = lowest-misfit retained MCMC sample (`m_best`); band = 2.5 / 97.5 % posterior, after
  post-hoc relabelling by x (label switching between identical slots).
