# Per-epoch classical SA+MCMC (adapter — Phase 2)

The inversion code stays in its own repo: `iclassic_insar.m`
(github.com/DennyChee/insar-classical-inversion; local copy path = `${MCMC_REPO}` in `paths.yaml`).

This folder will hold only the adapter: `export.py` (scenario -> the `.mat` input iclassic_insar expects,
per date) and `import.py` (iclassic result mats -> the common `params.npz`). Legacy glue for reference:
`export_pila_points.py`, `compare_pila_vs_classical.py`, `classical_comsol/` at the repo root.
