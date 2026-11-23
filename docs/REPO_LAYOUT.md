# Repository layout (transition period, from 2026-11-23)

The repo is being restructured for a fresh PILA / per-epoch MCMC / vbICA comparison
(plan: `PLAN_clean_repo_full_comparison.md`). During the transition the NEW and LEGACY trees live side by side.
Nothing legacy has been moved or deleted; it is retired only at the end of the project, with explicit approval.

## New (use these for all new work)
| Path | What |
|---|---|
| `pila/` | core package (model, physics Mogi/Sun69/Okada, trainer, data loading, utils) — copied from the legacy modules, imports `pila.*` |
| `pila/extensions/` | optional training behaviours, switched on by config keys (delayed penalty, source log, keep-lowest, basin hop, bias-only init) |
| `methods/pila/run_pila.py` | the ONE PILA training entry point (replaces `train_pila.py` + the `train_pila_*` launchers) |
| `methods/pila/dump_fit.py`, `plot_insar_results.py` | per-acquisition fit dump and verification figures |
| `methods/{mcmc,vbica,supervised}/` | adapters to the external MATLAB repos / supervised slot (Phase 2; supervised on hold) |
| `scenarios/`, `compare/`, `experiments/` | scenario format + manifest, shared scoring, campaign drivers (Phase 2+) |
| `paths.example.yaml` -> `paths.yaml` | machine-specific data roots used as `${NAME}` in configs (paths.yaml is gitignored) |
| `runs/` | all new outputs (gitignored) |
| `tests/regression/` | proof that the new code is bit-identical to the legacy launchers |

## Legacy (unchanged, still runnable)
Root-level `train_pila*.py`, `test_pila_*.py`, `generate_*`, `compare_*`, `plot_*`, ... and the core folders
`base/ model/ trainer/ data_loader/ datasets/ utils/ logger/ physics/ parse_config.py`, `synthetic/`, `configs/`, `jobs/`.
Old outputs: `saved_legacy_20261123/` (gitignored). Snapshot: git tag `legacy-final-20261123`.

## Config switches replacing the legacy launchers
Every optional behaviour is switched on EXPLICITLY under `trainer.extensions`; its settings block stays where it
always was. A settings block without its switch is ignored (with a warning), exactly as legacy `train_pila.py`
ignored it.

| Legacy launcher | `methods/pila/run_pila.py` with |
|---|---|
| `train_pila.py` | (nothing) |
| `train_pila_delayedpen.py` | `trainer.extensions.delayed_penalty: true` (+ `trainer.phys_vae.delayed_penalty` block) |
| `train_pila_logsources.py` | `trainer.extensions.log_per_epoch_sources: true` |
| `train_pila_keepbest.py` | `delayed_penalty: true` + `keep_lowest: true` |
| `train_pila_basinhop.py` | `delayed_penalty: true` + `basin_hop: true` (+ `trainer.phys_vae.basin_hop` block; forces keep_lowest) |
| `train_pila_biasinit.py` | `arch.phys_vae.init_sources_mode: "bias_only"` |

Only difference (more permissive): legacy keep-best / basin-hop crashed without a delayed_penalty block; here they
run without it (then no schedule and no per_epoch_sources.csv).

Proof of equivalence: `python tests/regression/compare_with_legacy.py` (16 cases; weights, optimizer state,
per-epoch CSVs, fit dumps and logs bit-identical to the legacy launchers).
