# Supervised method (ON HOLD — slot reserved)

Status 2026-11-23: kept in view (KIV). Nothing is trained or run yet. This folder exists so the
supervised network can be added later without touching the other methods.

Contract it will follow (same as every method, finalised in Phase 2):
- `train(manifest_split, cfg)` — train only on scenarios whose manifest `split` is `train`/`val`
  (made by the scenario generator from the SAME prior box and noise model, separate seeds;
  test scenarios are never seen, checked by hash).
- `predict(scenario) -> runs/supervised/<scenario>/<run_id>/{params.npz, recon.npz, timing.json, meta.json}`
  — per-date `[T, K, P]` parameters in SI units (p5/p50/p95 if the network outputs a distribution).

Starting point when switched on: the Stage-1 DETR-lite network in the legacy tree
(`model/multisource.py`, `train_multisource.py`, `configs/multisource/`), unchanged.
