# Comparison pipeline (Phase 2, 2026-11-23)

One scenario format, one output format, one scorer. Every method reads the same pixels and the same prior box;
the truth is read only by the scorer.

```
scenario npz + truth json  ──►  PILA   methods/pila/{make_config, run_pila, predict}.py
(scenarios/schema.py)      ──►  MCMC   methods/mcmc/{export.py, run_iclassic_perdate.m, import_results.py}
manifest.csv               ──►  vbICA  methods/vbica/{export.py, run_vbica.sh, export_ic_fields.m,
(scenarios/manifest.py)                                grouping.py, perepoch.py}  (+ the MCMC driver per IC group)
                                        │
                     runs/<method>/<scenario>/<run_id>/{params.npz, recon.npz, timing.json, meta.json}
                                        │  (methods/contract.py)
                     compare/compare.py ──► master_table.csv, summary_by_method.csv, figures/
                                        (compare/score.py, compare/figures.py)
```

## Guarantees and their tests
| Guarantee | Test |
|---|---|
| New PILA entry point == legacy launchers (16 cases, bit-identical) | `tests/regression/compare_with_legacy.py` |
| Shared scorer == legacy non-static scorer (51 legacy runs) | `tests/unit/test_score_vs_legacy.py` |
| Scorer edge cases (missed sources, NaN slots, K=0, peaks, slot order) | `tests/unit/test_score_edge_cases.py` |
| Shared prior box == PILA decoder range (Mogi, Sun69, Okada) | `tests/unit/test_priors_match_pila.py` |
| `iclassic_multi` (K=1, no offset) == `iclassic_insar` bit-for-bit; okadaMod/mogiMod == NumPy (centroid) | `methods/mcmc/gate_iclassic_multi.m` |
| Same pixel set for every method (fingerprint in meta.json) | `compare.py` warning + smoke test |
| Whole chain runs for all methods | `tests/integration/smoke_all_methods.sh` |

## Fixed rules (user decisions 2026-11-23)
* Oracle K for every method in the main track; free per-date constant offset for every method
  (MCMC parameter, vbICA per group, PILA residual) -- `--offset fit` is an explicit, required argument.
* MCMC noise scale = std(data) (original), to be revisited.
* Real data: each method reads MintPy itself; the manifest fixes the preparation settings.
* Rank on `rms_phys_vs_clean_mm` (per-date demeaned) and the parameter scores; `rms_recon_vs_data_mm` is
  reported but not comparable across methods.

## Open for Phase 4 (decide BEFORE generating scenarios / looking at scores)
1. PILA recipe: single-source standard cell vs K=2 overlay (`configs/templates/overlay_multisource_k2.json`).
2. Prior box: fixed (`configs/*_paras*.json`) vs data-scaled amplitude box; offset bounds rule.
3. Restart budget per method (PILA seeds / basin hop; MCMC `--n-starts`, `NBay`) and the truth-free rule to pick
   ONE run per method among several (the method's own loss).
4. vbICA: rule for N, IC-grouping rule (`overlap_cluster` is a candidate), its thresholds, convergence policy.
5. Scoring thresholds (1.5 km / 50 % / 50 %, activity 2 x noise RMS, run success 80 %).
6. Compute: vbICA and full MCMC runs need PBS (1 core here: vbICA > 28 min for 30 x 16384 at 6000 steps).
