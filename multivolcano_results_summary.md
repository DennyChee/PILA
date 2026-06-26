# PILA Multi-Volcano Results — Summary & Assessment

Date: 2026-06-18
Source: `saved/<config>/.../figures/` (latest run per config) + `comparison_out/comparison_table.csv`

## How to read the metrics

Each run reports **two** fit metrics (from the `*_maps.png` annotation box):

- **physics-only** (`x_P`): reconstruction from the physical source alone (Mogi / Okada / Sun69).
- **physics + residual** (`x_PB`): physics source **plus** the `z_aux` low-rank correction channel.

> **Important caveat:** `comparison_out/comparison_table.csv` currently logs the
> **physics-only** R²/RMSE, which *understates* the full PILA model. Where this
> doc shows both numbers, they were read directly from the run figures. Single
> epoch per config, amortized (per-epoch) inversion — not joint over the stack.

---

## Sierra Negra (validation baseline) — STRONG

| Model | depth/d | radius | dV (m³) | R² phys | RMSE phys | R² +resid | RMSE +resid |
|---|---|---|---|---|---|---|---|
| Mogi  | 2.40 km | —      | +32.4e6 | 0.917 | 22.2 mm | 0.942 | 19.4 mm |
| Sun69 | 2.09 km | 2.83 km| +30.9e6 | 0.961 | 15.9 mm | 0.965 | 15.1 mm |

- Single near-circular central uplift — the textbook case both models are built for.
- **Sun69 beats Mogi** (15.9 vs 22.2 mm physics-only), consistent with the penny-crack
  model being a better fit for a sill-like reservoir.
- This is the credibility anchor: where the source matches the physics, PILA fits excellently.

---

## Nyiragongo (2021-12 eruption) — GOOD, one failure

| Model | Track | depth/d | dV (m³) | R² phys (table) |
|---|---|---|---|---|
| Mogi  | TA174 | 4.61 km | +82.2e6 | 0.530 |
| Mogi  | TD21  | 6.93 km | +140.3e6| 0.553 |
| Okada | TA174 | 11.97 km| —       | **−0.068 (NULL source)** |
| Okada | TD21  | 3.92 km | —       | 0.671 (best of all runs) |
| Sun69 | TA174 | 5.13 km | +91.8e6 | 0.557 |
| Sun69 | TD21  | 11.58 km| +172.4e6| 0.507 |

- Strongest amplitudes in the set; Mogi/Sun69 capture the main lobe (R² ≈ 0.5–0.56).
- **Nyiragongo_Okada_TA174 is the null-Okada failure**: inferred `length=0.10 km,
  width=0.10 km, opening=0.00 m` → physics panel is **blank**, R²=−0.068. The clear
  observed dipole is missed entirely; only `z_aux` recovers anything (+resid R²=0.247).
  This is the known null-Okada bug — must be fixed before use.
- Okada_TD21, by contrast, is the **best fit overall** (R²=0.671), showing the Okada
  source *can* work — the TA174 collapse is an optimization/init failure, not a model limit.

---

## La Palma (2021 eruption) — MODERATE (single-source limited)

| Model | Track | depth | params | R² phys | R² +resid (where read) |
|---|---|---|---|---|---|
| Mogi  | TA60  | 2.23 km | dV=+5.9e6 | 0.234 | 0.453 |
| Mogi  | TD169 | 3.43 km | dV=+11.4e6| 0.038 | — |
| Okada | TA60  | 3.71 km | open=3.40 m | 0.397 | — |
| Okada | TD169 | 4.44 km | open=12.55 m, W=14 km | 0.204 | — |
| Sun69 | TA60  | 1.81 km | dV=+6.8e6 | 0.223 | — |
| Sun69 | TD169 | 5.16 km | dV=+13.1e6| 0.031 | — |

- **N=105–152 coherent cells** (after `multilook:20`) — deliberate downsampling, NOT a bug.
- Map fits *look* good (dominant uplift lobe + amplitude + location captured), but the
  **scatter reveals a systematic miss of the negative-LOS field** — single sources flatten
  the surrounding deflation. Physics+resid recovers part of it (Mogi TA60: 0.234 → 0.453).
- Scientifically expected: La Palma 2021 was a **dike + reservoir** system; a single
  Mogi/Okada/Sun69 source genuinely cannot fit it. This is a **model-dictionary limit,
  not a PILA failure** — verify classical single-source does no better.
- Okada generally edges out Mogi/Sun69 here (TA60 0.397), as expected for a dike-fed event.

---

## Etna (2019) — MODERATE/WEAK (complex, low-amplitude scenes)

| Model | Track | depth/d | params | R² phys | R² +resid (where read) |
|---|---|---|---|---|---|
| Mogi  | TA44  | 2.87 km | dV=+9.5e6 | 0.146 | — |
| Mogi  | TD124 | 6.08 km | dV=−17.9e6| **−0.206** | 0.187 |
| Okada | TA44  | 1.41 km | open=2.25 m | 0.449 | — |
| Okada | TD124 | 7.60 km | open=5.29 m | 0.190 | — |
| Sun69 | TA44  | 4.42 km | dV=+11.9e6| 0.092 | — |
| Sun69 | TD124 | 1.18 km | dV=−16.6e6| −0.064 | — |

- Lowest scores in the set. The observed fields are **complex, multi-lobe and
  low-amplitude** (±150 mm, patchy) — likely a mix of weak deformation + atmosphere.
- Etna_Mogi_TD124 (R²=−0.206) is **not railing** — dV=−17.9e6, d=6 km are plausible;
  the single Mogi just mislocates a complex deflation pattern. +resid lifts it to 0.187.
- Okada_TA44 is the best Etna run (0.449). Descending track (TD124) is consistently worse
  on all three models — possible geometry/coherence or referencing issue worth checking.

---

## Comparison with classical methods (SA & Bayesian MCMC)

Source: `comparison_out/comparison_table.csv` (run 2026-06-18 21:37). Three methods per
config: **PILA**, **SA** (simulated annealing), **Bayesian** (MCMC). All three are
**physics-only** source fits (no `z_aux`) → this is an **apples-to-apples** comparison of
the inverted physical source. Metric = R² (and RMSE_mm) of reconstructed vs observed LOS.

### R² by config

| Volcano | Model | Track | PILA | SA | Bayesian | Read |
|---|---|---|---|---|---|---|
| Nyiragongo | Mogi  | TA174 | 0.530 | 0.524 | 0.530 | tie |
| Nyiragongo | Mogi  | TD21  | 0.553 | 0.552 | 0.554 | tie |
| Nyiragongo | Sun69 | TA174 | **0.557** | 0.529 | 0.554 | PILA ≈ best |
| Nyiragongo | Sun69 | TD21  | 0.507 | 0.502 | 0.507 | tie |
| Nyiragongo | Okada | TD21  | 0.671 | 0.539 | 0.673 | PILA≈Bayes, both ≫ SA |
| Nyiragongo | Okada | TA174 | **0.605** | 0.573 | 0.605 | **FIXED** — was −0.068 (null-Okada); now = Bayesian |
| Etna | Mogi  | TA44  | 0.146 | 0.073 | 0.147 | PILA≈Bayes, beat SA |
| Etna | Mogi  | TD124 | −0.207 | −0.201 | −0.188 | all fail (hard scene) |
| Etna | Okada | TA44  | **0.449** | 0.229 | 0.435 | PILA best |
| Etna | Okada | TD124 | 0.190 | 0.135 | 0.192 | PILA≈Bayes |
| Etna | Sun69 | TA44  | 0.092 | 0.197 | **0.206** | **PILA worst** |
| Etna | Sun69 | TD124 | **−0.064** | −0.165 | −0.110 | PILA least-bad |
| LaPalma | Mogi  | TA60  | 0.233 | 0.167 | **0.291** | Bayesian best |
| LaPalma | Mogi  | TD169 | 0.038 | −0.148 | 0.041 | PILA≈Bayes ≫ SA |
| LaPalma | Okada | TA60  | **0.397** | 0.052 | 0.092 | **PILA ≫ classical** |
| LaPalma | Okada | TD169 | **0.204** | −0.018 | −0.383 | **PILA ≫ classical** |
| LaPalma | Sun69 | TA60  | **0.223** | 0.014 | 0.006 | **PILA ≫ classical** |
| LaPalma | Sun69 | TD169 | 0.031 | −0.144 | **0.034** | PILA≈Bayes ≫ SA |

### Headline findings

1. **PILA matches gold-standard Bayesian MCMC on most configs** — tie-or-better on ~13/18,
   despite being a single amortized forward pass vs per-scene MCMC. The amortization speedup
   therefore comes at **no systematic accuracy cost**. This is the core result.
2. **SA is consistently the weakest** of the three — frequently below both PILA and Bayesian
   (e.g. LaPalma Okada TA60 0.052, Nyiragongo Okada TD21 0.539). PILA ≫ SA almost everywhere.
3. **PILA decisively beats classical on the data-starved La Palma scenes** (Okada/Sun69:
   PILA 0.20–0.40 vs classical ≈0 or negative). With only **N=105–152 coherent cells**, the
   per-scene optimizers overfit / wander (note classical source locations drifting to
   28.6–28.7°N, off the edifice), while PILA's **learned prior regularizes** the inversion.
   This is direct evidence for the learned-prior advantage argued in
   `PILA_vs_classical_notes.md` — exactly where classical struggles, PILA wins.
4. **Nyiragongo_Okada_TA174 — FIXED (2026-06-18).** Was PILA R²=−0.068 (null-Okada collapse:
   opening=0, length/width railed to min) vs Bayesian 0.605 / SA 0.573. Root cause: physics-
   branch posterior collapse — the residual `z_aux` absorbed the signal while opening→0 made
   the Okada term (and its length/width gradients) vanish. All three designed safeguards were
   off in the config. Fix (config-only): `epochs_pretrain: 0→30` (synthetic physics bootstrap)
   + `edge_penalty_weight: 0.0→0.01` (repel z_phy→0 corner). **After retrain: opening=13.3 m,
   length=14.9 km, depth=6.7 km → physics-only R²=0.605 (exactly matches classical Bayesian),
   physics+resid R²=0.799.** Run `0618_215030`. Comparison table regenerated → TA174 row now
   0.605.

   **Rollout finding (do NOT apply blanket):** the same fix was tested on the other 5 Okada
   configs (none were collapsed). It mildly *degraded* the already-healthy fits — Etna_Okada_TA44
   0.449→0.222, Nyiragongo_Okada_TD21 0.671→0.601, LaPalma_Okada_TA60 0.397→0.335 — because
   `epochs_pretrain=30` spends epochs on the bootstrap and `edge_penalty=0.01` penalizes the
   legitimately near-bound solutions those scenes found. **Conclusion: the fix is a targeted
   remedy for collapse, not a universal improvement — apply only to collapse-prone configs.**
   The 5 healthy configs were reverted to original; their superseded runs are quarantined in
   `saved/_superseded_okadafix_20260618/`. Open question: whether a gentler setting
   (`edge_penalty≈0.001` / pretrain-only) is a safe universal null-guard.
5. **Hard scenes fail for everyone** — Etna_Mogi_TD124 is negative R² for all three methods;
   that epoch's signal is genuinely not single-source (or atmosphere-dominated), not a method
   failure. Honest to report as such.
6. **Caveat:** PILA's tabulated R² is physics-only and thus *understates* the deployed model
   (physics+`z_aux` is higher, e.g. LaPalma Mogi TA60 0.234→0.453). The classical methods
   have no equivalent residual channel, so physics-only is the correct comparison — but the
   paper should state that PILA's *operational* reconstruction is better still.

### What this enables for the paper

The comparison now supports the central claim: **PILA achieves classical-Bayesian-quality
source inversion at amortized (sub-second) inference cost, and is more robust than classical
optimizers on low-coherence / data-starved scenes.** Two things must ship first: (a) fix
null-Okada (Nyiragongo TA174); (b) add the classical per-scene timing alongside PILA inference
latency to quantify the speedup, since accuracy parity makes speed the decisive axis.

## Inference latency + physics+residual metrics (blockers b & c)

Source: `comparison_out/pila_inference_benchmark.csv` (`benchmark_pila_inference.py`,
50 timed forward passes/config, CPU). `r2_phys` reproduces the comparison table exactly
(same inference path) → validates both. `r2_full` = physics+`z_aux` (PILA's deployed output).

| Config | N | infer (ms, median) | R² phys | R² full | Δ(full−phys) |
|---|---|---|---|---|---|
| Nyiragongo Mogi TA174  | 2321 | 0.80 | 0.530 | 0.772 | +0.242 |
| Nyiragongo Mogi TD21   | 6369 | 1.18 | 0.553 | 0.696 | +0.143 |
| Nyiragongo Okada TA174 | 2321 | 5.05 | 0.605 | 0.799 | +0.194 |
| Nyiragongo Okada TD21  | 6369 | 5.64 | 0.671 | 0.769 | +0.098 |
| Nyiragongo Sun69 TA174 | 2321 | 1.13 | 0.557 | 0.788 | +0.231 |
| Nyiragongo Sun69 TD21  | 6369 | 3.92 | 0.507 | 0.655 | +0.148 |
| Etna Okada TA44        | 2096 | 4.43 | 0.449 | 0.685 | +0.236 |
| Etna Mogi TA44         | 2096 | 0.81 | 0.146 | 0.433 | +0.287 |
| LaPalma Okada TA60     |  105 | 4.45 | 0.397 | 0.597 | +0.200 |
| LaPalma Mogi TA60      |  105 | 0.70 | 0.234 | 0.453 | +0.219 |
| … (all 18 in CSV) | | | | | |

**Headline numbers:**
- **Median single-scene inference latency = 1.19 ms (CPU)**; range 0.7–5.6 ms. Okada is
  ~4–6× slower than Mogi/Sun69 (heavier rectangular-dislocation forward model), still ms-scale.
- **Mean R² uplift from the `z_aux` residual = +0.253.** PILA's *operational* reconstruction
  is materially better than the physics-only number used in the classical comparison — so the
  physics-only table is a conservative lower bound on deployed performance.

**Reporting guidance for the paper:** quote physics-only R² for the head-to-head vs classical
(fair), and physics+residual R² as PILA's deployed reconstruction (with the caveat that
classical has no residual channel). Inference latency is the amortized cost per new epoch.

**Classical timing — MEASURED (2026-06-22):** `iclassic_insar.m` (instrumented with `tic/toc`)
was rerun for all 18 configs (NBay=1e4, SEED=1; PBS 3620605–3620622) and now saves
`runtime_sa_sec`, `runtime_mcmc_sec`, `runtime_total_sec` into each `classic_out/*/*_result.mat`.
All 18 result.mat verified present with finite runtime fields (the PBS `*_matlab.log` files were
not retained in `classic_jobs/`, so completion was validated via mat-file integrity instead — all
18 present, runtimes finite, obs/model arrays populated). Wall-times read with `h5py` (v7.3/HDF5).
Joined table written to `comparison_out/classical_vs_pila_timing.csv`.

Per-config classical wall-time (single scene) vs PILA amortized inference latency
(`comparison_out/pila_inference_benchmark.csv`, per-config median, CPU):

| Config | SA (s) | MCMC (s) | Total (s) | PILA (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| Etna_Mogi_TA44_A | 0.46 | 8.81 | 9.28 | 0.806 | 11,509× |
| Etna_Mogi_TD124_A | 0.59 | 9.69 | 10.28 | 0.802 | 12,822× |
| Etna_Okada_TA44_A | 2.05 | 143.40 | 145.46 | 4.433 | 32,812× |
| Etna_Okada_TD124_A | 1.65 | 103.71 | 105.36 | 3.805 | 27,691× |
| Etna_Sun69_TA44_A | 0.95 | 30.93 | 31.89 | 1.226 | 26,009× |
| Etna_Sun69_TD124_A | 0.83 | 28.88 | 29.71 | 1.195 | 24,858× |
| LaPalma_Mogi_TA60_A | 0.31 | 2.26 | 2.56 | 0.699 | 3,666× |
| LaPalma_Mogi_TD169_A | 0.31 | 2.75 | 3.06 | 0.854 | 3,578× |
| LaPalma_Okada_TA60_A | 0.48 | 13.71 | 14.19 | 4.454 | 3,186× |
| LaPalma_Okada_TD169_A | 0.45 | 15.87 | 16.32 | 4.853 | 3,362× |
| LaPalma_Sun69_TA60_A | 0.46 | 13.98 | 14.44 | 0.993 | 14,546× |
| LaPalma_Sun69_TD169_A | 0.56 | 8.42 | 8.98 | 1.020 | 8,806× |
| Nyiragongo_Mogi_TA174_A | 1.29 | 15.86 | 17.15 | 0.797 | 21,521× |
| Nyiragongo_Mogi_TD21_A | 0.96 | 25.52 | 26.48 | 1.178 | 22,476× |
| Nyiragongo_Okada_TA174_A | 2.52 | 147.19 | 149.71 | 5.051 | 29,639× |
| Nyiragongo_Okada_TD21_A | 5.70 | 331.04 | 336.74 | 5.644 | 59,663× |
| Nyiragongo_Sun69_TA174_A | 1.41 | 37.08 | 38.49 | 1.132 | 34,000× |
| Nyiragongo_Sun69_TD21_A | 1.21 | 37.38 | 38.59 | 3.915 | 9,858× |

**Headline speedup:** median per-config amortized speedup = **18,033×**; equivalently
median classical 21.81 s/scene vs median PILA 1.19 ms/scene = **18,386×**. Range 3,186×
(LaPalma_Okada_TA60) → 59,663× (Nyiragongo_Okada_TD21). MCMC dominates SA by ~10–60×, and
Okada (rectangular dislocation, more parameters) is the heaviest classical model — exactly the
configs where PILA's amortized advantage is largest. The expected 10³–10⁵× speedup is now a
**measured ~10⁴×**.

**Sanity check (2026-06-22):** re-ran `compare_pila_vs_classical.py` on the post-timing-rerun
mat files and diffed SA/Bayesian R², RMSE and inverted source locations against the pre-rerun
`comparison_out/comparison_table.csv`. **Zero drift** (max |ΔR²|=0.0000, max |ΔRMSE|=0.0, source
lon/lat identical) across all 36 classical rows — same data + SEED=1 reproduced bit-identical
inversions, so only timing instrumentation changed. The accuracy table below is unaffected.

**Total cost INCLUDING training (2026-06-22):** the ~18,000× above is the *marginal* (per-extra-scene)
latency ratio and excludes PILA's one-time training. Training wall-time was read from each run's
`saved/.../models/compute_usage.json` (`train_sec`, all CPU, 150 epochs). Per-config training =
**median 47.0 s** (range 28.1–107.5 s; all 18 configs combined = 15.3 min). Each PILA model is
trained once per config/track over that track's *whole* time-series stack (`n_samples_epochs` =
28–65 epochs), then amortizes over every epoch.

- **Break-even** = `train_sec / classical_per_scene`: **median ~1.8 scenes**. For the expensive
  Okada configs it is **< 1 scene** (PILA wins on the first scene even counting training); only
  for the cheapest Mogi scenes (La Palma, ~2.6 s classical) does it reach 16–19 scenes.
- **Full-stack total cost** (invert all 28–65 epochs of a track, training included): PILA is
  **median ~27× faster end-to-end** (range 2.9× LaPalma_Mogi_TA60 → 159× Nyiragongo_Okada_TD21).

So even charging PILA its full training cost, it wins for any realistic time-series; classical only
wins on a *single* low-cost (Mogi) scene that is never reused. **Fairness caveat:** PILA
train+inference timings were on a CPU; classical SA/MCMC ran on PBS compute nodes (ppn=4) — same
order-of-magnitude hardware, not the identical machine, but the 10³–10⁴× gaps dwarf any few-×
hardware difference. Per-config numbers in `comparison_out/classical_vs_pila_timing.csv`; summary
figure in `comparison_out/pila_vs_classical_summary.{png,pdf}`.

## Cross-cutting findings

1. **Where the source matches the physics, PILA is excellent** (Sierra Negra R²≈0.92–0.96).
   This validates the method.
2. **Complex/multi-source events (La Palma, Etna) are single-source-limited**, not PILA
   failures — motivates the multi-source dictionary and/or joint multi-source inversion.
3. **Null-Okada bug FIXED** on Nyiragongo_Okada_TA174 (was opening→0). Cause: physics-branch
   collapse with all safeguards disabled. Fix: enable `epochs_pretrain=30` +
   `edge_penalty_weight=0.01` → R² −0.068 → 0.605 (matches classical). Roll out to other Okada.
4. **Metric reporting bug**: comparison table logs physics-only, understating PILA by
   ~+0.2 R² typically (LaPalma Mogi: 0.234 → 0.453). Fix to log physics+residual (or both).
5. **Classical comparison DONE** (SA + Bayesian, 18 configs) — PILA matches Bayesian MCMC
   on ~13/18 and beats classical on the data-starved La Palma scenes. See "Comparison with
   classical methods" above. Parity claim now supported.
6. **Track asymmetry**: descending tracks (Etna TD124, LaPalma TD169) generally fit worse —
   check LOS geometry / referencing / coherence per track.

## Publishability read

- **Sierra Negra + Nyiragongo (excl. the null-Okada run)**: results are solid and support a
  "fast amortized inversion recovers the dominant source" claim.
- **Classical comparison now strengthens the case**: PILA ≈ Bayesian MCMC accuracy at
  amortized cost, and **more robust than classical on low-coherence La Palma** — the
  learned-prior advantage is now demonstrated, not just argued.
- **La Palma / Etna**: frame honestly as single-source limits; classical does no better
  (and often worse) on the same scenes — which the table now shows.
- **Before submission (blockers):** (a) ~~fix null-Okada~~ **DONE** (TA174 now R²=0.605 via
  `epochs_pretrain=30` + `edge_penalty_weight=0.01`); roll the same config fix out to the other
  Okada runs as a safeguard and regenerate the comparison table; (b) log physics+residual
  metrics; (c) ~~add classical-vs-PILA **timing**~~ **DONE (2026-06-22)** — measured median
  amortized speedup **~18,000×** (classical median 21.8 s/scene vs PILA 1.19 ms; see "Classical
  timing — MEASURED" above); (d) explain track asymmetry; (e) ≥1 GNSS check.

---

# Downsampling / Preprocessing

Source: `datasets/preprocessing/insar_mintpy.py` (`load_insar_mintpy`) +
`datasets/displacementGPS.py` (`_standardized_los_series_h5`). Identical settings
across all 20 runs unless noted.

## Pipeline applied to every scene

1. **Input** — MintPy geocoded cumulative LOS time series
   (`geo_timeseries_SET_ERA5_ramp_demErr.h5`): already SET- + ERA5-atmosphere-,
   ramp-, and DEM-error-corrected. Referencing is **kept as MintPy produced it**
   (its `REF_DATE` / REF pixel); no extra re-referencing is applied.
2. **Multilooking** — **mask-aware block averaging over 20×20 pixel blocks**
   (`multilook: 20`). A coarse cell is kept only if its **valid (coherent & finite)
   fraction ≥ `coh_valid_frac = 0.5`**, i.e. ≥50% of the 400 native pixels in the
   block are coherent; otherwise the cell is dropped (incoherent). This both reduces
   the field (~1.4 M coherent native pixels → ~10²–10³ cells) and suppresses noise.
3. **Coherence masking** — applied via the MintPy `geo_maskTempCoh.h5` temporal-coherence
   mask; incoherent cells are excluded from the loss (point path) or zeroed (image path).
4. **Units** — metres → **mm** (to match the physics decoders' mm output).
5. **Standardization** — single **global, scene-wide z-score** (one shared mean/scale),
   NOT per-point. (Per-point z-scoring was rejected because it amplifies the ~94%
   quiescent cells' noise to signal scale and flattens the deformation → fit collapses to ~0.)
6. **Reference frame** — local ENU pegged at config `(lat0, lon0)` = (0,0) km origin;
   **all inferred source positions are km offsets from this peg.** Per-cell LOS unit
   vectors derived via MintPy `enu2los` (+LOS = motion toward satellite).
7. **Crop** — optional `bbox`; **`None` in all runs** (full geocoded extent used).

## Resulting coarse grids (post-20× multilook)

| Volcano | Track | Coarse grid (Hd×Wd) | Coherent cells N | lat0, lon0 (peg) |
|---|---|---|---|---|
| SierraNegra | A     | 147×126 | 3475 | −0.83, −91.17 |
| Nyiragongo  | TA174 | 49×100  | 2321 | −1.52, 29.25 |
| Nyiragongo  | TD21  | 89×107  | 6369 | −1.52, 29.25 |
| Etna        | TA44  | 50×103  | 2096 | 37.751, 14.993 |
| Etna        | TD124 | 47×101  | 1785 | 37.751, 14.993 |
| LaPalma     | TA60  | 59×100  |  105 | 28.61, −17.87 |
| LaPalma     | TD169 | 59×102  |  152 | 28.61, −17.87 |

- **N is post-multilook coherent cells, not full-resolution pixels.** With 20×20 blocks,
  each cell aggregates up to 400 native pixels (≥200 must be coherent to survive).
- **La Palma's very low N (105/152)** reflects severe coherence loss over the fresh 2021
  lava field — only ~100 coarse cells clear the 50% coherence bar. This is physically
  real, not a bug, but it makes La Palma the most data-starved inversion in the set →
  worth a **sensitivity test at multilook 10 / `coh_valid_frac` 0.3** to recover more cells.
- A coarser/finer `multilook` directly trades noise suppression against spatial detail and
  N; 20 was used uniformly, so the effective ground sampling differs per scene with pixel spacing.

---

# Processing & ML Diagnostics

Source: `saved/<config>/.../models/compute_usage.json` + `log/info.log` (latest run per config).

## Common configuration (identical across ALL runs)

| Setting | Value |
|---|---|
| device | **CPU** (no GPU used; `gpu_peak_mb = 0`) |
| encoder_type | `mlp` |
| n_train_epochs | 150 |
| batch_size | 8 |
| multilook | 20 |
| Sentinel-1 wavelength | C-band 0.0555 m (LOS: + = toward satellite) |

Notes on terminology:
- **`coherent_cells` / `n_obs_points`** = number of coherent multilooked pixels (after
  20× multilook) on the coarse grid = the **encoder input dimension**. Model size scales
  with this (MLP first layer), so pixel count directly drives parameter count and RAM.
- **`n_samples_epochs`** = number of InSAR acquisition epochs (displacement maps) used as
  training samples for the amortized encoder.
- **`train_sec`** = wall-clock for all 150 training epochs; inference (per new scene) is a
  single forward pass — sub-second, not separately timed here.

## Per-run resource table

| Volcano | Model | Track | Coh. pixels | Coarse grid | Train epochs (samples) | Model params | Train time (s) | s/epoch | samples/s | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| SierraNegra | Mogi  | A     | 3475 | 147×126 | 25 | 557,376 | 19.1 | 0.128 | 195.9 | 4998 |
| SierraNegra | Sun69 | A     | 3475 | 147×126 | 25 | 557,634 | 23.8 | 0.158 | 157.8 | 4995 |
| Nyiragongo  | Mogi  | TA174 | 2321 | 49×100  | 61 | 400,432 | 36.8 | 0.245 | 248.7 | 3300 |
| Nyiragongo  | Okada | TA174 | 2321 | 49×100  | 61 | 401,464 | 59.8 | 0.399 | 153.0 | 3298 |
| Nyiragongo  | Sun69 | TA174 | 2321 | 49×100  | 61 | 400,690 | 37.4 | 0.249 | 244.9 | 3301 |
| Nyiragongo  | Mogi  | TD21  | 6369 | 89×107  | 28 | 950,960 | 31.6 | 0.211 | 132.7 | 3031 |
| Nyiragongo  | Okada | TD21  | 6369 | 89×107  | 28 | 951,992 | 59.3 | 0.395 |  70.8 | 3034 |
| Nyiragongo  | Sun69 | TD21  | 6369 | 89×107  | 28 | 951,218 | 28.1 | 0.187 | 149.5 | 3034 |
| Etna        | Mogi  | TA44  | 2096 | 50×103  | 65 | 369,832 | 56.7 | 0.378 | 172.1 | 3636 |
| Etna        | Okada | TA44  | 2096 | 50×103  | 65 | 370,864 | 87.5 | 0.584 | 111.4 | 3635 |
| Etna        | Sun69 | TA44  | 2096 | 50×103  | 65 | 370,090 | 46.0 | 0.307 | 211.9 | 3639 |
| Etna        | Mogi  | TD124 | 1785 | 47×101  | 45 | 327,536 | 41.3 | 0.275 | 163.4 | 2483 |
| Etna        | Okada | TD124 | 1785 | 47×101  | 45 | 328,568 | 72.2 | 0.481 |  93.5 | 2490 |
| Etna        | Sun69 | TD124 | 1785 | 47×101  | 45 | 327,794 | 33.9 | 0.226 | 199.1 | 2488 |
| LaPalma     | Mogi  | TA60  |  105 | 59×100  | 55 |  99,056 | 48.2 | 0.322 | 171.0 | 3542 |
| LaPalma     | Okada | TA60  |  105 | 59×100  | 55 | 100,088 | 52.1 | 0.347 | 158.4 | 3548 |
| LaPalma     | Sun69 | TA60  |  105 | 59×100  | 55 |  99,314 | 44.8 | 0.298 | 184.3 | 3542 |
| LaPalma     | Mogi  | TD169 |  152 | 59×102  | 55 | 105,448 | 48.1 | 0.320 | 171.6 | 3613 |
| LaPalma     | Okada | TD169 |  152 | 59×102  | 55 | 106,480 | 35.4 | 0.236 | 233.2 | 3617 |
| LaPalma     | Sun69 | TD169 |  152 | 59×102  | 55 | 105,706 | 49.9 | 0.333 | 165.4 | 3612 |

## Diagnostic observations

1. **Entirely CPU-bound** — every run is `device: cpu`, `gpu_peak_mb: 0`. Training fits
   comfortably on a single CPU node; **no GPU was needed**. This is itself a scalability
   selling point (commodity hardware), but a GPU would cut the Okada times sharply.
2. **Fast to train** — full 150-epoch training is **~19–88 s** per config (median ~45 s).
   The whole multi-volcano set (20 configs) trains in well under an hour of CPU time.
3. **Okada is the compute bottleneck** — Okada forward model is consistently the slowest
   (`s/epoch` 0.35–0.58 vs 0.13–0.40 for Mogi/Sun69); e.g. Etna_Okada_TA44 = 87.5 s vs
   46–57 s for the other two on the same data. Rectangular-dislocation Green's functions
   are heavier than the analytic point/penny-crack sources.
4. **Model size is driven by pixel count, not physics** — params range 99 k (LaPalma,
   105 px) → 952 k (Nyiragongo TD21, 6369 px), because the MLP encoder's input layer = number
   of coherent cells. Same physics model differs ~10× in size across volcanoes purely from
   coherent-pixel count.
5. **Modest, flat memory** — peak RSS **2.5–5.0 GB**, only weakly tied to problem size
   (data load dominates). Runs anywhere with ≥8 GB RAM.
6. **Heavy decimation** — `multilook: 20` everywhere reduces full-res interferograms to
   coarse grids of ~50×100–150×126 cells. **Pixel counts (105–6369) are post-multilook
   coherent cells, not full-resolution pixels.** La Palma's tiny 105/152 is the steep coherence
   loss over fresh 2021 lava, not a bug — but worth a sensitivity check at lower multilook.
7. **Throughput** ~70–250 samples/s (CPU); Okada sits at the low end as expected.
8. **Caveat for the paper's scalability claim** — the amortization advantage is in
   *inference* (single forward pass per new epoch, sub-second), not training. Training cost
   here is small but is paid once per volcano/track/model. Report inference latency explicitly
   to make the "fast vs classical per-scene MCMC" comparison concrete.
