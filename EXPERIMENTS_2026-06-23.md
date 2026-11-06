# PILA Experiments — 2026-06-23 (double-confirm + audit)

Summary and verification of the four experiment families run on the PILA amortized InSAR
inversion framework on **2026-06-23 → early 2026-06-24**. Each section gives the
experimental **design**, what was **varied vs held fixed**, the **metrics**, the **key
result**, **caveats**, and an **audit verdict** (did it actually run correctly).

Compiled 2026-06-24. Read-only audit; no jobs were re-run and no code was changed.

**Shared conventions (all synthetic experiments).** Marapi idealized grid: 128×128 px over
40 km (312.5 m/px), descending Sentinel-1 geometry (incidence 33.14°, azimuth −101.98°,
losU ≈ cos33° ≈ 0.837), 76 epochs at 12-day cadence, cumulative referencing (epoch 0 ≈
quiescent). Noise = real precomputed **Marapi APS cube** (~46 mm RMS, trop-dominated ~44 mm)
loaded from the MATLAB synthetic-noise struct; SNR_dB = 20·log₁₀(signal_RMS / noise_RMS)
with signal_RMS at the peak epoch. **No inverse crime**: synthetic forward = independent
NumPy (`synthetic/forward_numpy.py`); PILA decoder = torch physics modules. Multilook=1
(full resolution) for all synthetic runs. PILA inference is **per-epoch / amortized** and
**self-supervised** (LOS reconstruction loss; ground truth used only for scoring).

---

## 1. Synthetic SNR breakdown test

**Provenance:** `synthetic/snr_sweep.py`, `synthetic/noise.py`,
`synthetic/generate_timeseries.py`, `synthetic/evaluate.py`,
`synthetic/snr_failure_analysis.py`. Outputs in `synthetic/snr_sweep_ml1/` (global
standardization) and `synthetic/snr_sweep_ml1_ff/` (far-field robust standardization variant).

**Design.** A **Mogi static buildup** (fixed source geometry: xcen = −13438 m, ycen = 6208 m,
depth = 3000 m) is inverted at 120 training epochs while the **signal amplitude is swept**:
`dV = 2e7 · 10^(SNR/20)`, giving **24 points spanning −12.06 to +16.89 dB**. The noise field
is held **fixed** (the real Marapi APS cube); only the signal strength changes, so the sweep
isolates the signal-to-noise limit of recovery with everything else constant.

| Varied | Held fixed |
|---|---|
| Signal amplitude dV (→ realized SNR, 24 levels) | Source geometry, noise cube, grid/geometry, cadence, training epochs |

**Metric stance.** Recovery is judged by **parameter error (location, dV), not LOS RMSE.**
LOS RMSE stays bounded (~50 mm) even at total breakdown when parameters are completely wrong —
it is a poor failure metric. Headline number = **SNR₅₀** from a logistic fit of
recovery-probability vs SNR.

**Key results.**
- Scene-wide **SNR₅₀ = 3.71 dB** (95% CI 3.27–4.11; AUC 0.95).
- Local signal-weighted SNR₅₀ = 6.18 dB (AUC 0.90) → **scene-wide SNR predicts recovery
  better than local SNR**. Mechanism: PILA's global z-score standardization inflates the
  scale with scene-wide noise, shrinking the standardized signal everywhere.
- **Far-field robust standardization** (center/scale from pixels >8 km from the source) was
  tested as a lever → SNR₅₀ = **4.22 dB**, i.e. *no improvement* (marginally worse, within CI).
  **Hypothesis rejected.**
- Breakdown is **probabilistic, not a hard step**: the per-epoch view (≈1380 inversions)
  shows individual epochs reconstructing or failing depending on that epoch's erratic
  tropospheric noise (logistic transition width ≈ 2.3 dB).
- **Multilook gives ~0 dB SNR gain** for APS-dominated noise (correlated components are
  immune to averaging; only white noise drops).
- Detectability: min detectable dV ≈ **2.5×10⁷ m³** at SNR₅₀ (noise_rms ≈ 37.6 mm).

**Caveats.** LOS RMSE is not a recovery metric (above). Recovery is meaningful only on
strong-signal epochs (≥50% of peak). The result is specific to PILA's global-standardization
design; noise-aware weighting remains an untested lever.

**Audit verdict — PASS.** Sweep complete (24 points, −12.06 to +16.89 dB).
`failure_analysis.json` present for both the global and far-field variants with matching
detectability constants; all per-amplitude training logs present.

---

## 2. Synthetic moving-source tracking test

**Provenance:** `synthetic/run_moving_tests.py`, `synthetic/trajectories.py`,
`synthetic/generate_timeseries.py`, `synthetic/evaluate.py`, `synthetic/plots.py`. Specs in
`synthetic/specs/marapi_{mogi_rising,sun69_sill,okada_dike}_{clean,combined}.json`. Outputs
in `synthetic/cubes/`, `synthetic/eval/`, `synthetic/figures_moving_source/` (`REPORT.md`).

**Design.** Three geophysically-motivated **time-evolving** sources, each generated **clean**
(noise-free) and **combined** (noisy, APS scaled to a realized **13.0 dB** SNR). Each epoch
genuinely has a different source geometry — a stringent test of whether PILA's per-epoch
amortized inversion can **track** motion, not just fit a static source.

| Scenario | What moves (start → end) | Profiles |
|---|---|---|
| **Mogi rising** | depth 7000→2000 m (ascends), drift ±6 km E / ±4 km N, dV 1e6→8e7 m³ | geometry linear, dV sigmoid |
| **Sun69 sill** | depth fixed 3500 m, radius 800→3500 m (grows), lateral migration, dV 1e6→1e8 m³ | geometry linear, dV sigmoid |
| **Okada dike** | fixed-tip propagation: xoff/yoff/length grow together, length 1→13 km, opening 0.1→1.5 m; dip/strike/width/depth fixed | xoff/yoff/length share one profile (affine fixed-tip invariant), opening sigmoid |

The cumulative single-source-per-epoch convention keeps every frame single-lobe so PILA's
single-source decoder can fit it (true multi-lobe migration is explicitly out of scope).

**Metrics.** Per-parameter recovery-vs-epoch, source trajectory map (truth vs inferred,
marker size ∝ signal), peak-epoch LOS observed / predicted / residual. Scored on
strong-signal epochs (RMS ≥ 50% of peak); early near-quiescent epochs are geometrically
unconstrained and excluded from headline metrics.

**Key results.**
- **Mogi and Sun69 track robustly** — peak horizontal location error **~15–16 m** on a 40 km
  grid, with simultaneous recovery of lateral migration, depth/size change, and inflation.
- **Okada dike under wide default bounds is non-identifiable**: it collapses to a shallow,
  high-opening, near-horizontal sill with a ~180° strike flip (location error ~2.2 km). This
  is **structural opening-mode (tensile dislocation) non-uniqueness in the physics, not a PILA
  failure** — the residuals are smooth and confident; a shallow high-opening sill and a steep
  modest-opening dike produce near-identical surface LOS.
- A **physically reasonable dike prior** (dip 60–90°, opening 0–3 m, width 0.5–8 km, depth
  0.5–6 km, strike 0–180°) recovers all 8 parameters (location error ~31 m, all <5.5% error).
  A **loose-bounds** variant (ranges roughly doubled) gives essentially identical recovery,
  proving the tight box did not trivially force the answer — recovery holds as long as the
  prior excludes the shallow-dip / large-opening pathological corner.

**Caveats.** Okada needs a physical prior to be identifiable; Mogi/Sun69 do not. Judge
recovery only on strong-signal epochs. Cumulative (not incremental) source convention by design.

**Audit verdict — PASS.** All three combined cubes realized **13.0 dB** (verified in
`*_truth.json`: 13.000 / 13.000 / 12.9999). Metrics JSONs present for all three scenarios
plus the **constrained** and **loose** Okada bound variants; `figures_moving_source/REPORT.md`
present.

---

## 3. Stride-decimation UQ ensemble test

**Provenance:** `generate_uq_jobs.py`, `aggregate_uq.py`; stride logic in
`datasets/preprocessing/insar_mintpy.py:376-398`. Training in `saved/stride_uq/`, aggregated
output in `comparison_out/uq_*`.

**Design.** Spatial-sampling uncertainty via **every-n-th-pixel decimation**, **stride n = 50**.
The full coherent-pixel set is partitioned (row-major) into **50 disjoint members** that
**jointly tile** the scene — member `offset` keeps pixels `{offset, offset+50, …}`. This is
**systematic decimation with fixed K=50, NOT a random bootstrap**. Each member is standardized
on its own subset (per-subset scaler). Run on **5 configurations × 50 members = 250 runs**:
Etna Okada TA44 / TD124, Nyiragongo Okada TA174 / TD21, SierraNegra Sun69.

**Metric stance.** Per-parameter median / mean / std / p5–p25–p75–p95 / min–max across the 50
members, at **one common reference epoch** (last cumulative date). Source location
back-projected to lon/lat via the scene ENU origin. The spread therefore measures
**sensitivity to which pixels are sampled**, NOT a formal posterior and NOT temporal variation.

**Caveats.** Because each member uses its own standardizer, a small amount of scaler
variability enters the spread. The spread is systematic (fixed disjoint subsets), so
"inside p5–p95" reads as *consistency*, not statistical significance. For Okada, length/width
can rail to prior bounds — read railed spreads as "unconstrained", not tight uncertainty.

**Audit verdict — PASS.** All 5 ensembles have **50/50** `model_best.pth` checkpoints in
`saved/stride_uq/`, and each `uq_params.csv` contains exactly **50 member rows**.

**⚠ Calibration check vs known truth (2026-06-24, NEW — never done before).** A 50-member
stride-50 ensemble was run on the synthetic okada-dike cube (known truth) — the first test
of whether p5–p95 actually *brackets* truth rather than just being self-consistent. Result:
**coverage = 4/8**, and the misses are systematic — the ensemble *median* sits in the
degenerate conjugate corner (dip≈20° floor, width≈600 m, opening≈7 m, strike≈227° flip) vs
truth (dip 85°, width 3000 m, opening 1.5 m, strike 45°). I.e. the strided UQ is **NOT a
calibrated posterior** here; its spread reflects optimization degeneracy. **Key caveat:** on
the small 128² synthetic grid stride-50 gives only ~327 px/member, vs 14k–50k px/member on
the real scenes (40–150× denser) — at 327 px the dip/opening/width trade-off is
unconstrained, so this is a member-*density* failure (the full-res single run recovers truth
cleanly), not proof the real ensembles are miscalibrated. Implication: read the real-data
"inside p5–p95" counts as *sampling-consistency among possibly-degenerate fits*, NOT
calibrated uncertainty, until verified at production-equivalent density (e.g. stride-20 or a
larger synthetic grid).

**Density sweep (stride 10/20/30/40/50, 2026-06-24).** Confirms a sharp calibration
threshold at **~800 px/member**: stride 20 (819 px) → 90% of members in the true steep-dike
basin, median dip/opening/width on truth (89.6°/1.75 m/3060 m); strides ≤546 px collapse to
the degenerate corner (dip→20°, opening→7 m, width→600 m). (Stride-10 is noisy — 10 members,
and all UQ members share the fixed SEED=123 so basin choice is partly subset-luck; "% in
true basin" is a more reliable metric than coverage/8 at small N.) Since the real ensembles
have **14k–50k px/member (17–60× above the ~800 px threshold)**, the production UQ is in the
calibrated regime; the stride-50-on-synthetic failure was a small-grid sparsity artifact.
Outputs: `comparison_out/uq_calibration_vs_stride/`.

---

## 4. Comparison experiments (UQ vs full-scene PILA; MCMC vs UQ)

**Provenance:** `compare_uq_vs_fullscene.py`, `compare_mcmc_vs_uq.py`,
`generate_definitive_jobs.py`. Definitive full-scene runs in `saved/full_scene_def/`; MCMC
source `comparison_out/inversion_params_pila_vs_mcmc.csv`. Outputs in
`comparison_out/uq_vs_fullscene{,_def}/` and `comparison_out/uq_vs_mcmc/`.

### 4a. UQ ensemble vs full-scene PILA
Same scene, **different spatial sampling**: full-scene = single run on the block-averaged grid
(multilook); strided UQ = 50 full-resolution disjoint subsets. Agreement therefore tests
**robustness to sampling scheme**, not bias against a ground truth. Two full-scene baselines:
the original **ML=20 (pre-okada-fix)** and the definitive **ML=5 (post-okada-fix,
`saved/full_scene_def/`, 5/5 runs present)**. `z` and "inside p5–p95" are consistency flags only.

### 4b. MCMC vs UQ
Classical **SA + Bayesian** posterior **medians** for **4 Okada configs only** (no classical
runs for Sun69/Mogi), unit-harmonized (km→m), strike flagged circular. Tests whether the
amortized PILA ensemble brackets the classical solution.

**Key results (verified counts).**

| Comparison | Inside ensemble p5–p95 |
|---|---|
| UQ vs full-scene **ML=20** (pre-fix) | **38 / 47** |
| UQ vs full-scene **ML=5** (definitive, post-fix) | **28 / 47** — *lower* |
| MCMC vs UQ | **23 / 32** MCMC params inside |

- **Okada non-uniqueness is clear in the MCMC comparison.** e.g. Etna TA44: PILA length
  8980 m / opening 2.25 m vs MCMC length 455 m / opening 14.1 m — different geometry,
  comparable moment (conjugate trade-off). Nyiragongo length/width rail near the ~15 km prior
  bound → unconstrained, not tight. Large strike z-scores often just mean the two methods
  picked conjugate planes.

**✅ RESOLVED 2026-06-24 — see §5.** The railing/non-identifiability is now traced to the
production Okada configs pointing at the *wide* prior; a gate-validated dike prior
(`configs/okada_dike_paras.json`: dip 20–90°, opening 0–10 m, width 0.5–15 km, strike
0–360°) has been wired into all 4 field Okada configs and their definitive + UQ jobs
regenerated. Re-runs pending submission. The original flag is preserved below for context.

**⚠ Audit flag — needs attention.** **Etna_Okada_TD124_A agreement collapses from 10/10
(ML=20 pre-fix) to 1/10 (ML=5 definitive, post-fix).** The other four configs are stable
(Nyiragongo TA174 8/10, TD21 8/10, SierraNegra Sun69 5/7, Etna TA44 6/10). Inspection of the
ML=5 TD124 full-scene solution shows it **railed to the prior-box corner** — xoff ≈ 14994 m,
yoff ≈ −14973 m, depth ≈ 11857 m, opening ≈ 16.4 m, length ≈ 12835 m — a **degenerate Okada
fit** sitting at the parameter bounds, which is why it falls outside the ensemble on nearly
every parameter. **The definitive ML=5 Etna TD124 run should not be trusted until this is
investigated** (likely a bad local minimum / bounds-railing; consider re-running with a
tighter physical Okada prior, cf. the moving-source dike finding in §2).

**Caveats.** No ground truth in either comparison. The UQ ensemble predates the okada-fix, so
the UQ-vs-definitive comparison mixes code versions on the Okada side — interpret the drop
from 38/47 to 28/47 partly as a version effect, not purely sampling. MCMC medians (not MAP)
are compared.

**Audit verdict — PASS with one flag.** All comparison overview CSVs present and counts
reproduce; the Etna TD124 ML=5 railing above is the one item requiring action.

---

## 5. Okada prior resolution (2026-06-24)

**Problem.** The production Okada field configs (`Etna_Okada_*`, `Nyiragongo_Okada_*`) still
pointed at the *wide* prior (`etna/nyiragongo_okada_paras.json`: dip 1–90°, opening 0–20 m,
strike 0–360°, width 0.1–15 km). The decoder maps each network output affinely into those
bounds (`model/model_phys_smpl.py:456-474`), so the inversion was free to enter the
shallow-dip / large-opening / strike-flipped degenerate corner — the cause of the
Etna TD124 railing in §4b. The §2 fix existed only for synthetic SierraNegra, never in
production.

**Approach.** Validation set to **synthetic-truth-primary** (MCMC itself sits in the
high-opening conjugate corner for some configs, so it is not a trusted reference). A
synthetic okada-dike recovery run on the shared 13 dB combined cube was used as a **gate**:
no field config was changed until a candidate prior recovered the 8 parameters.

**Gate results (strong-epoch MAE; truth: depth 2000 m, dip 85°, length 13 km, width 3000 m,
opening 1.5 m).**

| Prior | loc err | dip | width | opening | verdict |
|---|---|---|---|---|---|
| old wide (dip 1–90, open 0–20) | 2.21 km | 83.8° off | 2771 m | 12.9 m | collapse |
| wide-opening (dip 20–90, open 0–18) | 0.29 km | 4.3° | **2458 m** | **5.8 m** | dip cured, width/opening FAIL |
| **MID = production (dip 20–90, open 0–10)** | **0.15 km** | **4.6°** | **560 m** | **0.1 m** | **PASS** |
| loose (dip 45–90, open 0–6) | 0.11 km | 2.5° | 541 m | 0.1 m | PASS (reference) |

**Key finding.** The **opening cap is the knob that controls the opening↔width moment
trade-off.** With opening free to 18 m the inversion picks a thin/high-opening dike (same
moment, wrong geometry) — width/opening become non-identifiable even though the `dip≥20°`
floor cures the catastrophic near-horizontal collapse. Capping opening at **10 m** breaks
the trade-off and recovers width/opening (peak err 0.6% / 7.3%) as well as the loose 6 m
cap, while still **bracketing the one plausible high-opening MCMC fit (Nyiragongo TA174,
8.6 m)**. Etna's 14–17 m MCMC openings sit *outside* this prior **by design** — the gate
shows those are the moment-trade-off artifact, not real geometry. (The strike 180° flip at
near-vertical dip is the benign conjugate ambiguity, not the pathological collapse.)

**⚠ Real-data follow-up (2026-06-24): the prior is necessary but NOT sufficient — a second
failure mode (optimization) appeared on field data.** The 4 definitive ML5 runs on the new
prior (single hardcoded seed 123) gave: Etna TA44 clean dike (good); Nyiragongo TD21
improved (dip 1.2°→88.5°, R² 0.52→0.62, a prior win); but **Nyiragongo TA174 and Etna TD124
collapsed to NULL sources** (opening→0, all params railed, R² −0.07 and −0.41). This is an
**optimization (local-minimum) failure, not a prior failure**: for TA174 the old wide-prior
run found a good steep dike (R²=0.52) whose parameters all lie *inside* the new box — i.e.
the good solution is reachable, the single deterministic run just didn't find it.
**Resolution = multi-start**: `train_pila.py` now accepts `--seed` (top-level config key
`seed`); `generate_multistart_jobs.py` re-fits each Okada config over 5 seeds and
`select_multistart.py` keeps the best-reconstruction-loss (non-null) run. The 50-member
strided-UQ ensembles (varied data subsets) provide an independent robustness cross-check.

**Multi-start result (best-of-5 seeds, 2026-06-24).** Confirms the null collapse is purely
optimization (8/20 seed runs hit the null basin; best-of-5 escapes it for all 4 configs):
Etna TA44 seed3 R²=0.38 (clean dike, no railing); Etna TD124 seed5 R²=0.20 (no seed null,
but a single dike is a poor model here — adequacy limit); Nyiragongo TA174 seed2 R²=0.59
(rescued from −0.07; length wants >15 km); Nyiragongo TD21 seed2 R²=0.57. So the robust
recipe is **okada_dike prior + best-of-N seeds**. Remaining caveats: TD124 low R² (model
adequacy), Nyiragongo length pins at the 15 km cap (large-source signal, length
unconstrained), and the strike=360° RAILED flag is a benign 0≡360 wrap.

**Shared-prior MCMC vs PILA (2026-06-24, classical re-run on the okada_dike prior).** The
classical SA+MCMC was re-run on the SAME exported LOS points with the okada_dike bounds
(`classic_out_shared_prior/`, ML20; `classic_out_shared_prior_ml5/`, ML5 in progress;
medians in `comparison_out/mcmc_shared_prior/`). Putting both methods on one prior largely
DISSOLVES the original disagreement — confirming it was a prior artifact, not a method gap:
Etna TA44 MCMC opening 14.1→**1.28 m**, length 0.46→**7.77 km** (now matching PILA 1.69 m /
6.86 km); Etna TD124 opening 17.3→**8.79 m**, and depth/dip/length/width all within ~10% of
PILA. Nyiragongo TA174 agrees on depth/dip/length. **TD21 is the lone exception**: even on
the identical prior, MCMC (depth 3.7 km, dip 84°, opening 2.2 m) and PILA (depth 10.4 km,
dip 45°, opening 8.7 m) pick different geometries of comparable moment — genuine residual
non-uniqueness, not a prior or method bug. MCMC remains NOT ground truth (no field truth);
the synthetic gate stays the primary validator.

**Wiring (done 2026-06-24).** New `configs/okada_dike_paras.json` (dip 20–90, opening 0–10,
width 0.5–15, depth 0.5–12, strike 0–360, length 0.1–15). The 4 canonical field configs
repointed to it; `generate_definitive_jobs.py --multilook 5` + `generate_uq_jobs.py
--stride 50` regenerated (definitive configs + 200 UQ member configs verified pointing at
the new prior). `plot_insar_results.py` now writes the prior box + a per-parameter **RAILED**
flag into `*_params.txt` and a footnote on the maps figure (validated on the old railed
TD124 run: flags `xoff, yoff` at ±15 km). Field re-runs + re-validation pending submission.

## 6. Temporal-prior regularization: curvature × dV-L1 grid (2026-07-28)

**Context.** PILA inverts the InSAR cube **per epoch** (amortized), so each epoch's
source parameters are estimated independently — there is no built-in temporal coupling,
and the recovered trajectories carry large **incoherent epoch-to-epoch jitter**, worst
where the source is quiescent and the geometry is weakly identifiable. Two regularizers
were developed separately to inject temporal/physical priors without abandoning the
amortized encoder:

- **Temporal curvature smoothing** — penalise the 2nd time-difference
  `(z[t+1] − 2·z[t] + z[t−1])²` of the source parameters (`temporal_smoothness_mode:
  "curvature"`, `include_dv:true`), computed in the affine z-space over sequence windows.
  It penalises *jerk*, not level, so a smooth genuine ramp survives while incoherent
  jitter is suppressed. Needs the **Seq** loader (`InSARSeqh5DataLoader`, seq_len 7) for
  `sequence_len`. Weight must be **O(1e3–1e4)** because `rec_loss` sums over 16384 px.
- **dV-L1 in u-space** — an L1 pull of the dV parameter toward its center in
  **pre-sigmoid (u) space** (`dv_penalty_space:"u"`). In post-sigmoid z-space the gradient
  carries a `z(1−z)` factor that vanishes at the rail — powerless exactly where dV rails to
  0; in u-space the pull is constant and non-vanishing, so quiescent epochs collapse to
  dV≈0 while the late high-dV rise is preserved.

**Question.** Do the two combine as hoped — curvature kills the jitter, dV-L1 drags the
residual flat level the rest of the way to 0 — or does one dominate? Run the 2D grid.

**Cube.** `marapi_mogi_rising_combined` (76 epochs): a **moving + rising** Mogi source —
xcen −6000→+6000 m, ycen −4000→+4000 m, depth 7000→2000 m, dV a **sigmoid 1e6→8e7 m³**
(peak epoch 75), ECMWF/MatAPS-style APS noise.

**Setup.** `generate_combined_curvdvl1u_jobs.py` clones the validated single-penalty
template `configs/synth/marapi_mogi_rising_combined_tscurv_lam15000.json`, flipping only
the two penalty weights + `dv_penalty_space:"u"`, and emits 6 configs + 6 PBS scripts.
The two penalties are **independent additive loss terms**
(`trainer/trainer_phys_smpl.py:561,563`), so both are active in one run; the Seq train
loader / `InSARh5DataLoader` `type_test` pair is preserved (dV-L1 is per-sample, agnostic
to sequence structure; the per-epoch dump needs the non-Seq test loader). Grid = curvature
`temporal_smoothness_weight` ∈ **{5000, 15000}** × dV-L1 `dv_l1_weight` ∈ **{30, 100, 300}**.
PBS **3761875–3761880**, all clean (77-row `uspace_series.csv` + plots each).

**How the two loss terms are computed (exact).** Both are additive terms in the total
loss (`trainer/trainer_phys_smpl.py:561,563` and `:574,576`), scaled by their weights.
Let `z ∈ (0,1)` be the sigmoid of the u-space latent, `u = logit(z)` the pre-sigmoid
latent, and let the decoder map `z → physical` affinely (`z·(max−min)+min`, geometry
×1000, dV ×scale+shift).

- **Temporal curvature smoothness** (`_temporal_smoothness_loss`, `:1049-1109`). Reshape
  `z_phy` to `(batch, seq_len, dim_z_phy)`, keep the smoothed dims (Mogi geometry xcen,
  ycen, d; +dV since `include_dv:true`), take the **discrete 2nd time-difference** and
  mean-square it:
  ```
  L_curv = mean_over(batch, t, dims) [ ( z[t+1] − 2·z[t] + z[t−1] )² ]      (seq_len ≥ 3)
  ```
  It is computed in **z-space, not u-space**, on purpose: the z→physical map is affine, so
  "linear in physical units ⇔ linear in z", and the synthetic truths are piecewise-linear
  in physical units → **zero z-curvature for a constant-velocity move or a linear dV ramp**;
  only jitter/kinks are penalised. (u-space would penalise the true linear ramp because
  logit is nonlinear.) Needs the Seq loader for `sequence_len`.

- **dV-L1 in u-space** (`:507-515`). Penalise the **absolute deviation of the dV latent
  from its center in pre-sigmoid space**, averaged over the batch:
  ```
  L_dVL1 = mean_over(batch) | u_dV − u0 |,   u0 = logit(z0)
  ```
  where `u_dV` is the raw u-space posterior mean (`z_phy_stat['mean'][…, dv_dim]`) and `z0`
  is the physical-zero point of the affine dV map (`:127`, `z0 = ((−shift/scale) − min)/
  (max−min)`). Under `mogi_paras_symdV.json` (scale 1, shift 0, dV range ±2e8) `z0 = 0.5`,
  so `u0 = logit(0.5) = 0` and the term reduces to `mean |u_dV|`. Acting in **u-space** is
  the whole point: the z-space gradient carries a `z(1−z)` factor that vanishes at the rail
  (z→0/1), so a z-space L1 is powerless exactly where dV rails to 0; the u-space pull is
  constant and non-vanishing, collapsing quiescent epochs to dV≈0 while sparing the rise.

**Metric.** Fluctuation-RMSE vs truth per parameter (`compare_prior_variants.py`),
reported all-epoch (% of truth RMS) and **excluding gated** epochs (truth |dV| < 5% peak,
where geometry is unidentifiable). 4-way vs the two single-penalty baselines:

| param | Baseline (no prior) | Curvature 15000 | dV-L1(u) 100 | **Combined (curv15000 + dV-L1 300)** |
|---|---|---|---|---|
| xcen (m) | 6.75e3 (192%) \| 7.2e3 | 1.32e3 (37%) \| **402** | 2.83e3 (81%) \| 1.65e3 | 1.26e3 (36%) \| 585 |
| ycen (m) | 3.42e3 (146%) \| 2.28e3 | 1.43e3 (61%) \| 408 | 3.39e3 (145%) \| 1.67e3 | 1.45e3 (62%) \| **357** |
| d (m) | 4.5e3 (95%) \| 4.16e3 | 2.36e3 (50%) \| 1.74e3 | 3.72e3 (79%) \| 2.89e3 | **2.13e3 (45%) \| 1.57e3** |
| dV (m³) | 8.99e7 (169%) \| 8.42e7 | 3.3e7 (62%) \| 2.37e7 | 6.68e7 (126%) \| 5.05e7 | **3.12e7 (59%) \| 2.27e7** |

**Key findings.**
- **Curvature carries the result; the curvature *weight*, not dV-L1, is the dominant knob.**
  Every curv=15000 cell beats every curv=5000 cell on dV (curv=5000 dV: 5.36 / 4.95 / 4.62e7
  for dV-L1 30/100/300, vs curv=15000: 3.27 / 3.22 / 3.12e7).
- **The flagged over-smoothing risk did NOT materialise.** 15000 is *better* than 5000, and
  the combined trace still tracks the late sigmoid rise — its **gated** dV RMSE (dominated by
  the high-dV epochs) is the lowest of all variants, so curvature-15000 is not lagging or
  flattening the genuine dV onset.
- **dV-L1 adds a small, monotonic nudge** on top of curvature=15000 (dV all-epoch: 3.27 →
  3.22 → 3.12e7 for L1 = 30 → 100 → 300) — the anticipated "L1 drags the residual flat level
  further" effect, but the gain over curvature-only is **modest (~5%)**.
- **Winner = c15000_d300** (curvature 15000 + dV-L1(u) 300): lowest dV both all-epoch and
  excl-gated, plus best `d` and `ycen`. Curvature-only wins only `xcen`-excl-gated (402 vs
  585). **Practically, curvature-15000-only is ~95% of the way** — the combined prior is the
  marginal best but the extra dV-L1 buys little on this cube.

**Figures.** 4-way overlay `saved/compare_rising_combined_curvXdvl1u_4way.png/.pdf`;
per-cell param-recovery under
`saved/combined_curvdvl1u_marapi_mogi_rising_combined/curv{5000,15000}_dvl1u{30,100,300}/`.
All staged flat in `scp_combined_figs/` for scp.

**Reproduce.**
```bash
cd /eos-rs/INSAR_processing/denny/PILA
conda activate pila
# Regenerate the 6 configs + PBS (idempotent), then submit from a scheduler-capable shell:
python generate_combined_curvdvl1u_jobs.py
for f in curv5000_dvl1u30 curv5000_dvl1u100 curv5000_dvl1u300 \
         curv15000_dvl1u30 curv15000_dvl1u100 curv15000_dvl1u300; do
  qsub jobs/submit_combined_${f}.pbs
done
# Rank all cells vs the two single-penalty baselines + rebuild the 4-way figure:
CROOT=saved/combined_curvdvl1u_marapi_mogi_rising_combined
python compare_prior_variants.py \
  --truth synthetic/cubes/marapi_mogi_rising_combined/marapi_mogi_rising_combined_truth.json \
  --series "Baseline (no prior)=saved/tscurv_marapi_mogi_rising_combined/lam0/uspace_series.csv" \
           "Curvature 15000=saved/tscurv_marapi_mogi_rising_combined/lam15000/uspace_series.csv" \
           "dV-L1(u) 100=saved/dvl1u_marapi_mogi_rising_combined/lam100/uspace_series.csv" \
           "Combined (curv15000 + dV-L1 300)=$CROOT/curv15000_dvl1u300/uspace_series.csv" \
  --out saved/compare_rising_combined_curvXdvl1u_4way
```

## Reproduce the audit numbers

```bash
cd /eos-rs/INSAR_processing/denny/PILA
# Agreement counts (strip CR first)
tr -d '\r' < comparison_out/uq_vs_fullscene/uq_vs_fullscene_overview.csv     | awk -F, 'NR>1{t++;if($NF=="True")i++}END{print i"/"t}'   # 38/47
tr -d '\r' < comparison_out/uq_vs_fullscene_def/uq_vs_fullscene_overview.csv | awk -F, 'NR>1{t++;if($NF=="True")i++}END{print i"/"t}'   # 28/47
tr -d '\r' < comparison_out/uq_vs_mcmc/mcmc_vs_uq_overview.csv               | awk -F, 'NR>1{t++;if($11=="True")i++}END{print i"/"t}'   # 23/32
# SNR50 (global + far-field)
grep snr50_db synthetic/snr_sweep_ml1/marapi_mogi_buildup_combined_failure_analysis.json
grep snr50_db synthetic/snr_sweep_ml1_ff/marapi_mogi_buildup_combined_failure_analysis.json
# Moving-source realized SNR
grep realized_snr_db synthetic/cubes/marapi_{mogi_rising,sun69_sill,okada_dike}_combined/*_truth.json
# UQ member counts
for d in saved/stride_uq/*/; do echo "$d $(find "$d" -name model_best.pth | wc -l)"; done
```

## Bottom line
- Experiments **1, 2, 3** ran cleanly and the headline science is sound.
- Experiment **4** flagged the Etna_Okada_TD124_A railing; **§5 (2026-06-24) resolves it**:
  the cause was the wide production Okada prior, now replaced by the gate-validated
  `configs/okada_dike_paras.json` (dip 20–90°, opening 0–10 m). Definitive + UQ Okada jobs
  regenerated on the new prior; **re-runs + re-validation pending submission**, after which
  the §4b agreement counts should be re-computed.
- Experiment **6 (2026-07-28)** tested combining temporal curvature smoothing with u-space
  dV-L1 on the moving+rising cube. **Curvature smoothing is the dominant regularizer**
  (weight 15000 > 5000, no dV-onset lag); dV-L1 adds only a ~5% dV nudge. Best cell
  **c15000_d300**, but **curvature-15000-only captures ~95% of the benefit** — the combined
  prior is marginally best, not decisively so, on this synthetic cube.
