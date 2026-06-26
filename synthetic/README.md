# PILA Synthetic-Test Framework

Synthetic, ground-truth-controlled testing of PILA (the physics-informed VAE for InSAR
source inversion). We generate synthetic LOS time-series with a **known** deformation
source, optionally add realistic noise, train PILA, and quantify how well it recovers the
source — including the **SNR at which it fails**.

All code lives in `/eos-rs/INSAR_processing/denny/PILA/synthetic/`.
Conda env: **`pila`**. Subprocess training needs `MKL_THREADING_LAYER=GNU`.

---

## 1. Why / what

PILA is amortized and trained **per scene**: it learns an encoder over a *stack* of epochs,
then infers source parameters per epoch. So a synthetic test is:

```
generate synthetic LOS time-series  ->  write MintPy-format HDF5
   ->  train PILA (train_pila.py)  ->  compare inferred params to known truth
```

The pipeline reuses the production loader
`datasets/preprocessing/insar_mintpy.load_insar_mintpy` unchanged — the generator only
writes a new `timeseries` cube and reuses the real scene's geometry/mask, so synthetic data
flows through the exact same path as real data.

**No inverse crime:** synthetic deformation is built with an *independent* NumPy
reimplementation (`forward_numpy.py`), not PILA's own torch decoder; the full-res→multilook
discretization gap keeps the residual small-but-nonzero.

---

## 2. Pipeline stages & tools

| Stage | Module | What it does |
|------|--------|--------------|
| Forward models | `forward_numpy.py` | Independent NumPy Mogi / Sun69 / Okada (match torch decoders < 1e-11 mm). |
| Trajectories | `trajectories.py` | `static_buildup` (dV ramp) and `moving` source; paras-bounds check. |
| Noise | `noise.py` | Loads the Marapi APS component cube; SNR + scaling helpers; orientation fix. |
| Idealized grid | `make_marapi_grid.py` | Builds the 128×128 / 40 km Mode-B base scene (geometry/mask/dates). |
| **Generator** | `generate_timeseries.py` | Builds clean LOS per epoch, injects noise, writes HDF5 + truth JSON + sanity plots. |
| Config | `make_synth_config.py` | Templates a PILA config onto a synthetic cube (`--spec` supplies base geometry; `--multilook`). |
| Evaluate | `evaluate.py` | Inferred-vs-truth: location error (km), per-param error, LOS RMSE. |
| **Validation (gen)** | `validate_generation.py` | Per-epoch component decomposition + per-component all-epoch montages + sum-check. |
| **Validation (loader)** | `validate_loader.py` | Multilook suppression by component; the standardized points PILA ingests. |
| **Validation (inference)** | `validate_inference.py` | Observed / prediction / physics-only / residual montages (all epochs). |
| Orchestrator | `validate_all.py` | Runs all three validation stages for one scenario. |
| Sweep (local) | `snr_sweep.py` | generate→train→eval→validate per amplitude; resumable; breakdown curve. |
| Sweep (PBS) | `run_one_snr_point.py`, `generate_snr_jobs.py`, `aggregate_sweep.py` | Fan one job per SNR point across the cluster, then aggregate. |
| Per-epoch breakdown | `perepoch_snr_diagnostic.py` | Pools every epoch of every run as an independent inversion → dense recovery curve. |
| Failure analysis | `snr_failure_analysis.py` | Logistic detection limit, physical detectability, scene-vs-local SNR. |
| MATLAB | `matlab/regen_noise_cube.m` | Regenerates the noise cube with a stripe-free orbital ramp. |

**Run order (one scenario):**
```bash
python -m synthetic.generate_timeseries --spec <spec.json>
python -m synthetic.make_synth_config --template <cfg> --synth-h5 <cube.h5> \
       --out configs/synth/<name>.json --spec <spec.json> --multilook 1 --epochs 120
python train_pila.py -c configs/synth/<name>.json        # MKL_THREADING_LAYER=GNU
python -m synthetic.evaluate --truth <truth.json> --ckpt <model_best.pth>
python -m synthetic.validate_all --spec <spec.json> --multilook 1 --ckpt <model_best.pth>
```

---

## 3. Conventions (important)

- **Marapi** is the volcano name; the displacement data dir on disk is literally spelled
  `/eos-rs/INSAR_processing/denny/Merapi/S1_TA76/` — keep that path verbatim.
- **SNR** = `20·log10( signal_rms / noise_rms )`, signal at the peak epoch, noise the mean
  per-epoch RMS, over mask pixels (MATLAB `mogi_snr_vbica_bench.m` convention).
- **SNR is varied** by the source amplitude (dV) at **fixed noise**. Calibration:
  `dV = 2.0e7 · 10^(SNR/20)` (2e7→0 dB, 4e7→6 dB, 8e7→12 dB).
- **Orientation:** the MATLAB noise cube is row0=south; `noise.py` applies `flipud`
  (`NOISE_ORIENTATION='flipud'`) to make it north-up (Marapi summit top-left), validated
  against the DEM. **Real geocoded MintPy data is already north-up — do NOT flip it.**
- **Noise source:** `noise_cube_v2.mat` (stripe-free orbital), components
  `trop` (stratified ERA5/MatAPS), `turbulent` (sim_atm), `orbit`, `white`, `combined`.

---

## 4. Findings so far

### 4.1 Machinery validated (clean, no noise)
On Sierra Negra TD128 Sun69:
- **Static buildup:** LOS RMSE **1.2 mm** (signal 204 mm), source-location error **8 m**,
  params 0.5–4%.
- **Moving source:** location error **6.5 m** across a 5×4 km migration (per-epoch tracking works).

### 4.2 Noise is APS-dominated and realistic
The Marapi "combined" noise (≈46 mm RMS) is dominated by the **stratified ERA5/MatAPS `trop`**
component (~44 mm, erratic epoch-to-epoch, anticorrelates −0.45 with topography — physical);
turbulent ~6 mm, orbital ~4 mm, white ~2 mm.

### 4.3 Multilook only suppresses white noise
Mask-aware multilook ×M reduces **white** RMS by ×1/M (verified ×0.25 at M=4) but leaves
**correlated** components (trop, turbulent, orbital, deformation) ~unchanged. For
APS-dominated noise, multilook gives **~no SNR gain** (12.03 → 11.97 dB). The common
"multilook improves SNR" only holds for white/decorrelation noise.

### 4.4 Inference behaves correctly
PILA's prediction ≈ the true deformation; the residual (observed − prediction) ≈ the APS
noise. It fits the source and **does not absorb** the unmodeled atmosphere into the physics.

### 4.5 Cost (no multilook, 16,384 px, CPU)
Train **86.5 s** (0.72 s/epoch, 120 epochs, 2.31M params); infer **3.10 ms/epoch**.

### 4.6 SNR breakdown (Mogi, combined Marapi APS, full resolution)
- **Peak-epoch sweep** (24 runs): a sharp cliff between **3.5 and 4.0 dB**.
- **Per-epoch view** (1382 epoch-inversions pooled): the breakdown is **probabilistic**, not a
  step. Logistic fit: **SNR₅₀ = 3.71 dB** (95% CI 3.27–4.11), transition width 2.29 dB;
  ~90% recovery at **~8.7 dB**, ~0% below ~0 dB. *Within one run*, some epochs reconstruct the
  Mogi and some don't, because each epoch's instantaneous (erratic `trop`) noise gates it.

### 4.7 Physical detectability
Calibrated dV ↔ signal: **4.36×10⁵ m³ per mm** of signal RMS. For any scene:
```
min detectable dV (m³) = 4.36e5 · noise_rms(mm) · 10^(SNR50/20)
```
This scenario (noise ≈ 38 mm): 50%-detectable at signal RMS ≥ 58 mm → **dV ≥ 2.5×10⁷ m³**.

### 4.8 Failure mode: scene-wide SNR governs, not local
Tested scene-wide vs **local (signal-weighted)** SNR as recovery predictors:

| Predictor | SNR₅₀ | width | AUC |
|---|---|---|---|
| **Scene-wide** | 3.71 dB | 2.29 dB | **0.952** |
| Local (signal-weighted) | 6.18 dB | 4.38 dB | 0.899 |

Recovery is governed by **scene-wide** noise, *not* the noise local to the source. Mechanism:
PILA fits the whole field with **global z-score standardization** — high scene noise inflates
the global scale and shrinks the standardized signal everywhere — and large-scale stratified
APS contaminates globally (a broad gradient mimics source position). **Lever to improve:**
local/robust standardization or noise-aware weighting — but note far-field robust standardization
was tested (§5) and did **not** help, so the remaining candidate is noise-aware weighting / the
spatially-modulated-noise test.

### 4.9 LOS RMSE is the wrong failure metric
LOS RMSE stays bounded (and actually *peaks* ~52 mm at the breakdown) even when parameters
are completely wrong. Judge recovery by **parameter error** (location km, dV %), not LOS misfit.

---

## 5. Standardization: global vs far-field robust

Before PILA's encoder sees the LOS field, every value is rescaled
`x_std = (x − center) / scale`. The choice of `center`/`scale` strongly affects the
failure mode (§4.8). Implemented in `load_insar_mintpy(..., standardization=..., far_field=...)`;
both the dataset and the physics decoder read the same `x_mean_global`/`x_scale_global`
(memoized), so training and inference stay consistent. `n_points` is unchanged either way.

An InSAR scene = a compact deforming bullseye in a sea of atmospheric noise.

### `'global'` (default, legacy)
```
center = mean(all coherent cells, all epochs)
scale  = std (all coherent cells, all epochs)
```
`scale` absorbs (1) the **signal itself** (a big deformation inflates std → the signal partly
divides itself out) and (2) the **scene-wide noise** (a noisy scene shrinks the standardized
signal everywhere, even if the source patch is locally clean). This is why recovery tracks
**scene-wide** SNR. Measured: global scale = 80.7 mm — inflated above the ~46 mm noise floor.

### `'far_field_robust'`
Compute the statistics over only the **far field** — coherent cells *outside* a disk of radius
`R` around the (known/assumed) source — using **robust** statistics, then apply that single
scaler to the **whole** scene:
```
ff      = cells with distance(cell, source) > R          # signal-free
center  = median( far-field values, all epochs )
scale   = 1.4826 × MAD( far-field values )               # MAD -> sigma-equivalent
x_std   = (x − center) / scale         # applied to ALL cells, incl. the source patch
```
- **Where (far field):** the scale reflects the **noise floor**, not the signal+noise mixture.
- **Which (robust):** median/MAD ignore outliers (unwrapping errors, decorrelation spikes,
  residual deformation leaking into the far field); ×1.4826 makes MAD a Gaussian-σ equivalent.
- **Applied to all cells:** the source cells are large *relative to the far-field noise*, so they
  become large standardized values — the standardized field is ≈ **signal / noise_floor ≈ local SNR**.

Measured: far-field center = 0 mm, scale = 52.6 mm (≈ the noise floor), over 14,385/16,384 cells
(>8 km from source).

### Why it should help
1. **No signal self-suppression** — `scale` no longer contains the signal.
2. **Decoupled from scene-wide noise** — `scale` is the noise floor presented consistently, not a
   scene-dependent mixture; PILA's input no longer shrinks just because the scene is noisy elsewhere.
3. The encoder receives the deformation already in detectability (SNR-like) units.

Hypothesis under test (19 `_ff` PBS runs in `snr_sweep_ml1_ff/`): far-field standardization should
**lower SNR₅₀** and make recovery track **local** rather than scene-wide SNR (close the AUC gap).

**Result (2026-06-23, 19/19 `_ff` runs complete):** the hypothesis is **not** supported — far-field
standardization gives no meaningful improvement.

| Metric | GLOBAL | FAR-FIELD |
|---|---|---|
| scene SNR₅₀ | **3.71 dB** | 4.22 dB |
| transition width *s* | 2.29 dB | **1.52 dB** |
| AUC (scene SNR) | 0.952 | **0.966** |
| AUC (local SNR) | 0.899 | **0.912** |

SNR₅₀ shifts **−0.50 dB** (far-field needs *slightly more* SNR — marginally worse, but within the
bootstrap CIs, so the two are statistically indistinguishable on threshold). The transition is sharper
under far-field and the AUCs tick up slightly, but **scene SNR stays the better predictor than local
in both schemes** — the AUC gap does not close. **Conclusion: standardization is not the lever; do not
switch to far-field.** Reproduce with `bash synthetic/finish_ff_comparison.sh`. The next candidate
lever is the spatially-modulated-noise experiment (matched scene SNR, quiet vs noisy at the source) —
the decisive scene-vs-local test (see §6).

### Caveats
- Must know the far field (synthetic: trivial; real data: a ring away from the edifice, or iterate
  invert→mask→recompute).
- Does **not** beat the SNR limit — improves consistency/dynamic range, not information content.
- **Requires retraining** (the encoder learned the old scale).
- If the far field isn't truly quiescent, the noise floor is over-estimated → conservative
  (under-detection); robust stats only partly mitigate.

### How to switch between the two

The mode lives in the config's `insar` block. **Default is `global`** (omit the keys entirely, or
set `"std_mode": "global"`). For far-field, add both keys:
```json
"insar": {
    "timeseries": "...", "geometry": "...", "mask": "...", "lat0": ..., "lon0": ...,
    "std_mode": "far_field_robust",
    "far_field": {"source_xE_km": -13.438, "source_yN_km": 6.208, "radius_km": 8.0}
}
```
These must appear in **every** insar block (`arch.args.insar`, `data_loader.args.insar`,
`data_dir_valid`, `data_dir_test`) so the model and dataset use the same scaler.

You normally don't edit JSON by hand — the tooling sets it for you (the source location is read
from the spec's trajectory automatically):

```bash
# global (default) — no flags needed
python -m synthetic.make_synth_config --template <cfg> --synth-h5 <cube.h5> \
       --out configs/synth/<name>.json --spec <spec.json> --multilook 1

# far-field robust
python -m synthetic.make_synth_config --template <cfg> --synth-h5 <cube.h5> \
       --out configs/synth/<name>_ff.json --spec <spec.json> --multilook 1 \
       --std-mode far_field_robust --ff-radius 8

# whole SNR sweep, isolated in its own folder + tag suffix
python -m synthetic.generate_snr_jobs --snr-min -1 --snr-max 8 --snr-step 0.5 \
       --std-mode far_field_robust --ff-radius 8 --tag-suffix ff \
       --out synthetic/snr_sweep_ml1_ff --submit
# (run_one_snr_point takes the same --std-mode / --ff-radius / --tag-suffix flags for one point)
```
Programmatic: `make_config(..., std_mode='far_field_robust', far_field={...})`. Switching modes
**changes the input distribution, so retrain** — keep variants in separate folders (`--tag-suffix`,
`--out`) so global and far-field runs never collide.

**Note:** real geocoded MintPy data is already north-up — the `flipud` fix is only for the MATLAB
Marapi cube, not for real scenes.

---

## 6. Limitations / open items

- Results are for **one slice**: Mogi source, one depth, centered on Marapi, combined APS,
  full resolution, single (descending) track. A full capability chart needs sweeping depth,
  source type (Sun69/Okada), and noise structure.
- The independent forward model is the same *source family* as the inversion (true model
  mismatch — generate CDM/pCDM, invert Mogi — is a stretch goal).
- Local-SNR test used signal²-weighting (high-variance estimator); the decisive test is a
  **constructed** experiment (spatially modulate noise: quiet-at-source vs noisy-at-source at
  matched scene SNR) — not yet run.
- **Realistic noise from real processed scenes** (use a real timeseries' residual as the noise,
  inject synthetic deformation) is the next major extension — see the separate prompt.
- Single-track geometry cannot fully constrain 3-D source geometry (depth/dV, dip/opening
  trade-offs).

---

## 7. Key outputs

- SNR breakdown (peak-epoch): `snr_sweep_ml1/marapi_mogi_buildup_combined_breakdown_snr.png`
- Per-epoch breakdown + recovery probability:
  `snr_sweep_ml1/marapi_mogi_buildup_combined_perepoch_snr.png`,
  `..._perepoch_snr_recovery_fraction.png`
- Failure analysis (scene vs local, logistic, detectability):
  `snr_sweep_ml1/marapi_mogi_buildup_combined_scene_vs_local_snr.png`,
  `..._failure_analysis.json`
- Per-scenario validation plots: `validation/<tag>/{per_epoch,component_montages,loader,inference}/`
- Cubes + ground truth: `cubes/<tag>/<tag>.h5`, `<tag>_truth.json`

(Paths relative to `/eos-rs/INSAR_processing/denny/PILA/synthetic/`.)
