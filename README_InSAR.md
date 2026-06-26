# PILA for InSAR — MintPy-native source inversion (Mogi & Okada, point & full-field)

This document explains the InSAR extension of PILA: the theory, the scientific
decisions, what the code does, and how to run it. It is self-contained — a new user
should be able to read this top to bottom and understand both *why* and *how*.

For the general PILA architecture (the physics-informed low-rank VAE), see the main
[`README.md`](README.md). This file covers everything InSAR-specific.

---

## 1. What this does, in one paragraph

We take a **MintPy line-of-sight (LOS) displacement time series** (a
`geo_timeseries_*.h5` cube) and invert it for a **volcanic deformation source** — either a
**Mogi** point source (inflation/deflation of a magma reservoir) or an **Okada** opening
dislocation (a dike/sill). The inversion is done by a physics-informed variational
autoencoder (PILA): an encoder reads the displacement field and infers the *physical
source parameters*; a differentiable forward model (Mogi/Okada) re-renders the predicted
displacement; the mismatch trains the encoder. Two encoder flavours are provided — a
**point/MLP** model (downsampled observation points) and a **CNN full-field** model
(the whole displacement image) — so you can compare them on the same data.

---

## 2. Scientific background

### 2.1 InSAR LOS displacement
InSAR measures ground motion **projected onto the radar line of sight** — a single scalar
per pixel, not a 3-D vector. Convention used throughout (matching MintPy):

> **positive LOS displacement = motion *toward* the satellite** (range decrease).

The MintPy cube is already unwrapped, referenced (to a reference date and a reference
pixel), and in **metres**. We convert to **mm** because the physics decoders output mm.

### 2.2 The forward models
Both forward models predict the **3-D ENU surface displacement** `(u_E, u_N, u_U)` at a set
of ground locations, then we project to LOS.

- **Mogi (1958)** — a point pressure source at depth. Parameters: horizontal position
  `xcen, ycen` (km), depth `d` (km), volume change `dV` (m³). Produces a smooth,
  radially-symmetric uplift/subsidence bowl.
- **Okada (1985), opening-only** — a rectangular tensile dislocation (dike/sill).
  Parameters: centroid `xoff, yoff` (km), `depth` (km), `strike`, `dip` (deg),
  `length`, `width` (km), `opening` (m). Produces an asymmetric pattern set by the fault
  geometry.

### 2.3 Why we keep observations in LOS and project the *model* (not the data)
A single LOS measurement is **one** number per pixel; the ENU displacement is **three**
unknowns. Decomposing LOS → ENU is therefore **underdetermined** and would need either
multiple look geometries or a hard assumption (e.g. "vertical only"). We avoid that
entirely:

```
                forward (well-posed)
  z_phy  ──►  Mogi/Okada  ──►  (u_E, u_N, u_U)  ──►  d_LOS = e_E·u_E + e_N·u_N + e_U·u_U
                                                       ▲
                                          per-point LOS unit vector e=(e_E,e_N,e_U)
  measured d_LOS  ───────────────────────────────────►  compared here (LOS space)
```

So the data **stay in LOS**, and the model's ENU prediction is **projected into LOS** to
match them. The per-point unit vector `e` is the geometry that makes this projection
correct.

### 2.4 The LOS unit vector (and a fixed bug)
The unit vector is derived from MintPy's `incidenceAngle` and `azimuthAngle` using the
**exact MintPy `enu2los` convention** (verified against MintPy 1.6.1):

```
d_LOS = -u_E·sin(inc)·sin(az) + u_N·sin(inc)·cos(az) + u_U·cos(inc)
  ⇒  e_E = -sin(inc)·sin(az),  e_N = +sin(inc)·cos(az),  e_U = cos(inc)
```

where `az` is MintPy's `azimuthAngle` (LOS, ground→satellite, from north,
anticlockwise-positive) and positive LOS = toward satellite.

> ⚠️ The older `datasets/preprocessing/akutan_insar_prepare.py` fallback had `e_E`/`e_N`
> swapped and the wrong sign on `e_N`. That fallback runs whenever MintPy is **not**
> importable — which is exactly the `pila` training env. The new loader
> `datasets/preprocessing/insar_mintpy.py` hardcodes the **correct** formula and has no
> MintPy dependency. **Sanity check:** `mean(e_U) ≈ 0.7–0.85` for Sentinel-1
> (Sierra Negra S1_TD128 descending gives 0.837).

### 2.5 The local ENU frame — and why the origin is *not* the source
Pixel lon/lat are mapped to a local **East/North grid in km** via an azimuthal-equidistant
projection centred on an origin `(lat0, lon0)`. The origin maps to `(0,0) km`.

**The origin is just a coordinate peg, not the source location.** The source position
(`xcen/ycen`, `xoff/yoff`) is a **free parameter that the inversion solves for**, reported
*relative to* the origin (e.g. "3.2 km east, 1.1 km north of the peg"). The only
requirement is that the source-parameter ranges bracket the plausible source location in
that frame — so pick a fixed peg near the scene (we use the **caldera centre**) and keep
the ranges roughly symmetric about it.

---

## 3. Two inversion methods

Both share the same physics decoders and the same MintPy loader; they differ only in the
encoder and how the field is presented.

| | **Point / MLP** | **CNN full-field** |
|---|---|---|
| Input | N observation points (vector) | Whole scene as one image `[1,H,W]` |
| Encoder | MLP over the N LOS values | Conv stack (ported from the Conv-VAE) |
| Pixels | coherent field **multilooked** to a coarse grid; every coarse cell is a point (~3.5k at multilook=20) | coherent field multilooked to a **finer** grid (~368×315 at multilook=8), rendered as an image |
| Physics output | LOS at the N points `[B,N]` | LOS over all grid cells, flattened `[B,H·W]`, reshaped to the image |
| Loss | MSE in standardized LOS space | **masked** MSE (incoherent cells excluded) |
| Standardization | **global** z-score | **global** z-score |
| Decoder residual (`dim_z_aux`) | enabled (low-rank augmentation) | `0` (pure physics) |

Why two? The point/MLP path is light and is the original PILA design. The CNN path uses
the **whole** field (your "no downsampling" instinct) at higher spatial resolution; a
single global source produces one coherent deformation pattern, so the CNN must see the
whole scene at once (independent patches can't infer one global source).

> **Note on resolution:** for *source-parameter* inversion the deformation signal is smooth
> and km-scale (low spatial frequency), so native 14 m pixels are not needed. The CNN grid
> size is configurable via `multilook`; the conv encoder uses adaptive pooling so any grid
> size works.

### 3.1 PILA vs classical inversion — cost, scaling, and when each is worth it

PILA was benchmarked against the classical analytical pipeline (simulated annealing +
Bayesian MCMC) on 18 configs (3 volcanoes × 3 source models × 2 tracks). The classical jobs
were timed (`runtime_sa_sec` / `runtime_mcmc_sec` / `runtime_total_sec` in each
`classic_out/*/*_result.mat`); PILA inference latency is in
`comparison_out/pila_inference_benchmark.csv`; PILA training wall-time is in each run's
`saved/.../models/compute_usage.json` (`train_sec`). The joined table is
`comparison_out/classical_vs_pila_timing.csv` and the summary figure is
`comparison_out/pila_vs_classical_summary.{png,pdf}` (regenerate with
`python make_comparison_figure.py`). Full write-up: `multivolcano_results_summary.md`.

**Headline numbers (all CPU timing):**

| Metric | Value |
|---|---|
| Accuracy | PILA ≈ Bayesian MCMC physics-only R² (tie-or-better on ~13/18) — **no accuracy cost for the speed** |
| Per-scene latency | classical median **21.8 s** vs PILA **1.19 ms** → marginal speedup **median ~18,000×** |
| Training (one-time) | median **47 s/config** (range 28–108 s; 150 epochs) |
| Break-even | `train_sec / classical_per_scene` = median **~1.8 scenes** (< 1 for Okada) |
| Full-stack speedup incl. training | median **~27×** (range 2.9–159×) over a track's 28–65 epochs |

> **What "break-even" means.** PILA pays its training cost once, then inverts each new scene
> in ~1 ms; classical re-pays its full SA+MCMC cost on every scene. Break-even is the number
> of scenes you invert before PILA's training is repaid. Below it, classical is cheaper on
> wall-time; above it, PILA wins and the gap widens with stack size (the amortization /
> "scale-up" story).

**Decision guide — which to use:**

| Situation | Use | Why |
|---|---|---|
| Time series / ongoing monitoring | **PILA** | well past break-even; cost amortizes over every epoch |
| Expensive source model (Okada), even 1 scene | **PILA** | break-even < 1 scene → wins on speed immediately |
| Low-coherence / data-starved scene (e.g. La Palma), any N | **PILA** | learned prior wins on **accuracy**, ignore break-even |
| Model already trained for the region | **PILA** | training is sunk; a new scene costs ~1 ms |
| **One cheap (Mogi) scene, trained from scratch, speed-only** | **Classical** | break-even > 1 → classical SA+MCMC is faster for that single scene |

> **Caveats (for honest reporting).** (1) The ~18,000× is the *marginal* (per-extra-scene)
> ratio; the ~27× is the training-inclusive total-cost ratio over a full stack. (2)
> Amortization here is **per-config** — each PILA model is trained on, and reused within, one
> track's own time series; it is *not* a single model that generalizes zero-shot to new
> volcanoes. (3) PILA timings are CPU; classical ran on PBS nodes (ppn=4) — same
> order-of-magnitude hardware, and the 10³–10⁴× gaps dwarf the difference, but it is not the
> identical machine.

---

## 4. Data pipeline

`datasets/preprocessing/insar_mintpy.py :: load_insar_mintpy(...)` is the single source of
truth. Steps:

1. **Load** the LOS cube (`timeseries`, metres) and dates from `geo_timeseries_*.h5`.
2. **Coherence mask**: `geo_maskTempCoh.h5` `mask` ∧ finite-at-every-epoch.
3. **Mask-aware multilook**: block-average over `multilook × multilook` blocks using only
   coherent pixels; a coarse cell is kept if its coherent fraction ≥ `coh_valid_frac`
   (default 0.5). This reduces ~1.4M coherent pixels to a trainable count **and** suppresses
   noise. Geometry (`incidenceAngle`, `azimuthAngle`, lon, lat) is multilooked the same way.
4. **ENU coords + LOS unit vectors** per coarse cell (§2.4, §2.5).
5. **Referencing**: **kept exactly as MintPy produced it** (its REF_DATE / REF pixel). No
   extra re-referencing is applied.
6. **Standardization**: a single **global** (scene-wide) z-score for both paths, computed
   over all epochs so Stage A and Stage B share one scaler. (Per-point z-scoring is avoided
   for InSAR: with ~94% quiescent cells it inflates noise to signal scale and collapses the
   fit; the loader still exposes per-point stats but they are not used by default.)
7. **Memoization**: results are cached on `(paths, multilook, lat0, lon0, coh_valid_frac)`,
   so the dataset side and the model side get the **identical** points/grid + scaler from a
   single computation. (Deterministic multilook also means there is no random-sampling
   mismatch to coordinate.)

**Stages.** Both stages use the **whole time series** (each epoch is the cumulative LOS
field at that acquisition, referenced to epoch 0) — exactly like the GNSS PILA, which
trains on a time series, not a single field. A VAE needs many samples, so:
*Stage A* serves every epoch as an **independent** sample (mirrors `DisplacementGPS`).
*Stage B* serves **temporal sequences** that drive the smoothness regularizer, pinning the
inferred source location/geometry across consecutive epochs while amplitude is free
(mirrors `DisplacementGPSSeq`). The CNN path currently has only the independent-epoch mode
(no sequence/temporal-smoothness variant yet — see §8), so its Stage A and Stage B configs
are equivalent.

---

## 5. Code map

```
datasets/preprocessing/insar_mintpy.py   # THE loader: multilook, ENU, LOS vectors, std (memoized)
datasets/displacementGPS.py              # datasets:
   DisplacementInSARh5 / …Seqh5          #   point path, Stage A / Stage B
   DisplacementInSARImage                #   CNN path, full-field image (+ mask)
data_loader/data_loaders.py              # loaders:
   InSARh5DataLoader / …Seqh5DataLoader  #   point path
   InSARImageDataLoader                  #   CNN path
model/model_phys_smpl.py                 # model:
   _load_insar_inputs / _load_insar_grid_inputs   # points / grid from h5 (or legacy JSON)
   FeatureExtractor (encoder_type mlp|cnn)         # MLP or conv encoder
   Physics_Mogi_LOS  / Physics_Okada_LOS           # point-path LOS decoders
   Physics_Mogi_LOS_Grid / Physics_Okada_LOS_Grid  # CNN-path full-field LOS decoders
   _maybe_set_insar_input_dim                      # auto-sets input_dim from the data
trainer/trainer_phys_smpl.py             # added: optional masked-MSE for the CNN image path
configs/sierranegra_{mogi,okada}_paras.json        # Sierra Negra source-parameter ranges
configs/phys_smpl/SierraNegra_{Mogi,Okada}_{InSAR,CNN}_{A,B}.json   # 8 ready configs
```

**Data flow (training):** config → loader builds the dataset (LOS values) and the model
builds the physics decoder (coords + LOS vectors) — both call the *same* memoized loader →
encoder infers `z_phy` → physics decoder renders LOS → (masked) MSE → backprop.

---

## 6. Scientific decisions & assumptions (read before trusting results)

1. **LOS sign**: positive = toward satellite (MintPy). Uplift → positive LOS (since
   `e_U > 0`). For Sierra Negra the final-epoch field is strongly positive (inflation).
2. **Observations stay in LOS**; the model is projected into LOS (§2.3). No LOS→ENU
   decomposition is ever performed.
3. **Origin = fixed frame peg, not the source** (§2.5). Configs use the **caldera centre**
   `lat0=-0.83, lon0=-91.17`. **Confirm this matches your intended frame** — it must be the
   same frame the parameter ranges are defined in.
4. **Referencing** is left as MintPy set it. If you need a specific pre-event datum,
   re-reference the MintPy cube upstream.
5. **Multilooking** is mask-aware block averaging (not nearest-neighbour resize), so it
   preserves signal and reduces noise. The factor sets the resolution/cost trade-off.
6. **Standardization** is derived from the observations (physics-agnostic), so Mogi and
   Okada share the same scaler for a given scene/grid.
7. **Parameter ranges** in `sierranegra_*_paras.json` are **best-effort placeholders**.
   They define the physical search space; tune them to Sierra Negra after first runs.
   (Mogi `dV` is mapped internally as `dV_phys = dV·1e5 − 1e7` m³; Okada `xoff/yoff/depth/
   length/width` are in km, `opening` in m, angles in deg.)

---

## 7. How to use

### 7.1 Environment
The pipeline runs in the **`pila`** conda env (torch, pandas, h5py, pyproj, sklearn). It
does **not** require MintPy at runtime.

```bash
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila
```

### 7.2 Point the configs at your data
Each config has an `insar` block (repeated in `arch.args` and the `data_loader` keys — they
must match; the memoized loader guarantees consistency when they do):

```json
"insar": {
  "timeseries": ".../geo_timeseries_SET_ERA5_demErr.h5",
  "geometry":   ".../geo_geometryRadar.h5",
  "mask":       ".../geo_maskTempCoh.h5",
  "lat0": -0.83, "lon0": -91.17,    // caldera-centre frame peg
  "multilook": 20,                   // 20 → ~3.5k points (point path); 8 → 368×315 (CNN)
  "coh_valid_frac": 0.5
}
```

`input_dim` is **auto-set** from the multilook (you don't compute N by hand).

### 7.3 Run a training
```bash
# Point / MLP   (both stages use the whole time series)
python train_pila.py --config configs/phys_smpl/SierraNegra_Mogi_InSAR_A.json   # Stage A: epochs independent
python train_pila.py --config configs/phys_smpl/SierraNegra_Mogi_InSAR_B.json   # Stage B: temporal sequences
python train_pila.py --config configs/phys_smpl/SierraNegra_Okada_InSAR_A.json
python train_pila.py --config configs/phys_smpl/SierraNegra_Okada_InSAR_B.json

# CNN full-field
python train_pila.py --config configs/phys_smpl/SierraNegra_Mogi_CNN_A.json
python train_pila.py --config configs/phys_smpl/SierraNegra_Okada_CNN_B.json
# ... (Okada_CNN_A, Mogi_CNN_B exist too)
```
Checkpoints + logs land under the config's `trainer.save_dir`.

### 7.4 Adapt to a new scene
1. Set the three h5 paths and `lat0/lon0` (frame peg near the source).
2. Choose `multilook` (point: ~20; CNN: ~8). First run prints the grid size / N.
3. Copy `sierranegra_*_paras.json` → new ranges for your volcano and point the config at
   them. Keep Okada key order: `xoff, yoff, depth, strike, dip, length, width, opening`.

### 7.5 Reading results
The encoder's `z_phy` (in `[0,1]`) rescales to physical parameters via the decoder's
`rescale()`. Recover them by encoding a field and rescaling — e.g. via `test_pila_insar.py`
or by calling `model.physics_model.rescale(z_phy)`. Source location comes out **in km
relative to the origin**; convert back to lon/lat with the same AEQD projection if needed.

### 7.6 Visualizing the FULL reconstructed time series (GNSS-analog)

`plot_insar_results.py` renders a **single epoch** (`--epoch -1` = final cumulative): one
spatial map of actual vs reconstructed LOS. That is the spatial analog of *one time-slice*
of the GNSS result — useful, but not the whole series.

For the full series — the InSAR analog of the GNSS `plot_mogi_results.py` line graphs — use
`plot_insar_timeseries.py`. It runs the trained model over **every** epoch (one batched
pass) and writes three deliverables:

1. **`*_maps_over_time`** — actual / reconstruction / residual LOS maps across a set of
   epochs (small-multiples grid on a shared colour scale; add `--gif` for an animation over
   all epochs). This is the *temporal evolution of the spatial pattern* — the true analog of
   the GNSS per-station line graphs, because an InSAR epoch is a 2-D field, not a scalar.
2. **`*_pixel_timeseries`** — target vs predicted LOS(t) at representative points (peak /
   intermediate / far-field) — the 1:1 analog of the GNSS `timeseries_uz` figure.
3. **`*_parameters` / `*_scatter` / `*_fit_over_time`** — inferred source parameters over
   time (`dV(t)`, depth(t), …), pooled predicted-vs-target scatter, and per-epoch RMSE/R².

```bash
python plot_insar_timeseries.py \
    -r saved/sierranegra_mogi_insar_A/<ts>/models/model_best.pth \
    --n_map_epochs 6 --gif
```

> **Scientific note.** PILA inverts **per epoch** (amortized): each epoch gets its own
> source-parameter estimate, so `dV(t)` is a genuine time series, but a fixed source
> location is only *softly* enforced (Stage B temporal-smoothness reg), not guaranteed. The
> parameter / maps-over-time plots make any epoch-to-epoch geometry wobble visible — expect
> the early, near-zero-deformation epochs to be poorly constrained.

---

## 8. Known limitations / follow-ups

- **CNN temporal smoothness** is not implemented: both CNN stages feed each epoch as an
  **independent** image (the trainer can't batch 5-D sequence-of-images tensors), so
  `SierraNegra_*_CNN_A.json` and `_CNN_B.json` are currently equivalent. The point-path
  Stage B *does* have temporal smoothness. Wiring it for the CNN is a clean follow-up.
- **Parameter ranges** are placeholders (§6.7). The Okada point-path initial loss can be
  very large at random init because the opening/size ranges can produce off-scale
  displacement; it converges, but tune the ranges for stability.
- **Legacy paths** (CSV + `akutan_insar_prepare.py`, GNSS configs) are left intact and
  untouched; only the new h5 path was added.
- **`enu2los` fallback bug** still lives in `akutan_insar_prepare.py` (§2.4) — fix it there
  too if that script is ever revived.

---

## 9. Quick verification

```bash
conda activate pila
python - <<'PY'
from datasets.preprocessing.insar_mintpy import load_insar_mintpy
g='/eos-rs/INSAR_processing/denny/SierraNegra/S1_TD128/mintpy/geo/'
d=load_insar_mintpy(g+'geo_timeseries_SET_ERA5_demErr.h5', g+'geo_geometryRadar.h5',
                    g+'geo_maskTempCoh.h5', lat0=-0.83, lon0=-91.17, multilook=20)
print("grid", d.los_mm.shape, "| coherent cells", int(d.mask_d.sum()),
      "| mean e_U", round(float(d.losU_pts.mean()),3))   # expect ~0.84
PY
```
Expected: `grid (25, 147, 126) | coherent cells 3475 | mean e_U 0.837`.
