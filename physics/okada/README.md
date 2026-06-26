# Okada (1985) opening dislocation — PILA physics decoder

This directory provides Okada (1985) rectangular-dislocation forward models used
as physics-informed decoders in PILA, alongside the existing Mogi point source.
The volcano-facing model (`OkadaDike`) is an **opening-only** (tensile) source —
a dike or sill — intended to invert surface displacements for magmatic
intrusion geometry.

This README documents provenance, validation, conventions, and the scientific
assumptions/caveats that bound the model's validity. **Read the
[Scientific caveats](#scientific-caveats) section before interpreting any
inverted parameter.**

---

## Contents

| File | Role |
|------|------|
| `okada_dike.py` | **`OkadaDike`** — opening-only, batched, differentiable torch model. **This is the model PILA uses** (via `Physics_Okada`). Centroid reference convention. |
| `okada.py` | `Okada` — general three-mechanism (strike-slip / dip-slip / tensile) torch model. Used as a cross-check reference; **not** wired into PILA. Different (top-edge) convention — see below. |
| `validate_okada_dike.py` | **Regression + correctness checks for `OkadaDike`** (the PILA model): vs NumPy `okadaMod`, a frozen golden fixture, gradient flow, and finite gradients at degenerate geometry. |
| `validate_okada.py` | Validates `okada.py` against the original DC3D Fortran + a golden fixture + physical-sanity checks. |
| `plot_okada_check.py` | Visual ENU checks; contains a faithful NumPy port of `okadaMod.m`. |
| `plots/` | Generated figures (reference dike, dip sweep, strike sweep, ENU grid). |

Run anything here with the `fastai` conda env:
`/eos-rs/miniconda3/envs/fastai/bin/python physics/okada/<script>.py`.

---

## Provenance

- `okada.py` was written **from scratch** from the published Okada (1985)
  equations. It originally contained six transcription bugs (see
  [Bug history](#bug-history)); these were found and fixed by numerical
  validation, **not** inherited from a vetted source. Trust it only because of
  the validation below.
- `okada_dike.py` is a **faithful port of the user's `okadaMod.m`**
  (`.../matlab/synthetic_deformation/Volcano_Deformation/modified models/okadaMod.m`),
  which is **Beauducel's `okada85`** (IPGP, BSD-licensed, validated against
  Okada 1985) modified to be opening-only and vectorized. This is the
  authoritative reference and is the same forward model used in the MATLAB
  inversion workflow.

---

## Validation status

| Check | Result |
|-------|--------|
| `okada.py` vs **DC3D Fortran** (`okada_wrapper`), random dips 13°–89°, all 3 slip modes | < 1×10⁻⁷ m |
| `okada.py` tensile kernel vs `okadaMod.m` kernel (identical `ξ,η,q,dip`) | ~1×10⁻¹⁶ (machine ε) |
| **`okada_dike.py`** vs NumPy `okadaMod` port, random geometries, full ENU | **1.1×10⁻¹⁵ m** |
| Gradient flow through all 8 parameters | confirmed finite |
| Full `PHYS_VAE_SMPL` encode→decode with Okada decoder | runs; grads propagate |

DC3D is Okada's own published implementation, so agreement with it is an
authoritative correctness check. `okada_dike.py` (the PILA model) additionally
matches the user's MATLAB inversion model to machine precision.

**Outstanding visual check:** confirm `plots/okada_reference_synthetic.png`
matches the MATLAB `uv_plot.png` for the shared synthetic case.

---

## Conventions

**`OkadaDike` / `Physics_Okada` (the PILA model):**

| Quantity | Convention |
|----------|------------|
| `xoff, yoff, depth` | reference the fault **CENTROID** (not the top edge). `depth > 0`, positive down. |
| `strike` | degrees clockwise from North; fault **dips to the RIGHT of the trace**. |
| `dip` | degrees from horizontal, `[0, 90]`. |
| `length, width` | along-strike and down-dip extent, metres. |
| `opening` | tensile opening, metres; **positive = inflation**. |
| Output | `[East | North | Up]`, **millimetres**; positive Up. |
| Poisson's ratio | `ν = 0.25` (fixed). |
| Mechanism | **opening only** (strike-slip = dip-slip = 0). |

Edge depths derive by trigonometry:
`depth_top = depth − (W/2)·sinδ`, `depth_bottom = depth + (W/2)·sinδ`.

**Why centroid (and how it relates to Okada 1985).** Okada's paper references
the fault's **deeper (lower) edge**, not the centroid — his `d` is the depth of
that edge and is what the kernel's `p, q` substitution actually needs. The
exposed parameter is a free choice, related to Okada's edge by trigonometry:

| Exposed convention | → Okada deeper-edge depth `d` |
|---|---|
| Deeper edge (Okada 1985 native) | `d = depth` |
| **Centroid (this model)** | `d = depth + (W/2)·sinδ` |
| Top edge (`okada.py`) | `d = depth + W·sinδ` |

All three are exactly inter-convertible; none is "more correct". We expose the
**centroid** because it matches the user's `okadaMod.m` MATLAB inversion
(Beauducel), so inferred depths are directly comparable to that workflow, and
because it is symmetric (no asymmetric depth–width coupling). Inverted `depth`
must therefore be reported as **centroid depth**.

> **`okada.py` uses a different convention** (`depth` = depth to the *top* edge,
> `(xcen,ycen)` = top-edge centre, and an internal right-of-strike axis). The
> two models agree on the *physics* but **not** on the meaning of `depth` and
> the horizontal reference. Do not mix their inputs. PILA uses `OkadaDike`.

---

## Scientific caveats

These bound the scientific validity of any result. Several are assumptions
inherited from Okada (1985); others are choices specific to this integration.

### Physical-model assumptions (Okada 1985)
1. **Homogeneous, isotropic, linear-elastic half-space.** No crustal layering,
   no elastic heterogeneity, no viscoelastic/poroelastic effects. Real volcanic
   edifices violate this; inferred geometry is an *effective* source in an
   equivalent half-space.
2. **Flat free surface at z = 0.** Observation points are treated as if at sea
   level. **Topography is ignored.** The MATLAB inversion applies a per-station
   `depth + station_elevation` correction; **that correction is OMITTED here**
   (to match how the PILA Mogi model is wired). For a volcano with kilometre-
   scale relief this is a real approximation that biases inferred depth. Revisit
   if station elevations vary substantially.
3. **Fixed Poisson's ratio ν = 0.25** (λ = μ). Not inverted. ν controls the
   horizontal/vertical displacement ratio, so a wrong ν biases geometry.
4. **Uniform opening over a rectangular patch.** No opening distribution,
   tapering, or curvature. A real dike/sill has non-uniform opening.
5. **Static, small-strain linear elasticity.** No time dependence in the physics
   itself (PILA may add temporal features in the residual, not the source).

### Mechanism choice
6. **Opening-only (tensile).** Appropriate for magmatic intrusion (dikes/sills),
   **not** for co-eruptive faulting or flank slip. If shear deformation is
   present, this model cannot represent it and the residual will absorb it
   (possibly corrupting the inferred opening source).

### Interpretation / identifiability
7. **`depth` is the centroid depth, not the dike top.** The source MATLAB
   (`okadaMod.m`) carries a misleading inline comment ("depth representing top of
   dike"), but its math is the unmodified Beauducel centroid transform. Centroid
   and top differ by `(W/2)·sinδ` — for a vertical 1 km dike that is 500 m.
   Report inverted depth explicitly as **centroid depth**.
8. **Source non-uniqueness.** A Mogi sphere, a sill, a prolate spheroid, and a
   dike can fit similar surface data. Choosing the opening-Okada source is a
   modelling decision, not a data-driven certainty.
9. **Opening–area trade-off.** For a small/deep source only the product
   `opening × length × width` (the volume) is well constrained; `opening` and
   `(L, W)` are individually degenerate. Bound them tightly in
   `okada_paras.json` or constrain volume, or the encoder will wander.
10. **Strike degeneracy as dip → 0.** A horizontal sill has no strike
    dependence, so `strike` becomes unidentifiable for near-flat sources.
11. **PILA low-rank residual can mask model inadequacy.** The learnable residual
    `δ = (c·s)·Bᵀ` is added to the physics output to absorb misfit. If the
    residual is large relative to the physics term `x_P`, the inferred physical
    parameters (`z_phy`) are no longer trustworthy — inspect the residual
    magnitude and the warmup scale `r(t)` before interpreting `z_phy`.

### Numerical
12. **Singularities.** The on-fault-plane locus (`q = 0`) and `ξ = 0` are
    handled exactly as in `okadaMod.m` (the `atan` term and `I5` are zeroed
    there). Denominators carry a tiny `eps` (1e-15) guard that is negligible
    (< 1e-18 relative) but prevents NaNs at exact edges.
13. **Vertical fault (dip = 90°)** uses the analytic `cosδ → 0` limiting forms of
    the I-functions (`torch.where` branch), validated against DC3D at 89°+.

---

## Bug history (transparency)

`okada.py` shipped with six transcription bugs, all fixed and then caught/
confirmed by validation. Documented so the validation is auditable:

1. Tensile `uy/uz` used `(R+η)` instead of `(R+ξ)` in the leading term.
2. Vertical-fault `I5` limit set to 0 (should be `−k·ξ·sinδ/(R+d̃)`).
3. Vertical-fault `I1` limit denominator `(R+d̃)·R` (should be `(R+d̃)²`).
4. Vertical-fault `I3` limit denominator `(R+d̃)·R` (should be `(R+d̃)²`).
5. `I5` general form used `atan2` (should be `atan`).
6. `depth` was internally the deeper edge while the docstring claimed the top
   edge (dominant error; off by `(W/2)·sinδ`, and ~3–9× wrong magnitudes).

`okada_dike.py` was ported directly from the vetted `okadaMod.m` and validated
to machine precision, so it does not carry this history — but it shares all the
*physical* assumptions above.

---

## Status & remaining work before training

Built and validated: `OkadaDike`, `Physics_Okada` adapter, `okada_paras.json`,
`configs/phys_smpl/PILA_Okada_C.json`, `physics_init` `'Okada'` branch.

Not yet done (data/config, not physics):
1. **Training data** — `data/processed/okada/{train,valid,test}.csv` do not
   exist yet (config points there). Generate synthetic Okada displacements in
   the same 36-dim, 12-station format as the Mogi CSVs.
2. **Standardization** — `x_mean`/`x_scale` currently **reuse the Mogi files as
   a placeholder**. Compute Okada-specific stats from the Okada training data.
3. **Parameter ranges** — `okada_paras.json` values are **placeholders**, not
   tuned to a specific volcano (e.g. Agung). Adjust to the expected source and
   to mitigate caveats 9–10.

---

## References

- Okada, Y. (1985). *Surface deformation due to shear and tensile faults in a
  half-space.* BSSA 75(4), 1135–1154.
- Okada, Y. (1992). *Internal deformation due to shear and tensile faults in a
  half-space.* BSSA 82(2), 1018–1040. (DC3D Fortran used for validation.)
- Beauducel, F. `okada85` (MATLAB), IPGP — the reference `okadaMod.m` derives
  from this.
- Segall, P. (2010). *Earthquake and Volcano Deformation*, Ch. 3–4.
