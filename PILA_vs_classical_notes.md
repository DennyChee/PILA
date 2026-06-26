# PILA vs Classical Inversion — Conceptual Notes

Date: 2026-06-18

## What PILA is

PILA (physics-integrated VAE, `model/model_phys_smpl.py`) is a **constrained
generative inversion**, not a blind source separation method.

- Encoder maps an InSAR displacement field to a latent split into:
  - `z_phy` — parameters of an **explicit forward model** (Mogi, Okada dike,
    Sun69 penny-crack), with physical units (depth_m, dV_m3, geometry, ...)
  - `z_aux` — a small low-rank auxiliary latent for residual / unmodeled signal
- Decoder reconstructs as `physics_forward(z_phy) + aux_correction(z_aux)`

## PILA is NOT ICA / blind source separation

| | ICA | PILA |
|---|---|---|
| Assumption | Sources statistically independent, linear mixing | Sources obey a known physical Green's function / PDE |
| Output | Anonymous components (interpret post-hoc) | Named physical parameters with units |
| Identifiability | From data statistics | From forward model + priors |
| Small "hidden" signal | Can isolate it if independent & non-Gaussian | Captured only if it matches a source in the model dictionary; otherwise lands in `z_aux` residual |

ICA can isolate a low-amplitude independent component **precisely because it
ignores physics**. PILA cannot "name" an unmodeled source — it routes it into
`z_aux`. If a real source type is missing, the fix is to add it under
`physics/`, not to switch to ICA. For **discovery** of unexpected/unmodeled
signal, ICA / PCA / residual analysis is the right tool, not PILA.

## Advantages over classical Bayesian inversion (SA + MCMC)

**Speed via amortization is the headline** — but not the only advantage:

1. **Amortization / speed** — train once, then one forward pass (ms) per scene
   vs per-scene optimization (minutes–hours). Only matters at scale:
   full time-series (per-epoch), multi-volcano, multi-track, near-real-time
   monitoring. At that scale, speed is what makes the analysis possible at all.
2. **Amortized posterior / uncertainty** — `q(z_phy|data)` gives parameter
   uncertainties in the same forward pass; classical UQ means re-running a sampler.
3. **Learned prior across the dataset** — regularizes ill-posed / low-coherence
   scenes where a per-scene optimizer rails to a garbage minimum (helps exactly
   the dV-railing / null-Okada failures seen in the multi-volcano run).
4. **`z_aux` residual channel** — absorbs unmodeled signal so it doesn't corrupt
   the physics parameters; classical single-source inversion has nowhere to put it.

## When classical still wins

- Single, high-quality, well-posed scene where accuracy is everything:
  classical SA + MCMC is the gold standard. PILA won't beat it on one fit;
  use PILA output as an initialization at most.

## Bottom line

Time is the **necessary** condition that makes PILA worth it; the **learned
prior** and **amortized posterior** are what make it worth it for the right
scientific question. Validate the accuracy gap with `compare_pila_vs_classical.py`:
if PILA sits within the classical posterior's uncertainty on test scenes, the
speedup is effectively free.

PILA and ICA are complementary, not competing: ICA/residual analysis as a
front-end **discovery** step → PILA as the fast quantitative **inversion** once
the source dictionary is known.
