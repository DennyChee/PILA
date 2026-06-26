# Inversion parameters — PILA vs MCMC (Bayesian posterior median)

Inferred source parameters for the 18-config benchmark (3 volcanoes × 3 source models × 2
tracks), comparing **PILA** (amortized deep-learning inversion, from
`saved/.../figures/*_params.txt`) against **MCMC** (classical Bayesian posterior **median**,
`post_med` in `classic_out/*/*_result.mat`). Machine-readable source:
`comparison_out/inversion_params_pila_vs_mcmc.csv`. Accuracy (R²) for each config is in
`multivolcano_results_summary.md`; timing in `comparison_out/classical_vs_pila_timing.csv`.

**Units & conventions:** lon/lat in degrees; `d`/`depth`/`radius`/`length`/`width` in km;
`dV` in ×10⁶ m³ (**positive = inflation**, negative = deflation); `strike`/`dip` in degrees;
`opening` in m. LOS sign: positive = motion toward satellite (MintPy convention). Source
position is the inverted source lon/lat (the local-ENU peg `lat0`/`lon0` is *not* the source).

> **Scope:** these are physics-only source parameters (PILA's `z_aux` residual is excluded for
> a fair comparison with the analytical MCMC source). MCMC values are the posterior **median**;
> 16/84% credible intervals (`post_lo`/`post_hi`) are not tabulated here — for the low-R² scenes
> the posteriors are broad and the divergent parameters are correspondingly unconstrained.

## Mogi

| Config | Method | lon | lat | d (km) | dV (×10⁶ m³) |
|---|---|---|---|---|---|
| Etna_Mogi_TA44 | PILA | 14.9764 | 37.7223 | 2.87 | 9.50 |
| Etna_Mogi_TA44 | MCMC | 14.9748 | 37.7223 | 2.99 | 10.39 |
| Etna_Mogi_TD124 | PILA | 14.8907 | 37.7202 | 6.08 | −17.90 |
| Etna_Mogi_TD124 | MCMC | 14.8272 | 37.7191 | 11.76 | −37.10 |
| LaPalma_Mogi_TA60 | PILA | −17.8817 | 28.5825 | 2.23 | 5.90 |
| LaPalma_Mogi_TA60 | MCMC | −17.8986 | 28.5471 | 2.76 | 45.06 |
| LaPalma_Mogi_TD169 | PILA | −17.8848 | 28.5715 | 3.43 | 11.40 |
| LaPalma_Mogi_TD169 | MCMC | −17.8872 | 28.5688 | 3.59 | 11.95 |
| Nyiragongo_Mogi_TA174 | PILA | 29.2145 | −1.6336 | 4.61 | 82.20 |
| Nyiragongo_Mogi_TA174 | MCMC | 29.2147 | −1.6336 | 4.59 | 81.85 |
| Nyiragongo_Mogi_TD21 | PILA | 29.2847 | −1.6491 | 6.93 | 140.30 |
| Nyiragongo_Mogi_TD21 | MCMC | 29.2842 | −1.6494 | 6.79 | 135.36 |

## Sun69 (penny-shaped crack)

| Config | Method | lon | lat | depth (km) | radius (km) | dV (×10⁶ m³) |
|---|---|---|---|---|---|---|
| Etna_Sun69_TA44 | PILA | 14.9649 | 37.7210 | 4.42 | 2.84 | 11.90 |
| Etna_Sun69_TA44 | MCMC | 15.1290 | 37.7309 | 8.94 | 4.22 | −28.11 |
| Etna_Sun69_TD124 | PILA | 14.9261 | 37.7201 | 1.18 | 7.27 | −16.60 |
| Etna_Sun69_TD124 | MCMC | 14.9067 | 37.7188 | 7.45 | 3.32 | −21.63 |
| LaPalma_Sun69_TA60 | PILA | −17.8891 | 28.5782 | 1.81 | 3.18 | 6.80 |
| LaPalma_Sun69_TA60 | MCMC | −17.8566 | 28.7264 | 10.61 | 2.87 | −33.24 |
| LaPalma_Sun69_TD169 | PILA | −17.8819 | 28.5698 | 5.16 | 0.88 | 13.10 |
| LaPalma_Sun69_TD169 | MCMC | −17.8841 | 28.5666 | 5.25 | 1.34 | 13.71 |
| Nyiragongo_Sun69_TA174 | PILA | 29.2062 | −1.6329 | 5.13 | 5.45 | 91.80 |
| Nyiragongo_Sun69_TA174 | MCMC | 29.2083 | −1.6334 | 6.90 | 2.90 | 99.56 |
| Nyiragongo_Sun69_TD21 | PILA | 29.2948 | −1.6554 | 11.58 | 1.14 | 172.40 |
| Nyiragongo_Sun69_TD21 | MCMC | 29.2942 | −1.6543 | 11.28 | 1.35 | 167.15 |

## Okada (rectangular dislocation)

| Config | Method | lon | lat | depth (km) | strike (°) | dip (°) | length (km) | width (km) | opening (m) |
|---|---|---|---|---|---|---|---|---|---|
| Etna_Okada_TA44 | PILA | 14.9908 | 37.7274 | 1.41 | 355.9 | 63.2 | 8.98 | 1.41 | 2.25 |
| Etna_Okada_TA44 | MCMC | 15.0032 | 37.7200 | 1.22 | 0.3 | 53.1 | 0.46 | 5.70 | 14.12 |
| Etna_Okada_TD124 | PILA | 15.1035 | 37.6981 | 7.60 | 196.7 | 72.6 | 7.76 | 3.33 | 5.29 |
| Etna_Okada_TD124 | MCMC | 15.0984 | 37.7007 | 7.05 | 196.9 | 70.5 | 3.84 | 1.96 | 17.26 |
| LaPalma_Okada_TA60 | PILA | −17.9171 | 28.5572 | 3.71 | 27.8 | 44.6 | 10.05 | 1.35 | 3.40 |
| LaPalma_Okada_TA60 | MCMC | −17.9834 | 28.6982 | 9.83 | 209.1 | 66.8 | 7.97 | 5.62 | 12.45 |
| LaPalma_Okada_TD169 | PILA | −17.9634 | 28.5273 | 4.44 | 147.9 | 38.9 | 2.65 | 14.04 | 12.55 |
| LaPalma_Okada_TD169 | MCMC | −17.9585 | 28.5522 | 2.79 | 145.7 | 59.0 | 2.72 | 12.38 | 5.78 |
| Nyiragongo_Okada_TA174 | PILA | 29.1944 | −1.6425 | 6.74 | 6.4 | 23.5 | 14.94 | 0.88 | 13.30 |
| Nyiragongo_Okada_TA174 | MCMC | 29.1933 | −1.6417 | 6.98 | 10.0 | 24.3 | 14.47 | 1.47 | 8.59 |
| Nyiragongo_Okada_TD21 | PILA | 29.2488 | −1.6485 | 3.92 | 182.3 | 86.2 | 14.92 | 5.69 | 2.12 |
| Nyiragongo_Okada_TD21 | MCMC | 29.2482 | −1.6466 | 3.73 | 182.7 | 84.1 | 14.95 | 5.53 | 2.16 |

## How to read the agreement

- **Where both methods fit well, the parameters agree closely** — Nyiragongo across all three
  models (e.g. Mogi_TA174 d=4.61 vs 4.59 km, dV=82.2 vs 81.9; Okada_TD21 near-identical), and the
  well-constrained La Palma/Etna TD169 cases. This is the parameter-level confirmation of the R²
  parity reported in `multivolcano_results_summary.md`.
- **Where the scene is hard (low R²), parameters diverge — as expected.** Etna_Sun69_TA44 and
  LaPalma_Sun69_TA60 even **flip the sign of dV** (PILA inflation vs MCMC deflation), and the
  Okada cases trade off length↔width↔opening differently (Etna_Okada_TA44: PILA length 8.98 km /
  opening 2.25 m vs MCMC length 0.46 km / opening 14.12 m — comparable moment, very different
  geometry). These are the non-identifiable / multi-source scenes where neither inversion is
  trustworthy, consistent with their poor R². The disagreement reflects parameter
  non-uniqueness on weak data, not a PILA-vs-MCMC discrepancy per se.

## Excluded

- **Agung** (Okada, single-event test) and **Sierra Negra** (Mogi/Sun69) have no classical
  SA+MCMC run, so they are not in this comparison. PILA-only parameters live in their respective
  `saved/.../figures/*_params.txt`.
</content>
