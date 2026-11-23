# Plan — clean PILA repo + fresh 4-method comparison

Date: 2026-11-23 · Status: Phases 0-2 DONE 2026-11-23 (see docs/COMPARISON_PIPELINE.md); next = Phase 4 decisions

Decisions already made (user, 2026-11-23):
- Methods now: **PILA**, **per-epoch SA+MCMC** (MATLAB, `iclassic_insar.m`), **vbICA** (MATLAB).
  **Supervised NN is on hold (KIV).** The structure must still make it easy to add later (see Phase 3).
- vbICA and MCMC **stay in their own repos**. PILA talks to them only through adapters.
- Main fairness track: **K given to every method (oracle K)**. A "K unknown" track comes later.
- Approved: rename `saved/`, commit `Volcano_Deformation/`, tag the legacy state + branch `clean-restructure`.
- **GitHub cleanup** is part of Phase 0 (see 0.7).
- **All past runs are disregarded.** Every comparison is rerun from scratch with one scenario set and one scoring rule.
- Order: restructure → re-plan synthetic tests → QA gate → pilots → full submission.

---

## Phase 0 — Freeze and protect (no science yet)

**Gist:** take a snapshot of everything as it is now, so nothing can get lost while we move things around.

0.1 Commit `Volcano_Deformation/` (iclassic_insar.m and its gate) in `matlab/synthetic_deformation` (DennyChee/synthetic_data). It is currently **untracked**, and it was already lost once in Sept 2026. *(Needs your OK: it's another repo.)*
0.2 Commit or tag the current `vbICA` state, including `Code_denny/pila_bench`.
0.3 In PILA: commit the remaining useful untracked code on the current branch, then tag it `legacy-final-20261123`. That tag is the permanent pointer to every old campaign script.
0.4 Make a new branch `clean-restructure` from that tag. All refactoring happens there. `main` stays untouched until the end.
0.5 Mark old outputs as legacy:
   - rename `saved/` → `saved_legacy_20261123/` (a rename on the same filesystem: instant, nothing copied or deleted);
   - all new runs go to `runs/`, so old and new can never mix.
   *(Needs your OK: it's an `mv`.)*
0.6 Make a proposed list of what could be archived or deleted later (`.bak`, `scp_*`, tarballs, `_archive_*`, ~500 GB of closed experiments). This is a list only. Nothing is removed in this plan.

0.7 **GitHub cleanup (DennyChee/PILA).** History is linear: all 9 feature branches are ancestors of `multisource-k2-nonstatic-autopsy`, and 7 stacked PRs (#1–#7) are open.
   - fast-forward `main` to the legacy tag and push → GitHub marks PRs #1–#7 as merged;
   - delete the merged feature branches locally (`git branch -d`) and on myfork;
   - keep `origin` (yihshe) untouched;
   - keep both stashes (old main backups) until you say otherwise.

   From then on: `main` = legacy-final. The restructure lands on `main` through one PR from `clean-restructure`.

**Checks:** the tag exists on myfork; `git status` is clean apart from listed ignorables; `saved_legacy_*/` is readable.

---

## Phase 1 — Restructure the code (behaviour unchanged)

**Gist:** move the working parts into a tidy package and remove the copy-paste wrappers. The answers must stay identical.

Target layout:
```
pila/            core package: model/, physics/{mogi,sun69,okada}, trainer/, data/, utils/
scenarios/       generators + manifest.csv + truth schema + QA (no data in git)
methods/
  pila/          run_pila.py (one entry point, all options as config flags)
  mcmc/          export.py  -> iclassic_insar.m (external) -> import.py
  vbica/         export.py  -> vbICA (external) -> ic_to_source.py -> import.py
  supervised/    model.py, train.py, predict.py
compare/         score.py (single shared scoring), compare.py, figures/
experiments/<campaign>/   generator, configs, PBS templates (HPC-only, not for public)
configs/         priors (*_paras.json) + method templates only
paths.yaml       ONE place for data roots (MintPy dirs, vbICA/MCMC repo paths)
tests/           real unit + regression tests
```

1.1 Move the core (`base, model, physics, trainer, data_loader, datasets, utils, logger, parse_config`) into `pila/` and fix the imports. Leave out the RTM/HVAE/DPM upstream code (it stays reachable from the legacy tag).
1.2 Fold the 6 forked launchers into config flags of one `methods/pila/run_pila.py`. These are `train_pila_{keepbest,delayedpen,biasinit,logsources,basinhop}`, which today chain each other through `runpy` monkey-patching. The flags are `best_checkpoint`, `penalty_schedule`, `basin_hop`, `log_per_epoch_sources`, `init`.
1.3 Write a **regression test**: old launcher vs new flag on the same config and seed gives matching losses and params. If an old wrapper is not deterministic, check within tolerance instead.
1.4 Remove all hardcoded `/eos-rs/...` paths from code and configs. They come from `paths.yaml` (template committed, local copy gitignored).
1.5 Per-phase timing in every method (training / inference / decoding) in one `timing.json` format.

**Checks:** regression tests pass; a K=1 Mogi smoke run matches the legacy run of the same config; no `/eos-rs` string is left in `pila/`, `methods/` or `compare/`.

---

## Phase 2 — Common interfaces (the "fair comparison" contract)

**Gist:** every method reads the same input file and writes its answer in the same format, so one script can score them all the same way.

2.1 **Scenario format (one).** A scenario is an npz with:
   - LOS displacement `[T, H, W]` in **metres**, positive toward the satellite;
   - mask, dates, the LOS unit vector per pixel, local-ENU x/y (m), and the grid;
   - a `truth.json` in **one schema**: per-date `[T, K, P]` params in SI units plus model names (synthetic only).

   Real MintPy stacks are converted into this same format once, by `scenarios/real/import_mintpy.py` (applying multilook + ref_date as decided), so real and synthetic go down one path.
2.2 **Manifest** `scenarios/manifest.csv`: id, family, model(s), K, grid, track, noise model, SNR, file, truth, sha256.
2.3 **Method output format (one).** Every method writes `runs/<method>/<scenario>/<run_id>/`:
   - `params.npz`: per-date `[T, K, P]` in SI units, with posterior p5/p50/p95 where the method has them;
   - `recon.npz`: predicted LOS `[T, H, W]`;
   - `timing.json`, `meta.json` (code git hash, config, seed).
2.4 **vbICA → per-date params.** A moving source is split across **several ICs** (each IC has a fixed map). So the adapter:
   1. **Groups the ICs** that belong to the same deforming source.
   2. **Rebuilds that source's signal per date** as the sum of its ICs (spatial map × time course).
   3. **Inverts each date of that rebuilt signal** with the same per-epoch SA+MCMC used for the MCMC method.

   The grouping rule must be **honest (no truth)** and the same for every scenario. Candidates:
   - spatial overlap/correlation of the IC maps;
   - forward-model fit of the combined vs separate ICs;
   - a fixed misfit/ARD threshold.

   Noise/APS ICs are left out by the same rule. **We pick the rule together in Phase 4** and test it on the pilots. The vbICA run itself (N choice, S/T-mode) stays in the vbICA repo. The adapter only exports scenarios and reads `decomposedICA*.mat` back.
2.5 **Shared scoring** `compare/score.py`, generalised from `nonstatic_scoring.score_run`:
   - per-date per-source param errors (Hungarian matching for K≥2);
   - recon RMS vs data and vs clean truth;
   - detection (found / missed / spurious);
   - coverage of p5–p95 where a posterior exists;
   - timing.
2.6 **Standard figures, made automatically by compare.py for every scenario:** param recovery (truth / PILA / MCMC / vbICA / supervised, vbICA layout), recon maps, residuals, a timing table.

**Checks:** a round-trip export→import test for MCMC and vbICA on one clean scenario; every method on a noise-free K=1 scenario gives params within tight tolerance of truth.

---

## Phase 3 — Supervised method (ON HOLD — only make room for it now)

**Gist:** don't build it yet, but shape the code so adding it later is just "drop in one folder".

Done now, as part of Phases 1–2:
- `methods/supervised/` holds a README and an interface stub (`train(manifest_split, cfg)`, `predict(scenario) -> params.npz/recon.npz`).
- The scenario generator can produce **training sets** from the same prior box and noise model (a separate seed range, test hashes kept apart).
- The manifest has a `split` column (train/val/test).
- The existing DETR-lite (`model/multisource.py`, `train_multisource.py`) is kept in the legacy tag and moved under `methods/supervised/legacy_detr/`, not deleted.

Later, when we turn it on:

3.1 Extend the existing DETR-lite (`model/multisource.py`) to cover Mogi / Sun69 / Okada and K=0–3. Or, if simpler, use a separate head per model. *(ai-expert review.)*
3.2 Training data comes from the **same generator and the same prior box** as the synthetic scenarios, with noise drawn from each track's calibrated noise covariance. Test scenarios are never used for training: separate seeds, checked by hash.
3.3 Real data: one model per track/geometry, trained on synthetic data made with that track's LOS vectors, grid and noise. Applied per date (no labels on real data, as for every method).
3.4 Output in the common format (2.3). If the network gives a Gaussian per source, its mean and spread fill p50 and p5/p95.

**Checks:** held-out synthetic accuracy table; per-date prediction on one real track gives sensible maps.

---

## Phase 4 — Re-plan the synthetic tests (WITH YOU, before generating anything)

**Gist:** decide together which test scenarios we really need, then generate them all fresh with one generator.

To agree in a dedicated session:
- **Families:**
  - single-source static/moving (Mogi, Sun69, Okada);
  - multi-source K=2–3 (static + non-static);
  - SNR ladder;
  - COMSOL (non-analytic source, model-error test);
  - noise-only K=0 (false-alarm test).
- **Grid/geometry:** asc / desc / joint; noise model (ERA5-calibrated covariance + turbulence); number of seeds per cell.
- **Fairness rules applied to all methods** (proposed defaults):
  - the same prior box;
  - K given (oracle K) in the main track, with a "K unknown" track later;
  - no truth initialisation;
  - cold random starts;
  - the same number of restarts / best-of-N rule per method, chosen by each method's own misfit;
  - every method's wall-time reported, not equalised.

Then:
4.1 Regenerate every scenario with `scenarios/generate.py` from specs. Fill the manifest.
4.2 **QA gate** (independent agent + your review) before any inversion: units, sign, orientation, LOS geometry, truth round-trip, noise statistics vs real TA69.

---

## Phase 5 — Real case studies

**Gist:** prepare the real volcano datasets the same way for all four methods.

5.1 Confirm the list (SierraNegra TD128, Nyiragongo TA174/TD21, Etna TA44/TD124, LaPalma TD169/TA60, Krasheninnikov TA140/TD162, Agung TD32, COMSOL? Marapi TA69 needs reprocessing first, so it isn't on the list yet).
5.2 Use the `mintpy-inspector` agent to check each stack (network, tempCoh, reference point).
5.3 Import into the scenario format with recorded multilook, ref_date, reference pixel and model(s) to fit.

---

## Phase 6 — Pilots

**Gist:** run a few small jobs end-to-end for every method before spending the big compute.

6.1 Pilot matrix: one scenario per family plus one real track, × 3 methods (PILA, MCMC, vbICA), × 2 seeds.
6.2 Check: jobs finish; outputs follow the contract; the scorer and figures work; timings make sense; no method is helped or hurt by a pipeline bug. You review the pilot figures.
6.3 Fix, then freeze the code at a tagged commit. **Every full run records that hash.**

## Phase 7 — Full submission

**Gist:** submit everything from the frozen code, long jobs first, then score it all with the one compare script.

7.1 Generate the PBS jobs per method from the manifest. Show them to you, then submit (long independent jobs first, dependent ones last).
7.2 Run `compare.py` over the whole manifest: master table plus figures.

## Phase 8 — Public repo

**Gist:** publish a slim version others can install and run.

Contents: `pila/`, `methods/` adapters, `scenarios/` generators + specs, `compare/`, the example config, a small demo scenario, README / installation / `environment.yml`, tests. Leave out: `experiments/` PBS, legacy files, data, outputs. Squash into a fresh history on a new public repo or branch (your call later).

---

### Open decisions (later)
- vbICA IC-grouping rule (Phase 4).
- Public repo: fresh history vs `main` as-is (Phase 8).
