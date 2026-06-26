# CLAUDE.md — PILA (InSAR source inversion)

Project-specific instructions for Claude Code. My global style/units/plotting
rules live in `~/.claude/CLAUDE.md` and still apply — this file only adds what is
specific to **PILA** and, importantly, the **safety guardrails** for this repo.

---

## 0. SAFETY FIRST (read this before doing anything destructive)

I run Claude Code locally and want to be cautious. Two layers protect this repo:

1. **Enforced layer** — `.claude/settings.local.json` (permission deny/ask/allow).
   The harness blocks these regardless of intent. Highlights:
   - **DENIED (hard block):** `rm -rf`, `rmdir`, `git reset --hard`, `git clean`,
     `git push --force`, `git checkout -- / .`, `git branch -D`, `dd`, `mkfs`,
     `sudo`, recursive `chmod/chown`, and any **write/edit** into
     `data/raw/`, `pretrained/`, or `.git/`.
   - **ASK (confirm each time):** any other `rm`, `git push`, `reset`, `rebase`,
     `checkout`, `merge`, `stash`, `mv`, package installs (`pip`/`conda`/`mamba`),
     and network commands (`curl`, `wget`, `ssh`, `scp`, `rsync`).
   - **ALLOWED freely:** read-only commands, git read/add/commit, `python`,
     editing/creating files anywhere in the project except the protected dirs.

2. **Advisory layer** — this file. Rules I expect you to follow even where the
   enforced layer can't reach.

### Behavioural rules (advisory — follow even though python is allowed)
- **NEVER** delete, overwrite, or rename anything under `data/raw/`,
  `data/processed/`, or `pretrained/`. These are inputs and trained checkpoints.
  > Caveat I understand: `python` is allowed without prompts, so a script *can*
  > technically remove protected files (`os.remove`, `shutil.rmtree`). Do not
  > write or run any code that deletes/overwrites raw data or pretrained models.
  > If a task seems to need it, STOP and ask me first.
- **Confirm the conda env before any install.** This project uses env **`pila`**
  (`conda activate pila`). The shell often starts in `base` — check
  `echo $CONDA_DEFAULT_ENV` first. Never `pip install` into `base`.
- **Don't commit, push, or rewrite git history unless I explicitly ask.** Branch
  before committing if we're on `main`.
- Generated output dirs (`saved/`, `*_out_*/`, `comparison_out*/`, `figures/`,
  `experiment_figures/`, `_outputs_archive_*/`) are reproducible — but still ask
  before bulk-deleting them.
- When in doubt about anything destructive or scientifically consequential: ask,
  don't guess.

---

## 1. What this project is

**PILA** = Physics-Informed Low-Rank Augmentation: a physics-informed VAE that
inverts geophysical forward models for interpretable source parameters. Paper:
https://arxiv.org/abs/2405.18953

Two study domains live here:
- **InSAR volcanic source inversion** — Mogi (point pressure source) and Okada
  (dislocation/dike) models, inverted from MintPy LOS displacement. See
  `README_InSAR.md` for the full scientific writeup.
- **RTM forest inversion** — radiative-transfer model on Sentinel-2 spectra.

The encoder reads a displacement field and infers physical source parameters; a
differentiable forward model re-renders the prediction; the mismatch trains the
encoder, with a low-rank learnable residual absorbing model incompleteness.

---

## 2. Key directories

| Path | What it is | Protection |
|------|-----------|-----------|
| `data/raw/` | GNSS, S2, EVI, allometry inputs (~90M) | **write/edit DENIED** |
| `data/processed/` | preprocessed mogi/rtm tensors | do not overwrite |
| `pretrained/` | trained AE/NN checkpoints | **write/edit DENIED** |
| `configs/phys_smpl/` | PILA experiment configs (JSON) | edit OK |
| `model/`, `physics/`, `trainer/`, `data_loader/`, `datasets/` | source code | edit OK |
| `saved/` | training run outputs (models, logs, configs) | generated |
| `.claude/backups/`, `.claude/logs/` | Claude Code local state | leave alone |

---

## 3. Environment

```bash
conda activate pila          # NOT base — confirm with: echo $CONDA_DEFAULT_ENV
```
Stack: PyTorch, numpy/scipy, matplotlib, h5py, rasterio, pyproj (see
`environment.yml`). Sentinel-1 C-band wavelength 0.0555 m. LOS convention:
**positive = motion toward the satellite**. MintPy cubes are in metres; physics
decoders work in **mm** (conversion happens at load).

---

## 4. How to run

```bash
# Training (PILA) — config-driven
python train_pila.py --config configs/phys_smpl/PILA_Mogi_C.json
# optional: -r <model.pth> to resume, -d <device>

# Curated example commands (uncomment what you need)
./run.sh

# Evaluation
python test_pila_mogi.py \
  --config saved/mogi/<EXP>/<MMDD_HHMMSS>/models/config.json \
  --resume saved/mogi/<EXP>/<MMDD_HHMMSS>/models/model_best.pth
```
All entrypoints are config-driven argparse scripts. New experiments = new JSON in
`configs/phys_smpl/`, don't hardcode paths in source.

---

## 5. Working style for this repo

- Follow my global rules: verbose unit-bearing names, heavy WHY-comments, numbered
  progress prints, units on every plot axis, `RdBu_r`/`coolwarm` for displacement.
- After writing or modifying any script, delegate a review to the
  **`insar-code-reviewer`** subagent. Use **`ai-expert`** for VAE/training/arch
  work and **`mintpy-inspector`** for any MintPy HDF5 output.
- State scientific assumptions (units, reference frame, thresholds) explicitly.
- No multiprocessing unless I ask; don't load full large rasters into RAM.
