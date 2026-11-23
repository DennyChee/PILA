#!/bin/bash
# Usage:       tests/integration/smoke_all_methods.sh [scenario_id] [manifest]
#              (defaults: G1_N4_moving_n0000, runs/_pilot_scenarios/manifest.csv)
# Description: End-to-end smoke test of the comparison pipeline with TINY settings (plumbing, not science):
#              PILA (4 epochs) -> predict; MCMC (2 dates, NBay 2000, 1 start, offset fit) incl. a RESUME call that
#              must skip; vbICA (N=3, max_steps 800) -> IC fields -> grouping (oracle K) -> per-group MCMC (2 dates)
#              -> import; contract validation of every run; shared scorer; compare.py with figures and the
#              same-pixels check. Everything goes to a NEW directory runs/_smoke/<timestamp>/ (nothing overwritten).
# Requires:    conda env 'pila'; MATLAB at $MATLAB_BIN; paths.yaml with VBICA_REPO and MCMC_REPO.
# Date:        2026-11-23
set -e

SCENARIO="${1:-G1_N4_moving_n0000}"
MANIFEST="${2:-runs/_pilot_scenarios/manifest.csv}"
PILA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PILA_ROOT"
MATLAB_BIN="${MATLAB_BIN:-/usr/local/MATLAB/R2023b/bin/matlab}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="runs/_smoke/$STAMP"
RUNS="$OUT/runs"
MCMC_REPO=$(python3 -c "import sys; sys.path.insert(0,'.'); from pila.paths import resolve_tokens; print(resolve_tokens('\${MCMC_REPO}'))")
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1
echo "[0/7] smoke test of $SCENARIO -> $OUT (MCMC repo $MCMC_REPO)"
mkdir -p "$OUT"

echo "[1/7] PILA: config -> 4 epochs -> predict"
python methods/pila/make_config.py --scenario-id "$SCENARIO" --manifest "$MANIFEST" --epochs 4 --seed 1 \
    --save-dir "$OUT/pila_train" --out "$OUT/configs/pila.json" > /dev/null
python methods/pila/run_pila.py -c "$OUT/configs/pila.json" > "$OUT/pila_train.log" 2>&1
CKPT=$(ls -t "$OUT"/pila_train/"$SCENARIO"/*/models/model_best.pth | head -1)
python methods/pila/predict.py --checkpoint "$CKPT" --scenario-id "$SCENARIO" --out "$RUNS/pila/$SCENARIO/seed1" > /dev/null

echo "[2/7] MCMC: export -> 2 dates -> resume call (must skip) -> import"
MC="$RUNS/mcmc/$SCENARIO/smoke"
python -m methods.mcmc.export --scenario-id "$SCENARIO" --manifest "$MANIFEST" --offset fit --n-starts 1 --n-bay 2000 --out "$MC" > /dev/null
"$MATLAB_BIN" -batch "addpath('methods/mcmc'); run_iclassic_perdate('$MC', '$MCMC_REPO', [10 30])" > "$OUT/mcmc.log" 2>&1
"$MATLAB_BIN" -batch "addpath('methods/mcmc'); run_iclassic_perdate('$MC', '$MCMC_REPO', [10 30])" > "$OUT/mcmc_resume.log" 2>&1
grep -q "identical settings, skipping" "$OUT/mcmc_resume.log" || { echo "FAIL: resume did not skip"; exit 1; }
python -m methods.mcmc.import_results --run "$MC" --manifest "$MANIFEST" > /dev/null

echo "[3/7] vbICA: export -> decomposition N=3 (max_steps 800) -> IC fields"
VB="$RUNS/vbica/$SCENARIO/smoke"
python -m methods.vbica.export --scenario-id "$SCENARIO" --manifest "$MANIFEST" --out "$VB" --max-steps 800 > /dev/null
methods/vbica/run_vbica.sh "$VB" 3 1 > "$OUT/vbica.log" 2>&1

echo "[4/7] vbICA: grouping (oracle K) -> per-group MCMC (2 dates) -> import"
python -m methods.vbica.perepoch export --run "$VB" --manifest "$MANIFEST" --n-comp 3 --rule overlap_cluster \
    --offset fit --n-starts 1 --n-bay 2000 > "$OUT/vbica_group.log"
for GROUP in "$VB"/group_*; do
    "$MATLAB_BIN" -batch "addpath('methods/mcmc'); run_iclassic_perdate('$GROUP', '$MCMC_REPO', [10 30])" >> "$OUT/vbica_group.log" 2>&1
done
python -m methods.vbica.perepoch import --run "$VB" --manifest "$MANIFEST" >> "$OUT/vbica_group.log"

echo "[5/7] Contract validation"
for RUN in "$RUNS/pila/$SCENARIO/seed1" "$MC" "$VB"; do
    python -m methods.contract --validate "$RUN" | sed "s|^|  $(basename "$(dirname "$(dirname "$RUN")")"): |"
done

echo "[6/7] Shared scorer + compare.py (figures, same-pixels check)"
python -m compare.compare --runs-root "$RUNS" --out "$OUT/compare" --manifest "$MANIFEST" > "$OUT/compare.log" 2>&1
grep -E "WARNING|^ *(pila|mcmc|vbica) " "$OUT/compare.log" || true
python - "$OUT/compare/master_table.csv" <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1])
cols = ['method', 'n_points', 'pixel_mask_sha1', 'n_success', 'n_active', 'n_false_detection', 'rms_phys_vs_clean_mm', 'timing_total_s']
print(t[cols].to_string(index=False))
assert t['pixel_mask_sha1'].nunique() == 1 and t['n_points'].nunique() == 1, "methods fitted different pixel sets"
print("  same pixel set for all methods: PASS")
PY

echo "[7/7] Timing recorded per method"
python - "$RUNS" <<'PY'
import glob, json, sys
for path in sorted(glob.glob(f"{sys.argv[1]}/*/*/*/timing.json")):
    timing = json.load(open(path))
    print(f"  {path.split('/')[-4]:6s} total_s={timing.get('total_s')} train_s={timing.get('train_s')} infer_s={timing.get('infer_s')}")
PY
echo "Done. Output saved to $OUT/"
