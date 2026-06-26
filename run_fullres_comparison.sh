#!/bin/bash
# Usage:       ./run_fullres_comparison.sh
# Description: Final step for the FULL-RESOLUTION (multilook=1) definitive runs. Waits until
#              all 5 jobs have produced a checkpoint, then (a) re-runs the UQ-vs-full-scene
#              comparison against saved/full_scene_fullres/, and (b) prints each run's fit
#              R2/RMSE. Safe to run repeatedly; it only reads + writes comparison_out/.
# Date:        2026-06-24
set -e
cd /eos-rs/INSAR_processing/denny/PILA
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIGS=(SierraNegra_Sun69_InSAR_A Etna_Okada_TA44_A Etna_Okada_TD124_A \
         Nyiragongo_Okada_TA174_A Nyiragongo_Okada_TD21_A)
DIRS=(sierranegra_sun69_insar_a etna_okada_ta44_a etna_okada_td124_a \
      nyiragongo_okada_ta174_a nyiragongo_okada_td21_a)

# --- 1. Check all 5 full-res runs have finished (checkpoint + params.txt present) ---
echo "[1/3] Checking full-res run outputs ..."
missing=0
for d in "${DIRS[@]}"; do
  ck=$(find saved/full_scene_fullres/$d -name model_best.pth 2>/dev/null | sort | tail -1)
  pt=$(find saved/full_scene_fullres/$d -name "*_params.txt" 2>/dev/null | sort | tail -1)
  if [ -z "$ck" ] || [ -z "$pt" ]; then echo "  NOT READY: $d"; missing=1; else echo "  ok: $d"; fi
done
if [ "$missing" -ne 0 ]; then
  echo "Some runs are not finished yet (check 'qstat -u \$USER'). Re-run this script later."
  exit 1
fi

# --- 2. UQ (n=50) vs full-resolution full-scene comparison ---
echo "[2/3] Running UQ-vs-fullres comparison ..."
python compare_uq_vs_fullscene.py \
  --saved-root saved/full_scene_fullres \
  --out comparison_out/uq_vs_fullscene_fullres \
  --configs "${CONFIGS[@]}"

# --- 3. Fit quality (R2 / RMSE) per full-res run ---
echo "[3/3] Fit R2/RMSE per full-res run ..."
for d in "${DIRS[@]}"; do
  ck=$(find saved/full_scene_fullres/$d -name model_best.pth | sort | tail -1)
  echo "## $d"
  python plot_insar_results.py --resume "$ck" 2>&1 | grep -oE "R2=[-0-9.]+, RMSE=[0-9.]+ mm" | tail -1 | sed 's/^/   /'
done

echo "Done. Comparison in comparison_out/uq_vs_fullscene_fullres/"
