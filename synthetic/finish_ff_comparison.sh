#!/bin/bash
# Usage:       bash synthetic/finish_ff_comparison.sh
# Description: Resume the global-vs-far-field standardization comparison after the 19
#              far-field PBS jobs finish. The interactive-node monitor dies when the
#              session ends, but the PBS jobs complete on compute nodes regardless — so
#              this just re-runs the aggregation + analysis the monitor would have done.
#              Safe to run repeatedly. Run from the PILA repo root.
# Date:        2026-06-22
set -e
cd "$(dirname "$0")/.."          # PILA repo root
source /eos-rs/miniconda3/etc/profile.d/conda.sh
conda activate pila
export MKL_THREADING_LAYER=GNU

B=marapi_mogi_buildup_combined
FF=synthetic/snr_sweep_ml1_ff

echo "=== [1] far-field jobs status ==="
DONE=$(ls -d ${FF}/eval/${B}_*/ 2>/dev/null | wc -l)
echo "  far-field runs with metrics: ${DONE} / 19"
STILL=$(qstat -u "$USER" 2>/dev/null | grep -c "_ff" || true)
echo "  _ff jobs still in PBS queue: ${STILL}"
if [ "$STILL" -gt 0 ]; then
  echo "  WARNING: ${STILL} jobs still running — results will be partial. Re-run later for the full set."
fi

echo "=== [2] aggregate far-field sweep ==="
python -m synthetic.aggregate_sweep --out ${FF} --base-name ${B} --multilook 1 \
  2>&1 | grep -vE "insar_mintpy|Loaded|Fine|coarse|Sanity|Using mask|height|Standardization|object"

echo "=== [3] per-epoch recovery-probability curve (far-field) ==="
python -m synthetic.perepoch_snr_diagnostic --out ${FF} --base-name ${B} \
  --base-spec synthetic/specs/${B}.json --multilook 1 \
  2>&1 | grep -vE "insar_mintpy|Loaded|Fine|coarse|Sanity|Using mask|height|Standardization|object"

echo "=== [4] failure analysis (logistic + detectability + scene-vs-local), far-field ==="
python -m synthetic.snr_failure_analysis --base-spec synthetic/specs/${B}.json \
  --base-name ${B} --out ${FF} --multilook 1 \
  2>&1 | grep -E "SNR50|predictor|detectable|min_dV|wrote|points"

echo "=== [5] GLOBAL vs FAR-FIELD comparison ==="
python3 - <<'PY'
import json, os
g='synthetic/snr_sweep_ml1/marapi_mogi_buildup_combined_failure_analysis.json'
f='synthetic/snr_sweep_ml1_ff/marapi_mogi_buildup_combined_failure_analysis.json'
if not (os.path.exists(g) and os.path.exists(f)):
    print("  missing failure_analysis.json (global or ff) — rerun once both complete."); raise SystemExit
G=json.load(open(g)); F=json.load(open(f))
print(f"{'':12}{'SNR50(scene)':>14}{'width':>8}{'AUC(scene)':>12}{'AUC(local)':>12}")
for name,d in [('GLOBAL',G),('FAR-FIELD',F)]:
    print(f"{name:12}{d['scene']['snr50_db']:>14.2f}{d['scene']['width_s_db']:>8.2f}"
          f"{d['scene']['auc']:>12.3f}{d['local']['auc']:>12.3f}")
print(f"\nSNR50 shift (global - ff): {G['scene']['snr50_db']-F['scene']['snr50_db']:+.2f} dB "
      f"(positive = far-field recovers at LOWER SNR = better)")
print(f"local-AUC change: {F['local']['auc']-G['local']['auc']:+.3f} "
      f"(positive = recovery tracks LOCAL SNR more under far-field)")
PY
echo "=== DONE ==="
echo "Curves: ${FF}/${B}_breakdown_snr.png, ${B}_perepoch_snr_recovery_fraction.png, ${B}_scene_vs_local_snr.png"
