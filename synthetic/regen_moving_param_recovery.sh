#!/bin/bash
# Usage:       bash synthetic/regen_moving_param_recovery.sh
# Description: Regenerate the moving-source param-recovery figures (inference only,
#              no retraining) after the strong-signal shading was removed from
#              synthetic/plots.py. Re-runs synthetic.evaluate from each saved
#              checkpoint and copies the new *_param_recovery.{png,pdf} into
#              synthetic/figures_moving_source/inference/ with the published stems.
# Date:        2026-06-24
set -e

PROJ="/eos-rs/INSAR_processing/denny/PILA"
cd "$PROJ"

INFDIR="synthetic/figures_moving_source/inference"

# Each entry: <truth_json> | <checkpoint> | <eval_out_dir> | <published_stem>
# (constrained/loose reuse the combined truth sidecar -> stem renamed on copy)
CASES=(
  "synthetic/cubes/marapi_mogi_rising_combined/marapi_mogi_rising_combined_truth.json|saved/synth/marapi_mogi_rising_combined/marapi_mogi_rising_combined/0623_174715/models/model_best.pth|synthetic/eval/marapi_mogi_rising_combined|marapi_mogi_rising_combined_param_recovery"
  "synthetic/cubes/marapi_sun69_sill_combined/marapi_sun69_sill_combined_truth.json|saved/synth/marapi_sun69_sill_combined/marapi_sun69_sill_combined/0623_174908/models/model_best.pth|synthetic/eval/marapi_sun69_sill_combined|marapi_sun69_sill_combined_param_recovery"
  "synthetic/cubes/marapi_okada_dike_combined/marapi_okada_dike_combined_truth.json|saved/synth/marapi_okada_dike_combined/marapi_okada_dike_combined/0623_175128/models/model_best.pth|synthetic/eval/marapi_okada_dike_combined|marapi_okada_dike_combined_param_recovery"
  "synthetic/cubes/marapi_okada_dike_combined/marapi_okada_dike_combined_truth.json|saved/synth/marapi_okada_dike_constrained_combined/marapi_okada_dike_constrained_combined/0623_180851/models/model_best.pth|synthetic/eval/marapi_okada_dike_constrained_combined|marapi_okada_dike_CONSTRAINED_param_recovery"
  "synthetic/cubes/marapi_okada_dike_combined/marapi_okada_dike_combined_truth.json|saved/synth/marapi_okada_dike_loose_combined/marapi_okada_dike_loose_combined/0623_181643/models/model_best.pth|synthetic/eval/marapi_okada_dike_loose_combined|marapi_okada_dike_LOOSE_param_recovery"
)

for entry in "${CASES[@]}"; do
    IFS='|' read -r truth ckpt outdir stem <<< "$entry"
    echo ""
    echo "=== Regenerating: $stem ==="
    python -m synthetic.evaluate --truth "$truth" --ckpt "$ckpt" --out "$outdir"

    # The evaluate stem comes from truth['name'] (the combined cube name); copy the
    # freshly written recovery figure to the published name in the inference dir.
    src_stem="$outdir/$(python -c "import json,sys; print(json.load(open('$truth'))['name'])")_param_recovery"
    for ext in png pdf; do
        cp -f "${src_stem}.${ext}" "${INFDIR}/${stem}.${ext}"
        echo "  copied -> ${INFDIR}/${stem}.${ext}"
    done
done

echo ""
echo "Done. Regenerated 5 param-recovery figures in ${INFDIR}/"
