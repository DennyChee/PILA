#!/bin/bash
# Usage:       methods/vbica/run_vbica.sh <run_dir> <N_components> [whiten=1]
# Description: vbICA adapter, step 2: run ONE vbICA decomposition (vbICA's own driver, unchanged) inside the
#              self-contained working directory built by methods/vbica/export.py, then export every IC's
#              epoch-space field with vbICA's own reconstruction code (export_ic_fields.m). Nothing is written
#              into the vbICA repo. net_init = 'SVD' (deterministic) -> one seed, as the legacy bench.
# Outputs:     <run_dir>/workdir/Scenarios/PILA/<case>/matfiles_randinit_N<NN>_<wh|nowh>/decomposedICA_*.mat
#              <run_dir>/ic_fields_N<NN>.mat, <run_dir>/vbica_N<NN>.log
# Date:        2026-11-23
set -e

RUN_DIR="$(cd "$1" && pwd)"
N_COMPONENTS="$2"
WHITEN="${3:-1}"
PILA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MATLAB_BIN="${MATLAB_BIN:-/usr/local/MATLAB/R2023b/bin/matlab}"

echo "[1/3] Reading export.json in $RUN_DIR ..."
CASE=$(python3 -c "import json;print(json.load(open('$RUN_DIR/export.json'))['case'])")
SCEN_REL=$(python3 -c "import json;print(json.load(open('$RUN_DIR/export.json'))['scen_rel'])")
VBICA_REPO=$(python3 -c "import json;print(json.load(open('$RUN_DIR/export.json'))['vbica_repo'])")
NN=$(printf '%02d' "$N_COMPONENTS")
LOG="$RUN_DIR/vbica_N${NN}.log"

echo "[2/3] vbICA decomposition: case=$CASE N=$N_COMPONENTS whiten=$WHITEN (log $LOG) ..."
cd "$RUN_DIR/workdir"
export ICA_SCENARIO="$SCEN_REL" ICA_CASE="$CASE/" ICA_N_COMPONENTS="$N_COMPONENTS" ICA_WHITENING="$WHITEN" \
       ICA_SEEDS=1 ICA_FORCE_SERIAL=1
# vbICA's own review plot (plot_comp) saves figures under an ABSOLUTE path inside the vbICA repo, which does not
# exist for adapter cases, so the driver errors AFTER the decomposition is saved. Nothing is written into the vbICA
# repo; the plot is a review artefact. So: tolerate the driver's exit code, then REQUIRE the decomposition file.
"$MATLAB_BIN" -batch "addpath('$VBICA_REPO'); vbICA_driver_testing" > "$LOG" 2>&1 || \
    echo "  NOTE: vbICA driver exited non-zero (expected: plot_comp figure path); checking the decomposition file ..."
WH_TAG=$([ "$WHITEN" = "1" ] && echo wh || echo nowh)
DECOMP="$RUN_DIR/workdir/$SCEN_REL$CASE/matfiles_randinit_N${NN}_${WH_TAG}/decomposedICA_data_randinit_N${NN}_${WH_TAG}.mat"
RANK=$(grep -oE "Dimension reduced to [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$" || true)
if [ -n "$RANK" ] && [ "$RANK" -lt "$N_COMPONENTS" ]; then
    echo "ERROR: N exceeds data rank: vbICA kept only $RANK dimension(s) for N=$N_COMPONENTS; rerun with N <= $RANK"
    exit 2
fi
if [ ! -f "$DECOMP" ] || ! grep -q "driver total" "$LOG"; then
    echo "ERROR: vbICA decomposition missing ($DECOMP); see $LOG"
    exit 1
fi
if grep -q "Reached maximum of" "$LOG"; then
    echo "  WARNING: vbICA hit max_steps (not converged) -- recorded in ic_fields as converged=0"
fi

echo "[3/3] Exporting IC fields with vbICA's own reconstruction ..."
"$MATLAB_BIN" -batch "addpath(genpath('$VBICA_REPO/Code')); addpath('$PILA_ROOT/methods/vbica'); export_ic_fields('$RUN_DIR', $N_COMPONENTS, $WHITEN)" >> "$LOG" 2>&1
echo "Done. Output saved to $RUN_DIR/ic_fields_N${NN}.mat"
