#!/bin/bash
# Usage:       ./plot_insar_timeseries_batch.sh [VOLCANO_FILTER] [--gif]
#                e.g. ./plot_insar_timeseries_batch.sh lapalma --gif
#                     ./plot_insar_timeseries_batch.sh ""              # all volcanoes, no gif
# Description: Run plot_insar_timeseries.py on the LATEST model_best.pth of every InSAR
#              experiment under saved/ (optionally filtered by a substring of the saved/
#              subdir name). NO retraining — Stage A already trained on the whole series;
#              this only (re)generates the full-time-series figures.
# Date:        2026-06-18
set -e

# --- Config / args ---
VOLCANO_FILTER="${1:-}"                 # substring match on the saved/<exp> dir name (e.g. lapalma, etna)
GIF_FLAG=""
if [ "${2:-}" == "--gif" ] || [ "${1:-}" == "--gif" ]; then
    GIF_FLAG="--gif"
fi
# If the first arg was --gif, there is no volcano filter.
if [ "${VOLCANO_FILTER}" == "--gif" ]; then
    VOLCANO_FILTER=""
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVED_DIR="${PROJECT_DIR}/saved"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "Project:        ${PROJECT_DIR}"
echo "Volcano filter: '${VOLCANO_FILTER}' (empty = all)"
echo "GIF:            ${GIF_FLAG:-off}"

# --- Find the LATEST run per experiment that has a model_best.pth ---
# Layout: saved/<group>/<ExperimentName>/<timestamp>/models/model_best.pth
# We group runs by <ExperimentName> (the parent of the <timestamp> dir) and keep only the
# most recent <timestamp> (MMDD_HHMMSS, lexically sortable), so each experiment plots once.
echo "Scanning ${SAVED_DIR} for InSAR checkpoints (latest run per experiment) ..."
mapfile -t EXP_DIRS < <(find "${SAVED_DIR}" -type d -name models -exec test -f '{}/model_best.pth' ';' -print \
    | sed 's|/models$||' \
    | awk -F/ '{ grp=""; for(i=1;i<NF;i++) grp=grp $i "/"; ts=$NF;
                 if (ts > latest[grp]) { latest[grp]=ts; path[grp]=$0 } }
               END { for (g in path) print path[g] }' \
    | sort -u)

count=0
for EXP_DIR in "${EXP_DIRS[@]}"; do
    # Skip smoke-test dirs and any non-matching volcano.
    case "${EXP_DIR}" in
        *_smoke*) continue ;;
    esac
    if [ -n "${VOLCANO_FILTER}" ] && [[ "${EXP_DIR}" != *"${VOLCANO_FILTER}"* ]]; then
        continue
    fi

    CKPT="${EXP_DIR}/models/model_best.pth"
    # Only InSAR runs have an 'insar' block in their config; skip the rest (e.g. GNSS Mogi).
    CFG="${EXP_DIR}/models/config.json"
    if ! grep -q '"insar"' "${CFG}" 2>/dev/null; then
        continue
    fi

    echo "----------------------------------------------------------------"
    echo "[$((count + 1))] ${EXP_DIR}"
    "${PYTHON}" "${PROJECT_DIR}/plot_insar_timeseries.py" -r "${CKPT}" ${GIF_FLAG}
    count=$((count + 1))
done

echo "================================================================"
echo "Done. Generated full-time-series figures for ${count} InSAR run(s)."
echo "Figures are under each run's figures/ directory."
