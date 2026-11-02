#!/bin/bash
# Usage:       bash stage_k1_dvbox_figures.sh        (from the project root, env pila)
# Description: Stages figures showing the data-scaled dV box at work on the weak K=1 album (generate_k1_dvbox_jobs.py):
#              1. equiv/rule summary CSVs (summarize_k1_snr_runs.py)
#              2. per-run 4-panel PILA-vs-truth reports (plot_multisource_report.py) into
#                 figures/multisource_k1_dvbox/{equiv,rule}/<album>/seed<S>/  (same format as figures/multisource_k1_snr/)
#              3. fixed-vs-scaled comparison figures (plot_k1_dvbox_compare.py)
#              plot_multisource_report.py exits non-zero on noisy albums (its truth==data check fails by design),
#              so its exit status is not treated as an error; missing PNGs are reported instead.
# Date:        2026-10-22
set -e

SAVE_ROOT="saved/multisource_k1_dvbox"
FIG_ROOT="figures/multisource_k1_dvbox"
NOISY_DIR="synthetic/multisource/pila_stacks_v2/albums"
SNR_DIR="synthetic/multisource/pila_stacks_v2/albums_snr"

echo "[1/3] Scoring runs..."
for arm in equiv rule; do
    python summarize_k1_snr_runs.py --root "$SAVE_ROOT/$arm" --out-csv "$FIG_ROOT/${arm}_summary.csv"
done

echo "[2/3] Per-run report figures..."
for arm in equiv rule; do
    for album_root in "$SAVE_ROOT/$arm"/k1snr_seed*/*; do
        seed_tag="$(basename "$(dirname "$album_root")")"          # k1snr_seed<S>
        album="$(basename "$album_root")"
        case "$album" in *_n[0-9][0-9][0-9]) truth="$SNR_DIR/${album}_truth.json" ;; *) truth="$NOISY_DIR/${album}_truth.json" ;; esac
        out_dir="$FIG_ROOT/$arm/$album/${seed_tag#k1snr_}"
        mkdir -p "$out_dir"
        python plot_multisource_report.py --album-root "$album_root" --truth "$truth" --fit-k 1 --out-dir "$out_dir" \
            > "$out_dir/report.log" 2>&1 || true
        n_png=$(ls "$out_dir"/*.png 2>/dev/null | wc -l)
        [ "$n_png" -lt 4 ] && echo "  WARNING: $out_dir has only $n_png PNGs (see report.log)"
    done
done

echo "[3/3] Fixed-vs-scaled comparison figures..."
python plot_k1_dvbox_compare.py --fig-root "$FIG_ROOT"
echo "Done. Output saved to $FIG_ROOT/"
